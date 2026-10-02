"""ECL training model for OpenPI's PyTorch pi0.5 implementation.

This module keeps inference identical to ``PI0Pytorch`` and replaces only the
training loss.  The selector pass is gradient-free; factual and masked passes
share the same flow noise and timestep.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers.models.gemma import modeling_gemma

from openpi.models import tokenizer as _tokenizer
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks


TASKS_AND_COUNTS = (
    ("pick up the black bowl next to the plate and place it on the plate", 46),
    ("pick up the black bowl next to the cookie box and place it on the plate", 28),
    ("pick up the black bowl on the cookie box and place it on the plate", 19),
    ("pick up the ketchup and place it in the basket", 15),
    ("pick up the alphabet soup and place it in the basket", 11),
    ("push the plate to the front of the stove", 9),
    ("put the bowl on top of the cabinet", 8),
    ("put the cream cheese in the bowl", 7),
    ("put the wine bottle on top of the cabinet", 6),
    ("put the wine bottle on the rack", 5),
)


class PI05ECLPytorch(PI0Pytorch):
    """pi0.5 with the paper's ECL objective and relation-aware token masking."""

    def __init__(
        self,
        config,
        *,
        contrast_lambda: float = 0.3,
        mu: float = 0.3,
        top_ratio: float = 0.10,
        selector_last_n_layers: int = 4,
        warmup_steps: int = 0,
    ):
        if not config.pi05:
            raise ValueError("PI05ECLPytorch requires Pi0Config(pi05=True)")
        super().__init__(config)
        if not 0.0 < top_ratio <= 1.0:
            raise ValueError("top_ratio must lie in (0, 1]")
        if selector_last_n_layers <= 0:
            raise ValueError("selector_last_n_layers must be positive")
        self.ecl_lambda = float(contrast_lambda)
        self.ecl_mu = float(mu)
        self.ecl_top_ratio = float(top_ratio)
        self.ecl_selector_last_n_layers = int(selector_last_n_layers)
        self.ecl_warmup_steps = int(warmup_steps)

        tokenizer = _tokenizer.PaligemmaTokenizer(config.max_token_len)
        task_tokens = []
        task_masks = []
        counts = []
        for prompt, count in TASKS_AND_COUNTS:
            tokens, mask = tokenizer.tokenize(prompt)
            task_tokens.append(torch.as_tensor(tokens, dtype=torch.long))
            task_masks.append(torch.as_tensor(mask, dtype=torch.bool))
            counts.append(count)
        self.register_buffer("ecl_task_tokens", torch.stack(task_tokens), persistent=False)
        self.register_buffer("ecl_task_masks", torch.stack(task_masks), persistent=False)
        rarity = [(max(counts) - count) / (max(counts) - min(counts)) for count in counts]
        self.register_buffer("ecl_task_rarity", torch.tensor(rarity, dtype=torch.float32), persistent=False)

    def _rarity_from_prompt(self, tokens: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        expected_tokens = self.ecl_task_tokens.to(tokens.device)
        expected_masks = self.ecl_task_masks.to(masks.device)
        matches = (
            (tokens[:, None, :] == expected_tokens[None, :, :])
            & (masks[:, None, :] == expected_masks[None, :, :])
        ).all(dim=-1)
        if not bool(matches.any(dim=1).all()):
            bad = (~matches.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"ECL received prompts outside the ten LIBERO-Core tasks at batch rows {bad}")
        task_ids = matches.to(torch.int64).argmax(dim=1)
        return self.ecl_task_rarity.to(tokens.device)[task_ids]

    @torch.no_grad()
    def _relation_scores(
        self,
        prefix_embs: torch.Tensor,
        suffix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        adarms_cond: torch.Tensor,
        instruction_queries: torch.Tensor,
        action_queries: torch.Tensor,
        visual_token_count: int,
    ) -> torch.Tensor:
        """Average the last layers' instruction/action attention to visual tokens."""
        model_pair = [
            self.paligemma_with_expert.paligemma.language_model,
            self.paligemma_with_expert.gemma_expert.model,
        ]
        hidden_pair = [prefix_embs.detach(), suffix_embs.detach()]
        num_layers = self.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers
        start_layer = max(0, num_layers - self.ecl_selector_last_n_layers)
        instruction_sum = None
        action_sum = None

        for layer_idx in range(num_layers):
            queries, keys, values, gates = [], [], [], []
            for branch, hidden in enumerate(hidden_pair):
                layer = model_pair[branch].layers[layer_idx]
                normalized, gate = layer.input_layernorm(hidden, cond=[None, adarms_cond][branch])
                gates.append(gate)
                shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
                queries.append(layer.self_attn.q_proj(normalized).view(shape).transpose(1, 2))
                keys.append(layer.self_attn.k_proj(normalized).view(shape).transpose(1, 2))
                values.append(layer.self_attn.v_proj(normalized).view(shape).transpose(1, 2))

            query_states = torch.cat(queries, dim=2)
            key_states = torch.cat(keys, dim=2)
            value_states = torch.cat(values, dim=2)
            dummy = torch.zeros(
                query_states.shape[0], query_states.shape[2], query_states.shape[-1],
                device=query_states.device, dtype=query_states.dtype,
            )
            rotary = self.paligemma_with_expert.paligemma.model.language_model.rotary_emb
            cos, sin = rotary(dummy, position_ids)
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )
            attn_output, attn_weights = modeling_gemma.eager_attention_forward(
                model_pair[0].layers[layer_idx].self_attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                model_pair[0].layers[layer_idx].self_attn.scaling,
            )

            if layer_idx >= start_layer:
                mean_attention = attn_weights.float().mean(dim=1)
                visual_attention = mean_attention[:, :, :visual_token_count]

                def reduce_queries(query_mask: torch.Tensor) -> torch.Tensor:
                    weights = query_mask.to(visual_attention.dtype)
                    return (visual_attention * weights[:, :, None]).sum(dim=1) / weights.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1.0)

                instruction_layer = reduce_queries(instruction_queries)
                action_layer = reduce_queries(action_queries)
                instruction_sum = instruction_layer if instruction_sum is None else instruction_sum + instruction_layer
                action_sum = action_layer if action_sum is None else action_sum + action_layer

            batch_size = query_states.shape[0]
            attn_output = attn_output.reshape(batch_size, -1, query_states.shape[1] * query_states.shape[-1])
            next_pair = []
            offset = 0
            for branch, hidden in enumerate(hidden_pair):
                layer = model_pair[branch].layers[layer_idx]
                end = offset + hidden.shape[1]
                projected = layer.self_attn.o_proj(attn_output[:, offset:end].to(layer.self_attn.o_proj.weight.dtype))
                residual = modeling_gemma._gated_residual(hidden, projected, gates[branch])
                normalized, gate = layer.post_attention_layernorm(
                    residual, cond=[None, adarms_cond][branch]
                )
                normalized = normalized.to(layer.mlp.up_proj.weight.dtype)
                next_pair.append(modeling_gemma._gated_residual(residual, layer.mlp(normalized), gate))
                offset = end
            hidden_pair = next_pair

        layers_used = num_layers - start_layer
        instruction_score = instruction_sum / layers_used
        action_score = action_sum / layers_used
        instruction_score = instruction_score / instruction_score.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        action_score = action_score / action_score.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return instruction_score * action_score

    def _velocity(
        self,
        prefix_embs: torch.Tensor,
        suffix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :].float()
        return self.action_out_proj(suffix_out)

    def forward(self, observation, actions, *, global_step: int = 0):
        if global_step < self.ecl_warmup_steps:
            return super().forward(observation, actions)

        images, image_masks, language, language_mask, state = self._preprocess_observation(observation, train=True)
        noise = self.sample_noise(actions.shape, actions.device)
        time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * actions
        target_velocity = noise - actions

        prefix, prefix_pad, prefix_ar = self.embed_prefix(images, image_masks, language, language_mask)
        suffix, suffix_pad, suffix_ar, adarms_cond = self.embed_suffix(state, x_t, time)
        parameter_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        prefix = prefix.to(parameter_dtype)
        suffix = suffix.to(parameter_dtype)
        pad_mask = torch.cat([prefix_pad, suffix_pad], dim=1)
        ar_mask = torch.cat([prefix_ar, suffix_ar], dim=1)
        attention_mask = self._prepare_attention_masks_4d(make_att_2d_masks(pad_mask, ar_mask))
        position_ids = torch.cumsum(pad_mask, dim=1) - 1

        visual_count = prefix.shape[1] - language.shape[1]
        total_tokens = pad_mask.shape[1]
        instruction_queries = torch.zeros_like(pad_mask)
        instruction_queries[:, visual_count : prefix.shape[1]] = language_mask
        action_queries = torch.zeros_like(pad_mask)
        action_queries[:, prefix.shape[1] :] = suffix_pad
        relation = self._relation_scores(
            prefix, suffix, attention_mask, position_ids, adarms_cond,
            instruction_queries, action_queries, visual_count,
        )

        visual_valid = prefix_pad[:, :visual_count]
        relation = relation.masked_fill(~visual_valid, -torch.inf)
        selected = torch.zeros_like(visual_valid)
        for row in range(selected.shape[0]):
            k = max(1, round(int(visual_valid[row].sum()) * self.ecl_top_ratio))
            chosen = relation[row].topk(k=k, largest=True, sorted=False).indices
            selected[row, chosen] = True

        visual_tokens = prefix[:, :visual_count]
        keep = visual_valid & ~selected
        replacement = (visual_tokens * keep[:, :, None]).sum(dim=1, keepdim=True) / keep.sum(
            dim=1, keepdim=True
        ).clamp_min(1).to(visual_tokens.dtype)[:, :, None]
        masked_visual = torch.where(selected[:, :, None], replacement, visual_tokens)
        masked_prefix = torch.cat([masked_visual, prefix[:, visual_count:]], dim=1)

        factual_velocity = self._velocity(prefix, suffix, attention_mask, position_ids, adarms_cond)
        masked_velocity = self._velocity(masked_prefix, suffix, attention_mask, position_ids, adarms_cond)
        effect_velocity = factual_velocity - self.ecl_lambda * masked_velocity
        factual_loss = F.mse_loss(factual_velocity, target_velocity, reduction="none").mean(dim=(-1, -2))
        masked_loss = F.mse_loss(masked_velocity, target_velocity, reduction="none").mean(dim=(-1, -2))
        effect_loss = F.mse_loss(effect_velocity, target_velocity, reduction="none").mean(dim=(-1, -2))
        rarity = self._rarity_from_prompt(language, language_mask).to(effect_loss.dtype)
        return factual_loss + self.ecl_mu * masked_loss + rarity * effect_loss
