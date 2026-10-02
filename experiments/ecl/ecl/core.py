"""Core, model-agnostic operations used by ECL.

The functions in this file deliberately avoid importing the VLA repository so
that the mathematical parts can be unit-tested on CPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import re
from typing import Iterable, Iterator, Sequence

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


# LIBERO-Core contains ten fixed instructions.  Keeping the mapping explicit is
# preferable to a heuristic parser that may accidentally select the receptacle
# (for example, the rack) instead of the manipulated object (the wine bottle).
TASK_TARGETS = {
    "pick up the black bowl next to the plate and place it on the plate": "black bowl",
    "pick up the black bowl next to the cookie box and place it on the plate": "black bowl",
    "pick up the black bowl on the cookie box and place it on the plate": "black bowl",
    "pick up the ketchup and place it in the basket": "ketchup",
    "pick up the alphabet soup and place it in the basket": "alphabet soup",
    "push the plate to the front of the stove": "plate",
    "put the bowl on top of the cabinet": "bowl",
    "put the cream cheese in the bowl": "cream cheese",
    "put the wine bottle on top of the cabinet": "wine bottle",
    "put the wine bottle on the rack": "wine bottle",
}


def normalize_instruction(instruction: str) -> str:
    """Return a canonical, whitespace-normalized instruction."""

    return " ".join(instruction.strip().lower().rstrip(".").split())


def target_phrase_for_instruction(instruction: str) -> str:
    """Resolve the manipulated object phrase for a LIBERO-Core instruction.

    An exception is intentional: silently choosing a wrong object invalidates
    the relational masking claim and is worse than stopping early.
    """

    normalized = normalize_instruction(instruction)
    if normalized in TASK_TARGETS:
        return TASK_TARGETS[normalized]
    raise KeyError(
        f"Unsupported instruction: {instruction!r}. Add an audited mapping to "
        "TASK_TARGETS before using this task."
    )


def _subsequence_starts(sequence: Sequence[int], pattern: Sequence[int]) -> Iterator[int]:
    if not pattern or len(pattern) > len(sequence):
        return
    for start in range(len(sequence) - len(pattern) + 1):
        if list(sequence[start : start + len(pattern)]) == list(pattern):
            yield start


def object_token_mask(input_ids: torch.Tensor, tokenizer, target_phrase: str) -> torch.Tensor:
    """Locate a target phrase in an already-tokenized prompt.

    Several boundary variants are checked because byte-level tokenizers can
    tokenize a word differently at the beginning of a string and after a space.
    The longest successful variant is used.
    """

    if input_ids.ndim != 1:
        raise ValueError(f"Expected one token sequence, got shape={tuple(input_ids.shape)}")

    ids = input_ids.detach().cpu().tolist()
    variants = (
        target_phrase,
        f" {target_phrase}",
        f"the {target_phrase}",
        f" the {target_phrase}",
    )
    matches: list[tuple[int, int]] = []
    for text in variants:
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        for start in _subsequence_starts(ids, token_ids):
            matches.append((start, len(token_ids)))

    if not matches:
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        raise ValueError(
            f"Could not locate target phrase {target_phrase!r} in tokenized prompt: {decoded!r}"
        )

    start, length = max(matches, key=lambda item: item[1])
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mask[start : start + length] = True
    return mask


@dataclass(frozen=True)
class RelationalSelectorConfig:
    """Configuration for attention-derived relational token selection."""

    top_k: int = 0
    top_ratio: float = 0.10
    last_n_layers: int = 4
    eps: float = 1e-8

    def resolve_k(self, num_patches: int) -> int:
        if self.top_k > 0:
            return min(self.top_k, num_patches)
        if not 0.0 < self.top_ratio <= 1.0:
            raise ValueError("top_ratio must lie in (0, 1]")
        return max(1, min(num_patches, round(num_patches * self.top_ratio)))


def _original_to_multimodal(indices: torch.Tensor, num_patches: int) -> torch.Tensor:
    """Map text-token indices to Prismatic's [token0, image, rest] layout."""

    return torch.where(indices == 0, indices, indices + num_patches)


def relation_topk_mask(
    attentions: Iterable[torch.Tensor],
    object_mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_patches: int,
    action_token_begin_idx: int,
    action_token_end_idx: int,
    config: RelationalSelectorConfig,
    sample_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k relational patch mask and dense relational scores.

    The action query is the position *predicting* each action label (j-1), not
    the row of the ground-truth action token itself.  This prevents teacher-
    forcing label leakage in the selector.
    """

    usable = [attention for attention in attentions if attention is not None]
    if not usable:
        raise RuntimeError(
            "The model returned no attention tensors. ECL requires eager/SDPA "
            "attention with output_attentions=True; FlashAttention cannot be used for the selector."
        )
    selected = usable[-config.last_n_layers :] if config.last_n_layers > 0 else usable

    # Each layer is [batch, heads, query, key]. Accumulate instead of stacking
    # all layers to avoid an unnecessary large temporary tensor.
    averaged = None
    for layer_attention in selected:
        if layer_attention.ndim != 4:
            raise ValueError(f"Unexpected attention shape: {tuple(layer_attention.shape)}")
        layer_mean = layer_attention.detach().float().mean(dim=1)
        averaged = layer_mean if averaged is None else averaged + layer_mean
    averaged = averaged / len(selected)

    batch_size, sequence_length, key_length = averaged.shape
    if sequence_length != key_length:
        raise ValueError("ECL currently expects full causal self-attention matrices")
    if object_mask.shape != labels.shape:
        raise ValueError("object_mask and labels must use the original text-token layout")
    if sample_gate is None:
        sample_gate = torch.ones(batch_size, dtype=torch.bool, device=labels.device)
    else:
        sample_gate = sample_gate.to(device=labels.device, dtype=torch.bool)

    patch_slice = slice(1, 1 + num_patches)
    if 1 + num_patches > key_length:
        raise ValueError(
            f"num_patches={num_patches} is incompatible with attention length={key_length}"
        )

    dense_scores = torch.zeros(
        (batch_size, num_patches), dtype=torch.float32, device=averaged.device
    )
    topk_mask = torch.zeros_like(dense_scores, dtype=torch.bool)
    k = config.resolve_k(num_patches)

    for batch_index in range(batch_size):
        if not bool(sample_gate[batch_index]):
            continue

        object_indices = object_mask[batch_index].nonzero(as_tuple=False).flatten()
        action_label_indices = (
            (labels[batch_index] > action_token_begin_idx)
            & (labels[batch_index] < action_token_end_idx)
        ).nonzero(as_tuple=False).flatten()
        if object_indices.numel() == 0:
            raise ValueError(f"Sample {batch_index} has no target-object tokens")
        if action_label_indices.numel() == 0:
            raise ValueError(f"Sample {batch_index} has no supervised action tokens")

        # Causal logits at position j-1 predict the label at j.
        action_query_indices = torch.clamp(action_label_indices - 1, min=0)
        object_queries = _original_to_multimodal(object_indices, num_patches)
        action_queries = _original_to_multimodal(action_query_indices, num_patches)

        if object_queries.max() >= sequence_length or action_queries.max() >= sequence_length:
            raise ValueError("Mapped query index lies outside the multimodal attention matrix")

        object_to_patch = averaged[batch_index, object_queries, patch_slice].mean(dim=0)
        action_to_patch = averaged[batch_index, action_queries, patch_slice].mean(dim=0)
        object_to_patch = object_to_patch / object_to_patch.sum().clamp_min(config.eps)
        action_to_patch = action_to_patch / action_to_patch.sum().clamp_min(config.eps)
        relation = object_to_patch * action_to_patch
        dense_scores[batch_index] = relation
        chosen = relation.topk(k=k, largest=True, sorted=False).indices
        topk_mask[batch_index, chosen] = True

    return topk_mask, dense_scores


@contextmanager
def masked_projector_output(projector, patch_mask: torch.Tensor, mode: str = "mean"):
    """Temporarily replace selected projected visual tokens during one forward."""

    if mode not in {"mean", "zero"}:
        raise ValueError("mask mode must be 'mean' or 'zero'")

    def hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise TypeError("The Prismatic projector must return [batch, patches, hidden]")
        local_mask = patch_mask.to(device=output.device, dtype=torch.bool)
        if local_mask.shape != output.shape[:2]:
            raise ValueError(
                f"Patch mask shape {tuple(local_mask.shape)} != projector output {tuple(output.shape[:2])}"
            )
        if mode == "mean":
            keep = (~local_mask).to(output.dtype).unsqueeze(-1)
            replacement = (output * keep).sum(dim=1, keepdim=True) / keep.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
        else:
            replacement = torch.zeros_like(output[:, :1])
        return torch.where(local_mask.unsqueeze(-1), replacement, output)

    handle = projector.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def aligned_action_logits_and_labels(
    logits: torch.Tensor, labels: torch.Tensor, num_patches: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align Prismatic multimodal causal logits with original text labels."""

    # Original label j (j>=1) is inserted at multimodal position P+j and is
    # predicted by the logit at P+j-1. This is the same alignment used by the
    # upstream VLA training metrics.
    expected = labels.shape[1] - 1
    aligned_logits = logits[:, num_patches : num_patches + expected]
    aligned_labels = labels[:, 1 : 1 + aligned_logits.shape[1]].to(logits.device)
    return aligned_logits, aligned_labels


def action_token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_patches: int,
    action_token_begin_idx: int,
    action_token_end_idx: int,
    sample_gate: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mean CE over supervised action tokens, restricted to valid action bins."""

    aligned_logits, aligned_labels = aligned_action_logits_and_labels(logits, labels, num_patches)
    token_mask = (
        (aligned_labels > action_token_begin_idx)
        & (aligned_labels < action_token_end_idx)
    )
    if sample_gate is not None:
        token_mask &= sample_gate.to(token_mask.device, dtype=torch.bool).unsqueeze(1)
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be mean or sum")
    if not bool(token_mask.any()):
        return logits.sum() * 0.0

    vocab_start = action_token_begin_idx + 1
    action_logits = aligned_logits[..., vocab_start:action_token_end_idx]
    action_targets = aligned_labels - vocab_start
    return F.cross_entropy(
        action_logits[token_mask],
        action_targets[token_mask],
        reduction=reduction,
    )


@torch.no_grad()
def action_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_patches: int,
    action_token_begin_idx: int,
    action_token_end_idx: int,
) -> torch.Tensor:
    aligned_logits, aligned_labels = aligned_action_logits_and_labels(logits, labels, num_patches)
    mask = (
        (aligned_labels > action_token_begin_idx)
        & (aligned_labels < action_token_end_idx)
    )
    if not bool(mask.any()):
        return torch.zeros((), device=logits.device)
    vocab_start = action_token_begin_idx + 1
    predictions = aligned_logits[..., vocab_start:action_token_end_idx].argmax(dim=-1)
    predictions = predictions + vocab_start
    return (predictions[mask] == aligned_labels[mask]).float().mean()


def unwrap_model(model):
    """Return the underlying OpenVLA module from DDP/FSDP wrappers."""

    current = model
    seen: set[int] = set()
    while hasattr(current, "module") and id(current) not in seen:
        seen.add(id(current))
        current = current.module
    return current


def request_eager_attention(model) -> None:
    """Request an attention backend capable of returning attention weights."""

    base = unwrap_model(model)
    llm = base.llm_backbone.llm
    for config in (getattr(llm, "config", None), getattr(base, "config", None)):
        if config is None:
            continue
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "eager"
