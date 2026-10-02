"""ECL training loop built on the repository's DDP/FSDP strategies."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from .core import (
    RelationalSelectorConfig,
    action_token_accuracy,
    masked_projector_output,
    relation_topk_mask,
    request_eager_attention,
    unwrap_model,
)
from .runtime import move_to_device
from .losses import counterfactual_logits, weighted_action_ce


@dataclass(frozen=True)
class ECLTrainConfig:
    contrast_lambda: float = 0.3
    mu: float = 0.3
    stop_gradient: bool = False
    warmup_epochs: int = 2
    mask_mode: str = "mean"
    selector: RelationalSelectorConfig = RelationalSelectorConfig()
    save_interval: int = 2500
    max_steps: int | None = None


def _rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _write_metric(path: Path, record: dict) -> None:
    if not _rank_zero():
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _global_mean(value: torch.Tensor) -> float:
    result = value.detach().float().clone()
    if dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= dist.get_world_size()
    return result.cpu().item()


def _global_sum_int(value: torch.Tensor) -> int:
    result = value.detach().to(dtype=torch.long).clone()
    if dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return int(result.cpu().item())


def train_ecl(
    *,
    strategy,
    dataset,
    collator,
    action_tokenizer,
    run_dir: Path,
    epochs: int,
    config: ECLTrainConfig,
    start_step: int = 0,
) -> None:
    """Train ECL while retaining ordinary BC on every LT sample."""

    dataloader = DataLoader(
        dataset,
        batch_size=strategy.per_device_batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
        worker_init_fn=strategy.worker_init_fn,
    )
    steps_per_epoch = max(1.0, len(dataset) / strategy.global_batch_size)
    total_steps = max(1, (len(dataset) * epochs) // strategy.global_batch_size)
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)
    if not 0 <= start_step < total_steps:
        raise ValueError(f"start_step={start_step} must be in [0, {total_steps})")
    warmup_steps = int(config.warmup_epochs * steps_per_epoch)

    base_model = unwrap_model(strategy.vlm)
    projector = base_model.projector
    num_patches = int(base_model.vision_backbone.num_patches)
    request_eager_attention(strategy.vlm)
    device = torch.device("cuda", strategy.device_id)
    metrics_path = run_dir / "ecl_metrics.jsonl"

    strategy.vlm.train()
    strategy.optimizer.zero_grad(set_to_none=True)
    accumulation_steps = int(strategy.grad_accumulation_steps)
    if accumulation_steps < 1:
        raise ValueError(f"Invalid gradient accumulation steps: {accumulation_steps}")
    progress = tqdm(total=total_steps - start_step, disable=not _rank_zero(), desc="ECL")
    last_saved_step = start_step
    optimizer_step = start_step
    micro_step = 0

    for raw_batch in dataloader:
        if optimizer_step >= total_steps:
            break
        micro_step += 1
        global_step = optimizer_step + 1
        batch = move_to_device(raw_batch, device)
        method_active = global_step > warmup_steps
        gate = batch["contrast_gate"] if method_active else torch.zeros_like(batch["contrast_gate"])
        started = time.perf_counter()

        # All ranks execute the same selector/negative-forward schedule. This is
        # necessary for FSDP even when a phase-gated local batch has zero active samples.
        if method_active:
            with torch.no_grad(), torch.autocast(
                "cuda",
                dtype=strategy.mixed_precision_dtype,
                enabled=strategy.enable_mixed_precision_training,
            ):
                selector_output = strategy.vlm(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=None,
                    output_attentions=True,
                    return_dict=True,
                )
            patch_mask, relation_scores = relation_topk_mask(
                selector_output.attentions,
                batch["object_token_mask"],
                batch["labels"],
                num_patches=num_patches,
                action_token_begin_idx=action_tokenizer.action_token_begin_idx,
                action_token_end_idx=action_tokenizer.action_token_end_idx,
                config=config.selector,
                sample_gate=gate,
            )
            del selector_output
        else:
            patch_mask = torch.zeros(
                (batch["input_ids"].shape[0], num_patches), dtype=torch.bool, device=device
            )
            relation_scores = torch.zeros_like(patch_mask, dtype=torch.float32)

        with torch.autocast(
            "cuda",
            dtype=strategy.mixed_precision_dtype,
            enabled=strategy.enable_mixed_precision_training,
        ):
            positive = strategy.vlm(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                return_dict=True,
            )
            loss_full = positive.loss

            if method_active:
                with masked_projector_output(projector, patch_mask, config.mask_mode):
                    negative = strategy.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=None,
                        return_dict=True,
                    )

                loss_mask = weighted_action_ce(
                    negative.logits,
                    batch["labels"],
                    gate,
                    num_patches=num_patches,
                    action_tokenizer=action_tokenizer,
                )
                contrastive_logits = counterfactual_logits(
                    positive.logits, negative.logits,
                    config.contrast_lambda, config.stop_gradient,
                )
                loss_cf = weighted_action_ce(
                    contrastive_logits,
                    batch["labels"],
                    gate * batch["task_rarity"],
                    num_patches=num_patches,
                    action_tokenizer=action_tokenizer,
                )
                loss = loss_full + config.mu * loss_mask + loss_cf
                metric_logits = positive.logits
            else:
                loss_mask = loss_full.detach() * 0.0
                loss_cf = loss_full.detach() * 0.0
                loss = loss_full
                metric_logits = positive.logits

        (loss / accumulation_steps).backward()
        if micro_step % accumulation_steps != 0:
            continue
        strategy.clip_grad_norm()
        strategy.optimizer.step()
        strategy.lr_scheduler.step()
        strategy.optimizer.zero_grad(set_to_none=True)
        optimizer_step = global_step

        accuracy = action_token_accuracy(
            metric_logits.detach(),
            batch["labels"],
            num_patches=num_patches,
            action_token_begin_idx=action_tokenizer.action_token_begin_idx,
            action_token_end_idx=action_tokenizer.action_token_end_idx,
        )
        active_count = _global_sum_int(gate.sum())
        record = {
            "step": optimizer_step,
            "epoch": optimizer_step / steps_per_epoch,
            "loss": _global_mean(loss),
            "loss_full": _global_mean(loss_full),
            "loss_mask": _global_mean(loss_mask),
            "loss_cf_weighted": _global_mean(loss_cf),
            "rarity_mean": _global_mean(batch["task_rarity"].mean()),
            "action_accuracy": _global_mean(accuracy),
            "active_samples": active_count,
            "selected_patches_mean": _global_mean(patch_mask.sum(1).float().mean()),
            "relation_score_max_mean": _global_mean(relation_scores.max(1).values.mean()),
            "lr": strategy.lr_scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - started,
        }
        _write_metric(metrics_path, record)
        if _rank_zero():
            progress.update(1)
            progress.set_postfix(loss=f"{record['loss']:.4f}", acc=f"{record['action_accuracy']:.3f}")

        should_save = optimizer_step % config.save_interval == 0 or optimizer_step == total_steps
        if should_save:
            epoch_index = min(epochs - 1, int((optimizer_step - 1) // steps_per_epoch))
            strategy.save_checkpoint(
                run_dir,
                optimizer_step,
                epoch_index,
                record["loss"],
                only_trainable=False,
            )
            if dist.is_initialized():
                dist.barrier()
            last_saved_step = global_step

    if _rank_zero():
        progress.close()
    if last_saved_step != total_steps:
        raise RuntimeError("Training ended without writing its final checkpoint")
