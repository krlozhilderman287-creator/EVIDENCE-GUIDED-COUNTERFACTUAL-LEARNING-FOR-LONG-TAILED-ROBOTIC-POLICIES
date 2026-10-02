"""Repository discovery and VLA training bootstrap helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

import torch
from torch import nn


VLA_TYPE = "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt"


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "prismatic").is_dir() and (candidate / "vla_scripts" / "train.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find VLA-long-tail root. Put this folder directly under <project>/experiments/."
    )


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"step-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def checkpoint_step(path: Path) -> int:
    """Return the global step encoded in a step-N checkpoint filename."""

    step = _checkpoint_step(path)
    if step < 0:
        raise ValueError(f"Checkpoint filename does not contain step-N: {path}")
    return step


def resolve_checkpoint(project_root: Path, value: str) -> Path:
    if value != "auto":
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = project_root / path
        if path.is_dir():
            candidates = list((path / "checkpoints").glob("step-*.pt"))
            if not candidates:
                candidates = list(path.glob("step-*.pt"))
            if not candidates:
                raise FileNotFoundError(f"No step-*.pt checkpoint under {path}")
            path = max(candidates, key=_checkpoint_step)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()

    preferred = project_root / "pretrained/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt"
    if preferred.is_file():
        return preferred.resolve()
    candidates = list((project_root / "pretrained/minivla-libero90-prismatic/checkpoints").glob("step-*.pt"))
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover the MiniVLA pretrained checkpoint. Pass --pretrained-checkpoint."
        )
    return max(candidates, key=_checkpoint_step).resolve()


def resolve_data_root(project_root: Path, value: str) -> Path:
    candidates = []
    if value != "auto":
        given = Path(value).expanduser()
        candidates.append(given if given.is_absolute() else project_root / given)
    else:
        candidates.extend(
            [
                project_root / "tensorflow_datasets",
                project_root.parent / "tensorflow_datasets",
                Path("/tensorflow_datasets"),
            ]
        )
    for candidate in candidates:
        if (candidate / "libero_core_lt").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find tensorflow_datasets/libero_core_lt. Pass --data-root explicitly."
    )


def resolve_raw_data_root(project_root: Path, value: str, *, required: bool = True) -> Path | None:
    """Find the original LT HDF5 directory used by phase-aware metadata."""

    candidates: list[Path]
    if value != "auto":
        given = Path(value).expanduser()
        candidates = [given if given.is_absolute() else project_root / given]
    else:
        candidates = [
            project_root / "dataset_all/libero_core_lt_no_noops",
            project_root.parent / "dataset_all/libero_core_lt_no_noops",
        ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.hdf5")):
            return candidate.resolve()
    if required:
        raise FileNotFoundError(
            "Could not find dataset_all/libero_core_lt_no_noops. Pass --raw-data-root explicitly."
        )
    return None


def resolve_object_info_root(raw_data_root: Path) -> tuple[Path, ...]:
    """Locate all UUID metadata roots used by the three source task suites."""

    candidates = [
        raw_data_root / "object_infos",
        raw_data_root.parent / "libero_core_full_no_noops/object_infos",
        *sorted(raw_data_root.parent.glob("*/object_infos")),
    ]
    roots: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir() and candidate.resolve() not in roots:
            roots.append(candidate.resolve())
    if roots:
        return tuple(roots)
    raise FileNotFoundError(
        "Could not locate object_infos in the LT directory or its dataset_all siblings."
    )


def read_hf_token(project_root: Path) -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    token_file = project_root / ".hf_token"
    return token_file.read_text().strip() if token_file.is_file() else None


def _freeze_unused_timm_tail(featurizer: nn.Module) -> list[str]:
    """Freeze TIMM modules bypassed by second-to-last-layer feature extraction.

    Prismatic monkey-patches its TIMM vision featurizers to call
    ``get_intermediate_layers`` at ``len(blocks) - 2``.  Consequently the last
    transformer block, final normalization, and optional classification
    attention pool are structurally absent from every VLA forward graph.
    ``freeze_backbones(vla-full-train)`` otherwise marks them trainable, which
    makes DDP fail on the next iteration with an unfinished reduction.
    """

    frozen: list[str] = []
    blocks = getattr(featurizer, "blocks", None)
    if blocks is not None and len(blocks):
        blocks[-1].requires_grad_(False)
        frozen.append("blocks[-1]")
    for name in ("norm", "fc_norm", "attn_pool", "head"):
        module = getattr(featurizer, name, None)
        if isinstance(module, nn.Module):
            module.requires_grad_(False)
            frozen.append(name)
    return frozen


def freeze_structurally_unused_vision_parameters(vla) -> dict[str, list[str]]:
    """Exclude unreachable TIMM tail parameters from optimization and DDP."""

    vision = vla.vision_backbone
    featurizers = {
        name: getattr(vision, name)
        for name in ("featurizer", "dino_featurizer", "siglip_featurizer")
        if isinstance(getattr(vision, name, None), nn.Module)
    }
    return {
        name: frozen
        for name, featurizer in featurizers.items()
        if (frozen := _freeze_unused_timm_tail(featurizer))
    }


def determine_stage(vla, vla_cfg) -> str:
    if not vla_cfg.freeze_vision_backbone and not vla_cfg.freeze_llm_backbone:
        stage = "vla-full-train"
    elif vla_cfg.freeze_vision_backbone and not vla_cfg.freeze_llm_backbone:
        stage = "vla-train"
    elif not vla_cfg.freeze_vision_backbone and vla_cfg.freeze_llm_backbone:
        if not vla_cfg.unfreeze_last_llm_layer:
            raise ValueError("Frozen LLM requires unfreeze_last_llm_layer")
        stage = "vla-sandwich-train"
    else:
        if not vla_cfg.unfreeze_last_llm_layer:
            raise ValueError("At least the final LLM layer must be trainable")
        stage = "vla-last-layer-train"
    vla.freeze_backbones(stage)
    if vla.vision_backbone_requires_grad:
        freeze_structurally_unused_vision_parameters(vla)
    return stage


def create_strategy(
    *,
    name: str,
    vla,
    device_id: int,
    stage: str,
    epochs: int,
    max_steps: int | None,
    global_batch_size: int,
    per_device_batch_size: int,
    vla_cfg,
    worker_init_fn,
):
    if name == "ddp":
        from prismatic.training.strategies.ddp import DDPStrategy

        cls = DDPStrategy
        extra: dict[str, Any] = {}
    else:
        from prismatic.training.strategies.fsdp import FSDPStrategy

        cls = FSDPStrategy
        extra = {"sharding_strategy": "full-shard"}

    return cls(
        vlm=vla,
        device_id=device_id,
        stage=stage,
        epochs=epochs,
        max_steps=max_steps,
        global_batch_size=global_batch_size,
        per_device_batch_size=per_device_batch_size,
        learning_rate=vla_cfg.learning_rate,
        weight_decay=vla_cfg.weight_decay,
        max_grad_norm=vla_cfg.max_grad_norm,
        lr_scheduler_type=vla_cfg.lr_scheduler_type,
        warmup_ratio=vla_cfg.warmup_ratio,
        enable_gradient_checkpointing=vla_cfg.enable_gradient_checkpointing,
        enable_mixed_precision_training=vla_cfg.enable_mixed_precision_training,
        reduce_in_full_precision=vla_cfg.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
        save_every_n_steps=None,
        **extra,
    )


def write_compatible_config(
    run_dir: Path,
    pretrained_checkpoint: Path,
    vla_cfg,
    experiment: dict[str, Any],
) -> None:
    source_path = pretrained_checkpoint.parents[1] / "config.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing sibling config.json: {source_path}")
    source = json.loads(source_path.read_text())
    cfg_dict = asdict(vla_cfg) if is_dataclass(vla_cfg) else dict(vars(vla_cfg))
    # Keep the registered architecture ID from the loaded checkpoint. The
    # upstream fine-tuning default is a developer-local VLM directory.
    cfg_dict["base_vlm"] = source["vla"]["base_vlm"]
    source["vla"] = cfg_dict
    source["experiment"] = experiment
    (run_dir / "config.json").write_text(json.dumps(source, indent=2, default=str) + "\n")


def move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value
