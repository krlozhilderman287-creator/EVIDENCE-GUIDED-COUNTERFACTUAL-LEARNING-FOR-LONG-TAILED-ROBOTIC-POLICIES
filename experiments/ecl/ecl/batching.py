"""RLDS transform and collator adapters for ECL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from .core import object_token_mask, target_phrase_for_instruction
from .frequency import load_counts, rarity_for_instruction


def _decode_json(value, object_info_root: Path | Sequence[Path] | None = None):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    text = str(value)
    if text.lstrip().startswith("{"):
        return json.loads(text)
    if object_info_root is None:
        raise FileNotFoundError(
            "object_info stores an external JSON id, but no object-info directory was configured"
        )
    roots = [object_info_root] if isinstance(object_info_root, Path) else list(object_info_root)
    for root in roots:
        path = root / f"{text}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"Missing object metadata id={text!r} under {[str(root) for root in roots]}"
    )


def _match_object_key(keys, phrase: str) -> str | None:
    wanted = set(phrase.replace("-", " ").split())
    candidates = []
    for key in keys:
        normalized = set(str(key).lower().replace("-", "_").split("_"))
        overlap = len(wanted & normalized)
        if overlap == len(wanted):
            candidates.append((overlap, -len(normalized), str(key)))
    return max(candidates)[2] if candidates else None


class ECLBatchTransform:
    """Add target-token and optional phase gates to the upstream transform."""

    def __init__(
        self,
        base_transform,
        tokenizer,
        *,
        contrast_scope: str = "all",
        approach_distance: float = 0.12,
        object_info_root: Path | Sequence[Path] | None = None,
        task_counts: dict[str, int] | None = None,
    ) -> None:
        if contrast_scope not in {"all", "off", "approach", "pre_grasp", "post_grasp"}:
            raise ValueError(f"Unsupported contrast_scope={contrast_scope!r}")
        self.base_transform = base_transform
        self.tokenizer = tokenizer
        self.contrast_scope = contrast_scope
        self.approach_distance = approach_distance
        self.object_info_root = object_info_root
        self.task_counts = load_counts() if task_counts is None else task_counts

    def __call__(self, rlds_batch):
        output = self.base_transform(rlds_batch)
        instruction = rlds_batch["task"]["language_instruction"].decode().lower()
        target_phrase = target_phrase_for_instruction(instruction)
        output["object_token_mask"] = object_token_mask(
            output["input_ids"], self.tokenizer, target_phrase
        )

        gate = self.contrast_scope == "all"
        if self.contrast_scope not in {"all", "off"}:
            info = _decode_json(rlds_batch["object_info"], self.object_info_root)
            grasp_map = info.get("is_grasp", {})
            grasp_key = _match_object_key(grasp_map, target_phrase)
            if grasp_key is None:
                raise ValueError(
                    f"Could not map target phrase {target_phrase!r} to is_grasp keys={sorted(grasp_map)}"
                )
            grasped = bool(grasp_map[grasp_key])
            if self.contrast_scope == "pre_grasp":
                gate = not grasped
            elif self.contrast_scope == "post_grasp":
                gate = grasped
            elif self.contrast_scope == "approach":
                distance_map = info.get("gripper_to_obj_distance", {})
                distance_key = _match_object_key(distance_map, target_phrase)
                if distance_key is None:
                    raise ValueError(
                        f"Could not map target phrase {target_phrase!r} to distance keys={sorted(distance_map)}"
                    )
                distance = float(distance_map[distance_key])
                gate = (not grasped) and distance <= self.approach_distance
        output["contrast_gate"] = torch.tensor(gate, dtype=torch.bool)
        output["task_rarity"] = torch.tensor(
            rarity_for_instruction(instruction, self.task_counts), dtype=torch.float32
        )
        return output


class ECLCollator:
    """Delegate normal padding to upstream and collate ECL metadata."""

    def __init__(self, base_collator) -> None:
        self.base_collator = base_collator

    def __call__(self, instances):
        output = self.base_collator(instances)
        object_masks = pad_sequence(
            [item["object_token_mask"] for item in instances],
            batch_first=True,
            padding_value=False,
        )
        output["object_token_mask"] = object_masks[:, : output["input_ids"].shape[1]]
        output["contrast_gate"] = torch.stack([item["contrast_gate"] for item in instances])
        output["task_rarity"] = torch.stack([item["task_rarity"] for item in instances])
        return output
