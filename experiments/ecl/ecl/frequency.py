"""Demonstration counts and per-sample rarity (paper Eq. task rarity)."""
from __future__ import annotations

import json
from pathlib import Path

from .core import TASK_TARGETS, normalize_instruction

DEFAULT_COUNTS = dict(zip(TASK_TARGETS, [46, 28, 19, 15, 11, 9, 8, 7, 6, 5]))


def load_counts(path: str | Path | None = None) -> dict[str, int]:
    raw = DEFAULT_COUNTS if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
    counts = {normalize_instruction(k): v for k, v in raw.items()}
    if set(counts) != set(TASK_TARGETS):
        raise ValueError("Counts must cover exactly the ten LIBERO-Core instructions")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in counts.values()):
        raise ValueError("Demonstration counts must be positive integers")
    return counts


def rarity_for_instruction(instruction: str, counts: dict[str, int]) -> float:
    count = counts[normalize_instruction(instruction)]
    lower, upper = min(counts.values()), max(counts.values())
    # Balanced data are outside the paper's Nmax > Nmin case. Disable CF there.
    return (upper - count) / (upper - lower) if upper > lower else 0.0
