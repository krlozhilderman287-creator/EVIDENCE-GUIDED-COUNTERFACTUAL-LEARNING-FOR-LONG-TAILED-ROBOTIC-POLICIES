"""Static checks for the pinned π0.5 ECL overlay; requires only Python."""

from __future__ import annotations

import ast
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parent
MODEL = ROOT / "openpi_overlay/src/openpi/models_pytorch/pi0_ecl_pytorch.py"
PATCH = ROOT / "openpi_overlay.patch"
PINNED_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"


def main() -> None:
    model_text = MODEL.read_text(encoding="utf-8")
    patch_text = PATCH.read_text(encoding="utf-8")
    ast.parse(model_text, filename=str(MODEL))
    ast.parse((ROOT / "convert_core_lt_to_lerobot.py").read_text(encoding="utf-8"))

    tree = ast.parse(model_text)
    assignment = next(
        node for node in tree.body if isinstance(node, ast.Assign) and node.targets[0].id == "TASKS_AND_COUNTS"
    )
    tasks = ast.literal_eval(assignment.value)
    counts = [count for _, count in tasks]
    assert len(tasks) == 10
    assert counts == [46, 28, 19, 15, 11, 9, 8, 7, 6, 5]
    rarity = [(max(counts) - count) / (max(counts) - min(counts)) for count in counts]
    assert rarity[0] == 0.0 and rarity[-1] == 1.0
    assert all(left <= right for left, right in zip(rarity, rarity[1:]))

    required_model_fragments = (
        "contrast_lambda: float = 0.3",
        "mu: float = 0.3",
        "top_ratio: float = 0.10",
        "effect_velocity = factual_velocity - self.ecl_lambda * masked_velocity",
        "factual_loss + self.ecl_mu * masked_loss + rarity * effect_loss",
    )
    assert all(fragment in model_text for fragment in required_model_fragments)
    assert 'name="pi05_libero_ecl"' in patch_text
    assert "action_horizon=10" in patch_text
    assert "batch_size=256" in patch_text
    assert "warmup_steps=10_000" in patch_text
    assert "peak_lr=5e-5" in patch_text
    assert "num_train_steps=30_000" in patch_text
    assert re.search(r'PINNED_COMMIT="' + PINNED_COMMIT + r'"', (ROOT / "install_openpi.sh").read_text())
    print("π0.5 ECL static checks passed (model, objective, counts, config, pin).")


if __name__ == "__main__":
    main()
