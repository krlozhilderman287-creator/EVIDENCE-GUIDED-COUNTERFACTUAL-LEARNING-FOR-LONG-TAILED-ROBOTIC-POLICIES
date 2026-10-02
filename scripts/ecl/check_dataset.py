"""Verify task/demo counts and record selected demonstration IDs and frame counts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/ecl"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["full", "lt"], required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    import h5py
    from ecl.frequency import DEFAULT_COUNTS
    source = args.source or ROOT / f"dataset_all/libero_core_{args.variant}_no_noops"
    full_counts = [46, 47, 45, 42, 47, 39, 47, 39, 45, 38]
    records, counts = {}, {}
    for idx, (instruction, lt_count) in enumerate(DEFAULT_COUNTS.items()):
        path = source / (instruction.replace(" ", "_") + "_demo.hdf5")
        with h5py.File(path, "r") as data:
            demos = {k: int(data["data"][k]["actions"].shape[0]) for k in sorted(data["data"])}
        expected = lt_count if args.variant == "lt" else full_counts[idx]
        if len(demos) != expected:
            raise ValueError(f"{path.name}: expected {expected} demos, found {len(demos)}; inspect regeneration")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        counts[instruction] = len(demos)
        records[instruction] = {"file": path.name, "sha256": digest.hexdigest(), "demos": demos,
                                "frames": sum(demos.values())}
    (source / "manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (source / "task_counts.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(f"{args.variant}: {sum(counts.values())} demos, {sum(r['frames'] for r in records.values())} frames")
    print(f"Manifest: {source / 'manifest.json'}")


if __name__ == "__main__":
    main()
