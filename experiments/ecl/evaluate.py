#!/usr/bin/env python3
"""Evaluate MiniVLA using factual-only inference and the LIBERO-Core rollout."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--step", type=int, default=0, help="0 selects the latest numbered checkpoint")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seeds", default="7,14,21")
    parser.add_argument("--num-trials-per-task", type=int, default=75)
    parser.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--unnorm-key", default="libero_core_lt")
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    task_ids = [int(t) for t in args.task_ids.split(",")]
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0:
        parser.error("Seeds must be unique nonnegative integers")
    if not task_ids or len(task_ids) != len(set(task_ids)) or not set(task_ids) <= set(range(10)):
        parser.error("Task IDs must be unique and in 0..9")
    if args.num_trials_per_task < 1:
        parser.error("Trials must be positive")
    os.chdir(ROOT)
    run = Path(args.run_dir).expanduser().resolve()
    candidates = []
    for path in (run / "checkpoints").glob("step-*.pt"):
        match = re.match(r"step-(\d+)-", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No numbered checkpoints in {run}")
    step = args.step or max(s for s, _ in candidates)
    matches = [p for s, p in candidates if s == step]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one checkpoint at step {step}, got {matches}")
    for name in ("config.json", "dataset_statistics.json"):
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)
    output = Path(args.save_root) if args.save_root else ROOT / "results/ecl" / run.name / f"step_{step}" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("HF_TOKEN", "")
    os.environ.setdefault("PRISMATIC_DATA_ROOT", str(ROOT / "tensorflow_datasets"))

    from ecl.protocol import aggregate, initial_state_plan
    from vla_scripts.parallel_libero_evaluator import GenerateConfig, ParallelLiberoEvaluator, get_image_resize_size
    from libero.libero import benchmark
    from experiments.robot.robot_utils import set_seed_everywhere

    cfg = GenerateConfig(pretrained_checkpoint=str(matches[0]), load_step=step,
                         hf_token="HF_TOKEN", unnorm_key=args.unnorm_key,
                         task_suite_name="libero_core", num_trials_per_task=args.num_trials_per_task,
                         num_gpus=1, num_processes=1, center_crop=args.center_crop,
                         instruction_formatting=False)
    cfg.max_steps = 350
    # Exact checkpoint resolution above avoids the upstream filename-padding bug.
    evaluator = ParallelLiberoEvaluator.__new__(ParallelLiberoEvaluator)
    evaluator.cfg = cfg
    evaluator.resize_size = get_image_resize_size(cfg)
    evaluator.task_suite = benchmark.get_benchmark_dict()["libero_core"]()
    model, processor = evaluator._build_policy(args.gpu_id)
    records = []
    plans = {str(s): {str(t): initial_state_plan(len(evaluator.task_suite.get_task_init_states(t)),
              args.num_trials_per_task, s, t) for t in task_ids} for s in seeds}
    protocol = {**vars(args), "checkpoint": str(matches[0]), "step": step,
                "inference": "factual_only", "max_steps": 350, "settling_steps": 10,
                "initial_states": plans,
                "initial_state_sampling": "independently shuffled cycles without replacement within each cycle"}
    (output / "evaluation_config.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    for seed in seeds:
        cfg.seed = seed
        cfg.save_root = str(output / f"seed_{seed}")
        evaluator._set_results()
        evaluator._build_logger()
        set_seed_everywhere(seed)
        for task_id in task_ids:
            for episode, init_id in enumerate(plans[str(seed)][str(task_id)]):
                record = evaluator.evalute_single(model, processor, task_id, init_id, episode, False)
                record.update(seed=seed, initial_state_id=init_id)
                records.append(record)
                with (output / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
    summary = aggregate(records, seeds, task_ids, args.num_trials_per_task)
    summary.update(checkpoint=str(matches[0]), inference="factual_only")
    (output / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "aggregate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "task", "successes", "episodes", "mean", "binomial_std", "seed_std"])
        for task_id, row in summary["tasks"].items():
            writer.writerow([task_id, row["task"], row["successes"], row["episodes"], row["mean"], row["binomial_std"], row["seed_std"]])
        writer.writerow(["overall", "macro average", "", "", summary["overall"]["mean"], "", summary["overall"]["seed_std"]])
    print(f"RESULTS={output}\nMACRO_SUCCESS={summary['overall']['mean']:.4f}")


if __name__ == "__main__":
    main()
