#!/usr/bin/env python3
"""Train ECL on the original LIBERO-Core-LT RLDS dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from ecl.batching import ECLBatchTransform, ECLCollator
from ecl.core import RelationalSelectorConfig
from ecl.frequency import load_counts
from ecl.runtime import (
    VLA_TYPE,
    checkpoint_step,
    create_strategy,
    determine_stage,
    find_project_root,
    read_hf_token,
    resolve_checkpoint,
    resolve_data_root,
    resolve_object_info_root,
    resolve_raw_data_root,
    write_compatible_config,
)
from ecl.trainer import ECLTrainConfig, train_ecl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=None, help="JSON defaults; explicit CLI flags override them")
    parser.add_argument("--project-root", default="auto")
    parser.add_argument("--pretrained-checkpoint", default="auto")
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Resume model and optimizer from an ECL step-*.pt checkpoint into a new run",
    )
    parser.add_argument("--data-root", default="auto")
    parser.add_argument("--raw-data-root", default="auto")
    parser.add_argument("--run-root", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--strategy", choices=("fsdp", "ddp"), default="ddp")
    parser.add_argument("--per-device-batch-size", type=int, default=20)
    parser.add_argument("--global-batch-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means full requested epochs")
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lambda-value", type=float, default=0.3)
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--task-counts", default=None, help="Instruction -> demonstration count JSON")
    parser.add_argument("--stop-gradient", action=argparse.BooleanOptionalAction, default=False,
                        help="Detach the negative branch as an ablation; the paper formula defaults to no detach")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=0, help="0 uses --top-ratio")
    parser.add_argument("--top-ratio", type=float, default=0.10)
    parser.add_argument("--last-n-layers", type=int, default=4)
    parser.add_argument("--mask-mode", choices=("mean", "zero"), default="mean")
    parser.add_argument(
        "--contrast-scope",
        choices=("all", "off", "approach", "pre_grasp", "post_grasp"),
        default="all",
    )
    parser.add_argument("--approach-distance", type=float, default=0.12)
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        values = json.loads(Path(preliminary.config).read_text(encoding="utf-8"))
        unknown = set(values) - {action.dest for action in parser._actions}
        if unknown:
            parser.error(f"Unknown configuration keys: {sorted(unknown)}")
        parser.set_defaults(**values)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.per_device_batch_size < 1 or args.save_interval < 1:
        raise ValueError("epochs, batch size and save interval must be positive")
    if min(args.lambda_value, args.mu, args.warmup_epochs, args.max_steps) < 0:
        raise ValueError("Loss coefficients, warm-up and max steps must be nonnegative")
    project_root = (
        find_project_root(THIS_DIR)
        if args.project_root == "auto"
        else Path(args.project_root).expanduser().resolve()
    )
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)
    os.environ.setdefault("PRISMATIC_DATA_ROOT", str(project_root / "tensorflow_datasets"))
    task_counts = load_counts(args.task_counts)

    # Import only after the project root is available.
    from prismatic.conf import VLAConfig
    from prismatic.models import load_vla
    from prismatic.overwatch import initialize_overwatch
    from prismatic.util import set_global_seed
    from prismatic.vla import get_vla_dataset_and_collator
    from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

    overwatch = initialize_overwatch(__name__)
    if not dist.is_initialized():
        raise RuntimeError("Launch with torchrun, including for one GPU. See README_ZH.md.")
    device_id = overwatch.local_rank()
    torch.cuda.set_device(device_id)
    world_size = overwatch.world_size()

    resume_checkpoint = (
        resolve_checkpoint(project_root, args.resume_checkpoint)
        if args.resume_checkpoint is not None
        else None
    )
    if resume_checkpoint is not None and args.strategy != "ddp":
        raise ValueError("Existing ECL optimizer checkpoints currently resume only with --strategy ddp")
    checkpoint = resume_checkpoint or resolve_checkpoint(project_root, args.pretrained_checkpoint)
    start_step = checkpoint_step(resume_checkpoint) if resume_checkpoint is not None else 0
    data_root = resolve_data_root(project_root, args.data_root)
    raw_data_root = resolve_raw_data_root(
        project_root,
        args.raw_data_root,
        required=args.contrast_scope not in {"all", "off"},
    )
    object_info_root = (
        resolve_object_info_root(raw_data_root)
        if raw_data_root is not None and args.contrast_scope not in {"all", "off"}
        else None
    )
    run_root = (
        project_root / "runs/miniVLA_libero_core_lt_ecl"
        if args.run_root == "auto"
        else Path(args.run_root).expanduser().resolve()
    )
    if args.run_name is not None:
        run_name = args.run_name
    elif resume_checkpoint is not None:
        run_name = f"{resume_checkpoint.parents[1].name}_resume_s{start_step}_{datetime.now():%Y%m%d_%H%M%S}"
    else:
        # torchrun processes must agree even when launched across a second boundary.
        run_name = f"ecl_l{args.lambda_value}_k{args.top_ratio}_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    shared_name = [run_name if overwatch.is_rank_zero() else None]
    dist.broadcast_object_list(shared_name, src=0)
    run_name = shared_name[0]
    run_dir = run_root / run_name
    if overwatch.is_rank_zero():
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoints").mkdir()
    dist.barrier()

    worker_init_fn = set_global_seed(args.seed, get_worker_init_fn=True)
    hf_token = read_hf_token(project_root)
    vla_cfg = VLAConfig.get_choice_class(VLA_TYPE)()
    vla_cfg.expected_world_size = world_size
    vla_cfg.per_device_batch_size = args.per_device_batch_size
    physical_batch_size = args.per_device_batch_size * world_size
    vla_cfg.global_batch_size = args.global_batch_size or physical_batch_size
    if vla_cfg.global_batch_size < physical_batch_size or vla_cfg.global_batch_size % physical_batch_size:
        raise ValueError("global batch size must be a multiple of per-device-batch-size * world size")
    vla_cfg.epochs = args.epochs

    vla = load_vla(
        checkpoint,
        hf_token=hf_token,
        load_for_training=True,
        image_sequence_len=vla_cfg.image_sequence_len,
    )
    for parameter in vla.parameters():
        if parameter.dtype != torch.float32:
            raise TypeError(f"Expected FP32 loaded parameters, found {parameter.dtype}")
    stage = determine_stage(vla, vla_cfg)

    dataset, action_tokenizer, base_collator = get_vla_dataset_and_collator(
        data_root,
        vla_cfg.data_mix,
        image_transform=vla.vision_backbone.get_image_transform(),
        tokenizer=vla.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        default_image_resolution=vla.vision_backbone.default_image_resolution,
        shuffle_buffer_size=vla_cfg.shuffle_buffer_size,
        image_aug=False,
        action_tokenizer=vla_cfg.action_tokenizer,
        image_window_size=vla_cfg.image_sequence_len,
        use_wrist_image=vla_cfg.use_wrist_image,
    )
    dataset.batch_transform = ECLBatchTransform(
        dataset.batch_transform,
        vla.llm_backbone.get_tokenizer(),
        contrast_scope=args.contrast_scope,
        approach_distance=args.approach_distance,
        object_info_root=object_info_root,
        task_counts=task_counts,
    )
    collator = ECLCollator(base_collator)

    max_steps = args.max_steps or None
    strategy = create_strategy(
        name=args.strategy,
        vla=vla,
        device_id=device_id,
        stage=stage,
        epochs=args.epochs,
        max_steps=max_steps,
        global_batch_size=vla_cfg.global_batch_size,
        per_device_batch_size=vla_cfg.per_device_batch_size,
        vla_cfg=vla_cfg,
        worker_init_fn=worker_init_fn,
    )
    strategy.run_setup(run_dir=run_dir, n_train_examples=len(dataset))
    if resume_checkpoint is not None:
        resume_state = torch.load(resume_checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if "optimizer" not in resume_state:
            raise KeyError(f"Resume checkpoint has no optimizer state: {resume_checkpoint}")
        strategy.optimizer.load_state_dict(resume_state["optimizer"])
        if "lr_scheduler" in resume_state:
            strategy.lr_scheduler.load_state_dict(resume_state["lr_scheduler"])
        del resume_state

    selector = RelationalSelectorConfig(
        top_k=args.top_k,
        top_ratio=args.top_ratio,
        last_n_layers=args.last_n_layers,
    )
    method_cfg = ECLTrainConfig(
        contrast_lambda=args.lambda_value,
        mu=args.mu,
        stop_gradient=args.stop_gradient,
        warmup_epochs=args.warmup_epochs,
        mask_mode=args.mask_mode,
        selector=selector,
        save_interval=args.save_interval,
        max_steps=max_steps,
    )
    experiment = {
        **vars(args),
        "project_root": str(project_root),
        "data_root": str(data_root),
        "method": "ECL",
        "task_counts": task_counts,
        "inference": "factual_only",
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "resume_step": start_step,
    }
    if overwatch.is_rank_zero():
        write_compatible_config(run_dir, checkpoint, vla_cfg, experiment)
        save_dataset_statistics(dataset.dataset_statistics, run_dir)
        (run_dir / "method_config.json").write_text(json.dumps(asdict(method_cfg), indent=2) + "\n")
        (run_dir / "task_counts.json").write_text(json.dumps(task_counts, indent=2) + "\n")
        (THIS_DIR / "latest_run.txt").write_text(str(run_dir) + "\n")
        print(f"RUN_DIR={run_dir}", flush=True)
    dist.barrier()

    train_ecl(
        strategy=strategy,
        dataset=dataset,
        collator=collator,
        action_tokenizer=action_tokenizer,
        run_dir=run_dir,
        epochs=args.epochs,
        config=method_cfg,
        start_step=start_step,
    )
    dist.barrier()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
