# π₀.₅ + ECL on LIBERO-Core-LT

This directory adds the ECL training objective to OpenPI's PyTorch π₀.₅ implementation. It pins OpenPI to commit `215abfb217dbac7d5f1273282331b9b1866c0479` and installs a small, reviewable overlay instead of copying the upstream repository.

## What is implemented

For every training sample, the code:

1. runs a gradient-free selector pass and averages the last four transformer layers;
2. computes instruction-to-visual and action-to-visual attention, multiplies the two normalized scores, and selects the top 10% of valid visual tokens;
3. replaces selected tokens with the mean of the remaining valid visual tokens;
4. evaluates factual and masked branches with the same flow-matching noise and timestep;
5. applies `L = L_BC + μ L_mask + r_i L_CF`, where `v_effect = v_factual - λ v_masked` and `r_i = (N_max - N_i) / (N_max - N_min)`.

The defaults are `λ=0.3`, `μ=0.3`, mask ratio `0.10`, LT demonstration counts `[46, 28, 19, 15, 11, 9, 8, 7, 6, 5]`, action horizon `10`, batch size `256`, peak learning rate `5e-5`, learning-rate warmup `10,000`, `30,000` training steps, gradient clipping `1.0`, and the π₀.₅-Base initialization. ECL warmup defaults to zero and can be changed with `--ecl-warmup-steps`.

The selector pass is used only during training. Checkpoint inference follows the ordinary factual π₀.₅ policy.

## Install the pinned OpenPI tree

Run from `ECL-long-tail`:

```bash
bash supplementary/pi05/install_openpi.sh
cd supplementary/pi05/openpi
uv sync
```

The installer copies `PI05ECLPytorch` into OpenPI and applies `openpi_overlay.patch`, which adds the `pi05_libero_ecl` config and routes the PyTorch trainer through the ECL loss. Re-running the installer is safe when the overlay is already present.

Run the dependency-free package check with:

```bash
python supplementary/pi05/self_check.py
```

## Prepare data

First build `tensorflow_datasets/libero_core_lt/1.0.0` using the main README. Then convert only that 154-demonstration dataset to LeRobot format:

```bash
cd supplementary/pi05/openpi
uv pip install tensorflow tensorflow-datasets
uv run ../convert_core_lt_to_lerobot.py \
  --data_dir ../../../tensorflow_datasets
uv run scripts/compute_norm_stats.py --config-name pi05_libero_ecl
```

The converter writes the local LeRobot repository `ecl/libero_core_lt` and rejects a dataset whose episode count is not 154. Pass `--overwrite` only when intentionally rebuilding it.

## Convert π₀.₅-Base and train

OpenPI's PyTorch trainer consumes a converted base checkpoint:

```bash
cd supplementary/pi05/openpi
cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/

uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base \
  --config_name pi05_libero \
  --output_path checkpoints/pi05_base_pytorch

uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py pi05_libero_ecl \
  --exp_name ecl_core_lt_s7 \
  --seed 7 \
  --pytorch_weight_path checkpoints/pi05_base_pytorch
```

Repeat with seeds `14` and `21` and distinct experiment names. `train.sh` combines conversion, normalization, checkpoint conversion, and one training run; set `NUM_GPUS`, `BATCH_SIZE`, `SEED`, `RUN_NAME`, `BASE_JAX_CHECKPOINT`, or `BASE_TORCH_CHECKPOINT` as needed. After the first conversion, set `CONVERT_DATA=0` when reusing the local LeRobot dataset.

## Evaluate

Use the factual checkpoint through OpenPI's standard LIBERO policy server. The helper prints the two commands without starting hidden processes:

```bash
bash supplementary/pi05/evaluate.sh \
  supplementary/pi05/openpi/checkpoints/pi05_libero_ecl/ecl_core_lt_s7/30000
```

Run the printed server command in one terminal and the LIBERO client command in another. Preserve the checkpoint, command line, seed, normalization assets, rollout plan, and per-episode results for each run.

## Files

- `openpi_overlay/src/openpi/models_pytorch/pi0_ecl_pytorch.py`: attention selector, token intervention, frequency-aware ECL flow loss.
- `openpi_overlay.patch`: OpenPI configuration and trainer integration.
- `convert_core_lt_to_lerobot.py`: RLDS-to-LeRobot conversion for the generated LT split.
- `install_openpi.sh`, `train.sh`, `evaluate.sh`: installation, training, and evaluation entry points.
- `self_check.py`: static verification of the objective, task counts, OpenPI config, and pinned commit.
