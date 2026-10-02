# ECL configuration

| Detail | Configuration |
| --- | --- |
| Counterfactual loss weight | `r_i=(Nmax-Ni)/(Nmax-Nmin)` per sample |
| Counterfactual coefficient λ | `0.3` |
| Masked-branch coefficient μ | `0.3` |
| Visual-token masking ratio | `0.10` |
| Frequency information | Training supervision only |
| Inference | Factual `predict_action()` |
| Negative-branch gradient | Gradients through both branches; optional `--stop-gradient` ablation |
| Evaluation seeds | 7, 14, 21 |
| Evaluation trials | 225 per task: 75 for each of seeds 7, 14 and 21 |
| Default physical/global batch | 20/20 |

# Source locations

- `experiments/ecl/ecl/core.py`: object phrase mapping, causal attention selection, projector masking and logit/label alignment.
- `experiments/ecl/ecl/frequency.py`: demonstration counts and normalized rarity.
- `experiments/ecl/ecl/batching.py`: object-token masks and task rarity for every sample.
- `experiments/ecl/ecl/losses.py`: per-sample action CE and contrastive prediction with DDP mean correction.
- `experiments/ecl/ecl/trainer.py`: warm-up, selector, factual/masked forwards and weighted training objective.
- `experiments/ecl/evaluate.py`: factual-only evaluation, checkpoint selection and episode recording.
- `experiments/ecl/ecl/protocol.py`: initial-state sampling and result statistics.
- `supplementary/pi05/`: π₀.₅ OpenPI integration, data conversion, training and evaluation entry points.

# Implementation details

The language signal uses the manipulated-object phrase from a fixed mapping of the ten Core instructions. The selector averages the last four attention layers and all heads. Action-related attention is read from the causal positions predicting action labels (`j-1`), while previous teacher-forced actions remain context.

Top-K uses rounded patch counts (`0.10 × 256 → 26`). Selected projected visual tokens are replaced with the mean of unselected tokens. Selection is detached. BC retains the upstream full-vocabulary causal loss, while masked and CF CE are restricted to valid action bins.

Masking applies throughout each trajectory. Balanced demonstration counts produce zero rarity. The main configuration uses 36 epochs, 34,621 optimizer steps and a 1,923-step ECL warm-up for 19,234 frames with global batch 20.

The evaluation protocol uses 225 episodes per task: 75 episodes for each of seeds 7, 14 and 21. Seeded shuffled cycles extend the stored initial states while preserving the complete initial-state plan. Canonical task IDs are 0–9.

# Evaluation statistics

For each task, equally sized seed runs are pooled to obtain success rate `p`. The evaluator reports `sqrt(p*(1-p)/n)` as `binomial_std` and the sample standard deviation of seed-wise rates as `seed_std`. The overall row averages task rates within each seed, then reports the mean and sample standard deviation of those seed-wise macro rates. Output values are fractions.

The evaluator checks episode completeness and uniqueness before producing the aggregate summary.
