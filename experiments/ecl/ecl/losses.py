"""ECL action supervision: average per sample, without renormalizing rarity."""
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .core import aligned_action_logits_and_labels


def counterfactual_logits(positive, negative, coefficient, stop_gradient=False):
    reference = negative.detach() if stop_gradient else negative
    return positive - coefficient * reference


def weighted_action_ce(logits, labels, weights, *, num_patches, action_tokenizer):
    aligned, targets = aligned_action_logits_and_labels(logits, labels, num_patches)
    start = action_tokenizer.action_token_begin_idx + 1
    end = action_tokenizer.action_token_end_idx
    valid = (targets >= start) & (targets < end)
    # Invalid labels are replaced before CE and then excluded from each sample.
    targets = (targets - start).masked_fill(~valid, 0)
    token_losses = F.cross_entropy(
        aligned[..., start:end].float().transpose(1, 2), targets, reduction="none"
    )
    counts = valid.sum(1)
    if bool((counts == 0).any()):
        raise ValueError("Every sample must contain supervised action tokens")
    per_sample = (token_losses * valid).sum(1) / counts
    local_sum = (per_sample * weights.to(per_sample)).sum()
    denominator = torch.tensor(float(labels.shape[0]), device=logits.device)
    world_size = 1
    if dist.is_initialized():
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    # DDP averages gradients. Preserve mean_i[r_i * loss_i], including r_i=0.
    return local_sum * world_size / denominator
