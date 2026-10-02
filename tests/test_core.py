import unittest

import torch

from ecl.core import (
    RelationalSelectorConfig,
    action_token_cross_entropy,
    aligned_action_logits_and_labels,
    masked_projector_output,
    relation_topk_mask,
)


class CoreTests(unittest.TestCase):
    def test_alignment_matches_prismatic_insertion(self):
        logits = torch.arange(1 * 8 * 4).reshape(1, 8, 4).float()
        labels = torch.tensor([[-100, 1, 2, 3, 4]])
        aligned, targets = aligned_action_logits_and_labels(logits, labels, num_patches=3)
        self.assertTrue(torch.equal(aligned, logits[:, 3:7]))
        self.assertTrue(torch.equal(targets, labels[:, 1:]))

    def test_selector_uses_causal_prediction_rows(self):
        attention = torch.zeros(1, 1, 9, 9)
        attention[:, :, 5, 1:4] = torch.tensor([0.1, 0.8, 0.1])
        attention[:, :, 6:8, 1:4] = torch.tensor([0.1, 0.8, 0.1])
        object_mask = torch.tensor([[False, False, True, False, False, False]])
        labels = torch.tensor([[-100, -100, -100, -100, 91, 92]])
        mask, _ = relation_topk_mask(
            [attention],
            object_mask,
            labels,
            num_patches=3,
            action_token_begin_idx=90,
            action_token_end_idx=100,
            config=RelationalSelectorConfig(top_k=1),
        )
        self.assertEqual(mask.tolist(), [[False, True, False]])

    def test_action_ce_ignores_inactive_samples(self):
        logits = torch.zeros(2, 7, 10, requires_grad=True)
        labels = torch.tensor([[-100, -100, 7, 8, -100], [-100, -100, 7, 8, -100]])
        loss = action_token_cross_entropy(
            logits,
            labels,
            num_patches=2,
            action_token_begin_idx=5,
            action_token_end_idx=10,
            sample_gate=torch.tensor([True, False]),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(logits.grad[1].abs().sum().item(), 0)

    def test_mask_mean_excludes_selected_tokens(self):
        projector = torch.nn.Identity()
        values = torch.tensor([[[1.0], [100.0], [3.0]]])
        with masked_projector_output(projector, torch.tensor([[False, True, False]]), "mean"):
            result = projector(values)
        self.assertEqual(result.squeeze(-1).tolist(), [[1.0, 2.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
