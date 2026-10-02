import math
from types import SimpleNamespace
import unittest

import torch

from ecl.frequency import DEFAULT_COUNTS, load_counts, rarity_for_instruction
from ecl.losses import counterfactual_logits, weighted_action_ce
from ecl.protocol import aggregate, initial_state_plan


class ECLTests(unittest.TestCase):
    def test_rarity_is_demo_based_and_bounded(self):
        counts = load_counts()
        values = [rarity_for_instruction(k.upper() + ".", counts) for k in counts]
        self.assertEqual(sum(counts.values()), 154)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 1.0)
        self.assertAlmostEqual(values[1], 18 / 41)
        self.assertEqual(values, sorted(values))

    def test_balanced_extension_is_zero(self):
        counts = {k: 10 for k in DEFAULT_COUNTS}
        self.assertEqual(rarity_for_instruction(next(iter(counts)), counts), 0)

    def test_cf_is_per_sample_mean_not_weight_normalized(self):
        logits = torch.zeros(2, 5, 8, requires_grad=True)
        labels = torch.tensor([[-100, 5, 6, -100], [-100, 5, 6, 5]])
        tokenizer = SimpleNamespace(action_token_begin_idx=4, action_token_end_idx=7)
        loss = weighted_action_ce(logits, labels, torch.tensor([0., 0.25]),
                                 num_patches=1, action_tokenizer=tokenizer)
        self.assertAlmostEqual(loss.item(), math.log(2) * 0.25 / 2, places=6)
        loss.backward()
        self.assertEqual(logits.grad[0].abs().sum().item(), 0)
        self.assertGreater(logits.grad[1].abs().sum().item(), 0)

    def test_paper_gradient_and_legacy_ablation(self):
        for stop in [False, True]:
            pos = torch.tensor([2.], requires_grad=True)
            neg = torch.tensor([1.], requires_grad=True)
            result = counterfactual_logits(pos, neg, 0.3, stop)
            self.assertAlmostEqual(result.item(), 1.7, places=6)
            result.sum().backward()
            self.assertEqual(pos.grad.item(), 1)
            if stop:
                self.assertIsNone(neg.grad)
            else:
                self.assertAlmostEqual(neg.grad.item(), -0.3, places=6)

    def test_initial_states_support_75_and_225_trials(self):
        for trials in [75, 225]:
            plan = initial_state_plan(50, trials, 7, 0)
            self.assertEqual(len(plan), trials)
            self.assertEqual(len(set(plan[:50])), 50)
            self.assertEqual(plan, initial_state_plan(50, trials, 7, 0))
            self.assertNotEqual(plan, initial_state_plan(50, trials, 14, 0))
            self.assertTrue(all(0 <= i < 50 for i in plan))

    def test_aggregation_distinguishes_statistics_and_checks_completeness(self):
        rows = [{"seed": seed, "task_id": task, "task": str(task), "episode": ep,
                 "success": seed == 7} for seed in [7, 14] for task in [0, 1] for ep in range(2)]
        summary = aggregate(rows, [7, 14], [0, 1], 2)
        self.assertEqual(summary["overall"]["mean"], 0.5)
        self.assertAlmostEqual(summary["overall"]["seed_std"], math.sqrt(0.5))
        self.assertEqual(summary["tasks"]["0"]["binomial_std"], 0.25)
        for bad in [rows[:-1], rows + [rows[0]]]:
            with self.assertRaises(ValueError):
                aggregate(bad, [7, 14], [0, 1], 2)


if __name__ == "__main__":
    unittest.main()
