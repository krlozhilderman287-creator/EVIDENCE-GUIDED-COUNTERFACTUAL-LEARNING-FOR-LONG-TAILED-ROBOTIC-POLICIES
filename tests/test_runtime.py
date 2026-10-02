import unittest

import torch
from torch import nn

from ecl.runtime import freeze_structurally_unused_vision_parameters


class _Featurizer(nn.Module):
    def __init__(self, *, with_pool: bool = False):
        super().__init__()
        self.patch_embed = nn.Linear(2, 2)
        self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.norm = nn.LayerNorm(2)
        self.attn_pool = nn.Linear(2, 2) if with_pool else nn.Identity()


class _Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino_featurizer = _Featurizer()
        self.siglip_featurizer = _Featurizer(with_pool=True)


class _VLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = _Vision()


class RuntimeTests(unittest.TestCase):
    def test_freezes_only_structurally_unreachable_timm_tails(self):
        model = _VLA()
        frozen = freeze_structurally_unused_vision_parameters(model)

        self.assertEqual(set(frozen), {"dino_featurizer", "siglip_featurizer"})
        for featurizer in (
            model.vision_backbone.dino_featurizer,
            model.vision_backbone.siglip_featurizer,
        ):
            self.assertTrue(featurizer.patch_embed.weight.requires_grad)
            self.assertTrue(featurizer.blocks[0].weight.requires_grad)
            self.assertFalse(featurizer.blocks[-1].weight.requires_grad)
            self.assertFalse(featurizer.norm.weight.requires_grad)
        self.assertFalse(model.vision_backbone.siglip_featurizer.attn_pool.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
