"""Contract checks, no training experiments or checkpoint selection."""

import unittest

import torch

from nir.engine import seed_all
from nir.model import Capsules, Model, margin_loss


class Contracts(unittest.TestCase):
    def test_capsule_shapes_and_gradients(self):
        seed_all()
        head = Capsules()
        x = torch.randn(2, 1024, 7, 7, requires_grad=True)
        scores = head(x)
        self.assertEqual(tuple(scores.shape), (2, 2))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        margin_loss(scores, torch.tensor([0, 1]), torch.ones(2)).backward()
        self.assertGreater(head.transforms.grad.abs().sum().item(), 0)
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_frozen_backbone_boundary(self):
        model = Model(pretrained=False)
        model.set_stage(1)
        model.train()
        bn = model.features.norm0
        before = bn.running_mean.clone()
        output = model(torch.randn(2, 1, 224, 224))
        output.sum().backward()
        self.assertFalse(bn.training)
        self.assertTrue(torch.equal(before, bn.running_mean))
        self.assertIsNone(model.features.conv0.weight.grad)
        self.assertTrue(any(p.grad is not None for p in model.features.denseblock4.parameters()))


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
