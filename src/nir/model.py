"""DenseNet121 features and dynamic routing between capsules."""

import torch
import torchxrayvision as xrv
from torch import nn

from .config import CONFIG


def squash(x):
    squared = (x * x).sum(-1, keepdim=True)
    return squared / (1 + squared) * x / torch.sqrt(squared + 1e-8)


class Capsules(nn.Module):
    def __init__(self):
        super().__init__()
        self.primary = nn.Conv2d(1024, 8 * 8, kernel_size=1)
        self.transforms = nn.Parameter(torch.randn(392, 2, 8, 16) * 0.01)

    def forward(self, features):
        batch = features.shape[0]
        primary = self.primary(features).reshape(batch, 8, 8, 7, 7)
        primary = squash(primary.permute(0, 3, 4, 1, 2).reshape(batch, 392, 8).float())
        # Normalize primary capsules in float32; vote transforms follow AMP.
        votes = torch.einsum("bid,ijdo->bijo", primary, self.transforms.float())
        logits = votes.new_zeros(batch, 392, 2)
        for iteration in range(3):
            coefficients = logits.softmax(dim=-1)
            outputs = squash((coefficients[..., None] * votes).sum(dim=1))
            if iteration < 2:
                logits = logits + (votes * outputs[:, None, :, :]).sum(-1)
        return torch.linalg.vector_norm(outputs, dim=-1)


class Model(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = xrv.models.DenseNet(weights=CONFIG["weights"] if pretrained else None)
        self.features = backbone.features
        self.capsules = Capsules()
        self.stage = 0
        self.set_stage(0)

    def set_stage(self, stage):
        self.stage = stage
        for name, p in self.features.named_parameters():
            p.requires_grad = bool(
                stage and (name.startswith("denseblock4.") or name.startswith("norm5."))
            )

    def train(self, mode=True):
        super().train(mode)
        for module in self.features.modules():
            if isinstance(module, nn.BatchNorm2d) and not any(
                p.requires_grad for p in module.parameters()
            ):
                module.eval()
        return self

    def forward(self, images):
        return self.capsules(torch.relu(self.features(images)))


def margin_loss(scores, labels, class_weights):
    target = nn.functional.one_hot(labels, 2).float()
    terms = (
        target * torch.relu(0.9 - scores).square()
        + 0.5 * (1 - target) * torch.relu(scores - 0.1).square()
    )
    return (terms.sum(1) * class_weights[labels]).mean()
