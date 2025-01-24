import torch
import torch.nn as nn
import torch.nn.functional as F
from PrimaryCapsules import PrimaryCapsules
from DigitCapsules import DigitCapsules


class CapsuleNet(nn.Module):
    def __init__(self, num_classes):
        super(CapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=1)
        self.primary_caps = PrimaryCapsules(
            in_channels=64, out_channels=32, num_capsules=8, capsule_dim=8, kernel_size=9, stride=2
        )
        self.digit_caps = DigitCapsules(
            input_capsules=8 * 6 * 6, input_dim=8, num_classes=num_classes, capsule_dim=16
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        return x
    
class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def forward(self, outputs, labels):
        labels = F.one_hot(labels, num_classes=outputs.size(1)).float()
        v_c = torch.norm(outputs, dim=-1)
        left = F.relu(0.9 - v_c) ** 2
        right = F.relu(v_c - 0.1) ** 2
        loss = labels * left + 0.5 * (1.0 - labels) * right
        return loss.sum(dim=1).mean()