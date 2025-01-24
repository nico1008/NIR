import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, out_channels, num_capsules, capsule_dim, kernel_size, stride):
        super(PrimaryCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
            for _ in range(num_capsules)
        ])
        self.capsule_dim = capsule_dim

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)  # Shape: [batch_size, num_capsules, out_channels, h, w]
        u = u.view(x.size(0), self.num_capsules, -1)  # Flatten spatial dimensions
        return self.squash(u)

    @staticmethod
    def squash(x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return (x / (1 + norm ** 2) / torch.sqrt(norm ** 2 + 1e-9))