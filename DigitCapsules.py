import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitCapsules(nn.Module):
    def __init__(self, input_capsules, input_dim, num_classes, capsule_dim):
        super(DigitCapsules, self).__init__()
        self.num_classes = num_classes
        self.capsule_dim = capsule_dim
        self.weights = nn.Parameter(
            torch.randn(1, input_capsules, num_classes, capsule_dim, input_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(2)  # Add dimension for classes
        u_hat = torch.matmul(self.weights, x.unsqueeze(-1))  # Linear mapping
        u_hat = u_hat.squeeze(-1)  # Remove last dimension

        b_ij = torch.zeros(*u_hat.size()[:3]).to(x.device)

        for _ in range(3):  # Dynamic routing iterations
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)
            v_j = self.squash(s_j)
            b_ij = b_ij + (u_hat * v_j.unsqueeze(1)).sum(dim=-1)

        return v_j

    @staticmethod
    def squash(x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return (x / (1 + norm ** 2) / torch.sqrt(norm ** 2 + 1e-9))