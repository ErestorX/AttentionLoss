import torch


class AttentionProfileLoss(torch.nn.Module):
    def __init__(self):
        super(AttentionProfileLoss, self).__init__()

    def forward(self, x):
        return torch.mean(1.0 - torch.mean(x, dim=-1), dtype=torch.float32)
