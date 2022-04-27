import torch


class AttentionProfileLoss(torch.nn.Module):
    def __init__(self, is_t2t=False):
        super(AttentionProfileLoss, self).__init__()
        self.is_t2t = is_t2t

    def forward(self, x):
        # x = [batch_size, nb_blocks, nb_heads]
        x = torch.mean(x, dim=1)
        return torch.mean(1.0 - x, dtype=torch.float32)