import torch


class AttentionProfileLoss(torch.nn.Module):
    def __init__(self, policy='max', operation='mean'):
        super(AttentionProfileLoss, self).__init__()
        self.policy = policy
        self.operation = operation

    def forward(self, x):
        # x = [batch_size, nb_blocks, nb_heads]
        if self.operation == 'mean':
            x = torch.mean(x, dim=1)
        elif self.operation == 'std':
            x = torch.std(x)
        if self.policy == 'max':
            return torch.mean(1.0 - x, dtype=torch.float32)
        elif self.policy == 'min':
            return torch.mean(x)