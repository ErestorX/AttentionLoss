import numpy as np
import torch
import json
import os


class AttentionProfileLoss(torch.nn.Module):
    def __init__(self, policy='max'):
        super(AttentionProfileLoss, self).__init__()
        if policy not in ['max', 'min']:
            assert os.path.exists(policy)
            with open(policy, 'r') as f:
                self.policy = torch.tensor(np.asarray(json.load(f)), dtype=torch.float32).cuda()
        else:
            self.policy = policy

    def forward(self, x):
        if self.policy == 'max':
            return torch.mean(1.0 - torch.mean(x, dim=-1), dtype=torch.float32)
        elif self.policy == 'min':
            return torch.mean(torch.mean(x, dim=-1), dtype=torch.float32)
        else:
            return torch.mean(torch.mean(torch.abs(x - self.policy), dim=-1), dtype=torch.float32)


if __name__ == '__main__':
    policies = ['max', 'min', 'output/train/attentionProfile-T2T-ViT-t.json',
                'output/train/attentionProfile-ViT-T-16 Pretrained.json',
                'output/train/attentionProfile-ViT-S-32 Pretrained.json',
                'output/train/attentionProfile-ViT-S-16 Pretrained.json']
    for policy in policies:
        loss = AttentionProfileLoss(policy)
