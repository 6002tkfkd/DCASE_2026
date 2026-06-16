import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_factor = self.alpha
            if not torch.is_tensor(alpha_factor):
                alpha_factor = torch.tensor(alpha_factor, device=logits.device, dtype=logits.dtype)
            focal_loss = alpha_factor * focal_loss

        return focal_loss.mean()