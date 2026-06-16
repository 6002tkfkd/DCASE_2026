import torch
import torch.nn as nn

from .base_fusion import FusionSpec


class AttentionFusion(nn.Module):
    """Attention-style fusion (placeholder)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._spec = FusionSpec(name="attention", output_dim=input_dim)
        self.attn = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 2),
            nn.Softmax(dim=-1),
        )

    def spec(self) -> FusionSpec:
        return self._spec

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        weights = self.attn(torch.cat([a, b], dim=-1))
        return a * weights[:, 0:1] + b * weights[:, 1:2]
