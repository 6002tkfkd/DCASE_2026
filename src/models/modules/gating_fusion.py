import torch
import torch.nn as nn

from .base_fusion import FusionSpec


class GatingFusion(nn.Module):
    """Learned scalar gate to mix features."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._spec = FusionSpec(name="gating", output_dim=input_dim)
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid(),
        )

    def spec(self) -> FusionSpec:
        return self._spec

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([a, b], dim=-1))
        return gate * a + (1 - gate) * b
