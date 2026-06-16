import torch
import torch.nn as nn

from .base_fusion import FusionSpec


class ConcatFusion(nn.Module):
    """Concatenate features along the last dimension."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._spec = FusionSpec(name="concat", output_dim=input_dim * 2)

    def spec(self) -> FusionSpec:
        return self._spec

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b], dim=-1)
