import torch
import torch.nn as nn

from .base_fusion import FusionSpec


class SelectiveKernelFusion(nn.Module):
    """Selective-kernel-style fusion (minimal stub)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._spec = FusionSpec(name="selective_kernel", output_dim=input_dim)
        self.selector = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 2),
            nn.Softmax(dim=-1),
        )

    def spec(self) -> FusionSpec:
        return self._spec

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        weights = self.selector(torch.cat([a, b], dim=-1))
        return a * weights[:, 0:1] + b * weights[:, 1:2]
