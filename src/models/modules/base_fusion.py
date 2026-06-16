from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class FusionSpec:
    name: str
    output_dim: int


class BaseFusion(Protocol):
    """Interface for fusion modules used in model architecture."""

    def spec(self) -> FusionSpec:
        ...

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ...
