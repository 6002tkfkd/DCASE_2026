from typing import Dict

from .attention_fusion import AttentionFusion
from .concat_fusion import ConcatFusion
from .gating_fusion import GatingFusion
from .sk_fusion import SelectiveKernelFusion


def build_fusion_registry(input_dim: int):
    """Return a simple registry of fusion modules by name."""
    return {
        "concat": ConcatFusion(input_dim),
        "gating": GatingFusion(input_dim),
        "attention": AttentionFusion(input_dim),
        "selective_kernel": SelectiveKernelFusion(input_dim),
    }
