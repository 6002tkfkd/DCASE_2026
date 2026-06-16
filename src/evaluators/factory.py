from __future__ import annotations

from typing import Any

from src.evaluators.hierarchical_proxy_evaluator import HierarchicalProxyClassificationEvaluator
from src.evaluators.proxy_evaluator import ProxyClassificationEvaluator


def get_evaluator(evaluator_type: str, **kwargs: Any):
    if evaluator_type == "hierarchical_proxy_classification":
        return HierarchicalProxyClassificationEvaluator(**kwargs)
    if evaluator_type == "proxy_classification":
        return ProxyClassificationEvaluator(**kwargs)
    if evaluator_type in {"classification", "default", None}:
        return None
    raise ValueError(f"Unknown evaluator type: {evaluator_type}")