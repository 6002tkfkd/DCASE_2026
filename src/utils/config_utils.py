import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class ConfigBundle:
    """Container for merged configuration sections."""
    base: Dict[str, Any]
    strategy: Dict[str, Any]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge for now; deep merge will be added during migration."""
    merged = dict(base)
    merged.update(override)
    return merged


def load_config(base_path: str, strategy_path: str) -> Dict[str, Any]:
    base_cfg = load_yaml(base_path)
    strat_cfg = load_yaml(strategy_path)
    return merge_dicts(base_cfg, strat_cfg)
