"""
Apply paths.yaml path overrides to a training config dict.

Usage:
    from src.utils.paths_override import apply_paths_override
    config = yaml.safe_load(open("config.yaml"))
    apply_paths_override(config, "paths.yaml")  # mutates config in-place
"""
from __future__ import annotations
import yaml


def _load_paths(paths_file: str) -> dict:
    with open(paths_file, "r") as f:
        return yaml.safe_load(f) or {}


def _subst(value: str, rules: list) -> str:
    for prefix, replacement in rules:
        if value.startswith(prefix):
            return replacement + value[len(prefix):]
    return value


def _walk(obj, rules: list) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                obj[k] = _subst(v, rules)
            else:
                _walk(v, rules)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str):
                obj[i] = _subst(v, rules)
            else:
                _walk(v, rules)


def apply_paths_override(config: dict, paths_file: str) -> None:
    """Mutate config in-place, substituting path prefixes from paths_file.

    Replaces the following default prefixes with values from paths_file:
      ./data      → data_root
      ./embedding → embedding_root
      ./output    → output_root
      ./filter/   → filter_root/
      filter/     → filter_root/   (bare form used in some configs)
    """
    paths = _load_paths(paths_file)

    data_root      = paths.get("data_root",      "./data")
    embedding_root = paths.get("embedding_root", "./embedding")
    output_root    = paths.get("output_root",    "./output")
    filter_root    = paths.get("filter_root",    "./filter")
    filter_base    = filter_root.rstrip("/")

    # Order matters: check longer/more-specific prefixes first.
    rules = [
        ("./data",       data_root),
        ("./embedding",  embedding_root),
        ("./output",     output_root),
        ("./filter/",    filter_base + "/"),
        ("filter/",      filter_base + "/"),
    ]

    _walk(config, rules)
