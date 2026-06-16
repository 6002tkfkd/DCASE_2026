from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


def _resolve_output_dir(config: Dict[str, Any]) -> Path:
    paths_cfg = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}
    output_cfg = config.get("output", {}) if isinstance(config.get("output"), dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}

    explicit_output_path = config.get("output_path")
    if explicit_output_path:
        return Path(explicit_output_path)

    output_root = output_cfg.get("root") or paths_cfg.get("output_root") or "./output"
    model_output_dir = output_cfg.get("run_dir") or output_cfg.get("model_output_dir") or paths_cfg.get("model_output_dir") or "model_output"
    model_name = model_cfg.get("name") or "base_classifier"
    return Path(output_root) / model_output_dir / model_name


def archive_runtime_config(config: Dict[str, Any], source_path: str, prefix: str) -> Path:
    """Copy the loaded runtime config into the run output directory.

    The archived file name is timestamped and prefixed with the strategy name,
    for example: baseline_config_20260517T123456Z.yaml.
    """
    output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    archive_name = f"{prefix}_config_{stamp}.yaml"
    archive_path = output_dir / archive_name

    with open(source_path, "r", encoding="utf-8") as src_file:
        loaded = yaml.safe_load(src_file) or {}

    with open(archive_path, "w", encoding="utf-8") as dst_file:
        yaml.safe_dump(loaded, dst_file, sort_keys=False, allow_unicode=True)

    return archive_path
