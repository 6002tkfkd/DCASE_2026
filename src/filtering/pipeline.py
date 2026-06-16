from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import json
import os

import pandas as pd

from .review_actions import ReviewAction, load_review_actions, filter_sample_ids


@dataclass(frozen=True)
class FilteringConfig:
    enabled: bool
    review_actions_csv: Optional[str]
    report_dir: Optional[str]
    report_name_template: Optional[str]
    apply_to: str = "all"


def make_filtering_config(filtering_cfg: Optional[Dict[str, object]]) -> FilteringConfig:
    filtering_cfg = filtering_cfg or {}
    return FilteringConfig(
        enabled=bool(filtering_cfg.get("enabled", False)),
        review_actions_csv=filtering_cfg.get("review_actions_csv"),
        report_dir=filtering_cfg.get("report_dir"),
        report_name_template=filtering_cfg.get("report_name_template"),
        apply_to=str(filtering_cfg.get("apply_to", "all")),
    )


def make_filtering_config_by_stage(
    filtering_cfg: Optional[Dict[str, object]], stage: str = "stage1"
) -> FilteringConfig:
    """Extract and create FilteringConfig for a specific stage (stage1 or stage2).
    
    Args:
        filtering_cfg: Top-level filtering config with 'stage1' and 'stage2' keys
        stage: Either 'stage1' or 'stage2'
    
    Returns:
        FilteringConfig for the specified stage
    """
    filtering_cfg = filtering_cfg or {}
    stage_cfg = filtering_cfg.get(stage, {}) if isinstance(filtering_cfg.get(stage), dict) else {}
    
    return FilteringConfig(
        enabled=bool(stage_cfg.get("enabled", False)),
        review_actions_csv=stage_cfg.get("review_actions_csv"),
        report_dir=stage_cfg.get("report_dir"),
        report_name_template=stage_cfg.get("report_name_template"),
        apply_to=str(stage_cfg.get("apply_to", "all" if stage == "stage1" else "train")),
    )


def should_apply_filtering(config: FilteringConfig, stage: Optional[str] = None) -> bool:
    if not config.enabled:
        return False
    if config.apply_to == "all":
        return True
    if config.apply_to == "train":
        return stage == "train"
    if config.apply_to == "other":
        # apply to any stage that is not 'train' (e.g., 'val', 'test', 'load')
        return (stage is not None) and (stage != "train")
    return True


def build_report_path(config: FilteringConfig, tag: str, **format_kwargs: object) -> Optional[str]:
    if not config.report_dir or not config.report_name_template:
        return None
    filename = config.report_name_template.format(tag=tag, **format_kwargs)
    return os.path.join(config.report_dir, filename)


def apply_filtering(sample_ids: Iterable[str], config: FilteringConfig, stage: Optional[str] = None) -> List[str]:
    """Apply filtering stage to a list of sample ids.

    This is a standalone stage intended for: raw -> processed -> filtered -> train/eval.
    """
    if not should_apply_filtering(config, stage=stage):
        return list(sample_ids)

    actions = load_review_actions(config.review_actions_csv)
    return filter_sample_ids(sample_ids, actions)


def apply_filtering_df(
    df: pd.DataFrame,
    sample_id_col: str,
    config: FilteringConfig,
    report_path: Optional[str] = None,
    stage: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Apply filtering to a dataframe and optionally write a report.

    Returns filtered dataframe and a report dict with counts and excluded ids.
    """
    input_count = len(df)
    if not should_apply_filtering(config, stage=stage):
        report = {
            "input_count": input_count,
            "excluded_count": 0,
            "kept_count": input_count,
            "excluded": [],
        }
        if report_path:
            _write_report(report_path, report)
        return df, report

    actions = load_review_actions(config.review_actions_csv)
    action_map = {a.sample_id: a for a in actions}
    sample_ids = df[sample_id_col].astype(str).tolist()
    kept_ids = set(filter_sample_ids(sample_ids, actions))

    filtered_df = df[df[sample_id_col].astype(str).isin(kept_ids)].reset_index(drop=True)
    excluded = []
    for sample_id in sample_ids:
        if sample_id not in kept_ids:
            action = action_map.get(sample_id)
            excluded.append(
                {
                    "sample_id": sample_id,
                    "action": action.action if action else "exclude",
                    "reason": action.reason if action else "",
                }
            )

    report = {
        "input_count": input_count,
        "excluded_count": len(excluded),
        "kept_count": len(filtered_df),
        "excluded": excluded,
    }
    if report_path:
        _write_report(report_path, report)
    return filtered_df, report


def _write_report(path: str, report: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
