import json
import os
from typing import Any, Dict, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.hatr_dataset import HATRDataset
from src.datasets.prep import load_processed_dataset
from src.evaluators import get_evaluator
from src.evaluators.classification_evaluator import evaluate_model
from src.models.hatr import BaseClassifier
from src.models.hatr_proxy_anchor import BaseClassifierProxyAnchor
from src.utils.config_utils import load_config
from src.utils.core_utils import build_class_to_topclass_mapping, set_seed


def _load_default_config() -> Dict[str, Any]:
    base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "base.yaml")
    )
    strategy_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "configs",
            "strategy",
            "eval_only.yaml",
        )
    )
    return load_config(base_path, strategy_path)


def _resolve_checkpoint_path(eval_cfg: Dict[str, Any]) -> str:
    checkpoint_path = eval_cfg.get("checkpoint_path")
    if checkpoint_path:
        return checkpoint_path

    checkpoint_dir = eval_cfg.get("checkpoint_dir")
    mode = eval_cfg.get("mode")
    fold = eval_cfg.get("fold")
    if checkpoint_dir and mode is not None and fold is not None:
        return os.path.join(checkpoint_dir, mode, f"fold_{fold}", "best_model.pth")

    raise ValueError("eval_only requires checkpoint_path or checkpoint_dir with mode and fold.")


def _resolve_device(device_cfg: Optional[str]) -> torch.device:
    if device_cfg in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_split(df: pd.DataFrame, dataset_cfg: Dict[str, Any]) -> pd.DataFrame:
    splits_csv = dataset_cfg.get("splits_csv")
    split_name = dataset_cfg.get("split")
    if not splits_csv:
        return df
    if not split_name:
        raise ValueError("eval_only.dataset.split is required when splits_csv is provided.")

    splits_df = pd.read_csv(splits_csv)
    if "index" not in splits_df.columns or "split" not in splits_df.columns:
        raise ValueError("splits_csv must contain index and split columns.")

    selected_ids = set(
        splits_df.loc[splits_df["split"] == split_name, "index"].astype(str).tolist()
    )
    return df[df["index"].astype(str).isin(selected_ids)].reset_index(drop=True)


def _write_manifest(
    output_dir: str,
    config: Dict[str, Any],
    checkpoint_path: str,
    dataset_cfg: Dict[str, Any],
    filtering_cfg: Dict[str, Any],
    device: torch.device,
    input_rows: int,
    evaluated_rows: int,
) -> None:
    manifest = {
        "strategy": "eval_only",
        "source_strategy": config["eval_only"].get("source_strategy"),
        "checkpoint_path": checkpoint_path,
        "checkpoint_dir": config["eval_only"].get("checkpoint_dir"),
        "dataset": dataset_cfg,
        "filtering": filtering_cfg,
        "output_dir": output_dir,
        "mode": config["eval_only"].get("mode"),
        "fold": config["eval_only"].get("fold"),
        "device": str(device),
        "input_rows": input_rows,
        "evaluated_rows": evaluated_rows,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_config_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def run(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Evaluation-only pipeline entrypoint."""
    config = config or _load_default_config()
    eval_cfg = config.get("eval_only", {})
    if not eval_cfg.get("enabled", False):
        raise ValueError("eval_only.enabled must be true to run eval-only mode.")

    seed = eval_cfg.get("seed", 1821)
    set_seed(seed)

    dataset_cfg = eval_cfg.get("dataset", {})
    processed_csv = dataset_cfg.get("processed_csv")
    if not processed_csv:
        raise ValueError("eval_only.dataset.processed_csv is required.")

    checkpoint_path = _resolve_checkpoint_path(eval_cfg)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    class_dict_json = eval_cfg.get("class_dict_json")
    top_class_dict_json = eval_cfg.get("top_class_dict_json")
    if not class_dict_json or not top_class_dict_json:
        raise ValueError("eval_only requires class_dict_json and top_class_dict_json.")

    class_dict = _load_json(class_dict_json)
    top_class_dict = _load_json(top_class_dict_json)

    filtering_cfg = eval_cfg.get("filtering", config.get("filtering", {}))
    dataset_name = dataset_cfg.get("name", "dataset")
    source_strategy = eval_cfg.get("source_strategy", "eval_only")
    report_tag = f"{dataset_name}_{source_strategy}"
    full_df = load_processed_dataset(
        processed_csv,
        filtering_cfg=filtering_cfg,
        report_tag=report_tag,
    )
    eval_df = _select_split(full_df, dataset_cfg)
    if eval_df.empty:
        raise ValueError("eval_only selected zero rows for evaluation.")

    dataset = HATRDataset(eval_df, aug=False)
    device = _resolve_device(eval_cfg.get("device", "auto"))
    data_loader = DataLoader(
        dataset,
        batch_size=eval_cfg.get("batch_size", 128),
        shuffle=False,
        num_workers=eval_cfg.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
    )

    output_dir = eval_cfg.get("output_dir", "./output/eval_only")
    class_to_top_class = build_class_to_topclass_mapping(class_dict, top_class_dict)

    evaluator_cfg = config.get("evaluator_config", {}) if isinstance(config.get("evaluator_config"), dict) else {}
    evaluator_type = evaluator_cfg.get("type", "classification")
    if evaluator_type == "proxy_classification":
        proxy_eval = get_evaluator(
            evaluator_type,
            class_to_topclass=class_to_top_class,
            class_dict=class_dict,
        )
        metrics = proxy_eval.evaluate_model(
            BaseClassifierProxyAnchor,
            checkpoint_path,
            data_loader,
            device,
            output_dir=output_dir,
            fold_id=eval_cfg.get("fold"),
            class_dict=class_dict,
        )
    else:
        metrics = evaluate_model(
            checkpoint_path,
            data_loader,
            device,
            class_to_top_class,
            output_dir=output_dir,
            fold_id=eval_cfg.get("fold"),
            class_dict=class_dict,
        )

    if eval_cfg.get("save_manifest", True):
        _write_manifest(
            output_dir,
            config,
            checkpoint_path,
            dataset_cfg,
            filtering_cfg,
            device,
            input_rows=len(full_df),
            evaluated_rows=len(eval_df),
        )

    return metrics
