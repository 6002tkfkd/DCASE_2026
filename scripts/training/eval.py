#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Run evaluation from a single full config file.")
    parser.add_argument(
        "--config",
        default="configs/eval_only.yaml",
        help="Path to a full eval config YAML (no merge step).",
    )
    parser.add_argument(
        "--paths",
        default=None,
        help="Path to paths.yaml for external path overrides (see paths_example.yaml).",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    if args.paths:
        from src.utils.paths_override import apply_paths_override
        apply_paths_override(config, args.paths)

    from src.utils.config_archive import archive_runtime_config

    archived_path = archive_runtime_config(config, args.config, "eval_only")
    print(f"Archived runtime config to {archived_path}")

    from src.datasets.hatr_dataset import HATRDataset
    from src.datasets.prep import load_processed_dataset
    from src.evaluators import get_evaluator
    from src.evaluators.classification_evaluator import evaluate_model
    from src.utils.core_utils import build_class_to_topclass_mapping

    data_dir = None
    output_cfg = config.get("output", {}) if isinstance(config.get("output"), dict) else {}
    paths_cfg = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    artifact_cfg = config.get("artifact_names", {}) if isinstance(config.get("artifact_names"), dict) else {}
    model_name = model_cfg.get("name", "base_classifier")

    # Try new structure first
    if output_cfg.get("root") and output_cfg.get("run_dir"):
        data_dir = os.path.join(
            output_cfg.get("root", "./output"),
            output_cfg.get("run_dir", "model_output"),
            model_name,
        )
    # Fall back to legacy output_path
    elif config.get("output_path"):
        data_dir = config.get("output_path")
    # Fall back to old structure
    else:
        output_root = paths_cfg.get("output_root", "./output")
        model_output_dir = output_cfg.get("model_output_dir", "model_output")
        data_dir = os.path.join(output_root, model_output_dir, model_name)

    processed_basename = paths_cfg.get("processed_basename", "processed_dataset.csv")
    prepared_path = os.path.join(data_dir, processed_basename)

    filtering_cfg = config.get("filtering", {})
    df = load_processed_dataset(prepared_path, filtering_cfg=filtering_cfg, report_tag="eval")

    batch_size = int(config.get("evaluation", {}).get("batch_size", 128))
    dataset = HATRDataset(df, aug=False)
    num_workers = int(config.get("evaluation", {}).get("num_workers", 4))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_dict_path = os.path.join(data_dir, artifact_cfg.get("class_dict_json", "class_dict.json"))
    top_class_dict_path = os.path.join(data_dir, artifact_cfg.get("top_class_dict_json", "top_class_dict.json"))

    with open(class_dict_path, "r") as f:
        class_dict = json.load(f)
    with open(top_class_dict_path, "r") as f:
        top_class_dict = json.load(f)

    model_path = config.get("evaluation", {}).get("checkpoint_path")
    if not model_path:
        raise ValueError("evaluation.checkpoint_path is required in config")

    output_dir = config.get("evaluation", {}).get("output_dir") or os.path.dirname(model_path)
    fold_id = int(config.get("evaluation", {}).get("fold", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_to_topclass = build_class_to_topclass_mapping(class_dict, top_class_dict)

    checkpoint = torch.load(model_path, map_location="cpu")
    proxy_state = checkpoint.get("proxy_loss_state") or checkpoint.get("criterion_state")
    proxy_loss_name = str(checkpoint.get("proxy_loss_name", ""))

    if isinstance(proxy_state, dict) and {"child_proxies", "parent_proxies"}.issubset(proxy_state.keys()):
        evaluator = get_evaluator(
            "hierarchical_proxy_classification",
            class_to_topclass=class_to_topclass,
            class_dict=class_dict,
        )
        metrics = evaluator.evaluate_model(
            None,
            model_path,
            loader,
            device,
            output_dir=output_dir,
            fold_id=fold_id,
            class_dict=class_dict,
        )
    elif (isinstance(proxy_state, dict) and "proxies" in proxy_state) or proxy_loss_name == "Proxy_Anchor":
        evaluator = get_evaluator(
            "proxy_classification",
            class_to_topclass=class_to_topclass,
            class_dict=class_dict,
        )
        metrics = evaluator.evaluate_model(
            None,
            model_path,
            loader,
            device,
            output_dir=output_dir,
            fold_id=fold_id,
            class_dict=class_dict,
        )
    else:
        metrics = evaluate_model(
            model_path,
            loader,
            device,
            class_to_topclass,
            output_dir=output_dir,
            fold_id=fold_id,
            class_dict=class_dict,
            split="eval",
        )
    print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    main()
