from typing import Any, Dict, List, Optional, Tuple

import inspect
import json
import os
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.metrics.hierarchical import hierarchical_accuracy, hierarchical_prf_weighted
from src.models.hatr import BaseClassifier
from src.models.hatr_proxy_anchor import BaseClassifierProxyAnchor
from src.models.hatr_proxy_anchor_deep_shared import BaseClassifierProxyAnchorDeepShared
from src.models.hatr_proxy_anchor_simple import BaseClassifierProxyAnchorSimple
from src.utils.core_utils import build_id_to_class_mapping




def _get_class_labels(class_dict: Dict[str, int]) -> List[str]:
    return sorted(class_dict.keys(), key=lambda x: class_dict[x])


def _compute_normalized_confusion_matrix(
    y_true: List[int], y_pred: List[int], num_classes: int
) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)
    return cm_normalized


def _plot_confusion_matrix(
    cm_normalized: np.ndarray,
    class_labels: List[str],
    title: str,
    save_path: str,
    figsize=(14, 12),
) -> None:
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={"label": "Normalized Value (0-1)"},
        square=True,
        vmin=0,
        vmax=1,
    )

    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_classification_metrics_from_predictions(
    predictions: Dict[str, List[int]],
    class_to_topclass: Dict[int, int],
    class_dict: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    total = len(predictions["gt"])
    preds = predictions["pred"]
    gts = predictions["gt"]

    id_to_class = build_id_to_class_mapping(class_dict) if class_dict else {}
    pred_labels = [id_to_class.get(p, str(p)) for p in preds]
    gt_labels = [id_to_class.get(gt, str(gt)) for gt in gts]
    pred_gt_pairs = list(zip(pred_labels, gt_labels))
    classes = list(set(gt_labels))

    correct = sum(p == gt for p, gt in zip(preds, gts))
    top_correct = sum(
        class_to_topclass.get(gt) == class_to_topclass.get(p)
        for p, gt in zip(preds, gts)
        if class_to_topclass.get(gt) is not None and class_to_topclass.get(p) is not None
    )

    class_accs = []
    class_top_accs = []
    for c in classes:
        class_indices = [i for i, gt in enumerate(gt_labels) if gt == c]
        if not class_indices:
            continue
        class_correct = sum(1 for i in class_indices if preds[i] == gts[i])
        class_accs.append(class_correct / len(class_indices))

        class_top_correct = sum(
            1 for i in class_indices if class_to_topclass.get(gts[i]) == class_to_topclass.get(preds[i])
        )
        class_top_accs.append(class_top_correct / len(class_indices))

    macro_accuracy = np.mean(class_accs) if class_accs else 0
    macro_top_accuracy = np.mean(class_top_accs) if class_top_accs else 0

    h_accs = []
    for c in classes:
        try:
            acc = hierarchical_accuracy(c, pred_gt_pairs, lambda_param=0.5)
            if not np.isnan(acc):
                h_accs.append(acc)
        except Exception:
            continue
    hierarchical_acc = np.mean(h_accs) if h_accs else 0

    hps, hrs, hfs = [], [], []
    for c in classes:
        try:
            p, r, f = hierarchical_prf_weighted(c, pred_gt_pairs, lambda_param=0.75)
            if not (np.isnan(p) or np.isnan(r) or np.isnan(f)):
                hps.append(p)
                hrs.append(r)
                hfs.append(f)
        except Exception:
            continue

    hp = np.mean(hps) if hps else 0
    hr = np.mean(hrs) if hrs else 0
    hf = np.mean(hfs) if hfs else 0

    metrics = {
        "accuracy": 100 * correct / total if total > 0 else 0,
        "top_accuracy": 100 * top_correct / total if total > 0 else 0,
        "macro_accuracy": 100 * macro_accuracy if total > 0 else 0,
        "macro_top_accuracy": 100 * macro_top_accuracy if total > 0 else 0,
        "hierarchical_accuracy": 100 * hierarchical_acc if total > 0 else 0,
        "hierarchical_precision": 100 * hp if total > 0 else 0,
        "hierarchical_recall": 100 * hr if total > 0 else 0,
        "hierarchical_f1": 100 * hf if total > 0 else 0,
    }
    metrics["hierarchical_f1-score"] = metrics["hierarchical_f1"]
    return metrics


def collect_predictions_and_metrics(
    model,
    data_loader,
    device,
    class_to_topclass,
    class_dict: Dict[str, int] | None = None,
):
    model.eval()
    predictions = {"sound_id": [], "gt": [], "pred": [], "pred_score": []}

    with torch.no_grad():
        for data in data_loader:
            labels = data["class_idx"].to(device)
            sound_ids = data.get("sound_id", [None] * labels.size(0))

            audio_emb = data.get("audio_embedding", None)
            text_emb = data.get("text_embedding", None)

            if audio_emb is not None:
                audio_emb = audio_emb.to(device)
            if text_emb is not None:
                text_emb = text_emb.to(device)

            outputs = model(audio_emb, text_emb)
            if isinstance(outputs, dict):
                class_logits = outputs.get("logits")
            else:
                _, class_logits, _ = outputs
            probs = torch.softmax(class_logits, dim=1)

            top1 = torch.argmax(probs, dim=1)
            max_probs = probs.gather(1, top1.unsqueeze(1)).squeeze(1)

            for i in range(labels.size(0)):
                sid = sound_ids[i]
                if isinstance(sid, torch.Tensor):
                    sid = sid.item()
                predictions["sound_id"].append(sid)
                predictions["gt"].append(labels[i].item())
                predictions["pred"].append(top1[i].item())
                predictions["pred_score"].append(float(max_probs[i]))

    metrics = compute_classification_metrics_from_predictions(predictions, class_to_topclass, class_dict)
    return predictions, metrics


def _resolve_model_class_from_checkpoint(checkpoint: Dict[str, Any]):
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    state_dict = checkpoint.get("model_state", {}) if isinstance(checkpoint, dict) else {}
    model_name = str(config.get("model_name", "")).strip().lower()
    use_classifier = bool(config.get("use_classifier", True))

    state_keys = set(state_dict.keys()) if isinstance(state_dict, dict) else set()

    def has_any(prefixes):
        return any(any(key.startswith(prefix) for prefix in prefixes) for key in state_keys)

    if "deep_shared" in model_name or "deepshared" in model_name or "proxy_anchor_shared" in model_name:
        return BaseClassifierProxyAnchorDeepShared

    # Newer simplified proxy-anchor checkpoints should be recognized even if
    # they were saved before model_name metadata was added.
    if "proxy_anchor_simple" in model_name or (
        "latent_projector.weight" in state_keys
        and "class_predictor.weight" in state_keys
        and not has_any(["latent_projector.0.", "latent_projector.3.", "latent_projector.6.", "residual_classifier.", "proxy_projector."])
    ):
        return BaseClassifierProxyAnchorSimple

    # Only treat as a proxy-anchor model when proxy-specific keys or explicit
    # model_name indicate it. `residual_classifier.` is present in the original
    # `BaseClassifier` and should not be used to detect proxy models.
    if "proxy_anchor" in model_name or not use_classifier or has_any(["proxy_projector.", "proxy_"]):
        return BaseClassifierProxyAnchor

    return BaseClassifier


def evaluate_model(
    model_path,
    data_loader,
    device,
    class_to_topclass,
    output_dir,
    fold_id,
    class_dict=None,
    split: str = "test",
):
    # -------------------- Setup --------------------
    checkpoint = torch.load(model_path, map_location=device)
    model_class = _resolve_model_class_from_checkpoint(checkpoint)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    signature = inspect.signature(model_class.__init__)
    allowed_keys = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    }
    filtered_config = {key: value for key, value in config.items() if key in allowed_keys and value is not None}
    model = model_class(**filtered_config)
    load_result = model.load_state_dict(checkpoint["model_state"], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            f"[WARN] Loaded {model_class.__name__} with missing_keys={load_result.missing_keys} "
            f"unexpected_keys={load_result.unexpected_keys}"
        )
    model.to(device)
    model.eval()

    model_name = model.__class__.__name__

    predictions = {"sound_id": [], "gt": [], "pred": [], "pred_score": []}

    # -------------------- Inference --------------------
    with torch.no_grad():
        for data in data_loader:
            labels = data["class_idx"].to(device)
            sound_ids = data["sound_id"]

            audio_emb = data.get("audio_embedding", None)
            text_emb = data.get("text_embedding", None)

            if audio_emb is not None:
                audio_emb = audio_emb.to(device)
            if text_emb is not None:
                text_emb = text_emb.to(device)

            outputs = model(audio_emb, text_emb)
            if isinstance(outputs, dict):
                class_logits = outputs.get("logits")
            else:
                _, class_logits, _ = outputs
            probs = torch.softmax(class_logits, dim=1)

            # Top-1 prediction
            top1 = torch.argmax(probs, dim=1)
            max_probs = probs.gather(1, top1.unsqueeze(1)).squeeze(1)

            # Store all predictions and scores
            for i in range(labels.size(0)):
                sid = sound_ids[i]
                if isinstance(sid, torch.Tensor):
                    sid = sid.item()

                predictions["sound_id"].append(sid)
                predictions["gt"].append(labels[i].item())
                predictions["pred"].append(top1[i].item())
                predictions["pred_score"].append(float(max_probs[i]))

    # -------------------- Metrics --------------------
    def compute_metrics(predictions, class_to_topclass, class_dict):
        total = len(predictions["gt"])

        preds = predictions["pred"]
        gts = predictions["gt"]

        id_to_class = build_id_to_class_mapping(class_dict)
        pred_labels = [id_to_class.get(p, str(p)) for p in preds]
        gt_labels = [id_to_class.get(gt, str(gt)) for gt in gts]

        pred_gt_pairs = list(zip(pred_labels, gt_labels))
        classes = list(set(gt_labels))

        # ---------------- Standard accuracy (micro) ----------------
        correct = sum(p == gt for p, gt in zip(preds, gts))

        top_correct = sum(
            class_to_topclass.get(gt) == class_to_topclass.get(p)
            for p, gt in zip(preds, gts)
            if class_to_topclass.get(gt) is not None
            and class_to_topclass.get(p) is not None
        )

        # ---------------- Standard accuracy (macro) ----------------
        class_accs = []
        class_top_accs = []
        for c in classes:
            class_indices = [i for i, gt in enumerate(gt_labels) if gt == c]
            if not class_indices:
                continue
            class_correct = sum(1 for i in class_indices if preds[i] == gts[i])
            class_accs.append(class_correct / len(class_indices))

            class_top_correct = sum(
                1
                for i in class_indices
                if class_to_topclass.get(gts[i]) == class_to_topclass.get(preds[i])
            )
            class_top_accs.append(class_top_correct / len(class_indices))

        macro_accuracy = np.mean(class_accs) if class_accs else 0
        macro_top_accuracy = np.mean(class_top_accs) if class_top_accs else 0

        # ---------------- Hierarchical accuracy ----------------
        h_accs = []
        for c in classes:
            try:
                acc = hierarchical_accuracy(c, pred_gt_pairs, lambda_param=0.5)
                if not np.isnan(acc):
                    h_accs.append(acc)
            except Exception:
                continue

        hierarchical_acc = np.mean(h_accs) if h_accs else 0

        # ---------------- Hierarchical weighted PRF ----------------
        hps, hrs, hfs = [], [], []

        for c in classes:
            try:
                p, r, f = hierarchical_prf_weighted(c, pred_gt_pairs, lambda_param=0.75)
                if not (np.isnan(p) or np.isnan(r) or np.isnan(f)):
                    hps.append(p)
                    hrs.append(r)
                    hfs.append(f)
            except Exception:
                continue

        hp = np.mean(hps) if hps else 0
        hr = np.mean(hrs) if hrs else 0
        hf = np.mean(hfs) if hfs else 0

        metrics = {
            "accuracy": 100 * correct / total if total > 0 else 0,
            "top_accuracy": 100 * top_correct / total if total > 0 else 0,
            "macro_accuracy": 100 * macro_accuracy if total > 0 else 0,
            "macro_top_accuracy": 100 * macro_top_accuracy if total > 0 else 0,
            "hierarchical_accuracy": 100 * hierarchical_acc if total > 0 else 0,
            "hierarchical_precision": 100 * hp if total > 0 else 0,
            "hierarchical_recall": 100 * hr if total > 0 else 0,
            "hierarchical_f1": 100 * hf if total > 0 else 0,
        }
        metrics["hierarchical_f1-score"] = metrics["hierarchical_f1"]
        return metrics

    metrics = compute_metrics(predictions, class_to_topclass, class_dict)

    # -------------------- Save outputs --------------------
    id_to_class = build_id_to_class_mapping(class_dict) if class_dict else {}

    df = pd.DataFrame(
        {
            "sound_id": predictions["sound_id"],
            "ground_truth": [id_to_class.get(lbl, str(lbl)) for lbl in predictions["gt"]],
            "prediction": [id_to_class.get(lbl, str(lbl)) for lbl in predictions["pred"]],
            "prediction_score": [round(float(x), 4) for x in predictions["pred_score"]],
            "is_correct": [gt == pred for gt, pred in zip(predictions["gt"], predictions["pred"])],
            "split": [split] * len(predictions["sound_id"]),
        }
    )

    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    pred_path = os.path.join(eval_dir, "predictions.csv")
    df.to_csv(pred_path, index=False)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    metrics_to_log = {
        "Accuracy": "accuracy",
        "Top class accuracy": "top_accuracy",
        "Macro accuracy": "macro_accuracy",
        "Macro top class accuracy": "macro_top_accuracy",
        "Hierarchical accuracy": "hierarchical_accuracy",
        "Hierarchical precision": "hierarchical_precision",
        "Hierarchical recall": "hierarchical_recall",
        "Hierarchical F1": "hierarchical_f1",
        "Hierarchical f1-score": "hierarchical_f1-score",
    }

    for label, key in metrics_to_log.items():
        print(f"[{model_name} | Fold {fold_id}] {label}: {metrics[key]:.2f}%")

    with open(os.path.join(eval_dir, "results.txt"), "w") as f:
        for label, key in metrics_to_log.items():
            f.write(f"{key}: {metrics[key]:.2f}%\n")

    # -------------------- Confusion matrix outputs --------------------
    class_labels = _get_class_labels(class_dict)
    cm_normalized = _compute_normalized_confusion_matrix(
        predictions["gt"], predictions["pred"], num_classes=len(class_labels)
    )

    cm_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    cm_df = pd.DataFrame(cm_normalized, index=class_labels, columns=class_labels)
    cm_df.to_csv(cm_csv_path)

    cm_png_path = os.path.join(output_dir, "confusion_matrix.png")
    title = f"Confusion Matrix | Fold {fold_id}"
    _plot_confusion_matrix(cm_normalized, class_labels, title, cm_png_path)

    return metrics


def _infer_mode_and_fold_from_path(prediction_path: str, output_dir: str) -> tuple[str, str]:
    relative_path = Path(os.path.relpath(prediction_path, output_dir))
    parts = relative_path.parts
    mode = "N/A"
    fold_id = "N/A"
    if len(parts) >= 3 and parts[-2] == "evaluation":
        parent = parts[-3]
        if parent.startswith("fold_"):
            fold_id = parent.replace("fold_", "")
        else:
            fold_id = parent
        if len(parts) >= 4:
            mode = parts[-4]
    return mode, fold_id


def merge_ultimate_analysis(output_dir: str) -> Optional[str]:
    """Merge master splits and all prediction CSVs into one analysis file.

    Train/val rows stay fold-specific. Test rows are also preserved per-fold
    when prediction files include fold-specific outputs.
    """
    master_path = os.path.join(output_dir, "master_dataset_splits.csv")
    prediction_paths = sorted(
        glob.glob(os.path.join(output_dir, "**", "evaluation", "predictions.csv"), recursive=True)
    )

    if not os.path.exists(master_path):
        print(f"merge_ultimate_analysis: master file not found at {master_path}; skipping")
        return None

    master_df = pd.read_csv(master_path, dtype=str)
    master_df["sound_id"] = master_df["sound_id"].astype(str)
    master_mode = "N/A"
    if "mode" in master_df.columns and not master_df["mode"].dropna().empty:
        master_mode = str(master_df["mode"].dropna().astype(str).mode().iat[0])

    # Read all prediction files into a single dataframe
    prediction_frames = []
    for prediction_path in prediction_paths:
        pred_df = pd.read_csv(prediction_path)
        pred_df["sound_id"] = pred_df["sound_id"].astype(str)

        inferred_mode, inferred_fold_id = _infer_mode_and_fold_from_path(prediction_path, output_dir)
        # Prefer the mode inferred from the prediction file path; fall back to master mode when missing
        pred_df["mode"] = inferred_mode if inferred_mode != "N/A" else (master_mode if master_mode != "N/A" else inferred_mode)
        if "fold_id" not in pred_df.columns:
            pred_df["fold_id"] = inferred_fold_id
        if "split" not in pred_df.columns:
            pred_df["split"] = "test"
        if "is_correct" not in pred_df.columns:
            if "ground_truth" in pred_df.columns and "prediction" in pred_df.columns:
                pred_df["is_correct"] = (
                    pred_df["ground_truth"].astype(str) == pred_df["prediction"].astype(str)
                )
            else:
                pred_df["is_correct"] = False

        prediction_frames.append(pred_df)

    if not prediction_frames:
        print(f"merge_ultimate_analysis: no prediction files found under {output_dir}; skipping")
        return None

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    predictions_df["sound_id"] = predictions_df["sound_id"].astype(str)

    test_predictions_df = predictions_df[predictions_df["split"] == "test"].copy()
    other_predictions_df = predictions_df[predictions_df["split"] != "test"].copy()

    # Drop any 'test' predictions that were produced from pretrain outputs.
    # Pretrain should only contribute train/val rows; finetune provides test predictions.
    if not test_predictions_df.empty:
        test_predictions_df = test_predictions_df[~((test_predictions_df.get("mode", "") == "pretrain") & (test_predictions_df.get("split", "") == "test"))].copy()

    # Keep test predictions per-fold when fold information is available.
    if not test_predictions_df.empty:
        # Prefer grouping by sound_id, mode, and fold_id when available so each fold's
        # test predictions are preserved instead of collapsing to a single 'all' fold.
        test_group_keys = [key for key in ["sound_id", "mode", "fold_id"] if key in test_predictions_df.columns]

        grouped = test_predictions_df.groupby(test_group_keys, dropna=False, sort=False)
        test_summary_df = grouped.agg(
            ground_truth=("ground_truth", "first") if "ground_truth" in test_predictions_df.columns else ("sound_id", "first"),
            prediction=("prediction", lambda s: s.astype(str).value_counts().idxmax() if len(s) else "N/A"),
            prediction_score=("prediction_score", lambda s: round(float(pd.to_numeric(s, errors="coerce").mean()), 4)),
        ).reset_index()
        # Do not force mode/fold_id to master/"all"; keep inferred fold-level results.
        if "split" not in test_summary_df.columns:
            test_summary_df["split"] = "test"
        test_summary_df["is_correct"] = test_summary_df["ground_truth"].astype(str) == test_summary_df["prediction"].astype(str)
    else:
        test_summary_df = pd.DataFrame(columns=["sound_id", "mode", "fold_id", "split", "ground_truth", "prediction", "prediction_score", "is_correct"])

    # Merge the non-test rows using detailed keys when available.
    master_non_test_df = master_df[master_df["split"] != "test"].copy()
    master_test_df = master_df[master_df["split"] == "test"].copy()

    if not master_test_df.empty:
        master_test_df = master_test_df.drop_duplicates(subset=[col for col in ["sound_id", "mode"] if col in master_test_df.columns])

    merge_keys = ["sound_id", "mode", "fold_id", "split"]
    if not other_predictions_df.empty and all(key in master_non_test_df.columns for key in merge_keys) and all(key in other_predictions_df.columns for key in merge_keys):
        merged_other_df = master_non_test_df.merge(other_predictions_df, how="outer", on=merge_keys, suffixes=("_split", "_pred"))
    elif not other_predictions_df.empty:
        merged_other_df = master_non_test_df.merge(other_predictions_df, how="outer", on=["sound_id"], suffixes=("_split", "_pred"))
    else:
        merged_other_df = master_non_test_df.copy()

    if not master_test_df.empty or not test_summary_df.empty:
        test_meta_cols = [col for col in ["sound_id", "dataset_source", "mode"] if col in master_test_df.columns]
        if test_meta_cols:
            test_meta_df = master_test_df[test_meta_cols].drop_duplicates(subset=[col for col in ["sound_id", "mode"] if col in test_meta_cols])
        else:
            test_meta_df = pd.DataFrame(columns=["sound_id", "mode", "dataset_source"])

        if not test_meta_df.empty and not test_summary_df.empty:
            join_keys = [key for key in ["sound_id", "mode"] if key in test_meta_df.columns and key in test_summary_df.columns]
            # Keep fold-level prediction rows as the base and attach metadata when available.
            merged_test_df = test_summary_df.merge(test_meta_df, how="left", on=join_keys)
        elif not test_meta_df.empty:
            merged_test_df = test_meta_df.copy()
        else:
            merged_test_df = test_summary_df.copy()

        if "split" not in merged_test_df.columns:
            merged_test_df["split"] = "test"
        else:
            merged_test_df["split"] = "test"
    else:
        merged_test_df = pd.DataFrame()

    merged_df = pd.concat([merged_other_df, merged_test_df], ignore_index=True, sort=False)

    def _derive_filtering_status(split_value: object) -> str:
        split_text = str(split_value)
        if split_text.startswith("excluded_"):
            return split_text
        if split_text in {"train", "val", "test"}:
            return "included"
        return split_text

    if "split" in merged_df.columns:
        merged_df["filtering_status"] = merged_df["split"].apply(_derive_filtering_status)
    else:
        merged_df["filtering_status"] = "unknown"

    preferred_columns = [
        "sound_id",
        "dataset_source",
        "mode",
        "fold_id",
        "split",
        "filtering_status",
        "ground_truth",
        "prediction",
        "prediction_score",
        "is_correct",
    ]
    ordered_columns = [col for col in preferred_columns if col in merged_df.columns]
    remaining_columns = [col for col in merged_df.columns if col not in ordered_columns]
    merged_df = merged_df[ordered_columns + remaining_columns]

    # Sort by mode, numeric fold_id (when possible), split, then sound_id for a clear grouped ordering
    if "mode" in merged_df.columns:
        merged_df["mode"] = merged_df["mode"].astype(str)
    else:
        merged_df["mode"] = "N/A"

    if "fold_id" in merged_df.columns:
        merged_df["fold_sort"] = pd.to_numeric(merged_df["fold_id"], errors="coerce").fillna(9999).astype(int)
    else:
        merged_df["fold_id"] = "N/A"
        merged_df["fold_sort"] = 9999

    sort_keys = ["mode", "fold_sort", "split", "sound_id"]
    merged_df = merged_df.sort_values(by=sort_keys, kind="mergesort")
    merged_df = merged_df.drop(columns=["fold_sort"])

    # Save using a clearer, usage-focused filename
    ultimate_path = os.path.join(output_dir, "data_usage_summary.csv")
    merged_df.to_csv(ultimate_path, index=False)
    print(f"Saved data usage summary to {ultimate_path}")
    return ultimate_path


class ClassificationEvaluator:
    """Evaluator that returns metrics and writes confusion matrices."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def evaluate(self) -> Dict[str, Any]:
        raise NotImplementedError("Use evaluate_model() for now.")
