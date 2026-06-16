from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from src.losses import Proxy_Anchor
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


def _l2_norm(input_tensor: torch.Tensor) -> torch.Tensor:
    input_size = input_tensor.size()
    buffer = torch.pow(input_tensor, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    output = torch.div(input_tensor, norm.view(-1, 1).expand_as(input_tensor))
    return output.view(input_size)


def _resolve_model_class_from_checkpoint(checkpoint: Dict) -> type:
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    state_dict = checkpoint.get("model_state", {}) if isinstance(checkpoint, dict) else {}
    model_name = str(config.get("model_name", "")).strip().lower()
    state_keys = set(state_dict.keys()) if isinstance(state_dict, dict) else set()

    def has_any(prefixes):
        return any(any(key.startswith(prefix) for prefix in prefixes) for key in state_keys)

    if "deep_shared" in model_name or "deepshared" in model_name or "proxy_anchor_shared" in model_name:
        return BaseClassifierProxyAnchorDeepShared
    if "proxy_anchor_simple" in model_name or has_any(["latent_projector.6.", "class_predictor."]):
        return BaseClassifierProxyAnchorSimple
    if "proxy_anchor" in model_name or has_any(["proxy_projector.", "proxy_classifier."]):
        return BaseClassifierProxyAnchor
    return BaseClassifier


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
            from src.metrics.hierarchical import hierarchical_accuracy

            acc = hierarchical_accuracy(c, pred_gt_pairs, lambda_param=0.5)
            if not np.isnan(acc):
                h_accs.append(acc)
        except Exception:
            continue

    hierarchical_acc = np.mean(h_accs) if h_accs else 0

    hps, hrs, hfs = [], [], []
    for c in classes:
        try:
            from src.metrics.hierarchical import hierarchical_prf_weighted

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


@dataclass
class ProxyClassificationEvaluator:
    proxy_loss: Optional[torch.nn.Module] = None
    class_to_topclass: Optional[Dict[int, int]] = None
    class_dict: Optional[Dict[str, int]] = None

    def _get_proxies(self, device: torch.device) -> torch.Tensor:
        if self.proxy_loss is None or not hasattr(self.proxy_loss, "proxies"):
            raise ValueError("Proxy loss with self.proxies is required for proxy evaluation.")
        return self.proxy_loss.proxies.to(device)

    def _collect_predictions(self, model, data_loader, device):
        if self.class_to_topclass is None:
            raise ValueError("class_to_topclass is required for proxy evaluation.")

        model.eval()
        proxies = self._get_proxies(device)
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
                    z = outputs.get("z")
                else:
                    z, _, _ = outputs
                logits = F.linear(_l2_norm(z), _l2_norm(proxies))
                probs = torch.softmax(logits, dim=1)

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

        metrics = compute_classification_metrics_from_predictions(
            predictions,
            self.class_to_topclass,
            self.class_dict,
        )
        return predictions, metrics

    def collect_predictions_and_metrics(self, model, data_loader, device):
        return self._collect_predictions(model, data_loader, device)

    def _load_proxy_loss_from_checkpoint(self, checkpoint: Dict, device: torch.device):
        proxy_state = checkpoint.get("proxy_loss_state")
        proxy_config = checkpoint.get("proxy_loss_config", {}) or {}
        if proxy_state is None:
            if self.proxy_loss is None:
                raise ValueError("Checkpoint does not contain proxy_loss_state and no proxy_loss was provided.")
            return self.proxy_loss.to(device)

        proxies = proxy_state.get("proxies")
        if proxies is None:
            raise ValueError("proxy_loss_state does not contain proxies.")

        nb_classes, sz_embed = proxies.shape
        proxy_loss = Proxy_Anchor(
            nb_classes=nb_classes,
            sz_embed=sz_embed,
            mrg=proxy_config.get("mrg", 0.1),
            alpha=proxy_config.get("alpha", 32),
        )
        proxy_loss.load_state_dict(proxy_state)
        return proxy_loss.to(device)

    def evaluate_model(self, model_class, model_path, data_loader, device, output_dir, fold_id, class_dict=None, split="test"):
        checkpoint = torch.load(model_path, map_location=device)
        resolved_model_class = _resolve_model_class_from_checkpoint(checkpoint)
        if model_class is not None and model_class is not resolved_model_class:
            print(
                f"[WARN] Proxy evaluator resolved {resolved_model_class.__name__} from checkpoint, "
                f"overriding passed {getattr(model_class, '__name__', model_class)}"
            )

        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        signature = inspect.signature(resolved_model_class.__init__)
        allowed_keys = {
            name
            for name, parameter in signature.parameters.items()
            if name != "self" and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        }
        filtered_config = {key: value for key, value in config.items() if key in allowed_keys and value is not None}

        model = resolved_model_class(**filtered_config)
        load_result = model.load_state_dict(checkpoint["model_state"], strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                f"[WARN] Loaded {resolved_model_class.__name__} with missing_keys={load_result.missing_keys} "
                f"unexpected_keys={load_result.unexpected_keys}"
            )
        model.to(device)

        self.class_dict = class_dict or self.class_dict
        self.proxy_loss = self._load_proxy_loss_from_checkpoint(checkpoint, device)

        predictions, metrics = self._collect_predictions(model, data_loader, device)

        id_to_class = build_id_to_class_mapping(self.class_dict) if self.class_dict else {}

        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        eval_dir = os.path.join(output_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

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
        pred_path = os.path.join(eval_dir, "predictions.csv")
        df.to_csv(pred_path, index=False)

        class_labels = _get_class_labels(self.class_dict)
        cm_normalized = _compute_normalized_confusion_matrix(
            predictions["gt"], predictions["pred"], num_classes=len(class_labels)
        )

        cm_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
        cm_df = pd.DataFrame(cm_normalized, index=class_labels, columns=class_labels)
        cm_df.to_csv(cm_csv_path)

        cm_png_path = os.path.join(output_dir, "confusion_matrix.png")
        title = f"Confusion Matrix | Fold {fold_id}"
        _plot_confusion_matrix(cm_normalized, class_labels, title, cm_png_path)

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

        results_path = os.path.join(eval_dir, "results.txt")
        with open(results_path, "w") as f:
            for _, key in metrics_to_log.items():
                f.write(f"{key}: {metrics[key]:.2f}%\n")

        for label, key in metrics_to_log.items():
            print(f"[ProxyClassificationEvaluator | Fold {fold_id}] {label}: {metrics[key]:.2f}%")

        return metrics