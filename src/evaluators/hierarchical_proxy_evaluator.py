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

from src.evaluators.classification_evaluator import compute_classification_metrics_from_predictions
from src.losses import HierarchicalProxyLoss
from src.models.hatr import BaseClassifier
from src.models.hatr_proxy_anchor import BaseClassifierProxyAnchor
from src.models.hatr_proxy_anchor_deep_shared import BaseClassifierProxyAnchorDeepShared
from src.models.hatr_proxy_anchor_simple import BaseClassifierProxyAnchorSimple
from src.utils.core_utils import build_id_to_class_mapping


def _get_class_labels(class_dict: Dict[str, int]) -> List[str]:
    return sorted(class_dict.keys(), key=lambda x: class_dict[x])


def _compute_normalized_confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    return np.nan_to_num(cm_normalized)


def _plot_confusion_matrix(cm_normalized: np.ndarray, class_labels: List[str], title: str, save_path: str) -> None:
    plt.figure(figsize=(14, 12))
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


def _resolve_model_class_from_checkpoint(checkpoint: Dict) -> type:
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    state_dict = checkpoint.get("model_state", {}) if isinstance(checkpoint, dict) else {}
    model_name = str(config.get("model_name", "")).strip().lower()
    state_keys = set(state_dict.keys()) if isinstance(state_dict, dict) else set()

    def has_any(prefixes):
        return any(any(key.startswith(prefix) for prefix in prefixes) for key in state_keys)

    if "deep_shared" in model_name or "deepshared" in model_name or "proxy_anchor_shared" in model_name:
        return BaseClassifierProxyAnchorDeepShared
    if "proxy_anchor_simple" in model_name or (
        "latent_projector.weight" in state_keys and "class_predictor.weight" in state_keys
    ):
        return BaseClassifierProxyAnchorSimple
    if "proxy_anchor" in model_name or has_any(["proxy_projector."]):
        return BaseClassifierProxyAnchor
    return BaseClassifier


@dataclass
class HierarchicalProxyClassificationEvaluator:
    proxy_loss: Optional[torch.nn.Module] = None
    class_to_topclass: Optional[Dict[int, int]] = None
    class_dict: Optional[Dict[str, int]] = None
    parent_of_child: Optional[torch.Tensor] = None

    def _load_proxy_loss_from_checkpoint(self, checkpoint: Dict, device: torch.device):
        proxy_state = checkpoint.get("proxy_loss_state") or checkpoint.get("criterion_state")
        if proxy_state is None:
            if self.proxy_loss is None:
                raise ValueError("Checkpoint does not contain hierarchical proxy state and no proxy_loss was provided.")
            return self.proxy_loss.to(device)

        child_proxies = proxy_state.get("child_proxies")
        parent_proxies = proxy_state.get("parent_proxies")
        if child_proxies is None or parent_proxies is None:
            raise ValueError("Hierarchical proxy state must contain child_proxies and parent_proxies.")

        loss_cfg = checkpoint.get("proxy_loss_config", {}) or {}
        proxy_loss = HierarchicalProxyLoss(
            embedding_dim=child_proxies.shape[1],
            num_parents=parent_proxies.shape[0],
            num_children=child_proxies.shape[0],
            temperature=loss_cfg.get("temperature", 0.07),
            alpha=loss_cfg.get("alpha", 0.4),
            beta=loss_cfg.get("beta", 0.3),
            gamma=loss_cfg.get("gamma", 0.15),
            delta=loss_cfg.get("delta", 0.05),
            sibling_margin=loss_cfg.get("sibling_margin", 0.4),
            parent_margin=loss_cfg.get("parent_margin", 0.0),
        )
        proxy_loss.load_state_dict(proxy_state)
        return proxy_loss.to(device)

    def _collect_predictions(self, model, data_loader, device):
        if self.class_to_topclass is None:
            raise ValueError("class_to_topclass is required for hierarchical proxy evaluation.")
        if self.proxy_loss is None:
            raise ValueError("proxy_loss is required for hierarchical proxy evaluation.")

        model.eval()
        self.proxy_loss.eval()
        child_proxies = F.normalize(self.proxy_loss.child_proxies.to(device), dim=1)
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
                z = outputs.get("z") if isinstance(outputs, dict) else outputs[0]
                logits = torch.matmul(F.normalize(z, dim=1), child_proxies.T) / self.proxy_loss.temperature
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

    def evaluate_model(self, model_class, model_path, data_loader, device, output_dir, fold_id, class_dict=None, split="test"):
        checkpoint = torch.load(model_path, map_location=device)
        resolved_model_class = _resolve_model_class_from_checkpoint(checkpoint)
        if model_class is not None and model_class is not resolved_model_class:
            print(
                f"[WARN] Hierarchical proxy evaluator resolved {resolved_model_class.__name__} from checkpoint, "
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
        df.to_csv(os.path.join(eval_dir, "predictions.csv"), index=False)

        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        class_labels = _get_class_labels(self.class_dict)
        cm_normalized = _compute_normalized_confusion_matrix(
            predictions["gt"], predictions["pred"], num_classes=len(class_labels)
        )
        pd.DataFrame(cm_normalized, index=class_labels, columns=class_labels).to_csv(
            os.path.join(output_dir, "confusion_matrix.csv")
        )
        _plot_confusion_matrix(
            cm_normalized,
            class_labels,
            f"Confusion Matrix | Fold {fold_id}",
            os.path.join(output_dir, "confusion_matrix.png"),
        )

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
        with open(os.path.join(eval_dir, "results.txt"), "w") as f:
            for _, key in metrics_to_log.items():
                f.write(f"{key}: {metrics[key]:.2f}%\n")
        for label, key in metrics_to_log.items():
            print(f"[HierarchicalProxyClassificationEvaluator | Fold {fold_id}] {label}: {metrics[key]:.2f}%")

        return metrics
