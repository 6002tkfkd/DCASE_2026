import json
import os
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.plot_history import plot_history


def save_history_json(
    history: Dict[str, Any],
    output_dir: str,
    serializer: Callable[[Any], Any],
) -> str:
    """Persist only `history.json` without generating plots."""
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(serializer(history), f, indent=2)
    return history_path


def save_history_with_plots(
    history: Dict[str, Any],
    output_dir: str,
    serializer: Callable[[Any], Any],
    fold_label: Optional[str] = None,
) -> str:
    """Persist `history.json` and generate per-epoch plots."""
    history_path = save_history_json(history, output_dir, serializer)

    try:
        created = plot_history(history_path, out_dir=output_dir)
        if created:
            label = fold_label or os.path.basename(output_dir)
            names = ", ".join(os.path.basename(p) for p in created)
            print(f"Saved plots for {label}: {names}")
    except Exception as exc:
        label = fold_label or os.path.basename(output_dir)
        print(f"Warning: failed to create plots for {label}: {exc}")

    return history_path


def _plot_cm(cm: np.ndarray, labels: list[str], title: str, save_path: str) -> None:
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
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


def summarize_mode_outputs(mode_dir: str, mode_name: str, serializer: Callable[[Any], Any]) -> None:
    fold_dirs = sorted(
        [
            os.path.join(mode_dir, d)
            for d in os.listdir(mode_dir)
            if d.startswith("fold_") and os.path.isdir(os.path.join(mode_dir, d))
        ]
    )
    if not fold_dirs:
        print(f"No fold directories found for mode={mode_name}; skipping summary")
        return

    metrics_per_fold = []
    cm_mats = []
    cm_labels = None

    for fold_dir in fold_dirs:
        metrics_path = os.path.join(fold_dir, "metrics.json")
        cm_path = os.path.join(fold_dir, "confusion_matrix.csv")

        if os.path.isfile(metrics_path):
            with open(metrics_path, "r") as f:
                metrics_per_fold.append(json.load(f))

        if os.path.isfile(cm_path):
            cm_df = pd.read_csv(cm_path, index_col=0)
            cm_labels = cm_df.columns.tolist()
            cm_mats.append(cm_df.to_numpy(dtype=float))

    if not metrics_per_fold:
        print(f"No metrics.json found for mode={mode_name}; skipping metrics summary")
    else:
        metric_keys = sorted(metrics_per_fold[0].keys())
        metrics_summary = {
            "mode": mode_name,
            "num_folds": len(metrics_per_fold),
            "mean": {},
            "std": {},
        }
        for key in metric_keys:
            values = [float(m[key]) for m in metrics_per_fold if key in m]
            if values:
                metrics_summary["mean"][key] = float(np.mean(values))
                metrics_summary["std"][key] = float(np.std(values, ddof=0))

        summary_json_path = os.path.join(mode_dir, "summary_metrics.json")
        with open(summary_json_path, "w") as f:
            json.dump(serializer(metrics_summary), f, indent=2)

        summary_txt_path = os.path.join(mode_dir, "summary_results.txt")
        with open(summary_txt_path, "w") as f:
            for key in metric_keys:
                if key in metrics_summary["mean"] and key in metrics_summary["std"]:
                    f.write(
                        f"{key}: mean={metrics_summary['mean'][key]:.2f}%, std={metrics_summary['std'][key]:.2f}%\n"
                    )

        print(f"Saved mode summary metrics to {summary_json_path}")

    if not cm_mats:
        print(f"No confusion_matrix.csv found for mode={mode_name}; skipping CM summary")
    else:
        cm_stack = np.stack(cm_mats, axis=0)
        cm_mean = np.mean(cm_stack, axis=0)
        cm_std = np.std(cm_stack, axis=0, ddof=0)

        if cm_labels is None:
            cm_labels = [str(i) for i in range(cm_mean.shape[0])]

        cm_mean_csv = os.path.join(mode_dir, "confusion_matrix_mean.csv")
        cm_std_csv = os.path.join(mode_dir, "confusion_matrix_std.csv")
        pd.DataFrame(cm_mean, index=cm_labels, columns=cm_labels).to_csv(cm_mean_csv)
        pd.DataFrame(cm_std, index=cm_labels, columns=cm_labels).to_csv(cm_std_csv)

        cm_mean_png = os.path.join(mode_dir, "confusion_matrix_mean.png")
        _plot_cm(cm_mean, cm_labels, f"Mean Confusion Matrix | mode={mode_name}", cm_mean_png)

        print(f"Saved mean/std confusion matrices to {mode_dir}")
