import json
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _maybe_get(xs: Dict, key: str):
    return xs.get(key) or xs.get(key.replace('val_', '')) or xs.get(key.replace('train_', ''))


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def plot_history(history_path: str, out_dir: str = None, metrics_to_plot: List[str] = None) -> List[str]:
    """Read a `history.json` produced by trainers and save epoch-wise plots.

    Args:
        history_path: path to history.json
        out_dir: directory to save plots; if None uses same folder as history_path
        metrics_to_plot: optional list of metric keys to plot. If None, automatic groups are used.

    Returns:
        List of created plot file paths.
    """
    if not os.path.isfile(history_path):
        raise FileNotFoundError(history_path)

    with open(history_path, 'r') as f:
        history = json.load(f)

    if out_dir is None:
        out_dir = os.path.dirname(history_path)
    plots_dir = os.path.join(out_dir, 'plots')
    _ensure_dir(plots_dir)

    # Flatten simple sequences into numpy arrays where possible
    flat = {}
    for k, v in history.items():
        try:
            arr = np.array(v)
            if arr.ndim == 1:
                flat[k] = arr
        except Exception:
            continue

    created = []

    # Default groups
    loss_keys = [k for k in flat.keys() if 'loss' in k]
    acc_keys = [k for k in flat.keys() if 'acc' in k or 'accuracy' in k]
    lr_keys = [k for k in flat.keys() if 'lr' in k or 'learning_rate' in k or 'learning_rates' in k]

    keys_for_individual_plots = metrics_to_plot if metrics_to_plot else sorted(flat.keys())

    # Always create one figure per metric key.
    for key in keys_for_individual_plots:
        if key not in flat:
            continue
        x = np.arange(1, len(flat[key]) + 1)
        fig_path = os.path.join(plots_dir, f"{_safe_filename(key)}_per_epoch.png")
        plt.figure(figsize=(6, 4))
        plt.plot(x, flat[key], linestyle='-', linewidth=2)
        plt.title(f'{key} per epoch')
        plt.xlabel('epoch')
        plt.ylabel(key)
        plt.xticks(x)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        created.append(fig_path)

    return created
