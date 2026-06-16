from typing import Any, Dict, Iterable, Optional


def resolve_training_config(config: Dict[str, Any], stage_keys: Iterable[str] = ()) -> Dict[str, Any]:
    """Resolve the most specific training config block available.

    Stage-specific blocks such as config['pretrain']['training'] or
    config['finetune']['training'] take priority when stage_keys are provided.
    Falls back to the global config['training'] block.
    """
    for stage_key in stage_keys:
        stage_cfg = config.get(stage_key, {}) if isinstance(config.get(stage_key), dict) else {}
        stage_training_cfg = stage_cfg.get('training', {}) if isinstance(stage_cfg.get('training'), dict) else {}
        if stage_training_cfg:
            return stage_training_cfg

    global_training_cfg = config.get('training', {}) if isinstance(config.get('training'), dict) else {}
    return global_training_cfg


def normalize_monitor_metric(metric_name: object) -> str:
    """Normalize the early-stopping metric name to a canonical key.

    Supported values: accuracy, hF1.
    """
    normalized = 'accuracy' if metric_name is None else str(metric_name).strip().lower()
    aliases = {
        'accuracy': 'accuracy',
        'acc': 'accuracy',
        'top1': 'accuracy',
        'hf1': 'hierarchical_f1',
        'h_f1': 'hierarchical_f1',
        'hierarchical_f1': 'hierarchical_f1',
        'hierarchical_f1-score': 'hierarchical_f1',
        'hierarchical f1': 'hierarchical_f1',
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported early_stopping_metric='{metric_name}'. Expected 'accuracy' or 'hF1'."
        )
    return aliases[normalized]


def get_monitor_score(val_metrics: Dict[str, Any], metric_name: object) -> float:
    """Return the validation score used for checkpointing, scheduling, and early stopping."""
    monitor_key = normalize_monitor_metric(metric_name)
    if monitor_key == 'accuracy':
        return float(val_metrics['accuracy'])
    return float(val_metrics.get('hierarchical_f1', val_metrics.get('hierarchical_f1-score', 0.0)))


def append_validation_metrics(history: Dict[str, list], val_metrics: Dict[str, Any]) -> float:
    """Append common validation metrics to history and return accuracy."""
    val_accuracy = float(val_metrics["accuracy"])
    hierarchical_f1 = float(
        val_metrics.get("hierarchical_f1", val_metrics.get("hierarchical_f1-score", 0.0))
    )
    history["val_accuracy"].append(val_accuracy)
    history["val_top_accuracy"].append(float(val_metrics["top_accuracy"]))
    history["val_macro_accuracy"].append(float(val_metrics["macro_accuracy"]))
    history["val_macro_top_accuracy"].append(float(val_metrics["macro_top_accuracy"]))
    history["val_hierarchical_accuracy"].append(float(val_metrics["hierarchical_accuracy"]))
    history["val_hierarchical_precision"].append(float(val_metrics["hierarchical_precision"]))
    history["val_hierarchical_recall"].append(float(val_metrics["hierarchical_recall"]))
    history["val_hierarchical_f1"].append(hierarchical_f1)
    return val_accuracy


def format_epoch_metrics_line(
    epoch: int,
    num_epochs: int,
    val_metrics: Dict[str, Any],
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
) -> str:
    """Build a unified epoch log line across trainers."""
    prefix = f"Epoch [{epoch + 1}/{num_epochs}] - "
    if train_loss is not None:
        prefix += f"train_loss={train_loss:.4f} | "
    if val_loss is not None:
        prefix += f"val_loss={val_loss:.4f} | "

    return (
        prefix
        + f"acc={float(val_metrics['accuracy']):.2f}% | "
        + f"top={float(val_metrics['top_accuracy']):.2f}% | "
        + f"macro={float(val_metrics['macro_accuracy']):.2f}% | "
        + f"macro_top={float(val_metrics['macro_top_accuracy']):.2f}% | "
        + f"hier={float(val_metrics['hierarchical_accuracy']):.2f}% | "
        + f"hp={float(val_metrics['hierarchical_precision']):.2f}% | "
        + f"hr={float(val_metrics['hierarchical_recall']):.2f}% | "
        + f"hierarchical_f1-score={float(val_metrics.get('hierarchical_f1', val_metrics.get('hierarchical_f1-score', 0.0))):.2f}%"
    )
