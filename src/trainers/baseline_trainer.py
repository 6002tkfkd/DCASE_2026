import json
import os
import shutil
from collections import defaultdict
import collections.abc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from src.utils.loss_loader import get_loss_function
from src.utils.core_utils import (
    load_config,
    set_seed,
    build_class_to_topclass_mapping,
    build_class_to_topclass_tensor,
)
from src.models.hatr import BaseClassifier
from src.models.builder import build_model
from src.datasets.data_manager import HATRDataManager
from src.trainers.base_trainer import BaseTrainer
from src.evaluators import get_evaluator
from src.evaluators.classification_evaluator import (
    collect_predictions_and_metrics,
    evaluate_model,
    merge_ultimate_analysis,
)
from src.trainers.common_artifacts import (
    save_history_json,
    save_history_with_plots,
    summarize_mode_outputs as common_summarize_mode_outputs,
)
from src.trainers.common_metrics import (
    append_validation_metrics,
    format_epoch_metrics_line,
    normalize_monitor_metric,
    resolve_training_config,
)

# Provide module-level alias so legacy module functions can call it
make_serializable = BaseTrainer.make_serializable


hidden_size = None
class_dict = None
emb_size_audio = None
emb_size_text = None
dropout = None
mode = None



def _resolve_modes_from_config(config):
    """Resolve list of modes from config.trainer.mode.

    Accepts string ('both', 'audio', 'both,audio') or list. Returns list of valid modes.
    """
    trainer_cfg = config.get('trainer', {}) if isinstance(config.get('trainer'), dict) else {}
    mode_cfg = trainer_cfg.get('mode', None)
    if mode_cfg is None:
        return ['both', 'audio']
    if isinstance(mode_cfg, str):
        modes = [m.strip() for m in mode_cfg.split(',') if m.strip()]
    elif isinstance(mode_cfg, (list, tuple)):
        modes = [str(m) for m in mode_cfg]
    else:
        modes = [str(mode_cfg)]
    allowed = {'both', 'audio', 'text'}
    modes = [m for m in modes if m in allowed]
    return modes if modes else ['both', 'audio']


def _resolve_run_mode(config):
    model_cfg = config.get('model', {}) if isinstance(config.get('model'), dict) else {}
    training_cfg = config.get('training', {}) if isinstance(config.get('training'), dict) else {}
    run_mode = training_cfg.get('run_mode', model_cfg.get('run_mode'))
    if run_mode is not None:
        return str(run_mode).strip().lower()

    proxy_loss = training_cfg.get('proxy_loss') or training_cfg.get('proxy_loss_name')
    classifier_loss = training_cfg.get('classifier_loss') or training_cfg.get('classifier_loss_name')
    if proxy_loss and classifier_loss:
        return 'dual_head'
    if proxy_loss:
        return 'proxy_only'
    return 'standard'


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


def summarize_mode_outputs(mode_dir: str, mode_name: str) -> None:
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
            json.dump(make_serializable(metrics_summary), f, indent=2)

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


# Training/load utilities are provided by BaseTrainer; implementations centralized there.


class BaselineTrainer(BaseTrainer):
    """Baseline trainer using the original single-dataset workflow."""

    def __init__(self, config=None) -> None:
        # config can be a merged dict passed from scripts or None
        self.config = config if isinstance(config, dict) else (load_config() or {})

        # Resolve key paths and names from config
        self.dataset_name = self.config.get('active_dataset')
        datasets_cfg = self.config.get('datasets', {})
        self.dataset_path = datasets_cfg.get(self.dataset_name, {}).get('metadata_csv')

        # Data/output paths - simplified structure
        paths_cfg = self.config.get('paths', {}) if isinstance(self.config.get('paths'), dict) else {}
        output_cfg = self.config.get('output', {}) if isinstance(self.config.get('output'), dict) else {}
        model_cfg = self.config.get('model', {}) if isinstance(self.config.get('model'), dict) else {}
        artifact_cfg = self.config.get('artifact_names', {}) if isinstance(self.config.get('artifact_names'), dict) else {}

        output_root = output_cfg.get('root', './output')
        run_dir = output_cfg.get('run_dir', 'model_output')
        model_name = model_cfg.get('name', 'base_classifier')
        self.data_dir = os.path.join(output_root, run_dir, model_name)

        self.processed_basename = paths_cfg.get('processed_basename', 'processed_dataset.csv')
        self.prepared_dataset_path = os.path.join(self.data_dir, self.processed_basename)

        self.class_dict_json = os.path.join(self.data_dir, artifact_cfg.get('class_dict_json', 'class_dict.json'))
        self.top_class_dict_json = os.path.join(self.data_dir, artifact_cfg.get('top_class_dict_json', 'top_class_dict.json'))
        self.subclass_json = os.path.join(self.data_dir, artifact_cfg.get('top_class_subclass_dict_json', 'top_class_subclass_dict.json'))

    def run(self) -> None:
        global hidden_size
        global class_dict
        global emb_size_audio
        global emb_size_text
        global dropout
        global mode

        seed = set_seed()  # For reproducibility
        self.data_manager = HATRDataManager(self.config)

        class_dict, top_class_dict, class_to_top_class = self.data_manager.get_class_mappings(mode="baseline")

        modes = _resolve_modes_from_config(self.config)
        run_mode = _resolve_run_mode(self.config)

        # Resolve training hyperparameters from config with sensible defaults
        training_cfg = resolve_training_config(self.config, ('baseline',))
        batch_size = int(training_cfg.get('batch_size', 64))
        num_epochs = int(training_cfg.get('num_epochs', 100))
        learning_rate = float(training_cfg.get('lr', training_cfg.get('learning_rate', 0.001)))
        classification_weight = float(training_cfg.get('classification_weight', 1))
        proxy_weight = float(training_cfg.get('proxy_weight', 1))
        proxy_loss_name = training_cfg.get('proxy_loss') or training_cfg.get('proxy_loss_name')
        proxy_loss_params = training_cfg.get('proxy_loss_params', {}) if isinstance(training_cfg.get('proxy_loss_params'), dict) else {}
        classifier_loss_name = training_cfg.get('classifier_loss') or training_cfg.get('classifier_loss_name')
        classifier_loss_params = training_cfg.get('classifier_loss_params', {}) if isinstance(training_cfg.get('classifier_loss_params'), dict) else {}
        scheduler_type = training_cfg.get('scheduler_type', 'step')
        scheduler_params = training_cfg.get('scheduler_params', {}) if isinstance(training_cfg.get('scheduler_params'), dict) else {}
        patience = int(training_cfg.get('patience', 5))
        early_stopping_metric = training_cfg.get('early_stopping_metric', 'accuracy')
        monitor_metric_label = 'hF1' if normalize_monitor_metric(early_stopping_metric) == 'hierarchical_f1' else 'accuracy'
        k_folds = int(training_cfg.get('k_folds', 5))
        evaluator_cfg = self.config.get('evaluator_config', {}) if isinstance(self.config.get('evaluator_config'), dict) else {}
        evaluator_type = evaluator_cfg.get('type', 'classification')

        model_output = os.path.join(
            self.config.get('output', {}).get('root', self.config.get('paths', {}).get('output_root', './output')),
            self.config.get('output', {}).get('run_dir', 'model_output'),
            self.config.get('model', {}).get('name', 'base_classifier'),
        )

        loaders = self.data_manager.get_dataloaders(mode="baseline")

        for mode in modes:
            print(f"\n=== Running experiments: Dataset={self.dataset_name} | Mode={mode} ===")

            for fold, (train_loader, val_loader, test_loader) in enumerate(loaders):
                print(f"\n==== Fold {fold} ====")

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                emb_size_audio = 512 if mode in ['audio', 'both'] else 0
                emb_size_text = 512 if mode in ['text', 'both'] else 0

                hidden_size = 128
                dropout = 0.1
                use_batch_norm = True

                model = build_model(
                    config=self.config,
                    run_mode=run_mode,
                    hidden_size=128,
                    num_classes=len(class_dict),
                    emb_size_audio=emb_size_audio,
                    emb_size_text=emb_size_text,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    mode=mode,
                ).to(device)
                model_class = model.__class__


                output_dir = os.path.join(
                    model_output,
                    mode,
                    f"fold_{fold}",
                )
                os.makedirs(output_dir, exist_ok=True)

                model_path = os.path.join(output_dir, "best_model.pth")

                self.init_weights(model)

                best_score, history, trained_model = self.train_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    class_to_top_class,
                    class_dict=class_dict,
                    num_epochs=num_epochs,
                    lr=learning_rate,
                    classification_weight=classification_weight,
                    output_dir=output_dir,
                    scheduler_type=scheduler_type,
                    patience=patience,
                    run_mode=run_mode,
                    loss_name='CrossEntropyLoss',
                    loss_params={},
                    proxy_loss_name=proxy_loss_name,
                    proxy_loss_params=proxy_loss_params,
                    classifier_loss_name=classifier_loss_name,
                    classifier_loss_params=classifier_loss_params,
                    proxy_weight=proxy_weight,
                    evaluator_type=evaluator_type,
                    early_stopping_metric=early_stopping_metric,
                    scheduler_params=scheduler_params,
                )
                print(f"Best validation {monitor_metric_label}: {best_score:.2f}%")

                # Save updated history with model info
                history['model_info'] = {
                    'model_class': trained_model.__class__.__name__,
                    'hidden_size': hidden_size,
                    'num_classes': len(class_dict),
                    'emb_size_audio': emb_size_audio,
                    'emb_size_text': emb_size_text,
                    'dropout': dropout,
                    'use_batch_norm': True,
                    'mode': mode,
                    'early_stopping_metric': monitor_metric_label,
                    'num_folds': k_folds,
                    'fold_id': fold,
                    'batch_size': batch_size,
                    'random_seed': seed,
                }

                save_history_with_plots(
                    history,
                    output_dir,
                    serializer=self.make_serializable,
                    fold_label=f"fold_{fold}",
                )

                # Testing
                class_to_top_class = build_class_to_topclass_mapping(class_dict, top_class_dict)
                subclass_to_topclass_tensor = build_class_to_topclass_tensor(
                    class_dict, top_class_dict, device
                )

                if run_mode in {'proxy_only', 'dual_head'}:
                    proxy_eval_name = (
                        "hierarchical_proxy_classification"
                        if str(proxy_loss_name).split(".")[-1] == "HierarchicalProxyLoss"
                        else "proxy_classification"
                    )
                    proxy_evaluator = get_evaluator(
                        proxy_eval_name,
                        class_to_topclass=class_to_top_class,
                        class_dict=class_dict,
                    )
                    metrics = proxy_evaluator.evaluate_model(
                        None,
                        model_path,
                        test_loader,
                        device,
                        output_dir=output_dir,
                        fold_id=fold,
                        class_dict=class_dict,
                    )
                else:
                    metrics = evaluate_model(
                        model_path,
                        test_loader,
                        device,
                        class_to_top_class,
                        output_dir=output_dir,
                        fold_id=fold,
                        class_dict=class_dict,
                        split="test",
                    )

                print("\n===== Fold Results =====")
                print(f"Final model accuracy: {metrics['accuracy']:.2f}%")
                print(f"Final model top-level accuracy: {metrics['top_accuracy']:.2f}%")
                print("========================")

            mode_dir = os.path.join(model_output, mode)
            common_summarize_mode_outputs(
                mode_dir=mode_dir,
                mode_name=mode,
                serializer=make_serializable,
            )

        merge_ultimate_analysis(model_output)
        print("All experiments done!")
