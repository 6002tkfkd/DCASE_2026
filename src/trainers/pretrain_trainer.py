import json
import os
import shutil
from collections import defaultdict
import collections.abc

import numpy as np
import torch
import torch.nn as nn

from src.losses import CrossEntropyLoss
from src.utils.loss_loader import get_loss_function
from src.utils.core_utils import load_config, set_seed
from src.models.hatr import BaseClassifier
from src.models.builder import build_model
from src.trainers.base_trainer import BaseTrainer
from src.datasets.data_manager import HATRDataManager
from src.evaluators.classification_evaluator import collect_predictions_and_metrics
from src.evaluators.classification_evaluator import merge_ultimate_analysis
from src.trainers.common_artifacts import save_history_json, save_history_with_plots
from src.trainers.common_metrics import (
    append_validation_metrics,
    format_epoch_metrics_line,
    normalize_monitor_metric,
    resolve_training_config,
)

class_dict = None
mode = None


def _identify_aux_dataset(config):
    datasets_cfg = config.get('datasets', {})
    for name in datasets_cfg.keys():
        if '35k' in name.lower():
            return name
    return None


# Training/load utilities are provided by BaseTrainer; leave implementations centralized there.


def _resolve_training_config(config):
    return resolve_training_config(config, ('pretrain',))


def _resolve_model_config(config):
    pretrain_cfg = config.get('pretrain', {}) if isinstance(config.get('pretrain'), dict) else {}
    stage_model_cfg = pretrain_cfg.get('model', {}) if isinstance(pretrain_cfg.get('model'), dict) else {}
    global_model_cfg = config.get('model', {}) if isinstance(config.get('model'), dict) else {}
    return stage_model_cfg or global_model_cfg


def _resolve_run_mode(config):
    pretrain_cfg = config.get('pretrain', {}) if isinstance(config.get('pretrain'), dict) else {}
    model_cfg = config.get('model', {}) if isinstance(config.get('model'), dict) else {}
    run_mode = pretrain_cfg.get('run_mode', model_cfg.get('run_mode'))
    if run_mode is not None:
        return str(run_mode).strip().lower()

    training_cfg = _resolve_training_config(config)
    proxy_loss = training_cfg.get('proxy_loss') or training_cfg.get('proxy_loss_name')
    classifier_loss = training_cfg.get('classifier_loss') or training_cfg.get('classifier_loss_name')
    if proxy_loss and classifier_loss:
        return 'dual_head'
    if proxy_loss:
        return 'proxy_only'
    return 'standard'


def _resolve_pretrain_validation_size(config, total_samples):
    pretrain_cfg = config.get('pretrain', {}) if isinstance(config.get('pretrain'), dict) else {}
    validation_split_cfg = (
        pretrain_cfg.get('validation_split', {})
        if isinstance(pretrain_cfg.get('validation_split'), dict)
        else {}
    )

    validation_size = validation_split_cfg.get('size', pretrain_cfg.get('validation_size'))
    if validation_size is not None:
        return int(validation_size)

    validation_ratio = validation_split_cfg.get('ratio', pretrain_cfg.get('validation_ratio'))
    if validation_ratio is not None:
        ratio = float(validation_ratio)
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"pretrain.validation_ratio must be between 0 and 1, got {ratio}")
        return max(1, int(round(total_samples * ratio)))

    return 3500


class PretrainTrainer(BaseTrainer):
    """Pre-training on 35k noisy data."""

    def __init__(self, config=None) -> None:
        # accept merged config dict or fall back to file
        self.config = config if isinstance(config, dict) else (load_config() or {})

    def run(self) -> None:
        global class_dict
        global mode

        seed = set_seed()  # For reproducibility
        self.data_manager = HATRDataManager(self.config)

        class_dict, top_class_dict, class_to_top_class = self.data_manager.get_class_mappings(mode="pretrain")

        # Configuration for pre-training on 35k
        mode = 'both'
        model_output = os.path.join(
            self.config.get('output', {}).get('root', self.config.get('paths', {}).get('output_root', './output')),
            self.config.get('output', {}).get('pretrain_dir', 'model_output_pretrain'),
            self.config.get('model', {}).get('name', 'base_classifier'),
        )

        training_cfg = _resolve_training_config(self.config)
        model_cfg = _resolve_model_config(self.config)
        run_mode = _resolve_run_mode(self.config)
        batch_size = int(training_cfg.get('batch_size', 128))
        num_epochs = int(training_cfg.get('num_epochs', 20))
        learning_rate = float(training_cfg.get('lr', training_cfg.get('learning_rate', 1e-4)))
        classification_weight = float(training_cfg.get('classification_weight', 1))
        proxy_weight = float(training_cfg.get('proxy_weight', 1))
        proxy_loss_name = training_cfg.get('proxy_loss') or training_cfg.get('proxy_loss_name')
        proxy_loss_params = training_cfg.get('proxy_loss_params', {}) if isinstance(training_cfg.get('proxy_loss_params'), dict) else {}
        classifier_loss_name = training_cfg.get('classifier_loss') or training_cfg.get('classifier_loss_name')
        classifier_loss_params = training_cfg.get('classifier_loss_params', {}) if isinstance(training_cfg.get('classifier_loss_params'), dict) else {}
        scheduler_type = training_cfg.get('scheduler_type', 'plateau')
        scheduler_params = training_cfg.get('scheduler_params', {}) if isinstance(training_cfg.get('scheduler_params'), dict) else {}
        patience = int(training_cfg.get('patience', 5))
        early_stopping_metric = training_cfg.get('early_stopping_metric', 'accuracy')
        monitor_metric_label = 'hF1' if normalize_monitor_metric(early_stopping_metric) == 'hierarchical_f1' else 'accuracy'

        print(f"\n{'='*80}")
        print("STEP 1: Pre-training on 35k Noisy Data")
        print(f"{'='*80}\n")

        loaders = self.data_manager.get_dataloaders(mode="pretrain")
        train_loader, val_loader, _ = loaders

        train_df = train_loader.dataset.dataframe
        val_df = val_loader.dataset.dataframe
        print(f"Loaded 35k dataset: {len(train_df) + len(val_df)} samples")
        print(f"Training set: {len(train_df)} | Validation set: {len(val_df)}")
        print(f"Learning rate: {learning_rate} | Max epochs: {num_epochs}\n")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}\n")

        # Model setup — auto-detect audio embedding dim from first sample
        first_audio_path = train_loader.dataset.dataframe['audio_emb_filepath'].iloc[0]
        emb_size_audio = int(np.load(first_audio_path).shape[0])
        print(f"Auto-detected emb_size_audio={emb_size_audio} from {first_audio_path}")
        emb_size_text = 512
        hidden_size = 128
        dropout = 0.1
        use_batch_norm = True

        model = build_model(
            config=self.config,
            run_mode=run_mode,
            hidden_size=hidden_size,
            num_classes=len(class_dict),
            emb_size_audio=emb_size_audio,
            emb_size_text=emb_size_text,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            mode=mode,
        ).to(device)

        classification_criterion = CrossEntropyLoss()

        os.makedirs(model_output, exist_ok=True)

        self.init_weights(model)

        # Train model
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
            output_dir=model_output,
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
            early_stopping_metric=early_stopping_metric,
            scheduler_params=scheduler_params,
        )

        best_model_path = os.path.join(model_output, "best_model.pth")
        pretrained_model_path = os.path.join(model_output, "pretrained_model.pth")
        if os.path.exists(best_model_path):
            shutil.copy2(best_model_path, pretrained_model_path)

        save_history_with_plots(
            history,
            model_output,
            serializer=self.make_serializable,
            fold_label="pretrain",
        )

        # Save model info
        model_info = {
            'stage': 'pretrain_35k',
            'model_class': trained_model.__class__.__name__,
            'hidden_size': hidden_size,
            'num_classes': len(class_dict),
            'emb_size_audio': emb_size_audio,
            'emb_size_text': emb_size_text,
            'dropout': dropout,
            'use_batch_norm': True,
            'mode': mode,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_epochs': num_epochs,
            'actual_epochs': len(history['val_loss']),
            'best_validation_metric': best_score,
            'best_val_loss': best_score,
            'early_stopping_metric': monitor_metric_label,
            'random_seed': seed,
        }

        info_path = os.path.join(model_output, "pretrain_info.json")
        with open(info_path, "w") as f:
            json.dump(self.make_serializable(model_info), f, indent=2)

        print(f"\n{'='*80}")
        print("Pre-training completed!")
        print(f"Pretrained model saved to: {pretrained_model_path}")
        print(f"Best validation {monitor_metric_label}: {best_score:.2f}%")
        merge_ultimate_analysis(model_output)
        print(f"{'='*80}\n")
