import json
import os
from collections import defaultdict
import collections.abc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.losses import CrossEntropyLoss
from src.utils.loss_loader import get_loss_function
from src.utils.core_utils import (
    load_config,
    set_seed,
    build_class_to_topclass_mapping,
    build_class_to_topclass_tensor,
)
from src.models.hatr import BaseClassifier
from src.models.builder import build_model
from src.evaluators import get_evaluator
from src.datasets.data_manager import HATRDataManager
from src.trainers.base_trainer import BaseTrainer
from src.evaluators.classification_evaluator import (
    collect_predictions_and_metrics,
    evaluate_model,
    merge_ultimate_analysis,
)
from src.trainers.common_artifacts import (
    save_history_json,
    save_history_with_plots,
    summarize_mode_outputs,
)
from src.trainers.common_metrics import (
    append_validation_metrics,
    format_epoch_metrics_line,
    normalize_monitor_metric,
    resolve_training_config,
)

datasets_cfg = None
data_dir = None
processed_basename = None
prepared_dataset_main = None
class_dict_json = None
top_class_dict_json = None

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


def _resolve_training_config(config):
    return resolve_training_config(config, ('finetune',))


def _resolve_model_config(config):
    finetune_cfg = config.get('finetune', {}) if isinstance(config.get('finetune'), dict) else {}
    stage_model_cfg = finetune_cfg.get('model', {}) if isinstance(finetune_cfg.get('model'), dict) else {}
    global_model_cfg = config.get('model', {}) if isinstance(config.get('model'), dict) else {}
    return stage_model_cfg or global_model_cfg


def _resolve_run_mode(config):
    finetune_cfg = config.get('finetune', {}) if isinstance(config.get('finetune'), dict) else {}
    model_cfg = config.get('model', {}) if isinstance(config.get('model'), dict) else {}
    run_mode = finetune_cfg.get('run_mode', model_cfg.get('run_mode'))
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


def _identify_main_dataset(config):
    """Identify 10k dataset from config."""
    datasets_cfg = config.get('datasets', {})
    for name in datasets_cfg.keys():
        if '10k' in name.lower():
            return name
    return list(datasets_cfg.keys())[0] if datasets_cfg else None


class FinetuneTrainer(BaseTrainer):
    """Fine-tuning on 10k verified data."""

    def __init__(self, config=None) -> None:
        # accept merged config dict or fallback to file
        self.config = config if isinstance(config, dict) else (load_config() or {})

    def run(self) -> None:
        global hidden_size
        global class_dict
        global emb_size_audio
        global emb_size_text
        global dropout
        global mode

        seed = set_seed()  # For reproducibility
        self.data_manager = HATRDataManager(self.config)

        class_dict, top_class_dict, class_to_top_class = self.data_manager.get_class_mappings(mode="finetune")

        self.main_dataset_name = _identify_main_dataset(self.config)
        if self.main_dataset_name is None:
            raise ValueError("10k dataset not found in config!")

        print(f"\n{'='*80}")
        print("STEP 2: Fine-tuning on 10k Verified Data")
        print(f"{'='*80}\n")

        modes = _resolve_modes_from_config(self.config)
        model_output = os.path.join(
            self.config.get('output', {}).get('root', self.config.get('paths', {}).get('output_root', './output')),
            self.config.get('output', {}).get('run_dir', 'model_output'),
            self.config.get('model', {}).get('name', 'base_classifier'),
        )

        training_cfg = _resolve_training_config(self.config)
        model_cfg = _resolve_model_config(self.config)
        run_mode = _resolve_run_mode(self.config)
        batch_size = int(training_cfg.get('batch_size', 128))
        num_epochs = int(training_cfg.get('num_epochs', 100))
        learning_rate = float(training_cfg.get('lr', training_cfg.get('learning_rate', 1e-5)))
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

        loaders = self.data_manager.get_dataloaders(mode="finetune")

        for mode in modes:
            # Resolve pretrain checkpoint path from config output settings.
            output_cfg = self.config.get('output', {}) if isinstance(self.config.get('output'), dict) else {}
            paths_cfg = self.config.get('paths', {}) if isinstance(self.config.get('paths'), dict) else {}
            model_cfg = self.config.get('model', {}) if isinstance(self.config.get('model'), dict) else {}
            output_root = output_cfg.get('root', paths_cfg.get('output_root', './output'))
            pretrain_dirname = output_cfg.get('pretrain_dir', 'model_output_pretrain')
            model_name = model_cfg.get('name', 'base_classifier')

            pretrain_output_dir = os.path.join(output_root, pretrain_dirname, model_name)
            pretrained_model_path = os.path.join(pretrain_output_dir, 'pretrained_model.pth')

            # Backward-compatible fallback paths.
            if not os.path.exists(pretrained_model_path):
                best_model_candidate = os.path.join(pretrain_output_dir, 'best_model.pth')
                if os.path.exists(best_model_candidate):
                    pretrained_model_path = best_model_candidate
                else:
                    legacy_candidates = [
                        './model_output_pretrain_both/pretrained_model.pth',
                        './model_output_pretrain_audio/pretrained_model.pth',
                    ]
                    for cand in legacy_candidates:
                        if os.path.exists(cand):
                            pretrained_model_path = cand
                            break
            if not os.path.exists(pretrained_model_path):
                raise FileNotFoundError(f"Pre-trained model not found: {pretrained_model_path}")

            print(f"\n{'='*80}")
            print(f"Fine-tuning: Dataset={self.main_dataset_name} | Mode={mode}")
            print(f"{'='*80}\n")

            for fold, (train_loader, val_loader, test_loader) in enumerate(loaders):
                print(f"\n==== Fold {fold} ====")

                train_df = train_loader.dataset.dataframe
                val_df = val_loader.dataset.dataframe
                test_df = test_loader.dataset.dataframe

                print(
                    f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}"
                )

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                if mode in ['audio', 'both']:
                    first_audio_path = train_loader.dataset.dataframe['audio_emb_filepath'].iloc[0]
                    emb_size_audio = int(np.load(first_audio_path).shape[0])
                    print(f"Auto-detected emb_size_audio={emb_size_audio} from {first_audio_path}")
                else:
                    emb_size_audio = 0
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

                # Load pre-trained weights
                print(f"Loading pre-trained model from: {pretrained_model_path}")
                checkpoint = self.load_pretrained_weights(model, pretrained_model_path, device, mode)
                print("Pre-trained weights loaded successfully")

                classification_criterion = CrossEntropyLoss()

                output_dir = os.path.join(
                    model_output,
                    mode,
                    f"fold_{fold}",
                )
                os.makedirs(output_dir, exist_ok=True)

                model_path = os.path.join(output_dir, "best_model.pth")

                # Note: No weight re-initialization since we're using pre-trained weights

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
                    early_stopping_metric=early_stopping_metric,
                    scheduler_params=scheduler_params,
                )
                print(f"Best validation {monitor_metric_label}: {best_score:.2f}%")

                # Save updated history with model info
                history['model_info'] = {
                    'stage': 'finetune_10k',
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
                    'learning_rate': learning_rate,
                    'max_epochs': num_epochs,
                    'pretrained_from': '35k_noisy_data',
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
            summarize_mode_outputs(
                mode_dir=mode_dir,
                mode_name=mode,
                serializer=self.make_serializable,
            )

        merge_ultimate_analysis(model_output)
        print(f"\n{'='*80}")
        print("All fine-tuning experiments completed!")
        print(f"{'='*80}\n")
