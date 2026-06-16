import json
import os
from collections import defaultdict
import collections.abc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from torch.utils.data import DataLoader

from src.losses import CrossEntropyLoss
from src.utils.loss_loader import get_loss_function
from src.utils.core_utils import (
    load_config,
    set_seed,
    build_class_to_topclass_mapping,
    build_class_to_topclass_tensor,
)
from src.models.builder import build_model
from src.models.hatr import BaseClassifier
from src.trainers.base_trainer import BaseTrainer
from src.datasets.hatr_dataset import HATRDataset
from src.datasets.prep import build_multi_dataset, load_processed_dataset
from src.filtering.pipeline import apply_filtering_df, build_report_path, make_filtering_config, make_filtering_config_by_stage
from src.evaluators.classification_evaluator import (
    collect_predictions_and_metrics,
    evaluate_model,
    merge_ultimate_analysis,
)
from src.evaluators import get_evaluator
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
color_dict_path = None
top_color_dict_path = None

data_dir = None
processed_basename = None
prepared_dataset_main = None
prepared_dataset_aux = None
class_dict_json = None
top_class_dict_json = None
subclass_json = None

main_dataset_name = None
aux_dataset_name = None


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
    training_cfg = resolve_training_config(config, ('combined',))
    run_mode = training_cfg.get('run_mode', model_cfg.get('run_mode'))
    if run_mode is not None:
        return str(run_mode).strip().lower()

    proxy_loss = training_cfg.get('proxy_loss') or training_cfg.get('proxy_loss_name')
    classifier_loss = (
        training_cfg.get('classifier_loss')
        or training_cfg.get('classifier_loss_name')
        or training_cfg.get('loss')
    )
    if proxy_loss and classifier_loss:
        return 'dual_head'
    if proxy_loss:
        return 'proxy_only'
    return 'standard'





class CombinedTrainer(BaseTrainer):
    """Combined trainer using the original 10k + 35k workflow."""

    def __init__(self, config=None) -> None:
        # Accept merged config dict or load default from file
        self.config = config if isinstance(config, dict) else (load_config() or {})

        datasets_cfg_local = self.config.get('datasets', {})
        self.datasets_cfg = datasets_cfg_local

        # attempt to identify main and aux dataset keys
        all_dataset_names = list(datasets_cfg_local.keys())
        main_name = None
        aux_name = None
        for name in all_dataset_names:
            lname = name.lower()
            if '10k' in lname and main_name is None:
                main_name = name
            if '35k' in lname and aux_name is None:
                aux_name = name
        if main_name is None and all_dataset_names:
            main_name = all_dataset_names[0]

        self.main_dataset_name = main_name
        self.aux_dataset_name = aux_name

        # paths and filenames
        paths_cfg = self.config.get('paths', {}) if isinstance(self.config.get('paths'), dict) else {}
        output_cfg = self.config.get('output', {}) if isinstance(self.config.get('output'), dict) else {}
        model_cfg = self.config.get('model', {}) if isinstance(self.config.get('model'), dict) else {}
        artifact_cfg = self.config.get('artifact_names', {}) if isinstance(self.config.get('artifact_names'), dict) else {}

        output_root = output_cfg.get('root', './output')
        run_dir = output_cfg.get('run_dir', 'model_output')
        model_name = model_cfg.get('name', 'base_classifier')
        self.data_dir = os.path.join(output_root, run_dir, model_name)
        self.processed_basename = paths_cfg.get('processed_basename', 'processed_dataset.csv')
        base = os.path.splitext(self.processed_basename)[0]
        candidate_main = os.path.join(self.data_dir, f"{base}_{self.main_dataset_name}.csv") if self.main_dataset_name else None
        candidate_aux = os.path.join(self.data_dir, f"{base}_{self.aux_dataset_name}.csv") if self.aux_dataset_name else None

        if candidate_main and os.path.exists(candidate_main):
            self.prepared_dataset_main = candidate_main
        else:
            self.prepared_dataset_main = os.path.join(self.data_dir, self.processed_basename)

        if candidate_aux and os.path.exists(candidate_aux):
            self.prepared_dataset_aux = candidate_aux
        else:
            self.prepared_dataset_aux = None

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

        base = os.path.splitext(self.processed_basename)[0]
        candidate_main = (
            os.path.join(self.data_dir, f"{base}_{self.main_dataset_name}.csv")
            if self.main_dataset_name
            else None
        )
        fallback_main = os.path.join(self.data_dir, self.processed_basename)
        if candidate_main and os.path.isfile(candidate_main):
            expected_main = candidate_main
        elif os.path.isfile(fallback_main):
            expected_main = fallback_main
        else:
            expected_main = candidate_main or fallback_main

        candidate_aux = (
            os.path.join(self.data_dir, f"{base}_{self.aux_dataset_name}.csv")
            if self.aux_dataset_name
            else None
        )
        expected_aux = candidate_aux if candidate_aux and os.path.isfile(candidate_aux) else candidate_aux

        required_artifacts = [
            expected_main,
            self.class_dict_json,
            self.top_class_dict_json,
            self.subclass_json,
        ]
        if expected_aux:
            required_artifacts.append(expected_aux)

        missing_artifacts = [p for p in required_artifacts if p and not os.path.isfile(p)]
        if missing_artifacts:
            print("Processed artifacts not found. Building datasets from combined config...")
            build_multi_dataset(config=self.config)

            # Refresh expected paths after build
            if candidate_main and os.path.isfile(candidate_main):
                self.prepared_dataset_main = candidate_main
            elif os.path.isfile(fallback_main):
                self.prepared_dataset_main = fallback_main

            if candidate_aux and os.path.isfile(candidate_aux):
                self.prepared_dataset_aux = candidate_aux

            missing_after = [p for p in required_artifacts if p and not os.path.isfile(p)]
            if missing_after:
                missing_txt = "\n".join(f"- {p}" for p in missing_after)
                raise FileNotFoundError(
                    "Required processed artifacts are still missing after auto-prepare:\n"
                    + missing_txt
                )

        seed = set_seed()  # For reproducibility

        with open(self.class_dict_json, 'r') as f:
            class_dict = json.load(f)
        with open(self.top_class_dict_json, 'r') as f:
            top_class_dict = json.load(f)
        class_to_top_class = build_class_to_topclass_mapping(class_dict, top_class_dict)

        modes = _resolve_modes_from_config(self.config)
        # Resolve training/options from config
        training_cfg = resolve_training_config(self.config, ('combined',))
        batch_size = int(training_cfg.get('batch_size', 128))
        num_epochs = int(training_cfg.get('num_epochs', 100))
        learning_rate = float(training_cfg.get('lr', training_cfg.get('learning_rate', 0.001)))
        classification_weight = float(training_cfg.get('classification_weight', 1))
        proxy_weight = float(training_cfg.get('proxy_weight', 1))
        proxy_loss_name = training_cfg.get('proxy_loss') or training_cfg.get('proxy_loss_name')
        proxy_loss_params = training_cfg.get('proxy_loss_params', {}) if isinstance(training_cfg.get('proxy_loss_params'), dict) else {}
        classifier_loss_name = (
            training_cfg.get('classifier_loss')
            or training_cfg.get('classifier_loss_name')
            or training_cfg.get('loss')
        )
        classifier_loss_params = training_cfg.get('classifier_loss_params', {}) if isinstance(training_cfg.get('classifier_loss_params'), dict) else {}
        run_mode = _resolve_run_mode(self.config)
        scheduler_type = training_cfg.get('scheduler_type', 'step')
        scheduler_params = training_cfg.get('scheduler_params', {}) if isinstance(training_cfg.get('scheduler_params'), dict) else {}
        patience = int(training_cfg.get('patience', 5))
        early_stopping_metric = training_cfg.get('early_stopping_metric', 'accuracy')
        monitor_metric_label = 'hF1' if normalize_monitor_metric(early_stopping_metric) == 'hierarchical_f1' else 'accuracy'
        k_folds = int(training_cfg.get('k_folds', 5))

        model_output = os.path.join(
            self.config.get('output', {}).get('root', self.config.get('paths', {}).get('output_root', './output')),
            self.config.get('output', {}).get('run_dir', 'model_output'),
            self.config.get('model', {}).get('name', 'base_classifier'),
        )

        # Stage 1 filtering: Load and apply stage1 filtering to full dataset
        filtering_cfg = self.config.get('filtering', {})
        filtering_stage1 = make_filtering_config_by_stage(filtering_cfg, stage="stage1")
        full_df = load_processed_dataset(
            self.prepared_dataset_main,
            filtering_cfg=filtering_cfg,
            report_tag=self.main_dataset_name,
            stage="stage1",
            output_path=self.data_dir,
            save_stage1_csv=True,
        )
        print(f"\n[FILTERING STAGE 1] After stage1 filtering: {len(full_df)} samples")
        
        # Stage 2 filtering config
        filtering_stage2 = make_filtering_config_by_stage(filtering_cfg, stage="stage2")
        
        aux_df = (
            load_processed_dataset(
                self.prepared_dataset_aux,
                filtering_cfg=filtering_cfg,
                report_tag="aux",
                stage="stage1",
                output_path=self.data_dir,
                save_stage1_csv=False,
            )
            if self.prepared_dataset_aux is not None
            else None
        )

        print(f"\n=== Dataset: {self.main_dataset_name} (main) ===")
        database = full_df
        labels = database["class_idx"].tolist()

        # Hold-out test split - fixed test size
        sss_holdout = StratifiedShuffleSplit(n_splits=1, test_size=2192, random_state=seed)
        trainval_idx, test_idx = next(sss_holdout.split(np.zeros(len(labels)), labels))

        trainval_df = database.iloc[trainval_idx].reset_index(drop=True)
        test_df = database.iloc[test_idx].reset_index(drop=True)
        print(f"Hold-out Test size: {len(test_df)} | Remaining for K-fold: {len(trainval_df)}")

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

        for mode in modes:
            print(f"\n=== Running experiments: Dataset={self.main_dataset_name} | Mode={mode} ===")

            for fold, (train_idx, val_idx) in enumerate(
                skf.split(np.zeros(len(trainval_df)), trainval_df['class_idx'].tolist())
            ):
                print(f"\n==== Fold {fold} ====")

                train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
                val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

                # Append auxiliary dataset only to training set
                if aux_df is not None:
                    train_df = pd.concat([train_df, aux_df], ignore_index=True)

                # Stage 2 filtering: Apply to train split only
                if filtering_stage2.enabled and filtering_stage2.apply_to == "train":
                    train_report_path = build_report_path(
                        filtering_stage2,
                        self.main_dataset_name,
                        fold=fold,
                        split="train",
                    )
                    train_df_before = len(train_df)
                    train_df, _ = apply_filtering_df(
                        train_df,
                        "index",
                        filtering_stage2,
                        report_path=train_report_path,
                        stage="train",
                    )
                    print(f"[FILTERING STAGE 2] Train split: {train_df_before} -> {len(train_df)} samples (fold={fold})")
                    
                    # Save stage2 filtered train dataset
                    fold_output_dir = os.path.join(self.data_dir, f"fold_{fold}")
                    os.makedirs(fold_output_dir, exist_ok=True)
                    stage2_csv_path = os.path.join(fold_output_dir, "processed_dataset_stage2.csv")
                    train_df.to_csv(stage2_csv_path, index=False)
                    print(f"  → Saved stage2 train dataset to {stage2_csv_path}")

                print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

                train_dataset = HATRDataset(train_df, aug=True, mask_pct=0.7)
                val_dataset = HATRDataset(val_df, aug=False)
                test_dataset = HATRDataset(test_df, aug=False)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=4,
                    pin_memory=torch.cuda.is_available(),
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=torch.cuda.is_available(),
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=torch.cuda.is_available(),
                )

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
                        split="test",
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
        print("All experiments done!")
