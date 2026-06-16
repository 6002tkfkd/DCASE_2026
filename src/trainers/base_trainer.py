import os
import json
from collections import defaultdict
import collections.abc

import numpy as np
import torch
import torch.nn as nn

from src.utils.loss_loader import get_loss_function
from src.evaluators import get_evaluator
from src.evaluators.classification_evaluator import collect_predictions_and_metrics
from src.trainers.common_artifacts import save_history_json, save_history_with_plots
from src.trainers.common_metrics import (
    append_validation_metrics,
    format_epoch_metrics_line,
    get_monitor_score,
    normalize_monitor_metric,
)


class BaseTrainer:
    """Shared training utilities for all trainers."""

    @staticmethod
    def init_weights(model):
        if isinstance(model, nn.Conv2d):
            nn.init.kaiming_normal_(model.weight, mode='fan_out')
        elif isinstance(model, nn.Linear):
            nn.init.xavier_uniform_(model.weight)

    @staticmethod
    def make_serializable(obj, decimals=6):
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()
            return BaseTrainer.make_serializable(obj, decimals)
        elif isinstance(obj, np.ndarray):
            if obj.ndim == 0:
                return round(float(obj), decimals)
            else:
                return [BaseTrainer.make_serializable(x, decimals) for x in obj]
        elif isinstance(obj, float):
            return round(obj, decimals)
        elif isinstance(obj, int):
            return obj
        elif isinstance(obj, collections.abc.Mapping):
            return {k: BaseTrainer.make_serializable(v, decimals) for k, v in obj.items()}
        elif isinstance(obj, collections.abc.Iterable) and not isinstance(obj, (str, bytes)):
            return [BaseTrainer.make_serializable(x, decimals) for x in obj]
        else:
            return obj

    def load_pretrained_weights(self, model, checkpoint_path, device, mode):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state"]

        model_state = model.state_dict()

        def _filter_matching_tensors(source_state_dict):
            filtered = {}
            skipped = {}
            for key, value in source_state_dict.items():
                if key in model_state and model_state[key].shape == value.shape:
                    filtered[key] = value
                else:
                    skipped[key] = tuple(value.shape) if hasattr(value, "shape") else None
            return filtered, skipped

        if mode == "both":
            filtered_state_dict, skipped = _filter_matching_tensors(state_dict)
            load_result = model.load_state_dict(filtered_state_dict, strict=False)
            print("Loaded pretrained weights with relaxed matching for mode=both")
            if skipped:
                print(f"Skipped mismatched keys: {sorted(skipped.keys())}")
            print(f"Missing keys: {load_result.missing_keys}")
            print(f"Unexpected keys: {load_result.unexpected_keys}")
            return checkpoint

        if mode == "audio":
            filtered_state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("text_emb_extractor.") and not key.startswith("fusion.")
            }
            filtered_state_dict, skipped = _filter_matching_tensors(filtered_state_dict)
            load_result = model.load_state_dict(filtered_state_dict, strict=False)
            print("Loaded pretrained weights with audio-only compatibility")
            if skipped:
                print(f"Skipped mismatched keys: {sorted(skipped.keys())}")
            print(f"Missing keys: {load_result.missing_keys}")
            print(f"Unexpected keys: {load_result.unexpected_keys}")
            return checkpoint

        filtered_state_dict, skipped = _filter_matching_tensors(state_dict)
        load_result = model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded pretrained weights with relaxed matching for mode={mode}")
        if skipped:
            print(f"Skipped mismatched keys: {sorted(skipped.keys())}")
        print(f"Missing keys: {load_result.missing_keys}")
        print(f"Unexpected keys: {load_result.unexpected_keys}")
        return checkpoint

    def train_model(
        self,
        model,
        train_loader,
        val_loader,
        device,
        class_to_topclass,
        class_dict=None,
        num_epochs=100,
        lr=0.001,
        classification_weight=1.0,
        output_dir='model_output',
        scheduler_type='plateau',
        patience=10,
        run_mode='standard',
        loss_name='CrossEntropyLoss',
        loss_params=None,
        proxy_loss_name=None,
        proxy_loss_params=None,
        classifier_loss_name=None,
        classifier_loss_params=None,
        proxy_weight=1.0,
        evaluator_type='classification',
        early_stopping_metric='accuracy',
        scheduler_params=None,
    ):
        os.makedirs(output_dir, exist_ok=True)

        run_mode = str(run_mode or 'standard').strip().lower()
        scheduler_type = str(scheduler_type or 'plateau').strip().lower()
        loss_params = loss_params or {}
        proxy_loss_params = proxy_loss_params or {}
        classifier_loss_params = classifier_loss_params or {}
        scheduler_params = scheduler_params or {}
        monitor_metric = normalize_monitor_metric(early_stopping_metric)
        monitor_metric_label = 'accuracy' if monitor_metric == 'accuracy' else 'hF1'

        if run_mode == 'standard':
            proxy_loss_name = None
            classifier_loss_name = classifier_loss_name or 'CrossEntropyLoss'
        elif run_mode == 'proxy_only':
            proxy_loss_name = proxy_loss_name or 'Proxy_Anchor'
            classifier_loss_name = None
        elif run_mode == 'dual_head':
            proxy_loss_name = proxy_loss_name or 'Proxy_Anchor'
            classifier_loss_name = classifier_loss_name or 'CrossEntropyLoss'
        else:
            raise ValueError(
                f"Unsupported run_mode='{run_mode}'. Expected one of: standard, proxy_only, dual_head."
            )

        proxy_criterion = None
        classifier_criterion = None
        parent_of_child = None
        parent_count = 0
        if class_to_topclass:
            num_children = int(getattr(model, "num_classes", len(class_to_topclass)))
            parent_values = [int(v) for v in class_to_topclass.values() if v is not None]
            parent_count = max(parent_values) + 1 if parent_values else 0
            parent_of_child_values = [int(class_to_topclass.get(child_idx, 0)) for child_idx in range(num_children)]
            parent_of_child = torch.tensor(parent_of_child_values, dtype=torch.long, device=device)

        if proxy_loss_name is not None:
            proxy_lp = {}
            proxy_lp.update(proxy_loss_params)
            is_hierarchical_proxy_name = str(proxy_loss_name).split(".")[-1] == "HierarchicalProxyLoss"
            if is_hierarchical_proxy_name:
                if "num_children" not in proxy_lp:
                    proxy_lp["num_children"] = getattr(model, "num_classes", None)
                if "num_parents" not in proxy_lp:
                    proxy_lp["num_parents"] = parent_count
                if "embedding_dim" not in proxy_lp:
                    proxy_lp["embedding_dim"] = getattr(model, "embedding_dim", None)
            else:
                if "nb_classes" not in proxy_lp:
                    proxy_lp["nb_classes"] = getattr(model, "num_classes", None)
                if "sz_embed" not in proxy_lp:
                    proxy_lp["sz_embed"] = getattr(model, "embedding_dim", None)
            proxy_criterion = get_loss_function(proxy_loss_name, **proxy_lp)
            if proxy_criterion is not None:
                proxy_criterion = proxy_criterion.to(device)

        if classifier_loss_name is not None:
            classifier_lp = {}
            classifier_lp.update(classifier_loss_params)
            classifier_criterion = get_loss_function(classifier_loss_name, **classifier_lp)
            if classifier_criterion is not None:
                classifier_criterion = classifier_criterion.to(device)

        is_hierarchical_proxy = bool(
            proxy_criterion is not None and getattr(proxy_criterion, "requires_hierarchical_labels", False)
        )
        if is_hierarchical_proxy and parent_of_child is None:
            raise ValueError("HierarchicalProxyLoss requires class_to_topclass mapping.")

        def _compute_proxy_loss(z, labels):
            if not is_hierarchical_proxy:
                return proxy_criterion(z, labels), {}
            parent_labels = parent_of_child[labels]
            result = proxy_criterion(z, parent_labels, labels, parent_of_child)
            if isinstance(result, tuple):
                return result
            return result, {}

        proxy_evaluator = None
        if proxy_criterion is not None:
            evaluator_kwargs = {
                "proxy_loss": proxy_criterion,
                "class_to_topclass": class_to_topclass,
                "class_dict": class_dict,
            }
            evaluator_name = "proxy_classification"
            if is_hierarchical_proxy:
                evaluator_name = "hierarchical_proxy_classification"
                evaluator_kwargs["parent_of_child"] = parent_of_child
            proxy_evaluator = get_evaluator(evaluator_name, **evaluator_kwargs)

        optimizer_params = list(model.parameters())
        if proxy_criterion is not None:
            optimizer_params.extend(list(proxy_criterion.parameters()))
        if classifier_criterion is not None:
            optimizer_params.extend(list(classifier_criterion.parameters()))
        optimizer = torch.optim.Adam(optimizer_params, lr=lr, weight_decay=1e-5)
        if scheduler_type == 'plateau':
            plateau_factor = float(scheduler_params.get('factor', 0.5))
            plateau_patience = int(scheduler_params.get('patience', max(1, int(patience / 3))))
            plateau_threshold = float(scheduler_params.get('threshold', 1e-4))
            plateau_threshold_mode = str(scheduler_params.get('threshold_mode', 'rel')).strip().lower()
            plateau_cooldown = int(scheduler_params.get('cooldown', 0))
            plateau_min_lr = scheduler_params.get('min_lr', 0)
            plateau_eps = float(scheduler_params.get('eps', 1e-8))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=plateau_factor,
                patience=plateau_patience,
                threshold=plateau_threshold,
                threshold_mode=plateau_threshold_mode,
                cooldown=plateau_cooldown,
                min_lr=plateau_min_lr,
                eps=plateau_eps,
            )
        elif scheduler_type == 'step':
            step_size = int(scheduler_params.get('step_size', 20))
            gamma = float(scheduler_params.get('gamma', 0.5))
            last_epoch = int(scheduler_params.get('last_epoch', -1))
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma,
                last_epoch=last_epoch,
            )
        elif scheduler_type == 'cosine':
            cosine_t_max = int(scheduler_params.get('t_max', num_epochs))
            cosine_eta_min = float(scheduler_params.get('eta_min', 0.0))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_t_max,
                eta_min=cosine_eta_min,
            )
        elif scheduler_type in {'warmup_cosine', 'warmup_cosine_annealing', 'cosine_warmup'}:
            warmup_epochs = int(scheduler_params.get('warmup_epochs', 5))
            warmup_start_factor = float(scheduler_params.get('warmup_start_factor', 0.1))
            cosine_t_max = int(scheduler_params.get('t_max', max(1, num_epochs - warmup_epochs)))
            cosine_eta_min = float(scheduler_params.get('eta_min', 0.0))

            warmup_epochs = max(0, min(warmup_epochs, max(0, num_epochs - 1)))
            cosine_t_max = max(1, cosine_t_max)
            warmup_start_factor = min(max(warmup_start_factor, 0.0), 1.0)
            eta_min_factor = min(max(cosine_eta_min / lr if lr > 0 else 0.0, 0.0), 1.0)

            def _warmup_cosine_lambda(epoch_index):
                if warmup_epochs > 0 and epoch_index < warmup_epochs:
                    progress = epoch_index / float(max(1, warmup_epochs - 1)) if warmup_epochs > 1 else 0.0
                    progress = min(max(progress, 0.0), 1.0)
                    return warmup_start_factor + (1.0 - warmup_start_factor) * progress

                cosine_epoch = max(0, epoch_index - warmup_epochs)
                cosine_progress = min(cosine_epoch, cosine_t_max) / float(cosine_t_max)
                cosine_factor = 0.5 * (1.0 + np.cos(np.pi * cosine_progress))
                return eta_min_factor + (1.0 - eta_min_factor) * cosine_factor

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_cosine_lambda)
        else:
            scheduler = None

        best_monitor_score = float('-inf')
        epochs_without_improvement = 0
        history = defaultdict(list)

        for epoch in range(num_epochs):
            model.train()
            losses = defaultdict(float)
            total_samples = 0

            attn_audio_epoch = []
            attn_text_epoch = []

            for data in train_loader:
                class_labels = data['class_idx'].to(device)
                audio_emb = data.get('audio_embedding', None)
                text_emb = data.get('text_embedding', None)

                if audio_emb is not None:
                    audio_emb = audio_emb.to(device)
                if text_emb is not None:
                    text_emb = text_emb.to(device)

                optimizer.zero_grad()

                outputs = model(audio_emb, text_emb)
                z = outputs.get("z") if isinstance(outputs, dict) else outputs[0]
                class_logit = outputs.get("logits") if isinstance(outputs, dict) else outputs[1]
                attn_scores = outputs.get("attn_scores") if isinstance(outputs, dict) else outputs[2]

                if attn_scores is not None:
                    attn_audio_epoch.append(attn_scores[:, 0].detach().cpu())
                    attn_text_epoch.append(attn_scores[:, 1].detach().cpu())

                total_loss = 0.0

                batch_size = class_labels.size(0)
                total_samples += batch_size

                if proxy_criterion is not None:
                    proxy_loss, proxy_loss_parts = _compute_proxy_loss(z, class_labels)
                    losses['proxy'] += proxy_loss.item() * batch_size
                    for part_name, part_value in proxy_loss_parts.items():
                        if isinstance(part_value, torch.Tensor):
                            part_value = part_value.detach().item()
                        losses[f'proxy_{part_name}'] += float(part_value) * batch_size
                    total_loss += proxy_weight * proxy_loss

                if classifier_criterion is not None:
                    if class_logit is None:
                        raise ValueError("classifier_loss_name was provided, but model returned no logits.")
                    cls_loss = classifier_criterion(class_logit, class_labels)
                    losses['cls'] += cls_loss.item() * batch_size
                    total_loss += classification_weight * cls_loss

                total_loss.backward()
                optimizer.step()
                losses['total'] += total_loss.item() * batch_size

            if attn_audio_epoch:
                attn_audio_epoch = torch.cat(attn_audio_epoch, dim=0)
                attn_text_epoch = torch.cat(attn_text_epoch, dim=0)
                history["attention_audio"].append(attn_audio_epoch.mean(0).numpy())
                history["attention_text"].append(attn_text_epoch.mean(0).numpy())

            for k in losses:
                history[f'train_{k}_loss'].append(losses[k] / total_samples)
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            print(f"[Epoch {epoch+1}/{num_epochs}] lr={current_lr:.6g}")

            if proxy_evaluator is not None:
                _, val_metrics = proxy_evaluator.collect_predictions_and_metrics(model, val_loader, device)
            else:
                _, val_metrics = collect_predictions_and_metrics(
                    model,
                    val_loader,
                    device,
                    class_to_topclass=class_to_topclass,
                    class_dict=class_dict,
                )
            val_accuracy = append_validation_metrics(history, val_metrics)
            monitor_score = get_monitor_score(val_metrics, monitor_metric)

            val_loss = 0.0
            total_val_samples = 0
            with torch.no_grad():
                for data in val_loader:
                    labels = data['class_idx'].to(device)
                    audio_emb = data.get('audio_embedding', None)
                    text_emb = data.get('text_embedding', None)

                    if audio_emb is not None:
                        audio_emb = audio_emb.to(device)
                    if text_emb is not None:
                        text_emb = text_emb.to(device)

                    outputs = model(audio_emb, text_emb)
                    z = outputs.get("z") if isinstance(outputs, dict) else outputs[0]
                    class_logit = outputs.get("logits") if isinstance(outputs, dict) else outputs[1]

                    loss = 0.0
                    if proxy_criterion is not None:
                        proxy_val_loss, _ = _compute_proxy_loss(z, labels)
                        loss = loss + proxy_weight * proxy_val_loss
                    if classifier_criterion is not None:
                        if class_logit is None:
                            raise ValueError("classifier_loss_name was provided, but model returned no logits.")
                        loss = loss + classification_weight * classifier_criterion(class_logit, labels)

                    batch_size = labels.size(0)
                    val_loss += float(loss.item()) * batch_size
                    total_val_samples += batch_size

            avg_val_loss = val_loss / total_val_samples if total_val_samples > 0 else 0.0
            history['val_loss'].append(avg_val_loss)

            save_history_json(history, output_dir, serializer=BaseTrainer.make_serializable)

            train_loss = losses['total'] / total_samples if total_samples > 0 else 0.0
            print(format_epoch_metrics_line(epoch, num_epochs, val_metrics, train_loss=train_loss, val_loss=avg_val_loss))
            best_display_score = monitor_score if best_monitor_score == float('-inf') else best_monitor_score
            print(f"  Early stopping metric ({monitor_metric_label}): {monitor_score:.2f}% | best: {best_display_score:.2f}%")

            if scheduler:
                if scheduler_type == 'plateau':
                    scheduler.step(monitor_score)
                else:
                    scheduler.step()

            if monitor_score > best_monitor_score:
                best_monitor_score = monitor_score
                checkpoint = {
                    'model_state': model.state_dict(),
                    'config': {
                        'model_name': model.__class__.__name__,
                        'hidden_size': getattr(model, 'hidden_size', None),
                        'num_classes': getattr(model, 'num_classes', None),
                        'emb_size_audio': getattr(model, 'emb_size_audio', None),
                        'emb_size_text': getattr(model, 'emb_size_text', None),
                        'embedding_dim': getattr(model, 'embedding_dim', None),
                        'dropout': getattr(model, 'dropout', None),
                        'use_batch_norm': getattr(model, 'use_batch_norm', True),
                        'mode': getattr(model, 'mode', None),
                        'use_classifier': getattr(model, 'use_classifier', True),
                        'normalize_embedding': getattr(model, 'normalize_embedding', None),
                        'early_stopping_metric': monitor_metric_label,
                    },
                }

                if proxy_criterion is not None:
                    checkpoint['proxy_loss_state'] = proxy_criterion.state_dict()
                    checkpoint['proxy_loss_name'] = proxy_criterion.__class__.__name__
                    if is_hierarchical_proxy:
                        checkpoint['proxy_loss_config'] = {
                            'embedding_dim': getattr(proxy_criterion, 'embedding_dim', getattr(model, 'embedding_dim', None)),
                            'num_parents': getattr(proxy_criterion, 'num_parents', parent_count),
                            'num_children': getattr(proxy_criterion, 'num_children', getattr(model, 'num_classes', None)),
                            'temperature': getattr(proxy_criterion, 'temperature', 0.07),
                            'alpha': getattr(proxy_criterion, 'alpha', 0.4),
                            'beta': getattr(proxy_criterion, 'beta', 0.3),
                            'gamma': getattr(proxy_criterion, 'gamma', 0.15),
                            'delta': getattr(proxy_criterion, 'delta', 0.05),
                            'sibling_margin': getattr(proxy_criterion, 'sibling_margin', 0.4),
                            'parent_margin': getattr(proxy_criterion, 'parent_margin', 0.0),
                        }
                        checkpoint['parent_of_child'] = parent_of_child.detach().cpu()
                    else:
                        checkpoint['proxy_loss_config'] = {
                            'mrg': getattr(proxy_criterion, 'mrg', 0.1),
                            'alpha': getattr(proxy_criterion, 'alpha', 32),
                        }

                torch.save(checkpoint, os.path.join(output_dir, "best_model.pth"))
                print('  New best model saved')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print("Early stopping triggered.")
                    break

        return best_monitor_score, history, model
# End of file: previously contained a temporary BaseTrainer stub which
# was accidentally appended; removed to preserve the full implementation above.
