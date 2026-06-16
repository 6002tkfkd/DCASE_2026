import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Ensure `src` is importable when running the script from repo root
sys.path.insert(0, os.getcwd())

from src.utils.core_utils import load_config, set_seed
from src.datasets.data_manager import HATRDataManager
from src.models.builder import build_model
from src.trainers.base_trainer import BaseTrainer

PERPLEXITY = 30.0
N_ITER = 1000
SEED = 42

markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P', '1', '2', '3', '4']


def _resolve_modes_from_config(config):
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


def _resolve_training_cfg(config):
    finetune_cfg = config.get('finetune', {}) if isinstance(config.get('finetune'), dict) else {}
    stage_training_cfg = finetune_cfg.get('training', {}) if isinstance(finetune_cfg.get('training'), dict) else {}
    global_training_cfg = config.get('training', {}) if isinstance(config.get('training'), dict) else {}
    return stage_training_cfg or global_training_cfg


def _resolve_default_stages(config):
    strategy = str(config.get('strategy', '')).strip().lower()
    if strategy.startswith('two_stage'):
        return ['pretrain', 'finetune']
    return ['finetune']


def _resolve_finetune_checkpoint_path(model_output_root: str, mode: str, fold_id: int) -> str:
    return os.path.join(model_output_root, mode, f'fold_{fold_id}', 'best_model.pth')


def _resolve_model_spec_from_checkpoint(checkpoint, config, fallback_mode='both'):
    ckpt_cfg = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    model_cfg = config.get('model', {}) if isinstance(config.get('model'), dict) else {}
    training_cfg = config.get('training', {}) if isinstance(config.get('training'), dict) else {}

    def _safe_int(value, default):
        return default if value is None else int(value)

    def _safe_float(value, default):
        return default if value is None else float(value)

    hidden_size = _safe_int(ckpt_cfg.get('hidden_size', model_cfg.get('hidden_size', 128)), 128)
    num_classes = _safe_int(ckpt_cfg.get('num_classes', model_cfg.get('num_classes', 10)), 10)
    emb_size_audio = _safe_int(ckpt_cfg.get('emb_size_audio', model_cfg.get('emb_size_audio', 512)), 512)
    emb_size_text = _safe_int(ckpt_cfg.get('emb_size_text', model_cfg.get('emb_size_text', 512)), 512)
    dropout = _safe_float(ckpt_cfg.get('dropout', model_cfg.get('dropout', 0.1)), 0.1)
    use_batch_norm = bool(ckpt_cfg.get('use_batch_norm', model_cfg.get('use_batch_norm', True)))
    mode = ckpt_cfg.get('mode', model_cfg.get('mode', fallback_mode))
    embedding_dim = _safe_int(ckpt_cfg.get('embedding_dim', model_cfg.get('embedding_dim', hidden_size)), hidden_size)
    use_classifier = bool(ckpt_cfg.get('use_classifier', True))

    resolved_run_mode = training_cfg.get('run_mode', model_cfg.get('run_mode', None))
    if resolved_run_mode is None:
        resolved_run_mode = 'dual_head' if use_classifier and 'proxy' in str(model_cfg.get('name', '')).lower() else 'standard'
    resolved_run_mode = str(resolved_run_mode).strip().lower()
    if not use_classifier:
        resolved_run_mode = 'proxy_only'
    elif resolved_run_mode not in {'standard', 'proxy_only', 'dual_head'}:
        resolved_run_mode = 'dual_head' if 'proxy' in str(model_cfg.get('name', '')).lower() else 'standard'

    return resolved_run_mode, {
        'hidden_size': hidden_size,
        'num_classes': num_classes,
        'emb_size_audio': emb_size_audio,
        'emb_size_text': emb_size_text,
        'dropout': dropout,
        'use_batch_norm': use_batch_norm,
        'mode': mode,
        'embedding_dim': embedding_dim,
        'use_classifier': use_classifier,
    }


def run_tsne_and_plot(ax, vectors, df_meta, title, top_classes, sub_classes):
    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        max_iter=N_ITER,
        random_state=SEED,
        init='pca',
        learning_rate='auto',
    )
    vis_dims = tsne.fit_transform(np.vstack(vectors))

    temp_df = df_meta.copy()
    temp_df['x'], temp_df['y'] = vis_dims[:, 0], vis_dims[:, 1]

    colors = plt.cm.get_cmap('Set1', len(top_classes))
    sub_class_to_marker = {sub: markers[i % len(markers)] for i, sub in enumerate(sub_classes)}

    for i, tc in enumerate(top_classes):
        tc_df = temp_df[temp_df['top_class'] == tc]
        tc_color = colors(i)
        for sc in sorted(tc_df['sub_class'].unique()):
            sc_df = tc_df[tc_df['sub_class'] == sc]
            if len(sc_df) == 0:
                continue
            ax.scatter(
                sc_df['x'],
                sc_df['y'],
                color=tc_color,
                marker=sub_class_to_marker[sc],
                label=f"{tc} | {sc}",
                alpha=0.6,
                s=60,
                edgecolors='white',
                linewidths=0.3,
            )
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.2)


def _build_input_vector(audio_emb, text_emb, mode):
    if mode == 'both':
        parts = []
        if audio_emb is not None:
            parts.append(audio_emb)
        if text_emb is not None:
            parts.append(text_emb)
        if not parts:
            return None
        return torch.cat(parts, dim=-1)
    if mode == 'audio':
        return audio_emb
    if mode == 'text':
        return text_emb
    return None


def _collect_embeddings(model, data_loader, device, mode):
    zs = []
    raw_inputs = []
    labels = []

    with torch.no_grad():
        for batch in data_loader:
            audio_emb = batch.get('audio_embedding', None)
            text_emb = batch.get('text_embedding', None)
            class_idx = batch.get('class_idx')

            if audio_emb is not None:
                audio_emb = audio_emb.to(device)
            if text_emb is not None:
                text_emb = text_emb.to(device)

            outputs = model(audio_emb, text_emb)
            z = outputs.get('z') if isinstance(outputs, dict) else (outputs[0] if len(outputs) > 0 else None)
            if z is None:
                continue
            raw_input = _build_input_vector(audio_emb, text_emb, mode)
            z = z.detach().cpu().numpy()
            raw_input = raw_input.detach().cpu().numpy() if raw_input is not None else None
            for i in range(z.shape[0]):
                zs.append(z[i])
                if raw_input is not None:
                    raw_inputs.append(raw_input[i])
                labels.append(int(class_idx[i].item()))

    return zs, raw_inputs, labels


def _prepare_model_kwargs_for_build(model_kwargs, run_mode):
    prepared_kwargs = dict(model_kwargs)
    if str(run_mode).strip().lower() == 'standard':
        prepared_kwargs.pop('embedding_dim', None)
        prepared_kwargs.pop('use_classifier', None)
    return prepared_kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--fold', type=int, default=None, help='If set, only process this fold index')
    parser.add_argument(
        '--stages',
        type=str,
        default='auto',
        help='Comma-separated stages to render: finetune, pretrain, or both. Default is auto.',
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(args.seed))

    if args.stages == 'auto':
        stages = _resolve_default_stages(config)
    else:
        stages = [s.strip().lower() for s in args.stages.split(',')] if args.stages else _resolve_default_stages(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_manager = HATRDataManager(config)
    modes = _resolve_modes_from_config(config)

    model_output_root = data_manager._resolve_output_path()
    finetune_training_cfg = _resolve_training_cfg(config)
    finetune_k_folds = int(finetune_training_cfg.get('k_folds', 5))

    if 'finetune' in stages:
        fold_ids = [args.fold] if args.fold is not None else list(range(finetune_k_folds))
        for mode in modes:
            for fold_id in fold_ids:
                ckpt_path = _resolve_finetune_checkpoint_path(model_output_root, mode, fold_id)
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(
                        f"Finetune checkpoint not found for mode={mode}, fold={fold_id}: {ckpt_path}"
                    )

    # Precompute class mappings for finetune if needed
    if 'finetune' in stages:
        class_dict_ft, top_class_dict_ft, class_to_topclass_ft = data_manager.get_class_mappings(mode='finetune')
        id_to_class_ft = {v: k for k, v in class_dict_ft.items()}
        id_to_top_ft = {v: k for k, v in top_class_dict_ft.items()}
        loaders_ft = data_manager.get_dataloaders(mode='finetune')

    # Precompute finetune mappings/loaders for pretrain test visualization when needed.
    if 'pretrain' in stages and 'finetune' not in stages:
        class_dict_ft, top_class_dict_ft, class_to_topclass_ft = data_manager.get_class_mappings(mode='finetune')
        id_to_class_ft = {v: k for k, v in class_dict_ft.items()}
        id_to_top_ft = {v: k for k, v in top_class_dict_ft.items()}
        loaders_ft = data_manager.get_dataloaders(mode='finetune')

    # Precompute for pretrain if requested
    if 'pretrain' in stages:
        class_dict_pt, top_class_dict_pt, class_to_topclass_pt = data_manager.get_class_mappings(mode='pretrain')
        id_to_class_pt = {v: k for k, v in class_dict_pt.items()}
        id_to_top_pt = {v: k for k, v in top_class_dict_pt.items()}
        # pretrain dataloaders return (train_dl, val_dl, None)
        train_dl_pt, val_dl_pt, _ = data_manager.get_dataloaders(mode='pretrain')

    # FINETUNE: per-mode, per-fold test t-SNE
    if 'finetune' in stages:
        for mode in modes:
            for fold_id, (_, _, test_loader) in enumerate(loaders_ft):
                if args.fold is not None and fold_id != args.fold:
                    continue

                fold_output_dir = os.path.join(model_output_root, mode, f'fold_{fold_id}')
                os.makedirs(fold_output_dir, exist_ok=True)

                print(f"Processing mode={mode} fold={fold_id} -> output_dir={fold_output_dir}")

                num_classes = len(class_dict_ft)

                # Load best model weights
                ckpt_path = _resolve_finetune_checkpoint_path(model_output_root, mode, fold_id)
                checkpoint = torch.load(ckpt_path, map_location=device)
                resolved_run_mode, model_kwargs = _resolve_model_spec_from_checkpoint(checkpoint, config, fallback_mode=mode)
                model_kwargs['num_classes'] = num_classes
                model_kwargs['mode'] = mode
                model_kwargs = _prepare_model_kwargs_for_build(model_kwargs, resolved_run_mode)

                model = build_model(
                    config=config,
                    run_mode=resolved_run_mode,
                    **model_kwargs,
                ).to(device)

                try:
                    trainer_helper = BaseTrainer()
                    trainer_helper.load_pretrained_weights(model, ckpt_path, device, mode)
                except Exception:
                    # fallback to direct load
                    state = checkpoint.get('model_state', checkpoint)
                    model.load_state_dict(state, strict=False)

                model.eval()

                zs, raw_inputs, labels = _collect_embeddings(model, test_loader, device, mode)

                if not zs:
                    print(f"No embeddings extracted for fold {fold_id}; skipping")
                    continue


                df_meta = pd.DataFrame([
                    {
                        'sub_class': id_to_class_ft[l] if l in id_to_class_ft else str(l),
                        'top_class': (
                            id_to_top_ft[class_to_topclass_ft[l]]
                            if l in class_to_topclass_ft and class_to_topclass_ft[l] in id_to_top_ft
                            else (id_to_class_ft[l].split('-')[0] if l in id_to_class_ft else str(l))
                        ),
                    }
                    for l in labels
                ])

                top_classes = sorted(df_meta['top_class'].unique())
                sub_classes = sorted(df_meta['sub_class'].unique())

                fig, ax = plt.subplots(1, 1, figsize=(12, 10))
                run_tsne_and_plot(ax, zs, df_meta, f"TSNE Test Latent | mode={mode} | fold={fold_id}", top_classes, sub_classes)

                handles, lab = ax.get_legend_handles_labels()
                if handles:
                    fig.legend(handles, lab, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=6, fontsize='x-small', title='Testset Hierarchy')

                plt.tight_layout()
                out_png = os.path.join(fold_output_dir, 'tsne_test_embeddings.png')
                plt.savefig(out_png, dpi=200, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved TSNE plot: {out_png}")

                if raw_inputs:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
                    run_tsne_and_plot(
                        ax,
                        raw_inputs,
                        df_meta,
                        f"TSNE Test Input Embeddings | mode={mode} | fold={fold_id}",
                        top_classes,
                        sub_classes,
                    )

                    handles, lab = ax.get_legend_handles_labels()
                    if handles:
                        fig.legend(handles, lab, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=6, fontsize='x-small', title='Testset Hierarchy')

                    plt.tight_layout()
                    out_png = os.path.join(fold_output_dir, 'tsne_test_input_embeddings.png')
                    plt.savefig(out_png, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Saved TSNE plot: {out_png}")

    # PRETRAIN: visualize 10k test embeddings using the pretrain checkpoint.
    if 'pretrain' in stages:
        # Determine pretrain output directory similar to PretrainTrainer
        output_cfg = config.get('output', {}) if isinstance(config.get('output'), dict) else {}
        paths_cfg = config.get('paths', {}) if isinstance(config.get('paths'), dict) else {}
        output_root = output_cfg.get('root', paths_cfg.get('output_root', './output'))
        pretrain_dirname = output_cfg.get('pretrain_dir', 'model_output_pretrain')
        model_name = config.get('model', {}).get('name', 'base_classifier')
        pretrain_output_dir = os.path.join(output_root, pretrain_dirname, model_name)

        # prefer 'pretrained_model.pth' then 'best_model.pth'
        ckpt_candidates = [os.path.join(pretrain_output_dir, 'pretrained_model.pth'), os.path.join(pretrain_output_dir, 'best_model.pth')]
        ckpt_path = next((c for c in ckpt_candidates if os.path.exists(c)), None)
        if ckpt_path is None:
            print(f"Warning: no pretrain checkpoint found in {pretrain_output_dir}; skipping pretrain t-SNE")
        else:
            print(f"Processing pretrain 10k test embeddings using checkpoint: {ckpt_path}")
            ck = torch.load(ckpt_path, map_location=device)
            mode_pt = ck.get('config', {}).get('mode', 'both') if isinstance(ck, dict) else 'both'
            resolved_run_mode_pt, model_kwargs_pt = _resolve_model_spec_from_checkpoint(ck, config, fallback_mode=mode_pt)
            model_kwargs_pt['num_classes'] = len(class_dict_ft)
            model_kwargs_pt['mode'] = mode_pt
            model_kwargs_pt = _prepare_model_kwargs_for_build(model_kwargs_pt, resolved_run_mode_pt)

            model_pt = build_model(
                config=config,
                run_mode=resolved_run_mode_pt,
                **model_kwargs_pt,
            ).to(device)

            # Load checkpoint
            trainer_helper = BaseTrainer()
            try:
                trainer_helper.load_pretrained_weights(model_pt, ckpt_path, device, mode_pt)
            except Exception:
                state = ck.get('model_state', ck)
                model_pt.load_state_dict(state, strict=False)

            model_pt.eval()

            # Reuse the finetune test loader so we visualize the 10k test split through the pretrain model.
            test_loader_pt = loaders_ft[0][2]
            zs, raw_inputs, labels = _collect_embeddings(model_pt, test_loader_pt, device, mode_pt)

            if zs:
                df_meta = pd.DataFrame([
                    {
                        'sub_class': id_to_class_ft[l] if l in id_to_class_ft else str(l),
                        'top_class': (
                            id_to_top_ft[class_to_topclass_ft[l]]
                            if l in class_to_topclass_ft and class_to_topclass_ft[l] in id_to_top_ft
                            else (id_to_class_ft[l].split('-')[0] if l in id_to_class_ft else str(l))
                        ),
                    }
                    for l in labels
                ])

                top_classes = sorted(df_meta['top_class'].unique())
                sub_classes = sorted(df_meta['sub_class'].unique())

                fig, ax = plt.subplots(1, 1, figsize=(12, 10))
                run_tsne_and_plot(ax, zs, df_meta, f"TSNE Pretrain 10k Test Latent | fold=pretrain", top_classes, sub_classes)
                handles, lab = ax.get_legend_handles_labels()
                if handles:
                    fig.legend(handles, lab, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=6, fontsize='x-small', title='Hierarchy')
                out_png = os.path.join(pretrain_output_dir, 'tsne_pretrain_test_embeddings.png')
                plt.tight_layout()
                plt.savefig(out_png, dpi=200, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved pretrain test TSNE plot: {out_png}")

                if raw_inputs:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
                    run_tsne_and_plot(
                        ax,
                        raw_inputs,
                        df_meta,
                        f"TSNE Pretrain 10k Test Input Embeddings | fold=pretrain",
                        top_classes,
                        sub_classes,
                    )

                    handles, lab = ax.get_legend_handles_labels()
                    if handles:
                        fig.legend(handles, lab, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=6, fontsize='x-small', title='Hierarchy')

                    plt.tight_layout()
                    out_png = os.path.join(pretrain_output_dir, 'tsne_pretrain_test_input_embeddings.png')
                    plt.savefig(out_png, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Saved pretrain test TSNE plot: {out_png}")


if __name__ == '__main__':
    main()
