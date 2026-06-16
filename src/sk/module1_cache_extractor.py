"""
Module 1 — OOFCacheExtractor

Branch 1 모델(best_model.pth)을 로드하여 raw .npy 임베딩을 통과시킨 후
z (중간 특징)와 logits를 5-fold split 기준으로 .npz 파일에 저장.

dcase_ensemble에 이미 학습된 체크포인트를 심링크 없이 상대경로(legacy_data_root)로
참조한다. experiment_dir과 processed_dataset.csv 안의 audio/text_emb_filepath는
모두 legacy_data_root 기준 상대경로로 풀어서 읽는다 (src.sk.paths 참고).

출력 경로(MESH 안, 새로 생성됨):
    {cache_root}/{model_name}/fold_{k}/{split}.npz
    split: "train" | "val" | "test"

.npz 내부 키:
    sound_ids  (N,)       str  — sound_id, 오름차순 정렬
    z          (N, D_z)   f32  — latent embedding
    logits     (N, C)     f32  — class logits
    labels     (N,)       i64  — class_idx (leaf)
    top_labels (N,)       i64  — top_class_idx (parent; hF1 계산에 사용)
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.sk import paths as sk_paths
from src.sk.models import BaseClassifier, BaseClassifierProxyAnchorSimple

logger = logging.getLogger(__name__)

# best-combo 4개 모델 전부 config["model_name"]="BaseClassifier"로 저장되어 있음
# (실험 디렉토리명이 proxy_anchor_simple이라도 실제 클래스는 training.run_mode로 결정됨).
# BaseClassifierProxyAnchorSimple은 지금 당장 쓰이진 않지만, model_pool을 확장했을 때
# 바로 쓸 수 있도록 같은 패턴으로 등록해 둔다.
_MODEL_REGISTRY = {
    "BaseClassifier": BaseClassifier,
    "BaseClassifierProxyAnchorSimple": BaseClassifierProxyAnchorSimple,
}

_SPLITS = ("train", "val", "test")


# ─────────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────────

class OOFCacheExtractor:
    """
    Branch 1 체크포인트에서 z + logits를 추출해 .npz로 캐싱.

    Args:
        cache_root:       캐시 저장 루트 디렉토리 (MESH 안, 새로 생성됨).
        legacy_data_root: 체크포인트/임베딩이 실제로 있는 dcase_ensemble 루트
                           (MESH 루트 기준 상대경로, 예: "../dcase_ensemble").
        batch_size:        forward pass 배치 크기.
        num_workers:       DataLoader 워커 수.
    """

    def __init__(
        self,
        cache_root: str,
        legacy_data_root: str,
        batch_size: int = 256,
        num_workers: int = 4,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.legacy_data_root = legacy_data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def is_cached(self, model_name: str, fold_id: int, split: str) -> bool:
        return self._out_path(model_name, fold_id, split).exists()

    def extract_model(
        self,
        model_name: str,
        experiment_dir: str,
        n_folds: int = 5,
    ) -> None:
        """
        지정 모델의 모든 fold에 대해 캐시 추출.

        Args:
            model_name:      모델 식별자 (캐시 디렉토리명으로 사용).
            experiment_dir:  legacy_data_root 기준 상대경로
                              (예: "output/clap_music_as_chunk_mean/.../base_classifier_proxy_anchor_simple").
                              하위에 processed_dataset.csv 와 both/fold_{k}/ 가 있어야 함.
            n_folds:         fold 수.
        """
        exp_dir = sk_paths.resolve_legacy(experiment_dir, self.legacy_data_root)
        processed_csv = exp_dir / "processed_dataset.csv"
        if not processed_csv.exists():
            raise FileNotFoundError(f"processed_dataset.csv 없음: {processed_csv}")

        full_df = pd.read_csv(processed_csv)
        full_df = sk_paths.rewrite_embedding_columns(full_df, self.legacy_data_root)
        full_df["index"] = full_df["index"].astype(str)
        full_df = full_df.set_index("index")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("[M1] device=%s", device)

        for fold_id in range(n_folds):
            self._extract_fold(
                model_name, exp_dir, full_df, fold_id, device
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────────────────────────────────

    def _out_path(self, model_name: str, fold_id: int, split: str) -> Path:
        return self.cache_root / model_name / f"fold_{fold_id}" / f"{split}.npz"

    def _extract_fold(
        self,
        model_name: str,
        exp_dir: Path,
        full_df: pd.DataFrame,
        fold_id: int,
        device: torch.device,
    ) -> None:
        fold_dir = exp_dir / "both" / f"fold_{fold_id}"
        checkpoint_path = fold_dir / "best_model.pth"
        splits_csv = fold_dir / "splits.csv"

        # 세 split 모두 캐시됐으면 스킵
        if all(self.is_cached(model_name, fold_id, s) for s in _SPLITS):
            logger.info("[M1] SKIP %s/fold_%d (all cached)", model_name, fold_id)
            return

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트 없음: {checkpoint_path}")
        if not splits_csv.exists():
            raise FileNotFoundError(f"splits.csv 없음: {splits_csv}")

        logger.info("[M1] 로딩 %s/fold_%d ...", model_name, fold_id)
        model, mode = _load_model(checkpoint_path, device)

        split_ids = _parse_splits(splits_csv)  # {split_name: [sound_id, ...]}

        try:
            for split in _SPLITS:
                out_path = self._out_path(model_name, fold_id, split)
                if out_path.exists():
                    logger.info("[M1] SKIP %s/fold_%d/%s (cached)", model_name, fold_id, split)
                    continue

                ids = split_ids.get(split, [])
                if not ids:
                    logger.warning("[M1] split '%s' 에 sound_id 없음, fold=%d", split, fold_id)
                    continue

                # processed_dataset.csv에 없는 id 필터 (excluded 등)
                valid_ids = [sid for sid in ids if sid in full_df.index]
                if len(valid_ids) < len(ids):
                    logger.warning(
                        "[M1] %d/%d sound_id 가 processed_dataset.csv 에 없어 제외 (fold=%d, split=%s)",
                        len(ids) - len(valid_ids), len(ids), fold_id, split,
                    )

                split_df = full_df.loc[valid_ids].reset_index()  # 'index' → 컬럼으로 복원

                sound_ids, z, logits, labels, top_labels = _collect_features(
                    model, split_df, mode, device, self.batch_size, self.num_workers
                )

                # sound_id 오름차순 정렬 (alignment 보장)
                order = np.argsort(sound_ids)

                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_path,
                    sound_ids=sound_ids[order],
                    z=z[order],
                    logits=logits[order],
                    labels=labels[order],
                    top_labels=top_labels[order],
                )
                logger.info(
                    "[M1] 저장 %s/fold_%d/%s.npz  (n=%d, z_dim=%d)",
                    model_name, fold_id, split, len(order), z.shape[1],
                )

        finally:
            # fold 완료 후 VRAM 완전 해제
            del model
            gc.collect()
            _try_cuda_empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    """체크포인트에서 모델을 재구성하고 가중치를 로드."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model_cls = _MODEL_REGISTRY.get(cfg.get("model_name", "BaseClassifier"))
    if model_cls is None:
        raise ValueError(
            f"알 수 없는 model_name: '{cfg.get('model_name')}'. "
            f"지원: {list(_MODEL_REGISTRY.keys())}"
        )

    # 공통 파라미터 (None 제외)
    init_kwargs: dict = {}
    for key in ("hidden_size", "num_classes", "emb_size_audio", "emb_size_text",
                 "dropout", "use_batch_norm", "mode"):
        val = cfg.get(key)
        if val is not None:
            init_kwargs[key] = val

    # Proxy 계열 전용 파라미터 (BaseClassifier는 받지 않음)
    if model_cls is not BaseClassifier:
        for key in ("embedding_dim", "use_classifier"):
            val = cfg.get(key)
            if val is not None:
                init_kwargs[key] = val

    model = model_cls(**init_kwargs)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()

    mode = cfg.get("mode", "both")
    return model, mode


def _parse_splits(splits_csv: Path) -> dict[str, list[str]]:
    """splits.csv → {split_name: [sound_id, ...]}. excluded_stage2 제외."""
    df = pd.read_csv(splits_csv, dtype=str)
    result: dict[str, list[str]] = {}
    for split in _SPLITS:
        result[split] = df.loc[df["split"] == split, "sound_id"].tolist()
    return result


def _collect_features(
    model: torch.nn.Module,
    split_df: pd.DataFrame,
    mode: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """forward pass를 배치로 실행하여 (sound_ids, z, logits, labels, top_labels) 반환."""
    dataset = _EmbeddingDataset(split_df, mode)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    z_parts, logits_parts, label_parts, top_label_parts, id_parts = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            audio_emb = batch["audio_emb"].to(device) if "audio_emb" in batch else None
            text_emb = batch["text_emb"].to(device) if "text_emb" in batch else None

            out = model(audio_emb=audio_emb, text_emb=text_emb)

            z_parts.append(out["z"].cpu().numpy())
            logits_parts.append(out["logits"].cpu().numpy())
            label_parts.append(batch["label"].numpy())
            top_label_parts.append(batch["top_label"].numpy())
            id_parts.extend(batch["sound_id"])

    sound_ids = np.array(id_parts)
    z = np.concatenate(z_parts, axis=0)
    logits = np.concatenate(logits_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    top_labels = np.concatenate(top_label_parts, axis=0)
    return sound_ids, z, logits, labels, top_labels


class _EmbeddingDataset(Dataset):
    """processed_dataset.csv 기반 경량 Dataset."""

    def __init__(self, df: pd.DataFrame, mode: str) -> None:
        self.df = df
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        item: dict = {
            "sound_id": str(row["index"]),
            "label": int(row["class_idx"]),
            "top_label": int(row["top_class_idx"]),
        }
        if self.mode in ("audio", "both"):
            item["audio_emb"] = torch.tensor(
                np.load(row["audio_emb_filepath"]), dtype=torch.float32
            )
        if self.mode in ("text", "both"):
            item["text_emb"] = torch.tensor(
                np.load(row["text_emb_filepath"]), dtype=torch.float32
            )
        return item


def _try_cuda_empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
