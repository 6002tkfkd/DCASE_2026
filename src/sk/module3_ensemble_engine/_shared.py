"""
Module 3 공유 유틸리티: npz 정렬 로드, 메트릭 계산.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_npz_aligned(
    cache_root: str | Path,
    model_names: list[str],
    fold_id: int,
    split: str,
) -> list[dict[str, np.ndarray]]:
    """
    여러 모델의 .npz를 로드하고 sound_ids 교집합으로 정렬.

    모델마다 confidence 필터 등으로 train 샘플 수가 다를 수 있으므로,
    val/test는 strict 일치를 검증하고, train은 교집합으로 맞춤.

    Returns:
        각 모델에 대한 dict 리스트. 키: sound_ids, z, logits, labels, top_labels(있으면).
    Raises:
        ValueError: val/test에서 sound_ids 불일치 시, 또는 교집합이 비어 있을 때.
        FileNotFoundError: .npz 파일이 없을 때.
    """
    import logging
    _log = logging.getLogger(__name__)

    raw: list[dict[str, np.ndarray]] = []

    for model in model_names:
        path = Path(cache_root) / model / f"fold_{fold_id}" / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"캐시 파일 없음: {path}\n"
                f"  Module 1을 먼저 실행해 주세요 (scripts/sk/extract_cache.py)."
            )
        raw.append(dict(np.load(path, allow_pickle=False)))

    # val/test: 모든 모델이 동일한 ID를 가져야 함 (strict)
    if split != "train":
        ref_ids = raw[0]["sound_ids"]
        for i, (data, model) in enumerate(zip(raw[1:], model_names[1:]), 1):
            if not np.array_equal(ref_ids, data["sound_ids"]):
                raise ValueError(
                    f"sound_ids 불일치 ({split}, fold={fold_id})\n"
                    f"  기준: '{model_names[0]}'  문제: '{model}'"
                )
        return raw

    # train: 교집합으로 정렬
    common_ids = set(raw[0]["sound_ids"].tolist())
    for data in raw[1:]:
        common_ids &= set(data["sound_ids"].tolist())

    if not common_ids:
        raise ValueError(
            f"train sound_ids 교집합이 비어 있음 (fold={fold_id}). "
            f"각 크기: {[len(d['sound_ids']) for d in raw]}"
        )

    common_sorted = np.array(sorted(common_ids), dtype=raw[0]["sound_ids"].dtype)
    n_dropped = max(len(d["sound_ids"]) for d in raw) - len(common_sorted)
    if n_dropped > 0:
        _log.debug(
            "[align] fold_%d train: 교집합 %d개 (최대 대비 -%d)",
            fold_id, len(common_sorted), n_dropped,
        )

    aligned: list[dict[str, np.ndarray]] = []
    for data in raw:
        id_to_idx = {int(sid): i for i, sid in enumerate(data["sound_ids"])}
        sel = np.array([id_to_idx[int(sid)] for sid in common_sorted])
        aligned.append({k: v[sel] for k, v in data.items()})

    return aligned


def build_class_to_top(
    labels: np.ndarray,
    top_labels: np.ndarray,
) -> dict[int, int]:
    """class_idx → top_class_idx 매핑 구성. labels/top_labels 전체 split 데이터 권장."""
    return {int(c): int(t) for c, t in zip(labels, top_labels)}


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    top_labels: np.ndarray | None = None,
    class_to_top: dict[int, int] | None = None,
) -> dict[str, Any]:
    """
    분류 메트릭 계산.

    Args:
        logits:       (N, C) float — 앙상블 output logits
        labels:       (N,)   int   — leaf class_idx (정답)
        top_labels:   (N,)   int   — top_class_idx (정답). None이면 hF1/top_accuracy 생략.
        class_to_top: {class_idx: top_class_idx}. None이면 labels/top_labels로 자동 구성.

    Returns:
        dict with "accuracy", and optionally "top_accuracy", "hF1".
    """
    preds = np.argmax(logits, axis=1)
    acc = float((preds == labels).mean() * 100)
    metrics: dict[str, Any] = {"accuracy": acc}

    if top_labels is None:
        return metrics

    if class_to_top is None:
        class_to_top = build_class_to_top(labels, top_labels)

    top_preds = np.array([class_to_top.get(int(p), -1) for p in preds])
    top_acc = float((top_preds == top_labels).mean() * 100)

    hp, hr, hf = _hierarchical_prf(preds, labels, top_preds, top_labels, lambda_param=0.75)

    metrics["top_accuracy"] = top_acc
    metrics["hF1"] = float(hf * 100)

    return metrics


def _hierarchical_prf(
    preds: np.ndarray,
    labels: np.ndarray,
    top_preds: np.ndarray,
    top_labels: np.ndarray,
    lambda_param: float = 0.75,
) -> tuple[float, float, float]:
    """
    훈련 코드와 동일한 hierarchical P/R/F1 계산 (lambda=0.75).

    각 leaf 클래스 c에 대해:
      Precision (pred=c인 샘플):
        gt=c         → contribution 1.0
        top_gt=top_c → contribution lambda * 0.5
        otherwise    → 0.0
      Recall (gt=c인 샘플):
        pred=c       → contribution 1.0
        top_pred=top_c → contribution lambda * 0.5
        otherwise    → 0.0
    P_c, R_c 평균 후 F1_c = 2PR/(P+R), 전체 클래스 macro 평균.
    """
    same_top_contrib = lambda_param * 0.5
    classes = np.unique(labels)
    hps, hrs, hfs = [], [], []

    for c in classes:
        c_top = class_to_top_scalar(c, top_labels, labels)

        pred_mask = (preds == c)
        if pred_mask.any():
            gt_c_mask   = (labels[pred_mask] == c)
            top_eq_mask = (top_labels[pred_mask] == c_top) & ~gt_c_mask
            hpp = np.where(gt_c_mask, 1.0,
                  np.where(top_eq_mask, same_top_contrib, 0.0))
            class_p = float(hpp.mean())
        else:
            class_p = 0.0

        gt_mask = (labels == c)
        if gt_mask.any():
            pred_c_mask   = (preds[gt_mask] == c)
            top_eq_mask2  = (top_preds[gt_mask] == c_top) & ~pred_c_mask
            hrr = np.where(pred_c_mask, 1.0,
                  np.where(top_eq_mask2, same_top_contrib, 0.0))
            class_r = float(hrr.mean())
        else:
            class_r = 0.0

        class_f = (2 * class_p * class_r / (class_p + class_r)
                   if (class_p + class_r) > 0 else 0.0)
        hps.append(class_p)
        hrs.append(class_r)
        hfs.append(class_f)

    return float(np.mean(hps)), float(np.mean(hrs)), float(np.mean(hfs))


def class_to_top_scalar(c: int, top_labels: np.ndarray, labels: np.ndarray) -> int:
    """leaf class index c 에 대응하는 top class index를 반환."""
    idx = np.where(labels == c)[0]
    return int(top_labels[idx[0]]) if idx.size > 0 else -1


def try_cuda_empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
