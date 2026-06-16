"""
Module 3 — Option B: 동적 특징 수준 앙상블 (GatingNetwork 학습)

구조:
    train.npz의 z (latent features)만으로 GatingNetwork를 학습.
    val.npz로 early stopping (best epoch 선택).
    최종 평가는 오직 test.npz로만 수행.

Data leakage 방지:
    - GatingNetwork는 train z로만 파라미터 업데이트.
    - val은 early stopping 기준으로만 사용.
    - test는 학습/검증에 절대 사용하지 않음.

Alignment:
    모든 모델의 sound_ids가 np.array_equal로 검증.
    불일치 시 ValueError 발생.

메모리 정리:
    각 Job 완료 후 model, optimizer, GPU 텐서를 완전히 해제.
    (del + gc.collect() + cuda.empty_cache())
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax as _softmax

from ._shared import build_class_to_top, compute_metrics, load_npz_aligned, try_cuda_empty_cache
from .gating_network import GatingNetwork, StackingHead

logger = logging.getLogger(__name__)


def run_option_b(
    job: Any,
    cache_root: str | Path,
    weights_dir: str | Path,
    device: str = "auto",
    k_folds: int = 5,
) -> dict[str, Any]:
    """
    Option B GatingNetwork 학습 및 평가.

    Args:
        job:         ExperimentJob (option="B",
                     hyperparams={"hidden_size", "fusion_type", "dropout_rate",
                                  "num_epochs", "lr", "patience"}).
        cache_root:  .npz 캐시 루트 경로.
        weights_dir: 학습된 GatingNetwork .pt를 저장할 디렉토리.
        device:      "auto" | "cpu" | "cuda" | "cuda:0" 등.

    Returns:
        메트릭 dict: {"accuracy", "top_accuracy"(있을 때), "hF1"(있을 때)}.
    """
    hp = job.hyperparams
    model_type = hp.get("model_type", "gating")

    if model_type == "oof_stacking":
        return _run_oof_stacking(job, cache_root, k_folds)

    _device = _resolve_device(device)

    logger.info(
        "[M3-B] job_id=%s | models=%s | fold=%d | type=%s | hidden=%d | dr=%.1f",
        job.job_id, job.models, job.fold_id,
        model_type, hp["hidden_size"], hp["dropout_rate"],
    )

    # ── 데이터 로드 ──────────────────────────────────────────────────
    train_data = load_npz_aligned(cache_root, job.model_names, job.fold_id, "train")
    val_data   = load_npz_aligned(cache_root, job.model_names, job.fold_id, "val")
    test_data  = load_npz_aligned(cache_root, job.model_names, job.fold_id, "test")

    num_classes = int(train_data[0]["logits"].shape[1])
    n_models    = len(train_data)

    # class_to_top: train split이 전체 클래스 커버 보장
    class_to_top = None
    if "top_labels" in train_data[0]:
        class_to_top = build_class_to_top(
            train_data[0]["labels"], train_data[0]["top_labels"]
        )

    train_labels_gpu = torch.tensor(
        train_data[0]["labels"], dtype=torch.long, device=_device
    )
    val_labels_np = val_data[0]["labels"]

    # ── 모델 & 입력 구성 ──────────────────────────────────────────────
    if model_type == "stacking":
        # Logit 기반 StackingHead: softmax(logits) concat → MLP
        model = StackingHead(
            n_models=n_models,
            num_classes=num_classes,
            hidden_size=hp["hidden_size"],
            dropout=hp["dropout_rate"],
        ).to(_device)
        train_inputs = [
            torch.tensor(d["logits"], dtype=torch.float32, device=_device)
            for d in train_data
        ]
        val_inputs = [
            torch.tensor(d["logits"], dtype=torch.float32, device=_device)
            for d in val_data
        ]
        test_inputs = [
            torch.tensor(d["logits"], dtype=torch.float32, device=_device)
            for d in test_data
        ]
    else:
        # GatingNetwork: z 임베딩 기반 (기존 방식)
        z_dim = int(train_data[0]["z"].shape[1])
        for i, d in enumerate(train_data):
            if d["z"].shape[1] != z_dim:
                raise ValueError(
                    f"z_dim 불일치: '{job.model_names[0]}' (z_dim={z_dim}) vs "
                    f"'{job.model_names[i]}' (z_dim={d['z'].shape[1]})"
                )
        model = GatingNetwork(
            z_dim=z_dim,
            num_sources=n_models,
            hidden_size=hp["hidden_size"],
            num_classes=num_classes,
            fusion_type=hp.get("fusion_type", "softmax_gate"),
            dropout=hp["dropout_rate"],
        ).to(_device)
        train_inputs = [
            torch.tensor(d["z"], dtype=torch.float32, device=_device)
            for d in train_data
        ]
        val_inputs = [
            torch.tensor(d["z"], dtype=torch.float32, device=_device)
            for d in val_data
        ]
        test_inputs = [
            torch.tensor(d["z"], dtype=torch.float32, device=_device)
            for d in test_data
        ]

    optimizer = torch.optim.Adam(model.parameters(), lr=float(hp.get("lr", 0.001)))
    criterion = nn.CrossEntropyLoss()

    num_epochs = int(hp.get("num_epochs", 30))
    patience   = int(hp.get("patience", 5))

    best_val_acc: float = -1.0
    best_state: dict | None = None
    patience_counter = 0

    # ── 학습 루프 ────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        model.train()
        logits_out, _ = model(train_inputs)
        loss = criterion(logits_out, train_labels_gpu)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(val_inputs)
        val_preds = val_logits.argmax(dim=1).cpu().numpy()
        val_acc   = float((val_preds == val_labels_np).mean())

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("[M3-B] Early stop epoch %d (best val_acc=%.4f)", epoch + 1, best_val_acc)
                break

    if best_state is None:
        raise RuntimeError("학습이 한 epoch도 진행되지 않았습니다.")

    # ── 최적 가중치 복원 & Test 평가 ──────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_logits_gpu, _ = model(test_inputs)
    test_logits_np = test_logits_gpu.cpu().numpy()

    test_labels_np = test_data[0]["labels"]
    test_top       = test_data[0].get("top_labels")

    metrics = compute_metrics(test_logits_np, test_labels_np, test_top, class_to_top)
    logger.info("[M3-B] 완료: %s", {k: f"{v:.2f}" for k, v in metrics.items()})

    # ── 체크포인트 저장 ──────────────────────────────────────────────
    weights_path = Path(weights_dir) / f"{job.job_id}.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict":  best_state,
            "model_type":  model_type,
            "num_classes": num_classes,
            "n_models":    n_models,
            "model_names": job.model_names,
            "hyperparams": hp,
            "val_accuracy": best_val_acc,
        },
        weights_path,
    )
    logger.info("[M3-B] 저장: %s", weights_path)

    # ── 메모리 완전 해제 ─────────────────────────────────────────────
    del model, optimizer, criterion
    del train_inputs, val_inputs, test_inputs, test_logits_gpu, train_labels_gpu
    del best_state
    gc.collect()
    try_cuda_empty_cache()

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _run_oof_stacking(
    job: Any,
    cache_root: str | Path,
    k_folds: int = 5,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    OOF (Out-Of-Fold) Logistic Regression 스태킹.

    모든 fold의 val OOF 예측을 합쳐 메타-학습기를 훈련한 뒤,
    전 fold test 예측의 평균에 메타-학습기를 적용해 최종 성능 측정.
    train 데이터를 일절 사용하지 않아 overfitting 없음.

    save_path가 주어지면 학습된 분류기(.joblib)와 메타데이터(.json)를
    동일 stem으로 저장한다 (10k/35k/eval 추론에서 재사용 가능).
    """
    from sklearn.linear_model import LogisticRegression

    names = job.model_names

    X_oof_list, y_oof_list = [], []
    test_probs_per_fold: list[np.ndarray] = []

    for fold in range(k_folds):
        val_data  = load_npz_aligned(cache_root, names, fold, "val")
        test_data = load_npz_aligned(cache_root, names, fold, "test")
        X_oof_list.append(
            np.concatenate([_softmax(d["logits"], axis=1) for d in val_data], axis=1)
        )
        y_oof_list.append(val_data[0]["labels"])
        test_probs_per_fold.append(
            np.concatenate([_softmax(d["logits"], axis=1) for d in test_data], axis=1)
        )

    X_meta = np.vstack(X_oof_list)
    y_meta = np.concatenate(y_oof_list)

    clf = LogisticRegression(C=5.0, max_iter=2000, random_state=42, solver="lbfgs")
    clf.fit(X_meta, y_meta)

    fold_pred_probs = [clf.predict_proba(X_f) for X_f in test_probs_per_fold]
    ensemble_probs = np.mean(fold_pred_probs, axis=0)

    test_data_f0 = load_npz_aligned(cache_root, names, 0, "test")
    test_labels  = test_data_f0[0]["labels"]
    test_top     = test_data_f0[0].get("top_labels")

    class_to_top = None
    if test_top is not None:
        train_data_f0 = load_npz_aligned(cache_root, names, 0, "train")
        if "top_labels" in train_data_f0[0]:
            class_to_top = build_class_to_top(
                train_data_f0[0]["labels"], train_data_f0[0]["top_labels"]
            )

    metrics = compute_metrics(ensemble_probs, test_labels, test_top, class_to_top)
    logger.info("[M3-B/OOF] 완료: %s", {k: f"{v:.2f}" for k, v in metrics.items()})

    if save_path is not None:
        _save_oof_stacking_artifact(save_path, clf, job, k_folds, metrics)

    return metrics


def _save_oof_stacking_artifact(
    save_path: str | Path,
    clf: Any,
    job: Any,
    k_folds: int,
    metrics: dict[str, Any],
) -> None:
    """학습된 OOF stacking LogisticRegression + 메타데이터를 저장."""
    import json
    from datetime import datetime, timezone

    import joblib

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, save_path.with_suffix(".joblib"))

    meta = {
        "job_id":      job.job_id,
        "option":      "B",
        "model_type":  "oof_stacking",
        "models":      job.models,
        "model_names": job.model_names,
        "hyperparams": job.hyperparams,
        "k_folds":     k_folds,
        "metrics":     metrics,
        "created_at":  datetime.now(timezone.utc).isoformat(),
    }
    with save_path.with_suffix(".json").open("w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("[M3-B/OOF] 모델 저장: %s.{joblib,json}", save_path)
