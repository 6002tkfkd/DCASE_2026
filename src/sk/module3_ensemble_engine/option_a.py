"""
Module 3 — Option A: 정적 예측 평균 앙상블

전략:
    simple_average:   모든 모델의 test logits를 단순 평균.
    weighted_average: val set에서 scipy로 최적 가중치를 학습한 뒤 test에 적용.
                      Simplex constraint (w_i > 0, Σw_i = 1)는 softmax 변환으로 보장.

Data leakage 방지:
    - 가중치 학습은 오직 val.npz의 logits로만 수행.
    - test.npz는 최종 평가에만 사용.

Alignment:
    모든 모델의 sound_ids가 동일한지 np.array_equal로 검증.
    불일치 시 ValueError 발생.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from ._shared import build_class_to_top, compute_metrics, load_npz_aligned

logger = logging.getLogger(__name__)


def run_option_a(
    job: Any,
    cache_root: str | Path,
) -> dict[str, Any]:
    """
    Option A 앙상블 실행.

    Args:
        job:        ExperimentJob (option="A", hyperparams={"strategy": ...}).
        cache_root: .npz 캐시 루트 경로.

    Returns:
        메트릭 dict: {"accuracy": float, "top_accuracy": float (있을 때), "hF1": float (있을 때)}.
    """
    strategy = job.hyperparams["strategy"]
    logger.info(
        "[M3-A] job_id=%s | strategy=%s | models=%s | fold=%d",
        job.job_id, strategy, job.models, job.fold_id,
    )

    # ── 데이터 로드 ──────────────────────────────────────────────────
    train_data = load_npz_aligned(cache_root, job.model_names, job.fold_id, "train")
    val_data   = load_npz_aligned(cache_root, job.model_names, job.fold_id, "val")
    test_data  = load_npz_aligned(cache_root, job.model_names, job.fold_id, "test")

    # class_to_top: train split이 전체 클래스 커버 보장
    class_to_top = None
    if "top_labels" in train_data[0]:
        class_to_top = build_class_to_top(train_data[0]["labels"], train_data[0]["top_labels"])

    test_labels   = test_data[0]["labels"]
    test_top      = test_data[0].get("top_labels")

    # ── 앙상블 logits 계산 ───────────────────────────────────────────
    if strategy == "simple_average":
        ensemble_logits = _simple_average(test_data)

    elif strategy == "prob_average":
        # softmax 확률 평균: logit 직접 평균보다 overconfident 모델의 영향 줄임
        ensemble_logits = _prob_average(test_data)

    elif strategy == "temp_calibrated":
        # val로 모델별 최적 temperature 탐색 → test에 적용 후 확률 평균
        # T>1: logit 완화(sharper → flatter), T<1: 예측 강화
        val_logits_stack = np.stack([d["logits"] for d in val_data], axis=0)
        val_labels       = val_data[0]["labels"]
        temps = [
            _calibrate_temperature(val_logits_stack[i], val_labels)
            for i in range(len(val_data))
        ]
        logger.info("[M3-A] calibrated temperatures: %s", [f"{t:.3f}" for t in temps])
        test_logits_stack = np.stack([d["logits"] for d in test_data], axis=0)
        scaled_probs = np.stack([
            softmax(test_logits_stack[i] / temps[i], axis=1)
            for i in range(len(test_data))
        ])
        ensemble_logits = scaled_probs.mean(axis=0)

    elif strategy == "weighted_average":
        val_logits_stack = np.stack([d["logits"] for d in val_data], axis=0)  # (M, N_val, C)
        val_labels       = val_data[0]["labels"]
        weights = _learn_weights(val_logits_stack, val_labels)
        logger.info("[M3-A] 학습된 가중치: %s", np.round(weights, 4).tolist())

        test_logits_stack = np.stack([d["logits"] for d in test_data], axis=0)  # (M, N_test, C)
        ensemble_logits   = (test_logits_stack * weights[:, None, None]).sum(axis=0)

    else:
        raise ValueError(
            f"알 수 없는 strategy: '{strategy}'. "
            f"지원: simple_average, prob_average, weighted_average"
        )

    # ── 메트릭 계산 ─────────────────────────────────────────────────
    n = len(job.models)
    metrics = compute_metrics(ensemble_logits, test_labels, test_top, class_to_top)
    if strategy == "weighted_average":
        metrics["weights"] = weights.tolist()
    elif strategy == "temp_calibrated":
        # weights 컬럼에 temperature 저장 (가중치 대신 보정값으로 활용)
        metrics["weights"] = [round(t, 4) for t in temps]
    else:
        metrics["weights"] = [round(1.0 / n, 6)] * n
    logger.info("[M3-A] 완료: %s", {k: f"{v:.2f}" for k, v in metrics.items() if k != "weights"})

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _simple_average(data_list: list[dict]) -> np.ndarray:
    """모든 모델의 logits를 단순 평균. shape: (N, C)."""
    logits_stack = np.stack([d["logits"] for d in data_list], axis=0)  # (M, N, C)
    return logits_stack.mean(axis=0)


def _prob_average(data_list: list[dict]) -> np.ndarray:
    """
    Softmax 확률 평균 후 log 반환 (log-prob space로 compute_metrics에 전달).
    logit 직접 평균보다 overconfident 모델이 앙상블을 지배하는 현상 완화.
    """
    logits_stack = np.stack([d["logits"] for d in data_list], axis=0)  # (M, N, C)
    probs = softmax(logits_stack, axis=2)                                # (M, N, C)
    avg_probs = probs.mean(axis=0)                                       # (N, C)
    # compute_metrics는 argmax만 사용하므로 log 변환 불필요, avg_probs 직접 반환
    return avg_probs


def _calibrate_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    단일 모델의 val NLL을 최소화하는 temperature scalar T 탐색.
    logit / T 를 적용한 뒤 softmax하면 예측 분포가 보정됨.
    T > 1: 예측 분포 완화 (overconfident 완화)
    T < 1: 예측 분포 강화 (underconfident 완화)
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import log_softmax as _lsm

    idx = np.arange(len(labels))

    def nll(T: float) -> float:
        lp = _lsm(logits / max(T, 1e-6), axis=1)
        return -float(lp[idx, labels].mean())

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def _learn_weights(logits_stack: np.ndarray, val_labels: np.ndarray) -> np.ndarray:
    """
    val set cross-entropy를 최소화하는 모델별 가중치 학습.

    accuracy(discrete)가 아닌 cross-entropy(smooth)를 목적함수로 사용해야
    Nelder-Mead가 실제로 gradient 방향을 탐색할 수 있음.
    multiple random restarts로 local minima 회피.

    Args:
        logits_stack: (M, N_val, C) — 각 모델의 val logits.
        val_labels:   (N_val,)      — 정답 labels.

    Returns:
        (M,) — softmax로 정규화된 가중치 (합=1, 모두 양수).
    """
    from scipy.special import log_softmax as _log_softmax

    n_models = logits_stack.shape[0]
    if n_models == 1:
        return np.array([1.0])

    idx = np.arange(len(val_labels))

    def neg_ll(raw_w: np.ndarray) -> float:
        w = softmax(raw_w)
        ensemble = (logits_stack * w[:, None, None]).sum(axis=0)
        log_probs = _log_softmax(ensemble, axis=1)
        return -float(log_probs[idx, val_labels].mean())

    rng = np.random.default_rng(42)
    starts = [np.zeros(n_models)] + [rng.normal(0, 1, n_models) for _ in range(4)]

    best_val, best_x = np.inf, np.zeros(n_models)
    for x0 in starts:
        r = minimize(neg_ll, x0, method="Nelder-Mead",
                     options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
        if r.fun < best_val:
            best_val, best_x = r.fun, r.x

    return softmax(best_x)
