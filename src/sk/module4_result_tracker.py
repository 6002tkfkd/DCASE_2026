"""
Module 4 — ResultTracker

ExperimentJob + metrics 딕셔너리를 받아 summary.csv에 한 줄씩 기록.

원자적 쓰기 보장:
    - fcntl.flock(LOCK_EX): 여러 터미널에서 동시에 실행해도 CSV 손상 없음.
    - lock 획득 후 os.fstat()로 실제 파일 크기를 확인 → 헤더 중복 기록 방지.
    - f.flush(): 크래시 발생 시 마지막 row 유실 방지.

summary.csv 컬럼 (raw, per-fold):
    job_id, option, models_used, n_models, hyperparams, fold_id,
    hF1, accuracy, top_accuracy

summary_agg.csv 컬럼 (fold 평균, 자동 갱신):
    option, n_models, backbones, models_used, hyperparams,
    avg_hF1, std_hF1, avg_accuracy, std_accuracy,
    avg_top_accuracy, std_top_accuracy, n_folds
"""

from __future__ import annotations

import csv
import fcntl
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.sk.module2_combo_manager import ExperimentJob

logger = logging.getLogger(__name__)

_COLUMNS = [
    "job_id",
    "option",
    "models_used",
    "n_models",
    "hyperparams",
    "fold_id",
    "hF1",
    "accuracy",
    "top_accuracy",
    "weights",
]

_METRICS = ["hF1", "accuracy", "top_accuracy"]

_AGG_COLUMNS = [
    "option",
    "n_models",
    "backbones",
    "model_weights",
    "hyperparams",
    "avg_hF1",    "std_hF1",
    "avg_accuracy", "std_accuracy",
    "avg_top_accuracy", "std_top_accuracy",
    "n_folds",
    "models_used",
]


class ResultTracker:
    """
    앙상블 Job 결과를 summary.csv에 원자적으로 기록하고 조회하는 클래스.

    Args:
        summary_csv: 결과를 저장할 CSV 파일 경로.
    """

    def __init__(self, summary_csv: str | Path) -> None:
        self.summary_csv = Path(summary_csv)

    # ─────────────────────────────────────────────────────────────
    # 기록
    # ─────────────────────────────────────────────────────────────

    def record(self, job: ExperimentJob, metrics: dict[str, Any]) -> None:
        """
        Job 결과를 summary.csv에 한 줄 추가하고 summary_agg.csv를 갱신.

        동시 접근 안전: fcntl.flock(LOCK_EX)로 배타적 잠금 획득 후 기록.
        헤더 중복 방지: lock 내에서 os.fstat()으로 파일 실제 크기 확인.
        """
        self.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        row = self._build_row(job, metrics)

        with open(self.summary_csv, "a", newline="", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            write_header = os.fstat(f.fileno()).st_size == 0
            writer = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            f.flush()

        self._rebuild_agg()
        logger.debug("[M4] 기록: %s", job.job_id)

    # ─────────────────────────────────────────────────────────────
    # 조회
    # ─────────────────────────────────────────────────────────────

    def get_top_k(
        self,
        k: int = 10,
        metric: str = "hF1",
    ) -> pd.DataFrame:
        """summary_agg.csv에서 metric 기준 상위 k개 반환."""
        agg_path = self.summary_csv.parent / "summary_agg.csv"
        if agg_path.exists():
            df = pd.read_csv(agg_path)
        else:
            df = self._compute_agg()
        if df.empty:
            return df
        col = f"avg_{metric}"
        if col not in df.columns:
            return df
        return df.sort_values(col, ascending=False).head(k).reset_index(drop=True)

    def _rebuild_agg(self) -> None:
        """summary.csv 전체를 읽어 summary_agg.csv를 재생성."""
        agg = self._compute_agg()
        agg_path = self.summary_csv.parent / "summary_agg.csv"
        agg.to_csv(agg_path, index=False, float_format="%.4f")

    def _compute_agg(self) -> pd.DataFrame:
        """fold별 raw 결과를 집계해 DataFrame 반환."""
        if not self.summary_csv.exists():
            return pd.DataFrame(columns=_AGG_COLUMNS)

        df = pd.read_csv(self.summary_csv)
        if df.empty:
            return pd.DataFrame(columns=_AGG_COLUMNS)

        for m in _METRICS:
            if m in df.columns:
                df[m] = pd.to_numeric(df[m], errors="coerce")

        group_cols = ["option", "models_used", "n_models", "hyperparams"]
        agg_parts = []
        for m in _METRICS:
            if m not in df.columns:
                continue
            part = (
                df.dropna(subset=[m])
                .groupby(group_cols)[m]
                .agg(avg="mean", std="std", n_folds="count")
                .rename(columns={"avg": f"avg_{m}", "std": f"std_{m}", "n_folds": f"_n_{m}"})
            )
            agg_parts.append(part)

        if not agg_parts:
            return pd.DataFrame(columns=_AGG_COLUMNS)

        merged = agg_parts[0]
        for part in agg_parts[1:]:
            merged = merged.join(part, how="outer")
        merged["n_folds"] = merged[[c for c in merged.columns if c.startswith("_n_")]].max(axis=1).astype(int)
        merged = merged.drop(columns=[c for c in merged.columns if c.startswith("_n_")])
        merged = merged.reset_index()

        # backbones 컬럼: 모델명의 __ 앞 부분 (중복 제거)
        def _extract_backbones(models_used: str) -> str:
            bbs, seen = [], set()
            for m in models_used.split(","):
                bb = m.strip().split("__")[0]
                if bb not in seen:
                    bbs.append(bb)
                    seen.add(bb)
            return " | ".join(bbs)

        merged["backbones"] = merged["models_used"].apply(_extract_backbones)

        # model_weights 컬럼: 모델별 평균 가중치 (fold 평균)
        # weights 컬럼이 있는 경우에만 계산
        if "weights" in df.columns and df["weights"].notna().any():
            group_cols_full = ["option", "models_used", "n_models", "hyperparams"]

            def _avg_weights(series: pd.Series) -> str:
                # series.name = (option, models_used, n_models, hyperparams) tuple
                models_used_str = series.name[1] if isinstance(series.name, tuple) else ""
                valid = series.dropna()
                valid = valid[valid != ""]
                if valid.empty:
                    return ""
                parsed = []
                for v in valid:
                    try:
                        parsed.append(json.loads(v))
                    except Exception:
                        pass
                if not parsed:
                    return ""
                avg_w = np.mean(parsed, axis=0)
                models = models_used_str.split(",")
                return " | ".join(f"{m.strip()}={w:.3f}" for m, w in zip(models, avg_w))

            model_weights_map = (
                df.groupby(group_cols_full)["weights"]
                .apply(_avg_weights)
                .rename("model_weights")
                .reset_index()
            )
            merged = merged.merge(model_weights_map, on=group_cols_full, how="left")
        else:
            merged["model_weights"] = ""

        cols = [c for c in _AGG_COLUMNS if c in merged.columns]
        return merged[cols].sort_values("avg_hF1", ascending=False).reset_index(drop=True)

    def load_all(self) -> pd.DataFrame:
        """summary.csv 전체를 DataFrame으로 반환. 파일이 없으면 빈 DataFrame."""
        if not self.summary_csv.exists():
            return pd.DataFrame(columns=_COLUMNS)
        return pd.read_csv(self.summary_csv)

    # ─────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_row(job: ExperimentJob, metrics: dict[str, Any]) -> dict:
        weights = metrics.get("weights")
        return {
            "job_id":       job.job_id,
            "option":       job.option,
            "models_used":  ",".join(job.models),
            "n_models":     len(job.models),
            "hyperparams":  json.dumps(job.hyperparams, ensure_ascii=False, separators=(",", ":")),
            "fold_id":      job.fold_id,
            "hF1":          metrics.get("hF1", ""),
            "accuracy":     metrics.get("accuracy", ""),
            "top_accuracy": metrics.get("top_accuracy", ""),
            "weights":      json.dumps([round(float(w), 4) for w in weights], separators=(",", ":")) if weights else "",
        }
