"""
Module 2 — CombinationManager

search_config.yaml → ExperimentJob 리스트 생성 + 멱등성 보장.

job_id 규칙 (summary.csv Primary Key):
    Option A:  A__{models}___{strategy}___fold{k}
    Option B:  B__{models}___h{hs}_{fusion_short}_d{dr}___fold{k}

    models = short_name1+short_name2+...
    fusion_short: softmax_gate → soft | sigmoid_gate → sig
    dr: dropout × 10 (e.g. 0.1 → d01, 0.2 → d02)

    short_name: 모델 pool 항목의 'short_name' 필드 우선,
                없으면 'name' 필드 그대로 사용.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass
class ExperimentJob:
    job_id: str
    models: list[str]           # short_name 리스트
    model_names: list[str]      # 실제 모델 이름 (캐시 경로에 쓰임)
    option: str                 # "A" | "B"
    fold_id: int
    hyperparams: dict[str, Any] = field(default_factory=dict)


class CombinationManager:
    """
    search_config.yaml을 읽어 ExperimentJob 리스트를 생성하고,
    summary.csv와 대조해 미완료 Job만 반환.
    """

    def __init__(self, config_path: str) -> None:
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        search = self.config.get("search", {})
        self.k_folds: int = int(search.get("k_folds", 5))
        self.min_size: int = int(search.get("min_combo_size", 2))
        self.max_size: int = int(search.get("max_combo_size", 5))

        # 모델 풀: {name, short_name} 리스트
        pool = self.config.get("model_pool", [])
        self._pool: list[dict] = [
            {"name": m["name"], "short": m.get("short_name", m["name"])}
            for m in pool
        ]

        self._option_a_cfg: dict = search.get("option_a", {})
        self._option_b_cfg: dict = search.get("option_b", {})

    # ─────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────

    def generate_jobs(self) -> list[ExperimentJob]:
        """가능한 모든 ExperimentJob 생성."""
        jobs: list[ExperimentJob] = []
        max_k = min(self.max_size, len(self._pool))

        for k in range(self.min_size, max_k + 1):
            for combo in itertools.combinations(self._pool, k):
                short_names = [m["short"] for m in combo]
                real_names = [m["name"] for m in combo]

                if self._option_a_cfg.get("enabled", False):
                    jobs.extend(self._option_a_jobs(short_names, real_names))

                if self._option_b_cfg.get("enabled", False):
                    jobs.extend(self._option_b_jobs(short_names, real_names))

        return jobs

    def filter_pending(
        self,
        jobs: list[ExperimentJob],
        summary_csv: str | Path,
    ) -> list[ExperimentJob]:
        """summary.csv에 이미 기록된 job_id를 제외한 미완료 Job 반환."""
        summary_path = Path(summary_csv)
        if not summary_path.exists():
            return jobs

        done_ids = set(
            pd.read_csv(summary_path, usecols=["job_id"])["job_id"].tolist()
        )
        return [j for j in jobs if j.job_id not in done_ids]

    def print_summary(
        self,
        all_jobs: list[ExperimentJob],
        pending_jobs: list[ExperimentJob],
    ) -> None:
        done = len(all_jobs) - len(pending_jobs)
        a_pending = sum(1 for j in pending_jobs if j.option == "A")
        b_pending = sum(1 for j in pending_jobs if j.option == "B")
        print(
            f"전체 {len(all_jobs)}개 job  |  "
            f"완료 {done}개  |  "
            f"대기 {len(pending_jobs)}개  (A: {a_pending}, B: {b_pending})"
        )

    # ─────────────────────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────────────────────

    def _option_a_jobs(
        self,
        short_names: list[str],
        real_names: list[str],
    ) -> list[ExperimentJob]:
        strategies = self._option_a_cfg.get("strategies", ["simple_average"])
        jobs = []
        for strategy in strategies:
            for fold_id in range(self.k_folds):
                job_id = _make_job_id_a(short_names, strategy, fold_id)
                jobs.append(ExperimentJob(
                    job_id=job_id,
                    models=short_names,
                    model_names=real_names,
                    option="A",
                    fold_id=fold_id,
                    hyperparams={"strategy": strategy},
                ))
        return jobs

    def _option_b_jobs(
        self,
        short_names: list[str],
        real_names: list[str],
    ) -> list[ExperimentJob]:
        hidden_sizes = self._option_b_cfg.get("hidden_sizes", [128])
        fusion_types = self._option_b_cfg.get("fusion_types", ["softmax_gate"])
        dropout_rates = self._option_b_cfg.get("dropout_rates", [0.1])

        num_epochs = int(self._option_b_cfg.get("num_epochs", 30))
        lr = float(self._option_b_cfg.get("lr", 0.001))
        patience = int(self._option_b_cfg.get("patience", 5))

        jobs = []
        for hs in hidden_sizes:
            for ft in fusion_types:
                for dr in dropout_rates:
                    hp = {
                        "hidden_size": hs,
                        "fusion_type": ft,
                        "dropout_rate": dr,
                        "num_epochs": num_epochs,
                        "lr": lr,
                        "patience": patience,
                    }
                    for fold_id in range(self.k_folds):
                        job_id = _make_job_id_b(short_names, hs, ft, dr, fold_id)
                        jobs.append(ExperimentJob(
                            job_id=job_id,
                            models=short_names,
                            model_names=real_names,
                            option="B",
                            fold_id=fold_id,
                            hyperparams=hp,
                        ))
        return jobs


# ─────────────────────────────────────────────────────────────────────────────
# job_id 생성 헬퍼 (모듈 밖에서도 재현 가능하도록 순수 함수로 분리)
# ─────────────────────────────────────────────────────────────────────────────

def _make_job_id_a(short_names: list[str], strategy: str, fold_id: int) -> str:
    models_str = "+".join(short_names)
    return f"A__{models_str}___{strategy}___fold{fold_id}"


def _make_job_id_b(
    short_names: list[str],
    hidden_size: int,
    fusion_type: str,
    dropout_rate: float,
    fold_id: int,
) -> str:
    models_str = "+".join(short_names)
    fusion_short = "soft" if fusion_type == "softmax_gate" else "sig"
    dr_str = f"d{int(round(dropout_rate * 10)):02d}"
    return f"B__{models_str}___h{hidden_size}_{fusion_short}_{dr_str}___fold{fold_id}"
