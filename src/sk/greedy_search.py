"""
Greedy Ensemble Search

전략:
    1. Solo pass  — pool의 모든 모델을 1개씩 평가 → avg_hF1 획득
    2. Pool 필터  — backbone별 상위 pool_top_n개만 유지 (--pool-top-n)
    3. Greedy 확장 — 최고 solo 모델로 시작, 매 단계 margin 최대 후보를 추가
                    개선 없거나 max_k 도달 시 종료

Option A 최적화:
    combo logit sum을 fold별로 캐시 → 후보 추가 시 sum만 업데이트 (O(1))

Option B:
    GatingNetwork 전체를 매 후보마다 훈련 → 정확하지만 느림
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.sk.module2_combo_manager import ExperimentJob, _make_job_id_a, _make_job_id_b
from src.sk.module3_ensemble_engine._shared import (
    build_class_to_top,
    compute_metrics,
    load_npz_aligned,
)
from src.sk.module3_ensemble_engine.option_a import _learn_weights, run_option_a
from src.sk.module3_ensemble_engine.option_b import run_option_b, _run_oof_stacking
from src.sk.module4_result_tracker import ResultTracker

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

class GreedySearchRunner:
    """
    Args:
        config:      search_config.yaml 파싱 결과 dict
        cache_root:  .npz 캐시 루트
        output_root: summary.csv / weights 저장 루트
        option:      "A" | "B"
        max_k:       최대 앙상블 크기 (CLI --max-k 우선, 없으면 config max_combo_size)
        pool_top_n:  backbone당 solo 상위 N개 유지 (None = 전체)
        device:      "auto" | "cpu" | "cuda"
    """

    def __init__(
        self,
        config: dict,
        cache_root: Path,
        output_root: Path,
        option: str,
        max_k: int | None,
        pool_top_n: int | None,
        max_per_backbone: int = 2,
        device: str = "auto",
    ) -> None:
        self.config           = config
        self.cache_root       = Path(cache_root)
        self.output_root      = Path(output_root)
        self.option           = option.upper()
        self.pool_top_n       = pool_top_n
        self.max_per_backbone = max_per_backbone
        self.device           = device

        search = config.get("search", {})
        self.k_folds = int(search.get("k_folds", 5))
        self.max_k   = max_k if max_k is not None else int(search.get("max_combo_size", 8))

        self._option_a_cfg = search.get("option_a", {})
        self._option_b_cfg = search.get("option_b", {})

        # model_pool 파싱: {name, short, backbone}
        self._pool: list[dict] = []
        for m in config.get("model_pool", []):
            backbone = _extract_backbone(m.get("experiment_dir", ""))
            self._pool.append({
                "name":     m["name"],
                "short":    m.get("short_name", m["name"]),
                "backbone": backbone,
            })

        # 중복 job_id 확인용
        summary_csv = self.output_root / "summary.csv"
        self._tracker = ResultTracker(summary_csv)
        self._done_ids: set[str] = _load_done_ids(summary_csv)

    # ─────────────────────────────────────────────────────────────
    # Main
    # ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info("[Greedy] option=%s  pool=%d  pool_top_n=%s  max_k=%d  k_folds=%d",
                    self.option, len(self._pool), self.pool_top_n, self.max_k, self.k_folds)

        # ── Step 1: Solo pass ────────────────────────────────────
        logger.info("[Greedy] Step 1/3 — Solo evaluation (%d models)", len(self._pool))
        solo_scores = self._solo_pass()

        if not solo_scores:
            logger.warning("[Greedy] Solo 결과 없음. 중단.")
            return

        # ── Step 2: Pool 필터 ────────────────────────────────────
        filtered = self._filter_pool(solo_scores)
        logger.info("[Greedy] Step 2/3 — Pool 필터 완료: %d → %d 모델", len(self._pool), len(filtered))
        for m in filtered:
            logger.info("  [pool] %-50s  backbone=%-40s  solo_hF1=%.2f",
                        m["short"], m["backbone"], solo_scores.get(m["name"], 0.0))

        # ── Step 3: Greedy 확장 ──────────────────────────────────
        logger.info("[Greedy] Step 3/3 — Greedy expansion (max_k=%d)", self.max_k)
        self._greedy_expand(filtered, solo_scores)

        logger.info("[Greedy] 완료.")

    # ─────────────────────────────────────────────────────────────
    # Step 1: Solo pass
    # ─────────────────────────────────────────────────────────────

    def _solo_pass(self) -> dict[str, float]:
        """
        각 모델을 단독으로 평가. 이미 summary.csv에 있으면 재사용.
        Option B인 경우 solo ranking용으로는 simple_average(Option A) 점수를 사용.
        GatingNetwork solo 학습은 240×5=1200번 → 극히 느리고 순위 결과는 동일.
        """
        solo_scores: dict[str, float] = {}

        for i, m in enumerate(self._pool, 1):
            fold_scores = []
            for fold in range(self.k_folds):
                if self.option == "B":
                    # Option B solo: simple_average 점수로 빠르게 ranking
                    jid = _make_job_id_a([m["short"]], "simple_average", fold)
                else:
                    jid = self._make_jid([m["short"]], fold)

                if jid in self._done_ids:
                    score = self._read_score_from_csv(jid)
                    if score is not None:
                        fold_scores.append(score)
                        continue

                try:
                    if self.option == "B":
                        # solo ranking: simple_average로 빠르게 평가 (GatingNetwork 학습 생략)
                        test_data  = load_npz_aligned(self.cache_root, [m["name"]], fold, "test")
                        train_data = load_npz_aligned(self.cache_root, [m["name"]], fold, "train")
                        class_to_top = None
                        if "top_labels" in train_data[0]:
                            class_to_top = build_class_to_top(
                                train_data[0]["labels"], train_data[0]["top_labels"]
                            )
                        logits = test_data[0]["logits"]
                        metrics = compute_metrics(logits, test_data[0]["labels"],
                                                  test_data[0].get("top_labels"), class_to_top)
                        job = ExperimentJob(
                            job_id=jid, models=[m["short"]], model_names=[m["name"]],
                            option="A", fold_id=fold,
                            hyperparams={"strategy": "simple_average"},
                        )
                    else:
                        metrics = self._eval_single_fold([m], fold)
                        job = self._make_job([m], fold)

                    self._tracker.record(job, metrics)
                    self._done_ids.add(jid)
                    fold_scores.append(metrics.get("hF1", 0.0))
                except Exception:
                    logger.exception("[Greedy-Solo] 실패: %s fold=%d", m["name"], fold)

            if fold_scores:
                avg = float(np.mean(fold_scores))
                solo_scores[m["name"]] = avg
                logger.info("[Greedy-Solo] [%d/%d] %-50s  avg_hF1=%.2f",
                            i, len(self._pool), m["short"], avg)
            else:
                logger.warning("[Greedy-Solo] [%d/%d] %s — 전체 fold 실패", i, len(self._pool), m["short"])

        return solo_scores

    # ─────────────────────────────────────────────────────────────
    # Step 2: Pool 필터
    # ─────────────────────────────────────────────────────────────

    def _filter_pool(self, solo_scores: dict[str, float]) -> list[dict]:
        if self.pool_top_n is None:
            return [m for m in self._pool if m["name"] in solo_scores]

        # backbone별 그룹
        from collections import defaultdict
        groups: dict[str, list[dict]] = defaultdict(list)
        for m in self._pool:
            if m["name"] in solo_scores:
                groups[m["backbone"]].append(m)

        kept = []
        for backbone, members in sorted(groups.items()):
            ranked = sorted(members, key=lambda x: solo_scores.get(x["name"], 0.0), reverse=True)
            top = ranked[:self.pool_top_n]
            kept.extend(top)
            logger.info("[Greedy-Filter] backbone=%-45s  kept %d/%d",
                        backbone, len(top), len(members))

        return kept

    # ─────────────────────────────────────────────────────────────
    # Step 3: Greedy expansion
    # ─────────────────────────────────────────────────────────────

    def _greedy_expand(self, pool: list[dict], solo_scores: dict[str, float]) -> None:
        if not pool:
            logger.warning("[Greedy] pool이 비어 있음.")
            return

        # 시작 모델: solo 최고 (solo_scores는 test 기반 — pool ranking용)
        seed = max(pool, key=lambda m: solo_scores.get(m["name"], 0.0))
        selected = [seed]
        remaining = [m for m in pool if m["name"] != seed["name"]]

        logger.info("[Greedy] k=1 시작: %s  test_solo_hF1=%.2f",
                    seed["short"], solo_scores.get(seed["name"], 0.0))

        from collections import Counter

        for step in range(2, self.max_k + 1):
            if not remaining:
                logger.info("[Greedy] 후보 소진 → 종료 (k=%d)", step - 1)
                break

            # 현재 선택된 백본 카운트 → max_per_backbone 초과 후보 제외
            backbone_count = Counter(m["backbone"] for m in selected)
            eligible = [
                c for c in remaining
                if backbone_count.get(c["backbone"], 0) < self.max_per_backbone
            ]
            if not eligible:
                logger.info("[Greedy] 백본 다양성 제약으로 eligible 후보 없음 → 종료 (k=%d)", step - 1)
                break

            best_candidate = None
            best_fold_scores: list[float] = []

            logger.info("[Greedy] k=%d 탐색: %d개 후보 (backbone 제약 후 %d개)",
                        step, len(remaining), len(eligible))

            if self.option == "A":
                # 캐시 기반 빠른 탐색 (val 기반 — 후보 선택 bias 제거)
                combo_logit_sums, labels_per_fold, top_labels_per_fold = \
                    self._preload_logit_sums(selected)

                # val 기반 현재 선택 combo 점수 (baseline; 매 step마다 재계산)
                current_val_scores = []
                n_sel = len(selected)
                for fold in range(self.k_folds):
                    avg_l = combo_logit_sums[fold] / n_sel
                    m = compute_metrics(avg_l, labels_per_fold[fold], top_labels_per_fold[fold])
                    current_val_scores.append(m.get("hF1", 0.0))
                current_score = float(np.mean(current_val_scores))
                best_score = current_score

                for cand in eligible:
                    try:
                        fold_scores = self._try_add_a_cached(
                            cand, combo_logit_sums, labels_per_fold,
                            top_labels_per_fold, len(selected),
                        )
                        avg = float(np.mean(fold_scores))
                        if avg > best_score:
                            best_score     = avg
                            best_candidate = cand
                            best_fold_scores = fold_scores
                    except Exception:
                        logger.debug("[Greedy] 후보 실패: %s", cand["short"], exc_info=True)
            else:
                # Option B: 매 후보마다 GatingNetwork / OOF LogReg 평가
                b_model_type = self._default_b_hp().get("model_type", "gating")

                def _eval_b(specs: list[dict]) -> float:
                    if b_model_type == "oof_stacking":
                        dummy_job = ExperimentJob(
                            job_id="tmp", models=[m["short"] for m in specs],
                            model_names=[m["name"] for m in specs],
                            option="B", fold_id=0, hyperparams=self._default_b_hp(),
                        )
                        return _run_oof_stacking(
                            dummy_job, self.cache_root, self.k_folds
                        ).get("hF1", 0.0)
                    return float(np.mean([
                        self._eval_single_fold(specs, f).get("hF1", 0.0)
                        for f in range(self.k_folds)
                    ]))

                current_score = _eval_b(selected)
                best_score = current_score

                for j, cand in enumerate(eligible, 1):
                    trial = selected + [cand]
                    try:
                        avg = _eval_b(trial)
                        logger.info("  [B] (%d/%d) +%s → hF1=%.2f",
                                    j, len(remaining), cand["short"], avg)
                        if avg > best_score:
                            best_score     = avg
                            best_candidate = cand
                            best_fold_scores = [avg]
                    except Exception:
                        logger.debug("[Greedy] 후보 실패: %s", cand["short"], exc_info=True)

            if best_candidate is None:
                logger.info("[Greedy] k=%d: 개선 없음 → 조기 종료 (score=%.2f)", step, current_score)
                break

            selected.append(best_candidate)
            remaining.remove(best_candidate)
            current_score = best_score
            logger.info("[Greedy] k=%d 선택: +%s  avg_hF1=%.2f",
                        step, best_candidate["short"], current_score)

            # 확정 combo를 모든 fold에 대해 기록 (이미 done이면 skip)
            self._record_combo_all_folds(selected)

        # 최종 combo 요약
        final_names = [m["short"] for m in selected]
        logger.info("[Greedy] 최종 조합 (k=%d): %s  avg_hF1=%.2f",
                    len(selected), "+".join(final_names), current_score)

    # ─────────────────────────────────────────────────────────────
    # Option A: 캐시 기반 평가
    # ─────────────────────────────────────────────────────────────

    def _preload_logit_sums(self, selected: list[dict]) -> tuple:
        """
        현재 선택된 combo의 fold별 val logit sum 캐시.
        greedy 후보 탐색은 val로 수행 → selection bias 없이 test 점수 보고 가능.
        Returns: (logit_sums, labels_per_fold, top_labels_per_fold)
        """
        logit_sums:    dict[int, np.ndarray] = {}
        labels_pf:     dict[int, np.ndarray] = {}
        top_labels_pf: dict[int, np.ndarray | None] = {}

        for fold in range(self.k_folds):
            data = load_npz_aligned(self.cache_root, [m["name"] for m in selected], fold, "val")
            logit_sums[fold]    = np.stack([d["logits"] for d in data]).sum(axis=0)
            labels_pf[fold]     = data[0]["labels"]
            top_labels_pf[fold] = data[0].get("top_labels")

        return logit_sums, labels_pf, top_labels_pf

    def _try_add_a_cached(
        self,
        cand: dict,
        combo_logit_sums: dict[int, np.ndarray],
        labels_pf: dict[int, np.ndarray],
        top_labels_pf: dict[int, np.ndarray | None],
        n_selected: int,
    ) -> list[float]:
        """cand 1개를 기존 val sum에 더해 hF1을 계산 (full reload 없이)."""
        fold_scores = []
        for fold in range(self.k_folds):
            cand_data = load_npz_aligned(self.cache_root, [cand["name"]], fold, "val")
            new_sum    = combo_logit_sums[fold] + cand_data[0]["logits"]
            avg_logits = new_sum / (n_selected + 1)
            metrics = compute_metrics(avg_logits, labels_pf[fold], top_labels_pf[fold])
            fold_scores.append(metrics.get("hF1", 0.0))
        return fold_scores

    # ─────────────────────────────────────────────────────────────
    # 공통: 단일 fold 평가
    # ─────────────────────────────────────────────────────────────

    def _eval_single_fold(self, model_specs: list[dict], fold: int) -> dict[str, Any]:
        names = [m["name"] for m in model_specs]

        if self.option == "A":
            strategy = (self._option_a_cfg.get("strategies") or ["simple_average"])[0]
            val_data  = load_npz_aligned(self.cache_root, names, fold, "val")
            test_data = load_npz_aligned(self.cache_root, names, fold, "test")

            if strategy == "weighted_average" and len(names) > 1:
                val_stack = np.stack([d["logits"] for d in val_data])
                w = _learn_weights(val_stack, val_data[0]["labels"])
                test_stack = np.stack([d["logits"] for d in test_data])
                logits = (test_stack * w[:, None, None]).sum(axis=0)
            else:
                logits = np.stack([d["logits"] for d in test_data]).mean(axis=0)

            train_data = load_npz_aligned(self.cache_root, names, fold, "train")
            class_to_top = None
            if "top_labels" in train_data[0]:
                class_to_top = build_class_to_top(
                    train_data[0]["labels"], train_data[0]["top_labels"]
                )
            return compute_metrics(logits, test_data[0]["labels"],
                                   test_data[0].get("top_labels"), class_to_top)

        else:  # Option B
            hp = self._default_b_hp()
            job = self._make_job(model_specs, fold)
            weights_dir = self.output_root / "weights"
            return run_option_b(job, self.cache_root, weights_dir, device=self.device)

    def _default_b_hp(self) -> dict:
        cfg = self._option_b_cfg
        return {
            "model_type":    cfg.get("model_type", "stacking"),
            "hidden_size":   (cfg.get("hidden_sizes") or [64])[0],
            "fusion_type":   (cfg.get("fusion_types") or ["softmax_gate"])[0],
            "dropout_rate":  (cfg.get("dropout_rates") or [0.3])[0],
            "num_epochs":    int(cfg.get("num_epochs", 100)),
            "lr":            float(cfg.get("lr", 0.001)),
            "patience":      int(cfg.get("patience", 15)),
        }

    # ─────────────────────────────────────────────────────────────
    # 기록: 모든 fold + 모든 hyperparams
    # ─────────────────────────────────────────────────────────────

    def _record_combo_all_folds(self, model_specs: list[dict]) -> None:
        """확정된 combo를 모든 fold/hyperparams에 대해 평가하고 기록."""
        names  = [m["name"] for m in model_specs]
        shorts = [m["short"] for m in model_specs]

        if self.option == "A":
            strategies = self._option_a_cfg.get("strategies", ["simple_average"])
            for strategy in strategies:
                for fold in range(self.k_folds):
                    jid = _make_job_id_a(shorts, strategy, fold)
                    if jid in self._done_ids:
                        continue
                    try:
                        job = ExperimentJob(
                            job_id=jid, models=shorts, model_names=names,
                            option="A", fold_id=fold,
                            hyperparams={"strategy": strategy},
                        )
                        metrics = run_option_a(job, self.cache_root)
                        self._tracker.record(job, metrics)
                        self._done_ids.add(jid)
                    except Exception:
                        logger.exception("[Greedy] 기록 실패: %s", jid)

        else:  # Option B
            hp = self._default_b_hp()
            weights_dir = self.output_root / "weights"
            model_type = hp.get("model_type", "gating")

            if model_type == "oof_stacking":
                base_jid = _make_job_id_b(
                    shorts, hp["hidden_size"], hp["fusion_type"], hp["dropout_rate"], fold_id=0
                ) + "_oof"
                fold_jids = [f"{base_jid}_fold{f}" for f in range(self.k_folds)]
                if not all(jid in self._done_ids for jid in fold_jids):
                    try:
                        # oof_stacking은 5-fold 데이터를 내부에서 모두 합산해 단일 메트릭을 산출함.
                        # summary_agg.csv의 n_folds(=count) 집계가 실제 사용 fold 수(5)와
                        # 맞아떨어지도록, 동일 metrics를 fold_id=0~4 각각으로 기록한다.
                        rep_job = ExperimentJob(
                            job_id=fold_jids[0], models=shorts, model_names=names,
                            option="B", fold_id=0, hyperparams=hp,
                        )
                        save_path = weights_dir / base_jid
                        metrics = _run_oof_stacking(
                            rep_job, self.cache_root, self.k_folds, save_path=save_path
                        )
                        for fold, jid in enumerate(fold_jids):
                            if jid in self._done_ids:
                                continue
                            job = ExperimentJob(
                                job_id=jid, models=shorts, model_names=names,
                                option="B", fold_id=fold, hyperparams=hp,
                            )
                            self._tracker.record(job, metrics)
                            self._done_ids.add(jid)
                    except Exception:
                        logger.exception("[Greedy] 기록 실패: %s", base_jid)
            else:
                for fold in range(self.k_folds):
                    jid = _make_job_id_b(
                        shorts, hp["hidden_size"], hp["fusion_type"], hp["dropout_rate"], fold
                    )
                    if jid in self._done_ids:
                        continue
                    try:
                        job = ExperimentJob(
                            job_id=jid, models=shorts, model_names=names,
                            option="B", fold_id=fold, hyperparams=hp,
                        )
                        metrics = run_option_b(job, self.cache_root, weights_dir, device=self.device)
                        self._tracker.record(job, metrics)
                        self._done_ids.add(jid)
                    except Exception:
                        logger.exception("[Greedy] 기록 실패: %s", jid)

    # ─────────────────────────────────────────────────────────────
    # 헬퍼
    # ─────────────────────────────────────────────────────────────

    def _make_job(self, model_specs: list[dict], fold: int) -> ExperimentJob:
        shorts = [m["short"] for m in model_specs]
        names  = [m["name"]  for m in model_specs]
        if self.option == "A":
            strategy = (self._option_a_cfg.get("strategies") or ["simple_average"])[0]
            return ExperimentJob(
                job_id=_make_job_id_a(shorts, strategy, fold),
                models=shorts, model_names=names,
                option="A", fold_id=fold,
                hyperparams={"strategy": strategy},
            )
        else:
            hp = self._default_b_hp()
            return ExperimentJob(
                job_id=_make_job_id_b(
                    shorts, hp["hidden_size"], hp["fusion_type"], hp["dropout_rate"], fold
                ),
                models=shorts, model_names=names,
                option="B", fold_id=fold,
                hyperparams=hp,
            )

    def _make_jid(self, shorts: list[str], fold: int) -> str:
        if self.option == "A":
            strategy = (self._option_a_cfg.get("strategies") or ["simple_average"])[0]
            return _make_job_id_a(shorts, strategy, fold)
        else:
            hp = self._default_b_hp()
            return _make_job_id_b(
                shorts, hp["hidden_size"], hp["fusion_type"], hp["dropout_rate"], fold
            )

    def _read_score_from_csv(self, job_id: str) -> float | None:
        df = self._tracker.load_all()
        row = df[df["job_id"] == job_id]
        if row.empty:
            return None
        val = row["hF1"].iloc[0]
        try:
            return float(val)
        except (ValueError, TypeError):
            return None


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _extract_backbone(experiment_dir: str) -> str:
    """
    experiment_dir 에서 backbone 폴더명 추출.
    예) output/m2d_clap_vit_base_meanstd/two_stage_exp1_hloss/... → m2d_clap_vit_base_meanstd
    """
    parts = Path(experiment_dir).parts
    for i, p in enumerate(parts):
        if p == "output" and i + 1 < len(parts):
            return parts[i + 1]
    # fallback: 두 번째 파트
    return parts[1] if len(parts) > 1 else experiment_dir


def _load_done_ids(summary_csv: Path) -> set[str]:
    if not summary_csv.exists():
        return set()
    import pandas as pd
    try:
        return set(pd.read_csv(summary_csv, usecols=["job_id"])["job_id"].tolist())
    except Exception:
        return set()
