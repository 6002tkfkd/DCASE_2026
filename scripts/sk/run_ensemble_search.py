#!/usr/bin/env python3
"""
앙상블 탐색 메인 실행 루프 (sk: module2~4 + greedy_search 진입점).

두 가지 탐색 모드:

  --mode full   (기본) 전체 조합 열거 후 평가.
                  min_combo_size ~ max_combo_size 범위의 모든 조합을 시도.

  --mode greedy  Solo → Pool 필터 → 탐욕적 확장.
                  1. 모든 모델을 solo 평가해 avg_hF1 획득
                  2. backbone당 상위 N개만 pool로 유지 (--pool-top-n)
                  3. 최고 solo 모델 시작 → 매 단계 margin 최대 후보 추가
                  결과는 동일한 summary.csv에 기록되어 full 결과와 합산 비교 가능.

주의: dcase_ensemble/ensemble/run_ensemble_search.py 라는 동명의 구버전 파일이 따로
있는데, 그건 MESH scripts/ensemble/run_ensemble_search.py와 호환되는 별개 시스템
(Option A 단순/가중평균만 지원)이라 이 스크립트와는 무관함.

Usage:
    # 전체 조합 탐색 (지금 best_combo.yaml은 4개 모델 고정이라 combo 1개뿐)
    python scripts/sk/run_ensemble_search.py --config scripts/sk/configs/best_combo.yaml

    # Greedy 탐색 — A/B 각각 (model_pool을 더 큰 풀로 확장했을 때 유용)
    python scripts/sk/run_ensemble_search.py --config scripts/sk/configs/best_combo.yaml \\
        --mode greedy --option B --pool-top-n 5 --max-k 8

    # 건식 실행 (full 모드만 지원)
    python scripts/sk/run_ensemble_search.py --config scripts/sk/configs/best_combo.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sk.module2_combo_manager import CombinationManager
from src.sk.module3_ensemble_engine.option_a import run_option_a
from src.sk.module3_ensemble_engine.option_b import run_option_b
from src.sk.module4_result_tracker import ResultTracker
from src.sk.greedy_search import GreedySearchRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    args = _parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cache_root  = Path(args.cache_root  or config["meta"]["cache_root"])
    output_root = Path(args.output_root or config["meta"]["output_root"])
    summary_csv = output_root / "summary.csv"
    weights_dir = output_root / "weights"

    # ── Greedy 모드 ────────────────────────────────────────────────────
    if args.mode == "greedy":
        if args.option.upper() not in ("A", "B"):
            logger.error("greedy 모드에서는 --option A 또는 --option B 를 명시해야 합니다.")
            sys.exit(1)
        runner = GreedySearchRunner(
            config=config,
            cache_root=cache_root,
            output_root=output_root,
            option=args.option.upper(),
            max_k=args.max_k,
            pool_top_n=args.pool_top_n,
            max_per_backbone=args.max_per_backbone,
            device=args.device,
        )
        runner.run()
        _print_top_results(ResultTracker(summary_csv), metric=args.rank_metric, k=5)
        return

    # ── Full 모드 (전체 조합 탐색) ──────────────────────────
    manager  = CombinationManager(args.config)
    all_jobs = manager.generate_jobs()

    if args.option and args.option.upper() != "ALL":
        all_jobs = [j for j in all_jobs if j.option == args.option.upper()]

    pending = manager.filter_pending(all_jobs, summary_csv)

    manager.print_summary(all_jobs, pending)

    if not pending:
        logger.info("대기 중인 Job이 없습니다. 종료.")
        return

    if args.dry_run:
        print("\n[Dry-run] 실행 예정 Job 목록:")
        for job in pending:
            print(f"  {job.job_id}")
        return

    # ── 실행 루프 ────────────────────────────────────────────────────
    tracker = ResultTracker(summary_csv)
    n_total  = len(pending)
    n_done   = 0
    n_failed = 0

    logger.info("탐색 시작: %d개 Job | cache=%s | output=%s", n_total, cache_root, output_root)

    for i, job in enumerate(pending, 1):
        logger.info("[%d/%d] 시작: %s", i, n_total, job.job_id)

        try:
            if job.option == "A":
                metrics = run_option_a(job, cache_root)
            elif job.option == "B":
                metrics = run_option_b(job, cache_root, weights_dir, device=args.device)
            else:
                raise ValueError(f"알 수 없는 option: '{job.option}'")

            tracker.record(job, metrics)
            n_done += 1

            metric_str = "  ".join(f"{k}={v:.2f}" for k, v in metrics.items())
            logger.info("[%d/%d] 완료: %s | %s", i, n_total, job.job_id, metric_str)

        except KeyboardInterrupt:
            logger.warning("사용자 중단 (Ctrl+C). %d/%d 완료.", n_done, i - 1)
            break

        except Exception:
            n_failed += 1
            logger.exception("[%d/%d] 실패 (건너뜀): %s", i, n_total, job.job_id)
            continue

    # ── 결과 요약 ────────────────────────────────────────────────────
    logger.info(
        "탐색 완료: 성공 %d / 전체 %d / 실패 %d",
        n_done, n_total, n_failed,
    )

    _print_top_results(tracker, metric=args.rank_metric, k=3)


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _print_top_results(tracker: ResultTracker, metric: str, k: int) -> None:
    """완료 후 top-k 결과를 콘솔에 출력."""
    top = tracker.get_top_k(k=k, metric=metric)
    if top.empty:
        return

    print(f"\n{'='*60}")
    print(f"  Top-{k} 앙상블 조합 (기준: avg_{metric})")
    print(f"{'='*60}")
    for _, row in top.iterrows():
        avg_val = row.get(f"avg_{metric}", float("nan"))
        std_val = row.get(f"std_{metric}", float("nan"))
        print(
            f"  [{row['option']}] {row['models_used']}\n"
            f"      hyperparams={row['hyperparams']}\n"
            f"      avg_{metric}={avg_val:.2f}  std={std_val:.2f}  n_folds={int(row['n_folds'])}\n"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="앙상블 탐색 실행 루프 (sk)")
    parser.add_argument(
        "--config", default="scripts/sk/configs/best_combo.yaml",
        help="search_config.yaml 경로 (기본값: scripts/sk/configs/best_combo.yaml)",
    )
    parser.add_argument(
        "--mode", default="full",
        choices=["full", "greedy"],
        help="탐색 모드: full=전체 조합, greedy=탐욕적 확장 (기본값: full)",
    )
    parser.add_argument(
        "--option", default="all",
        choices=["A", "B", "all"],
        help="실행할 Option 종류 (기본값: all). greedy 모드에서는 A 또는 B 필수.",
    )
    parser.add_argument(
        "--pool-top-n", type=int, default=None,
        dest="pool_top_n",
        help="[greedy 전용] backbone당 solo 상위 N개를 pool로 유지. 미지정 시 전체.",
    )
    parser.add_argument(
        "--max-per-backbone", type=int, default=2,
        dest="max_per_backbone",
        help="[greedy 전용] 앙상블 조합에서 동일 backbone 최대 허용 수 (기본값: 2).",
    )
    parser.add_argument(
        "--max-k", type=int, default=None,
        dest="max_k",
        help="[greedy 전용] 최대 앙상블 크기. 미지정 시 config의 max_combo_size 사용.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="PyTorch device (기본값: auto — CUDA 있으면 cuda, 없으면 cpu)",
    )
    parser.add_argument(
        "--rank-metric", default="hF1",
        choices=["hF1", "accuracy", "top_accuracy"],
        help="완료 후 top-k 출력 기준 메트릭 (기본값: hF1)",
    )
    parser.add_argument(
        "--output-root", default=None,
        dest="output_root",
        help="결과 저장 경로 (미지정 시 config의 output_root 사용)",
    )
    parser.add_argument(
        "--cache-root", default=None,
        dest="cache_root",
        help="OOF 캐시 경로 (미지정 시 config의 cache_root 사용)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="[full 모드 전용] 실제 실행 없이 실행 예정 Job 목록만 출력",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
