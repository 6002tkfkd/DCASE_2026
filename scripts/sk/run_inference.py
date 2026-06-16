#!/usr/bin/env python3
"""
DCASE 제출 추론 스크립트 (sk: module5 진입점): eval .wav 폴더 → submission.csv

흐름:
    1. summary.csv에서 Job 선택 (--job-id 명시 또는 --top 자동 선택)
    2. InferenceEngine 실행:
       - Branch 1 모델별 순차 추출 (OOM 방지)
       - Option A: logits 평균, Option B: GatingNetwork 또는 OOF-stacking 적용
    3. submission.csv 저장

사전 조건:
    - eval 오디오의 .npy 임베딩이 search_config.yaml의 `eval:` 블록에 설정된
      (legacy_data_root 기준 상대) 경로에 이미 추출되어 있어야 합니다.

Usage:
    # 특정 Job ID로 추론
    python scripts/sk/run_inference.py \\
        --config scripts/sk/configs/best_combo.yaml \\
        --job-id "B__A+B+C+D___h64_soft_d03___fold0_oof" \\
        --audio-dir ./eval_audio

    # summary.csv 최고 성능 Job 자동 선택
    python scripts/sk/run_inference.py \\
        --config scripts/sk/configs/best_combo.yaml \\
        --top \\
        --audio-dir ./eval_audio \\
        --output-csv ./submission.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sk.module5_inference_engine import InferenceEngine

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

    output_root = Path(config["meta"]["output_root"])
    summary_csv = output_root / "summary.csv"
    batch_size  = int(config["meta"].get("batch_size", 256))

    # ── Job ID 결정 ─────────────────────────────────────────────────
    job_id = args.job_id if args.job_id else _pick_top_job(summary_csv, args.rank_metric)

    # ── 추론 실행 ────────────────────────────────────────────────────
    engine = InferenceEngine(
        config_path = args.config,
        summary_csv = summary_csv,
        output_root = output_root,
    )

    output_csv = Path(args.output_csv)
    submission = engine.run(
        job_id     = job_id,
        audio_dir  = args.audio_dir,
        output_csv = output_csv,
        eval_split = args.eval_split,
        batch_size = batch_size,
        device     = args.device,
    )

    # ── 결과 요약 ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Job ID  :  {job_id}")
    print(f"  파일 수 :  {len(submission)}")
    print(f"  저장    :  {output_csv.resolve()}")
    print(f"{'='*60}")
    print("\n[클래스별 예측 분포]")
    dist = submission["scene_label"].value_counts()
    for label, cnt in dist.items():
        bar = "█" * int(cnt / len(submission) * 30)
        print(f"  {label:<25} {cnt:>5}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _pick_top_job(summary_csv: Path, metric: str) -> str:
    """summary.csv에서 metric 기준 단일 최고 Job의 job_id 반환."""
    if not summary_csv.exists():
        raise FileNotFoundError(
            f"summary.csv 없음: {summary_csv}\n"
            f"  먼저 scripts/sk/run_ensemble_search.py를 실행하세요."
        )
    df = pd.read_csv(summary_csv)
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    valid = df.dropna(subset=[metric])
    if valid.empty:
        raise ValueError(f"summary.csv에 '{metric}' 값이 있는 완료된 Job이 없습니다.")

    best = valid.sort_values(metric, ascending=False).iloc[0]
    logger.info(
        "[M5] 자동 선택: %s  (%s=%.2f, fold=%s)",
        best["job_id"], metric, float(best[metric]), best["fold_id"],
    )
    return str(best["job_id"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DCASE 앙상블 추론 → submission.csv (sk)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="scripts/sk/configs/best_combo.yaml",
        help="search_config.yaml 경로",
    )
    parser.add_argument(
        "--audio-dir", required=True,
        help="평가용 .wav 파일 디렉토리",
    )
    parser.add_argument(
        "--output-csv", default="./submission.csv",
        help="submission CSV 저장 경로 (default: ./submission.csv)",
    )
    parser.add_argument(
        "--eval-split", default="eval",
        help="사용하지 않음 (호환성 유지용). eval 임베딩 위치는 config의 eval: 블록으로 결정.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="PyTorch device: auto | cpu | cuda | cuda:1 (default: auto)",
    )
    parser.add_argument(
        "--rank-metric", default="hF1",
        choices=["hF1", "accuracy", "top_accuracy"],
        help="--top 사용 시 자동 선택 기준 메트릭 (default: hF1)",
    )

    # Job 선택: 명시 또는 자동 중 하나 필수
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--job-id",
        help="실행할 특정 Job ID (summary.csv의 job_id 컬럼 값)",
    )
    group.add_argument(
        "--top", action="store_true",
        help="summary.csv에서 --rank-metric 기준 최고 성능 Job 자동 선택",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
