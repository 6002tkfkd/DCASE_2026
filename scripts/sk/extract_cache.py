#!/usr/bin/env python3
"""
Module 1 실행 엔트리 포인트: OOF 캐시 추출.

dcase_ensemble에 이미 학습된 best-combo 체크포인트를 legacy_data_root 상대경로로
참조해 MESH 안(scripts/sk/cache/)에 z/logits 캐시를 새로 생성한다.

Usage:
    python scripts/sk/extract_cache.py --config scripts/sk/configs/best_combo.yaml
    python scripts/sk/extract_cache.py --config scripts/sk/configs/best_combo.yaml --model m2d_clap_vit_base_meanstd__e1hl__foc_acc_grand_c
    python scripts/sk/extract_cache.py --config scripts/sk/configs/best_combo.yaml --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sk.module1_cache_extractor import OOFCacheExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 1: OOF 캐시 추출")
    parser.add_argument("--config", default="scripts/sk/configs/best_combo.yaml",
                        help="search_config.yaml 경로 (기본값: scripts/sk/configs/best_combo.yaml)")
    parser.add_argument("--model", default=None, help="특정 모델 하나만 처리 (미지정시 전체)")
    parser.add_argument("--dry-run", action="store_true", help="실제 추출 없이 경로/상태만 출력")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cache_root = config["meta"]["cache_root"]
    legacy_data_root = config["meta"]["legacy_data_root"]
    k_folds = config["search"].get("k_folds", 5)
    batch_size = config["meta"].get("batch_size", 256)
    num_workers = config["meta"].get("num_workers", 4)

    extractor = OOFCacheExtractor(
        cache_root, legacy_data_root, batch_size=batch_size, num_workers=num_workers
    )

    model_pool = config["model_pool"]
    if args.model:
        model_pool = [m for m in model_pool if m["name"] == args.model]
        if not model_pool:
            logger.error("모델 '%s' 을 model_pool 에서 찾을 수 없습니다.", args.model)
            sys.exit(1)

    for model_spec in model_pool:
        name = model_spec["name"]
        experiment_dir = model_spec["experiment_dir"]

        print(f"\n{'='*60}")
        print(f"[M1] 모델: {name}")
        print(f"     legacy_data_root: {legacy_data_root}")
        print(f"     experiment_dir:   {experiment_dir}")
        print(f"{'='*60}")

        if args.dry_run:
            for fold_id in range(k_folds):
                for split in ("train", "val", "test"):
                    cached = extractor.is_cached(name, fold_id, split)
                    status = "CACHED" if cached else "MISSING"
                    print(f"  fold_{fold_id}/{split}.npz  [{status}]")
            continue

        extractor.extract_model(
            model_name=name,
            experiment_dir=experiment_dir,
            n_folds=k_folds,
        )
        print(f"[M1] 완료: {name}")

    print("\n모든 모델 처리 완료.")


if __name__ == "__main__":
    main()
