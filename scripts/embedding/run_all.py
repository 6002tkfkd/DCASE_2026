#!/usr/bin/env python3
"""
Master embedding extraction runner — 전체 모델 한 번에 실행.

--mode all  : 각 모델 전체 grid search (BSD10k-v1.2 기준 약 533 sets)
--mode 35k  : 각 모델 best combos only (BSD35k-CS)

  BEATs       3 ckpt × 7 seq × 7 chunk = 147  (all) / 2 combos (35k)
  PaSST       2 ckpt ×          7 chunk =  14  (all) / 1 combo  (35k)
  fPaSST      3 ckpt × 7 seq × 7 chunk = 147  (all) / 1 combo  (35k)
  ATST-F      3 ckpt × 7 seq × 7 chunk = 147  (all) / 1 combo  (35k)
  EfficientAT 6 model×          7 chunk =  42  (all) / —
  CLAP        7 ckpt × (7 audio+1 text) =  56  (all) / 2 combos (35k)
  M2D                                    =  15  (all) / 1 combo  (35k)

Usage:
    python scripts/embedding/run_all.py --mode all
    python scripts/embedding/run_all.py --mode 35k
    python scripts/embedding/run_all.py --mode all --models beats clap
    python scripts/embedding/run_all.py --mode all --skip-done
    python scripts/embedding/run_all.py --mode all --dry-run
"""
import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SCRIPT_DIR   = os.path.dirname(__file__)

ALL_MODELS = ["beats", "passt", "fpasst", "atst", "efficientat", "clap", "m2d"]

RUNNER_MAP = {
    "beats":       "run_beats.py",
    "passt":       "run_passt.py",
    "fpasst":      "run_fpasst.py",
    "atst":        "run_atst.py",
    "efficientat": "run_efficientat.py",
    "clap":        "run_clap.py",
    "m2d":         "run_m2d.py",
}

# efficientat는 35k 미지원 (10k only)
NO_35K_MODELS = {"efficientat"}

EXPECTED_COUNTS = {
    "beats":       {"all": 147, "35k": 2},
    "passt":       {"all":  14, "35k": 1},
    "fpasst":      {"all": 147, "35k": 1},
    "atst":        {"all": 147, "35k": 1},
    "efficientat": {"all":  42, "35k": 0},
    "clap":        {"all":  56, "35k": 2},
    "m2d":         {"all":  15, "35k": 1},
}


def run_model(model, mode, extra_args, dry_run):
    script = os.path.join(SCRIPT_DIR, RUNNER_MAP[model])

    if mode == "35k" and model in NO_35K_MODELS:
        print(f"\n[{model.upper()}] 35k 미지원 — 건너뜀")
        return True

    cmd = [sys.executable, script, "--mode", mode] + extra_args
    expected = EXPECTED_COUNTS[model][mode]
    print(f"\n{'='*60}")
    print(f"  [{model.upper()}]  mode={mode}  expected: {expected} sets")
    print(f"{'='*60}")

    if dry_run:
        print(f"  (dry-run) {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode == 0

    t0 = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s   = divmod(rem, 60)
    print(f"\n  [{model.upper()}] 완료: {h:02d}:{m:02d}:{s:02d}  (returncode={result.returncode})")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="전체 모델 embedding 한 번에 추출")
    parser.add_argument("--mode", choices=["all", "35k"], required=True,
                        help="all: 전체 grid search / 35k: best combos only")
    parser.add_argument("--dry-run",   action="store_true", help="명령어만 출력, 실행 안 함")
    parser.add_argument("--skip-done", action="store_true", help="완료된 조합 건너뜀")
    parser.add_argument("--overwrite", action="store_true", help="기존 파일 덮어쓰기")
    parser.add_argument("--models", nargs="+", default=None, choices=ALL_MODELS,
                        help=f"실행할 모델 지정 (기본: 전체). 선택지: {ALL_MODELS}")
    args = parser.parse_args()

    models = args.models if args.models else ALL_MODELS

    extra_args = []
    if args.dry_run:
        extra_args.append("--dry-run")
    if args.skip_done:
        extra_args.append("--skip-done")
    if args.overwrite:
        extra_args.append("--overwrite")

    total_expected = sum(EXPECTED_COUNTS[m][args.mode] for m in models)
    print(f"실행할 모델: {models}")
    print(f"mode: {args.mode}  예상 embedding sets: {total_expected}")
    print(f"flags: dry_run={args.dry_run}  skip_done={args.skip_done}  overwrite={args.overwrite}")

    wall_start = time.time()
    failed = []

    for model in models:
        ok = run_model(model, args.mode, extra_args, args.dry_run)
        if not ok:
            failed.append(model)

    wall_elapsed = time.time() - wall_start
    h, rem = divmod(int(wall_elapsed), 3600)
    m, s   = divmod(rem, 60)

    print(f"\n{'='*60}")
    print(f"전체 완료: {h:02d}:{m:02d}:{s:02d}")
    print(f"성공: {len(models) - len(failed)}/{len(models)}")
    if failed:
        print(f"실패한 모델: {failed}")


if __name__ == "__main__":
    main()
