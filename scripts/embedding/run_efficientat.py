#!/usr/bin/env python3
"""
EfficientAT embedding extraction.
(sequence pooling N/A: EfficientAT outputs pre-pooled embeddings)

6 models × 7 chunk pooling = 42 combos on BSD10k-v1.2

Usage:
    python scripts/embedding/run_efficientat.py
    python scripts/embedding/run_efficientat.py --skip-done
    python scripts/embedding/run_efficientat.py --dry-run
"""
import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

EFFICIENTAT_REPO_DIR = "visualize/4efficientat_code/EfficientAT"
EMBED_ROOT           = os.path.join(PROJECT_ROOT, "visualize", "embedding", "efficientat_embedding")

MODELS = [
    "dymn04_as",
    "dymn10_as",
    "dymn20_as",
    "mn10_as",
    "mn20_as",
    "mn40_as",
]

POOLING_OPTIONS = ["mean", "max", "std", "mean+max", "mean+std", "max+std", "mean+max+std"]

DATASETS = {"BSD10k-v1.2": "10k"}


def pool_name(p):
    return p.replace("+", "")


def model_short(m):
    return m.split("_as")[0]


def combo_name(model, chunk_pool):
    return f"efficientat_{model_short(model)}_chunk_{pool_name(chunk_pool)}"


def is_done(name):
    for short in DATASETS.values():
        out_dir = os.path.join(EMBED_ROOT, name, short)
        if not os.path.isdir(out_dir) or len(os.listdir(out_dir)) == 0:
            return False
    return True


def build_cmds(model, chunk_pool, overwrite=False):
    name = combo_name(model, chunk_pool)
    cmds = []
    for ds_name, ds_short in DATASETS.items():
        out_dir = os.path.join(EMBED_ROOT, name, ds_short)
        cmd = [
            sys.executable,
            "visualize/extract_efficientat_embeddings.py",
            "--efficientat-repo-dir", EFFICIENTAT_REPO_DIR,
            "--model-name", model,
            "--chunk-pooling", chunk_pool,
            "--output-dir", out_dir,
            "--datasets", ds_name,
            "--batch-size", "8",
        ]
        if overwrite:
            cmd.append("--overwrite")
        cmds.append((ds_short, cmd))
    return cmds


def main():
    parser = argparse.ArgumentParser(description="EfficientAT embedding extraction")
    parser.add_argument("--dry-run",   action="store_true", help="명령어만 출력, 실행 안 함")
    parser.add_argument("--skip-done", action="store_true", help="이미 완료된 조합 건너뜀")
    parser.add_argument("--overwrite", action="store_true", help="기존 .npy 파일 덮어쓰기")
    args = parser.parse_args()

    experiments = [
        (model, chunk_pool)
        for model in MODELS
        for chunk_pool in POOLING_OPTIONS
    ]
    total   = len(experiments)
    skipped = 0
    failed  = []

    print(f"[EfficientAT] 총 실험: {total}  ({len(MODELS)} models × {len(POOLING_OPTIONS)} chunk)")
    print(f"저장 경로: {EMBED_ROOT}/{{combo}}/10k/")
    print()

    for idx, (model, chunk_pool) in enumerate(experiments, 1):
        name = combo_name(model, chunk_pool)

        if args.skip_done and is_done(name):
            skipped += 1
            print(f"[{idx:2d}/{total}] SKIP  {name}")
            continue

        print(f"\n[{idx:2d}/{total}] {'DRY ' if args.dry_run else ''}START  {name}")
        print(f"  model: {model}")
        print(f"  chunk: {chunk_pool}")

        cmds = build_cmds(model, chunk_pool, args.overwrite)

        if args.dry_run:
            for ds_short, cmd in cmds:
                print(f"  [{ds_short}] {' '.join(cmd)}")
            continue

        t0 = time.time()
        experiment_failed = False
        for ds_short, cmd in cmds:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT)
            if result.returncode != 0:
                failed.append(f"{name}/{ds_short}")
                print(f"  [{ds_short}] FAILED (returncode={result.returncode})")
                experiment_failed = True
            else:
                print(f"  [{ds_short}] OK")
        print(f"  → {'FAILED' if experiment_failed else 'OK'} ({time.time() - t0:.0f}s)")

    print(f"\n{'='*60}")
    succeeded = total - len(failed) - skipped
    print(f"완료: {succeeded}/{total}  (skipped={skipped}, failed={len(failed)})")
    if failed:
        print("실패한 실험:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
