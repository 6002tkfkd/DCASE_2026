#!/usr/bin/env python3
"""
PaSST embedding extraction.
(sequence pooling N/A: PaSST outputs pre-pooled embeddings)

--mode all  : 2 ckpt × 7 chunk = 14 combos on BSD10k-v1.2
--mode 35k  : top 1 combo (ensemble_search 결과) on BSD35k-CS
                1. passt_kd_ap486_chunk_max  (3/19)

Usage:
    python scripts/embedding/run_passt.py --mode all
    python scripts/embedding/run_passt.py --mode 35k
    python scripts/embedding/run_passt.py --mode all --skip-done
    python scripts/embedding/run_passt.py --mode all --dry-run
"""
import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

EMBED_ROOT = os.path.join(PROJECT_ROOT, "visualize", "embedding", "passt_embedding")

CHECKPOINTS = {
    "swa_ap476": "passt_s_swa_p16_128_ap476",
    "kd_ap486":  "passt_s_kd_p16_128_ap486",
}

POOLING_OPTIONS = ["mean", "max", "std", "mean+max", "mean+std", "max+std", "mean+max+std"]

# ensemble_search.csv top-19 frequency 분석 결과
TOP_COMBOS = [
    ("kd_ap486", "passt_s_kd_p16_128_ap486", "max"),
]

DATASETS_ALL = {"BSD10k-v1.2": "10k"}
DATASETS_35K = {"BSD35k-CS":   "35k"}


def pool_name(p):
    return p.replace("+", "")


def combo_name(arch_short, chunk_pool):
    return f"passt_{arch_short}_chunk_{pool_name(chunk_pool)}"


def is_done(name, datasets):
    for short in datasets.values():
        out_dir = os.path.join(EMBED_ROOT, name, short)
        if not os.path.isdir(out_dir) or len(os.listdir(out_dir)) == 0:
            return False
    return True


def build_cmds(arch_short, arch_full, chunk_pool, datasets, overwrite=False):
    name = combo_name(arch_short, chunk_pool)
    cmds = []
    for ds_name, ds_short in datasets.items():
        out_dir = os.path.join(EMBED_ROOT, name, ds_short)
        cmd = [
            sys.executable,
            "visualize/extract_passt_embeddings.py",
            "--arch", arch_full,
            "--chunk-pooling", chunk_pool,
            "--output-dir", out_dir,
            "--datasets", ds_name,
            "--batch-size", "4",
        ]
        if overwrite:
            cmd.append("--overwrite")
        cmds.append((ds_short, cmd))
    return cmds


def run_experiments(experiments, datasets, args):
    total   = len(experiments)
    skipped = 0
    failed  = []

    for idx, (arch_short, arch_full, chunk_pool) in enumerate(experiments, 1):
        name = combo_name(arch_short, chunk_pool)

        if args.skip_done and is_done(name, datasets):
            skipped += 1
            print(f"[{idx:2d}/{total}] SKIP  {name}")
            continue

        print(f"\n[{idx:2d}/{total}] {'DRY ' if args.dry_run else ''}START  {name}")
        print(f"  arch : {arch_full}")
        print(f"  chunk: {chunk_pool}")

        cmds = build_cmds(arch_short, arch_full, chunk_pool, datasets, args.overwrite)

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


def main():
    parser = argparse.ArgumentParser(description="PaSST embedding extraction")
    parser.add_argument("--mode", choices=["all", "35k"], required=True,
                        help="all: 전체 grid search (10k) / 35k: best combos only (35k)")
    parser.add_argument("--dry-run",   action="store_true", help="명령어만 출력, 실행 안 함")
    parser.add_argument("--skip-done", action="store_true", help="이미 완료된 조합 건너뜀")
    parser.add_argument("--overwrite", action="store_true", help="기존 .npy 파일 덮어쓰기")
    args = parser.parse_args()

    if args.mode == "all":
        experiments = [
            (arch_short, arch_full, chunk_pool)
            for arch_short, arch_full in CHECKPOINTS.items()
            for chunk_pool in POOLING_OPTIONS
        ]
        datasets = DATASETS_ALL
        print(f"[PaSST] mode=all  총 실험: {len(experiments)}  "
              f"({len(CHECKPOINTS)} ckpt × {len(POOLING_OPTIONS)} chunk)")
    else:
        experiments = list(TOP_COMBOS)
        datasets = DATASETS_35K
        print(f"[PaSST] mode=35k  top combos: {len(experiments)}개")

    print(f"저장 경로: {EMBED_ROOT}/{{combo}}/{{{'10k' if args.mode == 'all' else '35k'}}}/")
    print()
    run_experiments(experiments, datasets, args)


if __name__ == "__main__":
    main()
