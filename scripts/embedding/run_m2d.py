#!/usr/bin/env python3
"""
M2D embedding extraction.

--mode all  : 15 combos on BSD10k-v1.2
                - 7 non-flat (3840-dim) : pooling × 7
                - 7 flat     ( 768-dim) : pooling × 7
                - 1 CLAP projected audio embedding
--mode 35k  : top 1 combo (ensemble_search 결과) on BSD35k-CS
                1. m2d_clap_vit_base_meanstd  pooling=mean+std  (12/19)

Usage:
    python scripts/embedding/run_m2d.py --mode all
    python scripts/embedding/run_m2d.py --mode 35k
    python scripts/embedding/run_m2d.py --mode all --skip-done
    python scripts/embedding/run_m2d.py --mode all --dry-run
"""
import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

CHECKPOINT = "visualize/7m2d_code/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth"
EMBED_ROOT  = os.path.join(PROJECT_ROOT, "visualize", "embedding", "m2d_embedding")

POOLING_OPTIONS = ["mean", "max", "std", "mean+max", "mean+std", "max+std", "mean+max+std"]

# ensemble_search.csv top-19 frequency 분석 결과
TOP_COMBOS = [
    ("m2d_clap_vit_base_meanstd", "mean+std"),
]

DATASETS_ALL = {"BSD10k-v1.2": "10k"}
DATASETS_35K = {"BSD35k-CS":   "35k"}


def pool_name(p):
    return p.replace("+", "")


def is_done(combo_name, datasets):
    for short in datasets.values():
        out_dir = os.path.join(EMBED_ROOT, combo_name, short)
        if not os.path.isdir(out_dir) or len(os.listdir(out_dir)) == 0:
            return False
    return True


def build_all_experiments():
    exps = []
    for pool in POOLING_OPTIONS:
        exps.append({"name": f"m2d_clap_vit_base_{pool_name(pool)}",
                     "pooling": pool, "flat": False, "clap": False})
    for pool in POOLING_OPTIONS:
        exps.append({"name": f"m2d_clap_vit_base_flat_{pool_name(pool)}",
                     "pooling": pool, "flat": True,  "clap": False})
    exps.append({"name": "m2d_clap_vit_base_clap",
                 "pooling": None, "flat": False, "clap": True})
    return exps


def build_cmds(exp, datasets, overwrite=False):
    cmds = []
    for ds_name, ds_short in datasets.items():
        out_dir = os.path.join(EMBED_ROOT, exp["name"], ds_short)
        cmd = [
            sys.executable,
            "visualize/extract_m2d_embeddings.py",
            "--checkpoint", CHECKPOINT,
            "--output-dir", out_dir,
            "--datasets", ds_name,
        ]
        if exp["clap"]:
            cmd.append("--clap-audio")
        else:
            cmd += ["--pooling", exp["pooling"]]
            if exp["flat"]:
                cmd.append("--flat-features")
        if overwrite:
            cmd.append("--overwrite")
        cmds.append((ds_short, cmd))
    return cmds


def run_all_mode(args):
    datasets    = DATASETS_ALL
    experiments = build_all_experiments()
    total       = len(experiments)
    skipped     = 0
    failed      = []

    print(f"[M2D] mode=all  총 실험: {total}  (non-flat×7 + flat×7 + clap×1)")
    print(f"저장 경로: {EMBED_ROOT}/{{combo}}/10k/")
    print()

    for idx, exp in enumerate(experiments, 1):
        name = exp["name"]

        if args.skip_done and is_done(name, datasets):
            skipped += 1
            print(f"[{idx:2d}/{total}] SKIP  {name}")
            continue

        mode_tag = "clap" if exp["clap"] else ("flat" if exp["flat"] else "non-flat")
        print(f"\n[{idx:2d}/{total}] {'DRY ' if args.dry_run else ''}START  {name}  [{mode_tag}]")

        cmds = build_cmds(exp, datasets, args.overwrite)

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


def run_35k_mode(args):
    datasets = DATASETS_35K
    total    = len(TOP_COMBOS)
    skipped  = 0
    failed   = []

    print(f"[M2D] mode=35k  top combos: {total}개")
    print(f"저장 경로: {EMBED_ROOT}/{{combo}}/35k/")
    print()

    for idx, (name, pooling) in enumerate(TOP_COMBOS, 1):
        if args.skip_done and is_done(name, datasets):
            skipped += 1
            print(f"[{idx}/{total}] SKIP  {name}")
            continue

        print(f"\n[{idx}/{total}] {'DRY ' if args.dry_run else ''}START  {name}")
        print(f"  pooling: {pooling}")

        exp = {"name": name, "pooling": pooling, "flat": False, "clap": False}
        cmds = build_cmds(exp, datasets, args.overwrite)

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
    parser = argparse.ArgumentParser(description="M2D embedding extraction")
    parser.add_argument("--mode", choices=["all", "35k"], required=True,
                        help="all: 전체 grid search (10k) / 35k: best combos only (35k)")
    parser.add_argument("--dry-run",   action="store_true", help="명령어만 출력, 실행 안 함")
    parser.add_argument("--skip-done", action="store_true", help="이미 완료된 조합 건너뜀")
    parser.add_argument("--overwrite", action="store_true", help="기존 .npy 파일 덮어쓰기")
    args = parser.parse_args()

    if args.mode == "all":
        run_all_mode(args)
    else:
        run_35k_mode(args)


if __name__ == "__main__":
    main()
