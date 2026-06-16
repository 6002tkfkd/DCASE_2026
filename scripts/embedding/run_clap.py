#!/usr/bin/env python3
"""
CLAP embedding extraction.

--mode all  : 7 ckpt × (1 text + 7 audio) = 56 runs on BSD10k-v1.2
              --audio-only / --text-only 로 부분 실행 가능
--mode 35k  : top 2 audio combos (ensemble_search 결과) on BSD35k-CS
                1. clap_music_as_chunk_mean    (19/19)
                2. clap_630k_chunk_meanmax     (6/19)

Usage:
    python scripts/embedding/run_clap.py --mode all
    python scripts/embedding/run_clap.py --mode all --audio-only
    python scripts/embedding/run_clap.py --mode all --text-only
    python scripts/embedding/run_clap.py --mode 35k
    python scripts/embedding/run_clap.py --mode all --skip-done
    python scripts/embedding/run_clap.py --mode all --dry-run
"""
import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

CHECKPOINT_DIR   = os.path.join(PROJECT_ROOT, "visualize", "5clap_code", "checkpoints")
AUDIO_EMBED_ROOT = os.path.join(PROJECT_ROOT, "visualize", "embedding", "clap_embedding", "audio")
TEXT_EMBED_ROOT  = os.path.join(PROJECT_ROOT, "visualize", "embedding", "clap_embedding", "text")

# (short_name, filename, amodel, enable_fusion)
CHECKPOINTS = [
    ("630k_as",         "630k-audioset-best.pt",                      "HTSAT-tiny", False),
    ("630k_as_fusion",  "630k-audioset-fusion-best.pt",                "HTSAT-tiny", True),
    ("630k",            "630k-best.pt",                                 "HTSAT-tiny", False),
    ("630k_fusion",     "630k-fusion-best.pt",                          "HTSAT-tiny", True),
    ("music_as",        "music_audioset_epoch_15_esc_90.14.pt",         "HTSAT-base", False),
    ("music_speech",    "music_speech_epoch_15_esc_89.25.pt",           "HTSAT-base", False),
    ("music_speech_as", "music_speech_audioset_epoch_15_esc_89.98.pt",  "HTSAT-base", False),
]

POOLING_OPTIONS = ["mean", "max", "std", "mean+max", "mean+std", "max+std", "mean+max+std"]

# ensemble_search.csv top-19 frequency 분석 결과 (audio only)
TOP_COMBOS = [
    ("music_as", "music_audioset_epoch_15_esc_90.14.pt", "HTSAT-base", False, "mean"),
    ("630k",     "630k-best.pt",                          "HTSAT-tiny", False, "mean+max"),
]

DATASETS_ALL = {"BSD10k-v1.2": "10k"}
DATASETS_35K = {"BSD35k-CS":   "35k"}


def pool_name(p):
    return p.replace("+", "")


def audio_combo_name(ckpt_short, chunk_pool):
    return f"clap_{ckpt_short}_chunk_{pool_name(chunk_pool)}"


def text_combo_name(ckpt_short):
    return f"clap_{ckpt_short}"


def is_audio_done(name, datasets):
    for short in datasets.values():
        out_dir = os.path.join(AUDIO_EMBED_ROOT, name, short)
        if not os.path.isdir(out_dir) or len(os.listdir(out_dir)) == 0:
            return False
    return True


def is_text_done(name, datasets):
    for short in datasets.values():
        out_dir = os.path.join(TEXT_EMBED_ROOT, name, short)
        if not os.path.isdir(out_dir) or len(os.listdir(out_dir)) == 0:
            return False
    return True


def base_cmd(mode, ckpt_file, amodel, enable_fusion, output_dir, ds_name, overwrite):
    cmd = [
        sys.executable,
        "visualize/extract_clap_embeddings.py",
        "--mode", mode,
        "--checkpoint", os.path.join(CHECKPOINT_DIR, ckpt_file),
        "--amodel", amodel,
        "--output-dir", output_dir,
        "--datasets", ds_name,
        "--batch-size", "8",
    ]
    if enable_fusion:
        cmd.append("--enable-fusion")
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def build_text_cmds(ckpt_short, ckpt_file, amodel, enable_fusion, datasets, overwrite=False):
    name = text_combo_name(ckpt_short)
    cmds = []
    for ds_name, ds_short in datasets.items():
        out_dir = os.path.join(TEXT_EMBED_ROOT, name, ds_short)
        cmds.append((ds_short, base_cmd("text_per_sample", ckpt_file, amodel, enable_fusion,
                                        out_dir, ds_name, overwrite)))
    return name, cmds


def build_audio_cmds(ckpt_short, ckpt_file, amodel, enable_fusion, chunk_pool, datasets, overwrite=False):
    name = audio_combo_name(ckpt_short, chunk_pool)
    cmds = []
    for ds_name, ds_short in datasets.items():
        out_dir = os.path.join(AUDIO_EMBED_ROOT, name, ds_short)
        cmd = base_cmd("audio", ckpt_file, amodel, enable_fusion, out_dir, ds_name, overwrite)
        cmd += ["--chunk-pooling", chunk_pool]
        cmds.append((ds_short, cmd))
    return name, cmds


def run_cmds(cmds, dry_run):
    if dry_run:
        for ds_short, cmd in cmds:
            print(f"  [{ds_short}] {' '.join(cmd)}")
        return True
    ok = True
    for ds_short, cmd in cmds:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"  [{ds_short}] FAILED (returncode={result.returncode})")
            ok = False
        else:
            print(f"  [{ds_short}] OK")
    return ok


def run_all_mode(args):
    datasets = DATASETS_ALL
    do_text  = not args.audio_only
    do_audio = not args.text_only
    failed   = []

    missing = [f for _, f, _, _ in CHECKPOINTS
               if not os.path.isfile(os.path.join(CHECKPOINT_DIR, f))]
    if missing and not args.dry_run:
        print("경고: 아래 checkpoint 파일이 없습니다:")
        for f in missing:
            print(f"  {CHECKPOINT_DIR}/{f}")
        print()

    n_text  = len(CHECKPOINTS) if do_text else 0
    n_audio = len(CHECKPOINTS) * len(POOLING_OPTIONS) if do_audio else 0
    print(f"[CLAP] mode=all  text: {n_text}  audio: {n_audio}  합계: {n_text + n_audio}")
    print()

    for ckpt_short, ckpt_file, amodel, enable_fusion in CHECKPOINTS:
        if do_text:
            name, cmds = build_text_cmds(ckpt_short, ckpt_file, amodel, enable_fusion,
                                         datasets, args.overwrite)
            if args.skip_done and is_text_done(name, datasets):
                print(f"[TEXT ] SKIP  {name}")
            else:
                print(f"\n[TEXT ] {'DRY ' if args.dry_run else ''}START  {name}")
                print(f"  ckpt: {ckpt_file}  amodel: {amodel}")
                t0 = time.time()
                ok = run_cmds(cmds, args.dry_run)
                if not args.dry_run:
                    print(f"  → {'OK' if ok else 'FAILED'} ({time.time() - t0:.0f}s)")
                if not ok:
                    failed.append(f"[text] {name}")

        if do_audio:
            for chunk_pool in POOLING_OPTIONS:
                name, cmds = build_audio_cmds(ckpt_short, ckpt_file, amodel, enable_fusion,
                                              chunk_pool, datasets, args.overwrite)
                if args.skip_done and is_audio_done(name, datasets):
                    print(f"[AUDIO] SKIP  {name}")
                    continue
                print(f"\n[AUDIO] {'DRY ' if args.dry_run else ''}START  {name}")
                print(f"  ckpt: {ckpt_file}  chunk: {chunk_pool}")
                t0 = time.time()
                ok = run_cmds(cmds, args.dry_run)
                if not args.dry_run:
                    print(f"  → {'OK' if ok else 'FAILED'} ({time.time() - t0:.0f}s)")
                if not ok:
                    failed.append(f"[audio] {name}")

    print(f"\n{'='*60}")
    print(f"실패: {len(failed)}")
    if failed:
        for f in failed:
            print(f"  {f}")


def run_35k_mode(args):
    datasets = DATASETS_35K
    total   = len(TOP_COMBOS)
    skipped = 0
    failed  = []

    missing = [f for _, f, _, _, _ in TOP_COMBOS
               if not os.path.isfile(os.path.join(CHECKPOINT_DIR, f))]
    if missing and not args.dry_run:
        print("경고: 아래 checkpoint 파일이 없습니다:")
        for f in missing:
            print(f"  {CHECKPOINT_DIR}/{f}")
        print()

    print(f"[CLAP] mode=35k  top combos: {total}개")
    print(f"저장 경로: {AUDIO_EMBED_ROOT}/{{combo}}/35k/")
    print()

    for idx, (ckpt_short, ckpt_file, amodel, enable_fusion, chunk_pool) in enumerate(TOP_COMBOS, 1):
        name, cmds = build_audio_cmds(ckpt_short, ckpt_file, amodel, enable_fusion,
                                      chunk_pool, datasets, args.overwrite)
        if args.skip_done and is_audio_done(name, datasets):
            skipped += 1
            print(f"[{idx}/{total}] SKIP  {name}")
            continue

        print(f"\n[{idx}/{total}] {'DRY ' if args.dry_run else ''}START  {name}")
        print(f"  ckpt : {ckpt_file}  amodel: {amodel}")
        print(f"  chunk: {chunk_pool}")

        t0 = time.time()
        ok = run_cmds(cmds, args.dry_run)
        if not args.dry_run:
            print(f"  → {'OK' if ok else 'FAILED'} ({time.time() - t0:.0f}s)")
        if not ok:
            failed.append(name)

    print(f"\n{'='*60}")
    succeeded = total - len(failed) - skipped
    print(f"완료: {succeeded}/{total}  (skipped={skipped}, failed={len(failed)})")
    if failed:
        print("실패한 실험:")
        for f in failed:
            print(f"  {f}")


def main():
    parser = argparse.ArgumentParser(description="CLAP embedding extraction")
    parser.add_argument("--mode", choices=["all", "35k"], required=True,
                        help="all: 전체 grid search (10k) / 35k: best combos only (35k)")
    parser.add_argument("--dry-run",    action="store_true", help="명령어만 출력, 실행 안 함")
    parser.add_argument("--skip-done",  action="store_true", help="이미 완료된 조합 건너뜀")
    parser.add_argument("--overwrite",  action="store_true", help="기존 .npy 파일 덮어쓰기")
    parser.add_argument("--audio-only", action="store_true", help="(mode=all 전용) audio만 추출")
    parser.add_argument("--text-only",  action="store_true", help="(mode=all 전용) text만 추출")
    args = parser.parse_args()

    if args.mode == "35k" and (args.audio_only or args.text_only):
        parser.error("--audio-only / --text-only 는 --mode all 에서만 사용 가능합니다")

    if args.mode == "all":
        run_all_mode(args)
    else:
        run_35k_mode(args)


if __name__ == "__main__":
    main()
