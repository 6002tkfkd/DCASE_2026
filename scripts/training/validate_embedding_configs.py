#!/usr/bin/env python3
"""
270개 embedding config의 경로 유효성을 빠르게 체크.
실제 학습 없이 몇 초 안에 전체 스캔.

체크 항목:
  - audio_emb_folder 존재 여부
  - .npy 파일 최소 1개 이상 있는지

Usage:
    python scripts/validate_embedding_configs.py
    python scripts/validate_embedding_configs.py --verbose
"""
import argparse
import os
import yaml

PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EXP_DIRS = ["two_stage_exp1_hloss", "two_stage_exp2_proxy"]
MODELS = [
    "baseline",
    "atst_weak_seq_maxstd_chunk_meanstd",
    "beats_iter3plus_as2m_ft1_seq_meanstd_chunk_mean",
    "beats_iter3plus_as2m_seq_meanmaxstd_chunk_max",
    "clap_630k_chunk_meanmax",
    "clap_music_as_chunk_mean",
    "fpasst_weak_seq_meanmax_chunk_maxstd",
    "m2d_clap_vit_base_meanstd",
    "passt_kd_ap486_chunk_max",
]


def check_emb_folder(path_raw):
    """./embedding/... 경로를 절대 경로로 변환 후 체크."""
    if path_raw.startswith("./"):
        path = os.path.join(PIPELINE_ROOT, path_raw[2:])
    else:
        path = path_raw

    if not os.path.isdir(path):
        return False, f"디렉토리 없음: {path}"

    npy_files = [f for f in os.listdir(path) if f.endswith(".npy")]
    if not npy_files:
        return False, f".npy 파일 없음: {path}"

    return True, f"OK ({len(npy_files)}개)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="OK인 것도 출력")
    args = parser.parse_args()

    total = ok = fail = 0
    fail_list = []

    for exp_dir in EXP_DIRS:
        for model in MODELS:
            model_dir = os.path.join(PIPELINE_ROOT, "configs", exp_dir, model)
            if not os.path.isdir(model_dir):
                print(f"[WARN] 디렉토리 없음: configs/{exp_dir}/{model}")
                continue

            for yaml_file in sorted(os.listdir(model_dir)):
                if not yaml_file.endswith(".yaml"):
                    continue

                total += 1
                cfg_path = os.path.join(model_dir, yaml_file)
                label = f"[{exp_dir}/{model}/{yaml_file}]"

                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)

                errors = []
                for ds_name, ds_cfg in cfg.get("datasets", {}).items():
                    if not isinstance(ds_cfg, dict):
                        continue
                    folder = ds_cfg.get("audio_emb_folder")
                    if not folder:
                        continue
                    passed, msg = check_emb_folder(folder)
                    if not passed:
                        errors.append(f"  {ds_name}: {msg}")
                    elif args.verbose:
                        print(f"  OK  {ds_name}: {msg}")

                if errors:
                    fail += 1
                    fail_list.append(label)
                    print(f"FAIL {label}")
                    for e in errors:
                        print(e)
                else:
                    ok += 1
                    if args.verbose:
                        print(f"OK   {label}")

    print()
    print("=" * 60)
    print(f"전체: {total}  OK: {ok}  FAIL: {fail}")
    if fail_list:
        print("\n실패 목록:")
        for f in fail_list:
            print(f"  {f}")
    else:
        print("모든 embedding 경로 정상!")
    print("=" * 60)


if __name__ == "__main__":
    main()
