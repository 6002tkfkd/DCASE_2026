#!/usr/bin/env python3
"""
기존 configs를 9개 embedding 모델별로 복제 생성.

변경 사항:
  - datasets.BSD10k-v1.2.audio_emb_folder → ./embedding/{model}/10k
  - datasets.BSD35k-CS.audio_emb_folder   → ./embedding/{model}/35k
  - output.run_dir     → {model}/{original_run_dir}
  - output.pretrain_dir → {model}/{original_pretrain_dir}
  - strategy           → {model}/{original_strategy}

출력 위치:
  configs/{exp_dir}/{model}/{config_name}.yaml

Usage:
    python scripts/generate_embedding_configs.py
    python scripts/generate_embedding_configs.py --dry-run
"""
import argparse
import os
import shutil

import yaml

PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS_ROOT = os.path.join(PIPELINE_ROOT, "configs")

EMBEDDING_MODELS = [
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

EXP_DIRS = [
    "two_stage_exp1_hloss",
    "two_stage_exp2_proxy",
]


def patch_config(cfg: dict, model: str, exp_dir: str, config_name: str) -> dict:
    import copy
    c = copy.deepcopy(cfg)

    # audio_emb_folder 교체
    datasets = c.get("datasets", {})
    for ds_name, ds_cfg in datasets.items():
        if not isinstance(ds_cfg, dict):
            continue
        if "10k" in ds_name.lower():
            ds_cfg["audio_emb_folder"] = f"./embedding/{model}/10k"
        elif "35k" in ds_name.lower():
            ds_cfg["audio_emb_folder"] = f"./embedding/{model}/35k"

    # output 경로에 model prefix 추가
    output = c.get("output", {})
    if isinstance(output, dict):
        for key in ("run_dir", "pretrain_dir"):
            if key in output:
                output[key] = f"{model}/{output[key]}"

    # strategy에 model prefix 추가
    if "strategy" in c:
        c["strategy"] = f"{model}/{c['strategy']}"

    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="생성할 파일 경로만 출력")
    args = parser.parse_args()

    generated = 0
    skipped = 0

    for exp_dir in EXP_DIRS:
        src_dir = os.path.join(CONFIGS_ROOT, exp_dir)
        if not os.path.isdir(src_dir):
            print(f"[SKIP] {exp_dir} 디렉토리 없음")
            continue

        yaml_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".yaml"))

        for model in EMBEDDING_MODELS:
            dst_dir = os.path.join(CONFIGS_ROOT, exp_dir, model)

            for yaml_file in yaml_files:
                src_path = os.path.join(src_dir, yaml_file)
                dst_path = os.path.join(dst_dir, yaml_file)

                if args.dry_run:
                    print(f"  {dst_path}")
                    generated += 1
                    continue

                os.makedirs(dst_dir, exist_ok=True)

                with open(src_path, "r") as f:
                    cfg = yaml.safe_load(f)

                patched = patch_config(cfg, model, exp_dir, yaml_file)

                with open(dst_path, "w") as f:
                    yaml.dump(patched, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

                generated += 1

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}생성 완료: {generated}개")


if __name__ == "__main__":
    main()
