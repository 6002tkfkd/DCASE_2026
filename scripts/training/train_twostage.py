#!/usr/bin/env python3
import argparse
import sys
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run two-stage training with a single full config file.")
    parser.add_argument(
        "--config",
        default="configs/two_stage.yaml",
        help="Path to a full two-stage config YAML (no merge step).",
    )
    parser.add_argument(
        "--paths",
        default=None,
        help="Path to paths.yaml for external path overrides (see paths_example.yaml).",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    # Ensure repo root is importable
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    if args.paths:
        from src.utils.paths_override import apply_paths_override
        apply_paths_override(config, args.paths)

    from src.utils.config_archive import archive_runtime_config

    archived_path = archive_runtime_config(config, args.config, "two_stage")
    print(f"Archived runtime config to {archived_path}")

    # Run pretraining then finetune sequentially
    from src.trainers.pretrain_trainer import PretrainTrainer
    from src.trainers.finetune_trainer import FinetuneTrainer

    pretrainer = PretrainTrainer(config=config)
    pretrainer.run()

    finetuner = FinetuneTrainer(config=config)
    finetuner.run()

if __name__ == "__main__":
    main()
