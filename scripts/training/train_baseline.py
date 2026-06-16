#!/usr/bin/env python3
import argparse
import sys
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run baseline training with a single full config file.")
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to a full baseline config YAML (no merge step).",
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

    archived_path = archive_runtime_config(config, args.config, "baseline")
    print(f"Archived runtime config to {archived_path}")

    from src.trainers.baseline_trainer import BaselineTrainer

    trainer = BaselineTrainer(config=config)
    trainer.run()

if __name__ == "__main__":
    main()
