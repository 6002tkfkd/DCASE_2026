#!/usr/bin/env python3
"""Prepare processed datasets (wrapper for build_single_dataset/build_multi_dataset).

Usage:
    python scripts/prepare_data.py --single
    python scripts/prepare_data.py --multi

This avoids using `python -c` in docs and provides a single command for data preparation.
"""
import argparse
import sys
from pathlib import Path

import yaml

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', action='store_true', help='Build single processed dataset for active dataset')
    parser.add_argument('--multi', action='store_true', help='Build processed datasets for all registered datasets')
    parser.add_argument(
        '--config',
        default='configs/baseline.yaml',
        help='Path to a full config YAML used for dataset preparation.',
    )
    args = parser.parse_args()

    if not args.single and not args.multi:
        print('Specify --single or --multi')
        parser.print_help()
        sys.exit(1)

    from src.datasets.prep import build_single_dataset, build_multi_dataset

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f) or {}

    if args.single:
        build_single_dataset(config=config)
        print('build_single_dataset() completed')
    if args.multi:
        build_multi_dataset(config=config)
        print('build_multi_dataset() completed')
