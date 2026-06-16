#!/usr/bin/env python3
"""Generate epoch-wise plots from saved `history.json` files under a model_output directory.

Usage:
    python scripts/plot_fold_metrics.py --model_output ./output/model_output --mode both
"""
import argparse
import os
from src.utils.plot_history import plot_history


def find_histories(model_output: str, mode: str):
    mode_dir = os.path.join(model_output, mode)
    if not os.path.isdir(mode_dir):
        raise FileNotFoundError(mode_dir)
    histories = []
    for d in os.listdir(mode_dir):
        if d.startswith('fold_'):
            h = os.path.join(mode_dir, d, 'history.json')
            if os.path.isfile(h):
                histories.append(h)
    return histories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_output', default='./output/model_output')
    parser.add_argument('--mode', default='both')
    args = parser.parse_args()

    histories = find_histories(args.model_output, args.mode)
    if not histories:
        print('No history.json files found')
        return

    for h in histories:
        try:
            created = plot_history(h, out_dir=os.path.dirname(h))
            print(f'Created plots for {h}: {created}')
        except Exception as e:
            print(f'Failed to plot {h}: {e}')


if __name__ == '__main__':
    main()
