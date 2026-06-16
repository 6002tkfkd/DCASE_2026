#!/usr/bin/env python3
"""
M2D embedding extraction script.

Usage (컨테이너 /workspace/dcase2026_task1_baseline):
    python visualize/extract_m2d_embeddings.py \
        --checkpoint visualize/7m2d_code/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth \
        --pooling mean \
        --output-dir visualize/embedding/m2d_embedding/m2d_clap_vit_base_mean/10k

    # flat features (768-dim)
    python visualize/extract_m2d_embeddings.py \
        --checkpoint ... --pooling mean --flat-features \
        --output-dir visualize/embedding/m2d_embedding/m2d_clap_vit_base_flat_mean/10k

    # CLAP projected audio embedding
    python visualize/extract_m2d_embeddings.py \
        --checkpoint ... --clap-audio \
        --output-dir visualize/embedding/m2d_embedding/m2d_clap_vit_base_clap/10k
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.io import wavfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "7m2d_code", "examples")))

from utils import get_subconfig
from portable_m2d import PortableM2D, Config

TARGET_SR = 16000


def load_audio(path):
    try:
        waveform, sr = torchaudio.load(path)
    except Exception:
        sr, audio = wavfile.read(path)
        audio = np.asarray(audio)
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        else:
            audio = audio.astype(np.float32)
        waveform = torch.from_numpy(audio.T if audio.ndim > 1 else audio).unsqueeze(0)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    return waveform.squeeze(0)  # [T]


def pool_frames(x, pooling):
    """x: [T, D] → pooled vector"""
    parts = []
    for method in pooling.split("+"):
        method = method.strip()
        if method == "mean":
            parts.append(x.mean(dim=0))
        elif method == "max":
            parts.append(x.max(dim=0).values)
        elif method == "std":
            parts.append(x.std(dim=0, unbiased=False))
        else:
            raise ValueError(f"Unknown pooling: {method}")
    return torch.cat(parts, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pooling", default="mean",
                        help="Pooling over time frames: mean/max/std/mean+max/mean+std/max+std/mean+max+std")
    parser.add_argument("--flat-features", action="store_true",
                        help="Use flat ViT token embeddings (768-dim) instead of stacked (3840-dim)")
    parser.add_argument("--clap-audio", action="store_true",
                        help="Use CLAP projected audio embedding (audio_proj)")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load M2D model
    cfg = Config()
    cfg.flat_features = args.flat_features
    model = PortableM2D(args.checkpoint, cfg=cfg).to(device)
    model.eval()
    print(f"M2D loaded  flat={args.flat_features}  feature_d={model.cfg.feature_d}")
    if args.clap_audio:
        print("Mode: CLAP projected audio embedding")
    else:
        print(f"Mode: seq pooling={args.pooling}")

    # Dataset
    active_dataset = get_subconfig("active_dataset")
    ds_names = [n.strip() for n in args.datasets.split(",")] if args.datasets else \
               (get_subconfig("combined_datasets") if active_dataset == "combined" else [active_dataset])

    for ds_name in ds_names:
        ds_cfg = get_subconfig("datasets")[ds_name]
        meta = pd.read_csv(ds_cfg["metadata_csv"])
        meta["sound_id"] = meta["sound_id"].astype(str).str.strip()
        audio_dir = os.path.join("data", ds_name, "audio")

        os.makedirs(args.output_dir, exist_ok=True)
        written = skipped = missing = failed = 0

        if args.limit:
            meta = meta.head(args.limit)

        for _, row in meta.iterrows():
            sid = str(row["sound_id"])
            audio_path = os.path.join(audio_dir, f"{sid}.wav")
            out_path = os.path.join(args.output_dir, f"{sid}.npy")

            if os.path.exists(out_path) and not args.overwrite:
                skipped += 1
                continue
            if not os.path.isfile(audio_path):
                missing += 1
                continue

            try:
                waveform = load_audio(audio_path).to(device)  # [T]

                with torch.no_grad():
                    batch = waveform.unsqueeze(0)  # [1, T]

                    if args.clap_audio:
                        emb = model.encode_clap_audio(batch)   # [1, proj_dim]
                        emb = emb.squeeze(0)
                    else:
                        frames = model.encode(batch)            # [1, time_frames, D]
                        frames = frames.squeeze(0)              # [time_frames, D]
                        emb = pool_frames(frames, args.pooling) # [D] or [D*k]

                np.save(out_path, emb.cpu().float().numpy())
                written += 1

                if written % 500 == 0:
                    print(f"  {written} written...")

            except Exception as e:
                failed += 1
                print(f"  FAILED {sid}: {e}")

        total = written + skipped + missing + failed
        print(f"[{ds_name}] total={total}  written={written}  skipped={skipped}  missing={missing}  failed={failed}")
        if written > 0:
            sample = np.load(os.path.join(args.output_dir, f"{sid}.npy"))
            print(f"  embedding dim: {sample.shape[0]}")


if __name__ == "__main__":
    main()
