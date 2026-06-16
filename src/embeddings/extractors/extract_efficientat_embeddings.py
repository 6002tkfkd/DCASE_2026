import argparse
import contextlib
import io
import os
import sys
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.io import wavfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_subconfig


TARGET_SAMPLE_RATE = 32000


def load_efficientat_model(repo_dir, model_name, device, args):
    repo_dir = os.path.abspath(repo_dir)
    sys.path.insert(0, repo_dir)

    for module_name in list(sys.modules):
        if module_name == "models" or module_name.startswith("models."):
            del sys.modules[module_name]
        if module_name == "helpers" or module_name.startswith("helpers."):
            del sys.modules[module_name]

    old_cwd = os.getcwd()
    os.chdir(repo_dir)
    try:
        from helpers.utils import NAME_TO_WIDTH
        from models.preprocess import AugmentMelSTFT

        if model_name.startswith("dymn"):
            import models.dymn.model as dymn_model

            dymn_model.model_dir = os.path.join(repo_dir, "resources")
            model = dymn_model.get_model(
                width_mult=NAME_TO_WIDTH(model_name),
                pretrained_name=model_name,
                strides=args.strides,
            )
        else:
            import models.mn.model as mn_model

            mn_model.model_dir = os.path.join(repo_dir, "resources")
            model = mn_model.get_model(
                width_mult=NAME_TO_WIDTH(model_name),
                pretrained_name=model_name,
                strides=args.strides,
                head_type=args.head_type,
                input_dim_f=args.n_mels,
                input_dim_t=int(args.chunk_seconds * TARGET_SAMPLE_RATE / args.hop_size) + 1,
            )
    finally:
        os.chdir(old_cwd)

    model.eval()
    model.to(device)

    mel = AugmentMelSTFT(
        n_mels=args.n_mels,
        sr=TARGET_SAMPLE_RATE,
        win_length=args.window_size,
        hopsize=args.hop_size,
        freqm=0,
        timem=0,
    )
    mel.eval()
    mel.to(device)

    return model, mel


def pool_tensor(x, dim, pooling):
    pooled = []
    for method in pooling.split("+"):
        method = method.strip()
        if method == "mean":
            pooled.append(x.mean(dim=dim))
        elif method == "max":
            pooled.append(x.max(dim=dim).values)
        elif method == "std":
            pooled.append(x.std(dim=dim, unbiased=False))
        else:
            raise ValueError(f"Unknown pooling method: {method}")
    if len(pooled) == 1:
        return pooled[0]
    return torch.cat(pooled, dim=-1)


def load_audio(audio_path):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception:
        sample_rate, audio = wavfile.read(audio_path)
        audio = np.asarray(audio)
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        else:
            audio = audio.astype(np.float32)
        if audio.ndim == 1:
            waveform = torch.from_numpy(audio).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio.T)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE,
        )

    return waveform.squeeze(0)


def make_chunks(waveform, chunk_samples, hop_samples):
    chunks = []
    audio_len = waveform.numel()

    if audio_len == 0:
        waveform = torch.zeros(1, dtype=torch.float32)
        audio_len = 1

    starts = list(range(0, max(1, audio_len - chunk_samples + 1), hop_samples))
    if not starts:
        starts = [0]
    last_start = max(0, audio_len - chunk_samples)
    if starts[-1] != last_start:
        starts.append(last_start)

    for start in starts:
        chunk = waveform[start:start + chunk_samples]
        if chunk.numel() < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.numel()))
        chunks.append(chunk)

    return torch.stack(chunks)


def filter_metadata(df):
    df = df.copy()
    df["sound_id"] = df["sound_id"].astype(str).str.strip()
    s = df["class_idx"].astype(str)
    return df[~((s.str.len() == 3) & (s.str.endswith("99") | s.str.endswith("00")))].copy()


def resolve_dataset_names(args):
    if args.datasets:
        return [name.strip() for name in args.datasets.split(",") if name.strip()]

    active_dataset = get_subconfig("active_dataset")
    if active_dataset == "combined":
        return get_subconfig("combined_datasets")
    return [active_dataset]


def extract_dataset(model, mel, dataset_name, args, device):
    dataset_config = get_subconfig("datasets")[dataset_name]
    metadata = filter_metadata(pd.read_csv(dataset_config["metadata_csv"]))
    audio_dir = os.path.join("data", dataset_name, "audio")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join("data", dataset_name, "features", args.output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    if args.limit is not None:
        metadata = metadata.head(args.limit)

    chunk_samples = int(args.chunk_seconds * TARGET_SAMPLE_RATE)
    hop_samples = int(args.hop_seconds * TARGET_SAMPLE_RATE)
    use_amp = device.type == "cuda" and not args.no_amp

    written = 0
    skipped = 0
    missing = 0
    failed = 0

    for _, row in metadata.iterrows():
        sound_id = str(row["sound_id"])
        audio_path = os.path.join(audio_dir, f"{sound_id}.wav")
        output_path = os.path.join(output_dir, f"{sound_id}.npy")

        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            continue

        if not os.path.isfile(audio_path):
            missing += 1
            continue

        try:
            waveform = load_audio(audio_path)
            chunks = make_chunks(waveform, chunk_samples, hop_samples).to(device)

            chunk_embeddings = []
            for start in range(0, chunks.shape[0], args.batch_size):
                batch = chunks[start:start + args.batch_size]
                with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
                        spec = mel(batch)
                        _, features = model(spec.unsqueeze(1))

                if features.ndim != 2:
                    raise RuntimeError(f"Unexpected feature shape: {tuple(features.shape)}")
                chunk_embeddings.append(features.float())

            chunk_embeddings = torch.cat(chunk_embeddings, dim=0)
            embedding = pool_tensor(chunk_embeddings, dim=0, pooling=args.chunk_pooling)
            np.save(output_path, embedding.detach().cpu().numpy().astype(np.float32))
            written += 1
        except Exception as exc:
            failed += 1
            print(f"[{dataset_name}] failed {sound_id}: {exc}")

        done = written + skipped + missing + failed
        if done % args.log_every == 0:
            print(
                f"[{dataset_name}] {done}/{len(metadata)} "
                f"written={written} skipped={skipped} missing={missing} failed={failed}"
            )

    print(
        f"[{dataset_name}] done: written={written}, skipped={skipped}, "
        f"missing={missing}, failed={failed}, output={output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description="Extract EfficientAT audio embeddings as .npy files.")
    parser.add_argument("--efficientat-repo-dir", default="visualize/4efficientat_code/EfficientAT")
    parser.add_argument("--model-name", default="dymn10_as")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-subdir", default="efficientat_audio_embeddings")
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--hop-seconds", type=float, default=10.0)
    parser.add_argument(
        "--chunk-pooling",
        default="mean",
        choices=["mean", "max", "std", "mean+max", "mean+std", "max+std", "mean+max+std"],
        help="Pooling over chunk embeddings for one audio file.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strides", nargs=4, default=[2, 2, 2, 2], type=int)
    parser.add_argument("--head-type", default="mlp")
    parser.add_argument("--window-size", type=int, default=800)
    parser.add_argument("--hop-size", type=int, default=320)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, mel = load_efficientat_model(args.efficientat_repo_dir, args.model_name, device, args)
    for dataset_name in resolve_dataset_names(args):
        extract_dataset(model, mel, dataset_name, args, device)


if __name__ == "__main__":
    main()
