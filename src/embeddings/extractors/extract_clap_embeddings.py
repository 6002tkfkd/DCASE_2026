import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.io import wavfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "5clap_code", "CLAP", "src")))
from utils import get_subconfig


TARGET_SAMPLE_RATE = 48000  # CLAP requires 48kHz


def load_clap_model(ckpt_path, amodel, enable_fusion, device):
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=amodel, device=device)
    model.load_ckpt(ckpt_path)
    model.eval()
    return model


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

    chunks = []
    for start in starts:
        chunk = waveform[start:start + chunk_samples]
        if chunk.numel() < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.numel()))
        chunks.append(chunk)

    return torch.stack(chunks)


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


def extract_audio(model, dataset_name, args, device):
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
            chunks = make_chunks(waveform, chunk_samples, hop_samples)

            chunk_embeddings = []
            for start in range(0, chunks.shape[0], args.batch_size):
                batch = chunks[start:start + args.batch_size].to(device)
                with torch.no_grad():
                    emb = model.get_audio_embedding_from_data(batch, use_tensor=True)
                if emb.ndim != 2:
                    raise RuntimeError(f"Unexpected embedding shape: {tuple(emb.shape)}")
                chunk_embeddings.append(emb.float())

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


def extract_text(model, dataset_name, args):
    dataset_config = get_subconfig("datasets")[dataset_name]
    metadata = filter_metadata(pd.read_csv(dataset_config["metadata_csv"]))

    # class_key → description 매핑
    desc_df = pd.read_csv(dataset_config["class_names"])
    desc_df["class_key"] = desc_df["class_key"].astype(str).str.strip()
    class_to_desc = dict(zip(desc_df["class_key"], desc_df["description"].fillna("")))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join("data", dataset_name, "features", args.output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    if args.limit is not None:
        metadata = metadata.head(args.limit)

    # 클래스별로 text embedding 미리 계산 (중복 방지)
    unique_classes = metadata["class"].unique().tolist()
    texts = [class_to_desc.get(str(c).strip(), str(c).strip()) for c in unique_classes]
    print(f"[{dataset_name}] computing text embeddings for {len(texts)} classes...")

    with torch.no_grad():
        class_embeddings = model.get_text_embedding(texts, use_tensor=True)

    class_emb_map = {
        str(c).strip(): class_embeddings[i].float().cpu().numpy().astype(np.float32)
        for i, c in enumerate(unique_classes)
    }

    written = 0
    skipped = 0

    for _, row in metadata.iterrows():
        sound_id = str(row["sound_id"])
        output_path = os.path.join(output_dir, f"{sound_id}.npy")

        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            continue

        class_key = str(row["class"]).strip()
        if class_key not in class_emb_map:
            print(f"[{dataset_name}] missing class description for {class_key}, skipping {sound_id}")
            continue

        np.save(output_path, class_emb_map[class_key])
        written += 1

    print(
        f"[{dataset_name}] text done: written={written}, skipped={skipped}, output={output_dir}"
    )


def build_sample_text(row, text_field):
    """샘플별 텍스트 구성. text_field: tags / description / tags_and_description"""
    tags = str(row.get("tags", "") or "").strip().replace(",", " ")
    desc = str(row.get("description", "") or "").strip()
    if text_field == "tags":
        return tags
    if text_field == "description":
        return desc
    # tags_and_description
    parts = [p for p in [tags, desc] if p]
    return " ".join(parts)


def extract_text_per_sample(model, dataset_name, args):
    """샘플별 tags/description을 CLAP으로 인코딩해 {sound_id}.npy로 저장."""
    dataset_config = get_subconfig("datasets")[dataset_name]
    metadata = filter_metadata(pd.read_csv(dataset_config["metadata_csv"]))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join("data", dataset_name, "features", args.output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    if args.limit is not None:
        metadata = metadata.head(args.limit)

    # 이미 완료된 파일 건너뜀
    rows_to_process = []
    skipped = 0
    for _, row in metadata.iterrows():
        output_path = os.path.join(output_dir, f"{row['sound_id']}.npy")
        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
        else:
            rows_to_process.append(row)

    print(f"[{dataset_name}] text_per_sample: {len(rows_to_process)} to process, {skipped} skipped")
    print(f"[{dataset_name}] text_field={args.text_field}, batch_size={args.batch_size}")

    written = 0
    failed = 0

    for batch_start in range(0, len(rows_to_process), args.batch_size):
        batch_rows = rows_to_process[batch_start:batch_start + args.batch_size]
        texts = [build_sample_text(r, args.text_field) for r in batch_rows]

        # 빈 텍스트 fallback
        texts = [t if t.strip() else "sound" for t in texts]

        try:
            with torch.no_grad():
                embeddings = model.get_text_embedding(texts, use_tensor=True)
            embeddings = embeddings.float().cpu().numpy().astype(np.float32)

            for i, row in enumerate(batch_rows):
                output_path = os.path.join(output_dir, f"{row['sound_id']}.npy")
                np.save(output_path, embeddings[i])
                written += 1
        except Exception as exc:
            failed += len(batch_rows)
            print(f"[{dataset_name}] batch {batch_start} failed: {exc}")

        done = batch_start + len(batch_rows)
        if done % args.log_every == 0 or done == len(rows_to_process):
            print(f"[{dataset_name}] {done}/{len(rows_to_process)} written={written} failed={failed}")

    print(
        f"[{dataset_name}] text_per_sample done: written={written}, skipped={skipped}, "
        f"failed={failed}, output={output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description="Extract CLAP audio/text embeddings as .npy files.")
    parser.add_argument(
        "--mode",
        default="audio",
        choices=["audio", "text", "text_per_sample"],
        help=(
            "audio: 오디오 임베딩 추출 / "
            "text: 클래스 설명 임베딩 추출 (클래스당 1개, 같은 클래스는 동일 벡터) / "
            "text_per_sample: 샘플별 tags+description 임베딩 추출 (샘플마다 고유 벡터)"
        ),
    )
    parser.add_argument("--checkpoint", required=True, help="CLAP checkpoint 파일 경로 (.pt)")
    parser.add_argument(
        "--amodel",
        default="HTSAT-tiny",
        choices=["HTSAT-tiny", "HTSAT-base"],
    )
    parser.add_argument("--enable-fusion", action="store_true")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-subdir", default="clap_audio_embeddings")
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--hop-seconds", type=float, default=10.0)
    parser.add_argument(
        "--chunk-pooling",
        default="mean",
        choices=["mean", "max", "std", "mean+max", "mean+std", "max+std", "mean+max+std"],
        help="오디오 모드에서 청크 간 pooling 방식",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--text-field",
        default="tags_and_description",
        choices=["tags", "description", "tags_and_description"],
        help="text_per_sample 모드에서 사용할 텍스트 필드",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"mode: {args.mode}  checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"amodel: {args.amodel}  fusion: {args.enable_fusion}")

    model = load_clap_model(args.checkpoint, args.amodel, args.enable_fusion, device)

    for dataset_name in resolve_dataset_names(args):
        if args.mode == "audio":
            extract_audio(model, dataset_name, args, device)
        elif args.mode == "text":
            extract_text(model, dataset_name, args)
        else:
            extract_text_per_sample(model, dataset_name, args)


if __name__ == "__main__":
    main()
