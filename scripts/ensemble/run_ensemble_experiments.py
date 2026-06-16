#!/usr/bin/env python3
"""Run ensemble-only experiments with fold-safe OOF softmax caching.

This script never trains a meta-model. For each base run, it loads the fold_k
checkpoint and runs inference only on fold_k test samples from the saved
splits.csv. The resulting OOF softmax cache is then averaged for the requested
ensemble experiments.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

np = None
torch = None
F = None
BaseClassifier = None
BaseClassifierProxyAnchor = None
BaseClassifierProxyAnchorDeepShared = None
BaseClassifierProxyAnchorSimple = None
hierarchical_prf_weighted = None


def load_numpy() -> None:
    global np
    if np is not None:
        return
    import numpy as numpy_mod
    np = numpy_mod


def load_runtime_imports() -> None:
    global torch, F, BaseClassifier, BaseClassifierProxyAnchor
    global BaseClassifierProxyAnchorDeepShared, BaseClassifierProxyAnchorSimple
    global hierarchical_prf_weighted
    load_numpy()
    if torch is not None:
        return
    import torch as torch_mod
    import torch.nn.functional as functional_mod
    from src.metrics.hierarchical import hierarchical_prf_weighted as hprf
    from src.models.hatr import BaseClassifier as classifier_cls
    from src.models.hatr_proxy_anchor import BaseClassifierProxyAnchor as proxy_cls
    from src.models.hatr_proxy_anchor_deep_shared import BaseClassifierProxyAnchorDeepShared as deep_shared_cls
    from src.models.hatr_proxy_anchor_simple import BaseClassifierProxyAnchorSimple as simple_cls

    torch = torch_mod
    F = functional_mod
    BaseClassifier = classifier_cls
    BaseClassifierProxyAnchor = proxy_cls
    BaseClassifierProxyAnchorDeepShared = deep_shared_cls
    BaseClassifierProxyAnchorSimple = simple_cls
    hierarchical_prf_weighted = hprf

MODELS = [
    "m2d_clap_vit_base_meanstd",
    "clap_630k_chunk_meanmax",
    "clap_music_as_chunk_mean",
    "beats_iter3plus_as2m_seq_meanmaxstd_chunk_max",
    "passt_kd_ap486_chunk_max",
    "fpasst_weak_seq_meanmax_chunk_maxstd",
    "atst_weak_seq_maxstd_chunk_meanstd",
    "beats_iter3plus_as2m_ft1_seq_meanstd_chunk_mean",
]

FOLD_SUBDIR = Path("base_classifier_proxy_anchor_simple") / "both"


@dataclass(frozen=True)
class RunSpec:
    model: str
    exp: str
    config: str
    rank: int
    hf1: float

    @property
    def key(self) -> str:
        return f"{self.model}/{self.exp}/{self.config}"


def read_ranked_runs(summary_csv: Path, models: Sequence[str]) -> Dict[str, List[RunSpec]]:
    by_model: Dict[str, List[RunSpec]] = {m: [] for m in models}
    with summary_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            if model not in by_model:
                continue
            by_model[model].append(
                RunSpec(
                    model=model,
                    exp=row["exp"],
                    config=row["config"],
                    rank=int(row["rank"]),
                    hf1=float(row["hierarchical_f1_mean"]),
                )
            )
    for model, runs in by_model.items():
        runs.sort(key=lambda r: (-r.hf1, r.rank))
        if not runs:
            raise RuntimeError(f"No ranked runs found for model: {model}")
    return by_model


def run_base_dir(spec: RunSpec, output_roots: Sequence[Path]) -> Path:
    rel = Path(spec.model) / spec.exp / spec.config / "base_classifier_proxy_anchor_simple"
    for root in output_roots:
        candidate = root / rel
        if (candidate / "both" / "fold_0" / "best_model.pth").exists():
            return candidate
    roots = ", ".join(str(r) for r in output_roots)
    raise FileNotFoundError(f"Missing completed checkpoint for {spec.key} under: {roots}")


def cache_path(output_dir: Path, spec: RunSpec) -> Path:
    return output_dir / "cache" / "oof_softmax" / spec.model / spec.exp / spec.config / "oof_softmax.npz"


def metadata_path(output_dir: Path, spec: RunSpec) -> Path:
    return cache_path(output_dir, spec).with_suffix(".json")


def torch_load(path: Path, map_location="cpu"):
    load_runtime_imports()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_model_class_from_checkpoint(ckpt: dict):
    load_runtime_imports()
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    state = ckpt.get("model_state", {}) if isinstance(ckpt, dict) else {}
    model_name = str(cfg.get("model_name", "")).strip().lower()
    use_classifier = bool(cfg.get("use_classifier", True))
    state_keys = set(state.keys()) if isinstance(state, dict) else set()

    def has_any(prefixes: Sequence[str]) -> bool:
        return any(any(key.startswith(prefix) for prefix in prefixes) for key in state_keys)

    if "deep_shared" in model_name or "deepshared" in model_name or "proxy_anchor_shared" in model_name:
        return BaseClassifierProxyAnchorDeepShared

    if "proxy_anchor_simple" in model_name or (
        "latent_projector.weight" in state_keys
        and "class_predictor.weight" in state_keys
        and not has_any(["latent_projector.0.", "latent_projector.3.", "latent_projector.6.", "residual_classifier.", "proxy_projector."])
    ):
        return BaseClassifierProxyAnchorSimple

    if "proxy_anchor" in model_name or not use_classifier or has_any(["proxy_projector.", "proxy_"]):
        return BaseClassifierProxyAnchor

    return BaseClassifier


def build_model(ckpt: dict, device: torch.device):
    load_runtime_imports()
    cfg = ckpt.get("config", {})
    model_class = resolve_model_class_from_checkpoint(ckpt)
    signature = inspect.signature(model_class.__init__)
    allowed_keys = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    }
    filtered_cfg = {key: value for key, value in cfg.items() if key in allowed_keys and value is not None}
    model = model_class(**filtered_cfg)
    load_result = model.load_state_dict(ckpt["model_state"], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            f"[WARN] Loaded {model_class.__name__} with "
            f"missing_keys={load_result.missing_keys} unexpected_keys={load_result.unexpected_keys}"
        )
    model.eval()
    return model.to(device)


def resolve_existing_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    candidates: List[Path] = []
    marker = "dcase2026-task1-training-pipeline/"
    if marker in raw_path:
        candidates.append(PROJECT_ROOT / raw_path.split(marker, 1)[1])

    if "/embedding/" in raw_path:
        rel = raw_path.split("/embedding/", 1)[1]
        candidates.append(PROJECT_ROOT / "embedding" / rel)
        env_root = os.environ.get("DCASE_EMBEDDING_ROOT")
        if env_root:
            candidates.append(Path(env_root) / rel)

    if "/data/" in raw_path:
        rel = raw_path.split("/data/", 1)[1]
        data_roots = [
            PROJECT_ROOT / "data",
            PROJECT_ROOT.parent / "data",
            PROJECT_ROOT.parent / "dcase2026_task1_baseline" / "data",
            PROJECT_ROOT.parent.parent / "dcase2026_task1_baseline" / "data",
            Path("/workspace/dcase2026_task1_baseline/data"),
            Path("/workspace/mimi/dcase2026_task1_baseline/data"),
            Path("/home/hansarang/dcase/dcase2026_task1_baseline/data"),
        ]
        env_data_root = os.environ.get("DCASE_DATA_ROOT")
        if env_data_root:
            data_roots.insert(0, Path(env_data_root))
        candidates.extend(root / rel for root in data_roots)

    if "clap_text_embeddings" in raw_path:
        filename = path.name
        text_roots = [
            PROJECT_ROOT / "data" / "BSD10k-v1.2" / "features" / "clap_text_embeddings",
            PROJECT_ROOT.parent / "dcase2026_task1_baseline" / "data" / "BSD10k-v1.2" / "features" / "clap_text_embeddings",
            PROJECT_ROOT.parent.parent / "dcase2026_task1_baseline" / "data" / "BSD10k-v1.2" / "features" / "clap_text_embeddings",
            Path("/workspace/dcase2026_task1_baseline/data/BSD10k-v1.2/features/clap_text_embeddings"),
            Path("/workspace/mimi/dcase2026_task1_baseline/data/BSD10k-v1.2/features/clap_text_embeddings"),
            Path("/home/hansarang/dcase/dcase2026_task1_baseline/data/BSD10k-v1.2/features/clap_text_embeddings"),
            Path("/home/hansarang/dcase/eval/clap_text_embeddings"),
        ]
        env_text_root = os.environ.get("DCASE_TEXT_EMB_ROOT")
        if env_text_root:
            text_roots.insert(0, Path(env_text_root))
        candidates.extend(root / filename for root in text_roots)

    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate

    tried = "\n  ".join(str(c) for c in candidates[:12])
    if len(candidates) > 12:
        tried += f"\n  ... ({len(candidates) - 12} more)"
    raise FileNotFoundError(f"{raw_path}\nTried:\n  {tried}")


def load_processed_mapping(base_dir: Path) -> Dict[str, Tuple[str, str]]:
    csv_path = base_dir / "processed_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    mapping: Dict[str, Tuple[str, str]] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row["index"]).strip()
            mapping[sid] = (row["audio_emb_filepath"], row["text_emb_filepath"])
    return mapping


def load_class_dict(base_dir: Path) -> Dict[str, int]:
    path = base_dir / "class_dict.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def read_fold_test_split(base_dir: Path, fold: int) -> List[Tuple[str, str]]:
    path = base_dir / "both" / f"fold_{fold}" / "splits.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows: List[Tuple[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == "test":
                rows.append((str(row["sound_id"]).strip(), row["class"]))
    if not rows:
        raise RuntimeError(f"No test rows in {path}")
    return rows


def infer_fold(
    model: BaseClassifier,
    fold_rows: Sequence[Tuple[str, str]],
    processed: Dict[str, Tuple[str, str]],
    device: torch.device,
    batch_size: int,
) -> Tuple[List[str], np.ndarray]:
    load_runtime_imports()
    sound_ids: List[str] = []
    probs: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(fold_rows), batch_size):
            batch_rows = fold_rows[start:start + batch_size]
            valid_ids: List[str] = []
            audio_list: List[np.ndarray] = []
            text_list: List[np.ndarray] = []

            for sid, _ in batch_rows:
                if sid not in processed:
                    raise KeyError(f"sound_id {sid} missing from processed_dataset.csv")
                audio_path, text_path = processed[sid]
                audio_list.append(np.load(resolve_existing_path(audio_path)))
                text_list.append(np.load(resolve_existing_path(text_path)))
                valid_ids.append(sid)

            audio_t = torch.tensor(np.stack(audio_list), dtype=torch.float32, device=device)
            text_t = torch.tensor(np.stack(text_list), dtype=torch.float32, device=device)
            out = model(audio_emb=audio_t, text_emb=text_t)
            batch_probs = F.softmax(out["logits"], dim=-1).detach().cpu().numpy().astype(np.float32)
            sound_ids.extend(valid_ids)
            probs.append(batch_probs)

    return sound_ids, np.concatenate(probs, axis=0)


def ensure_oof_cache(
    spec: RunSpec,
    output_roots: Sequence[Path],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    force: bool,
    dry_run: bool,
) -> Path:
    out_path = cache_path(output_dir, spec)
    if out_path.exists() and not force:
        return out_path

    base_dir = run_base_dir(spec, output_roots)
    if dry_run:
        print(f"[DRY cache] {spec.key} <- {base_dir}")
        return out_path

    load_runtime_imports()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed = load_processed_mapping(base_dir)
    class_dict = load_class_dict(base_dir)
    id2class = {v: k for k, v in class_dict.items()}

    all_ids: List[str] = []
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []

    for fold in range(5):
        fold_dir = base_dir / "both" / f"fold_{fold}"
        ckpt_path = fold_dir / "best_model.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)

        fold_rows = read_fold_test_split(base_dir, fold)
        ckpt = torch_load(ckpt_path, map_location="cpu")
        model = build_model(ckpt, device)
        ids, probs = infer_fold(model, fold_rows, processed, device, batch_size)
        labels = [class_dict[label] for _, label in fold_rows]

        all_ids.extend(ids)
        all_labels.extend(labels)
        all_probs.append(probs)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  cache {spec.key} fold {fold}: {len(ids)} samples")

    probs_arr = np.concatenate(all_probs, axis=0).astype(np.float32)
    labels_arr = np.asarray(all_labels, dtype=np.int64)
    ids_arr = np.asarray(all_ids, dtype=str)

    np.savez_compressed(out_path, sound_ids=ids_arr, probs=probs_arr, labels=labels_arr)
    meta = {
        "model": spec.model,
        "exp": spec.exp,
        "config": spec.config,
        "rank": spec.rank,
        "hf1": spec.hf1,
        "base_dir": str(base_dir),
        "num_samples": int(len(ids_arr)),
        "class_dict": class_dict,
        "id2class": {str(k): v for k, v in id2class.items()},
        "oof_policy": "fold_k checkpoint evaluated only on fold_k test split",
    }
    with metadata_path(output_dir, spec).open("w") as f:
        json.dump(meta, f, indent=2)
    return out_path


def top_level(label: str) -> str:
    return label.split("-", 1)[0] if isinstance(label, str) else str(label)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, id2class: Dict[int, str]) -> Dict[str, float]:
    load_runtime_imports()
    preds = probs.argmax(axis=1)
    scores = probs.max(axis=1)
    gt_labels = [id2class[int(x)] for x in labels]
    pred_labels = [id2class[int(x)] for x in preds]
    total = len(labels)

    accuracy = 100.0 * float(np.mean(preds == labels))
    top_accuracy = 100.0 * float(np.mean([top_level(p) == top_level(g) for p, g in zip(pred_labels, gt_labels)]))

    class_accs = []
    class_top_accs = []
    for cls in sorted(set(labels.tolist())):
        idx = labels == cls
        class_accs.append(float(np.mean(preds[idx] == labels[idx])))
        class_top_accs.append(float(np.mean([
            top_level(pred_labels[i]) == top_level(gt_labels[i])
            for i in np.where(idx)[0]
        ])))

    pred_gt = list(zip(pred_labels, gt_labels))
    hfs = []
    for cls_label in sorted(set(gt_labels)):
        _, _, hf = hierarchical_prf_weighted(cls_label, pred_gt, lambda_param=0.75)
        if not np.isnan(hf):
            hfs.append(hf)

    return {
        "num_samples": int(total),
        "accuracy": accuracy,
        "top_accuracy": top_accuracy,
        "macro_accuracy": 100.0 * float(np.mean(class_accs)),
        "macro_top_accuracy": 100.0 * float(np.mean(class_top_accs)),
        "hierarchical_f1": 100.0 * float(np.mean(hfs)) if hfs else 0.0,
        "mean_prediction_score": 100.0 * float(np.mean(scores)),
    }


def average_duplicate_sound_ids(sound_ids: np.ndarray, probs: np.ndarray, labels: np.ndarray):
    positions: Dict[str, List[int]] = defaultdict(list)
    for idx, sid in enumerate(sound_ids.astype(str).tolist()):
        positions[sid].append(idx)

    if all(len(indices) == 1 for indices in positions.values()):
        return sound_ids.astype(str), probs.astype(np.float32), labels.astype(np.int64)

    ordered_ids = list(positions.keys())
    avg_probs = []
    avg_labels = []
    for sid in ordered_ids:
        idx = np.asarray(positions[sid], dtype=np.int64)
        label_values = labels[idx]
        if len(set(label_values.astype(int).tolist())) != 1:
            raise ValueError(f"Label mismatch across duplicate sound_id={sid}")
        avg_probs.append(probs[idx].mean(axis=0))
        avg_labels.append(int(label_values[0]))

    return (
        np.asarray(ordered_ids, dtype=str),
        np.stack(avg_probs).astype(np.float32),
        np.asarray(avg_labels, dtype=np.int64),
    )


def load_cache(output_dir: Path, spec: RunSpec):
    load_numpy()
    path = cache_path(output_dir, spec)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    with metadata_path(output_dir, spec).open() as f:
        meta = json.load(f)
    id2class = {int(k): v for k, v in meta["id2class"].items()}
    sound_ids, probs, labels = average_duplicate_sound_ids(
        data["sound_ids"].astype(str),
        data["probs"].astype(np.float32),
        data["labels"].astype(np.int64),
    )
    return sound_ids, probs, labels, id2class


def normalize_weights(weights: Sequence[float]) -> List[float]:
    arr = [float(w) for w in weights]
    total = sum(arr)
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return [w / total for w in arr]


def combine_specs(output_dir: Path, specs: Sequence[RunSpec], weights: Sequence[float]):
    load_numpy()
    weights_arr = normalize_weights(weights)
    per_run = []
    common_ids = None
    id2class_ref = None

    for spec in specs:
        ids, probs, labels, id2class = load_cache(output_dir, spec)
        mapping = {sid: i for i, sid in enumerate(ids.tolist())}
        per_run.append((spec, mapping, probs, labels, id2class))
        ids_set = set(mapping.keys())
        common_ids = ids_set if common_ids is None else common_ids & ids_set
        if id2class_ref is None:
            id2class_ref = id2class
        elif id2class_ref != id2class:
            raise ValueError(f"class_dict mismatch for {spec.key}")

    if not common_ids:
        raise RuntimeError("No common sound_ids across ensemble inputs")

    ordered_ids = sorted(common_ids, key=lambda x: int(x) if x.isdigit() else x)
    combined = None
    labels_ref = None

    for weight, (spec, mapping, probs, labels, _) in zip(weights_arr, per_run):
        idx = np.asarray([mapping[sid] for sid in ordered_ids], dtype=np.int64)
        part_probs = probs[idx]
        part_labels = labels[idx]
        if combined is None:
            combined = weight * part_probs.astype(np.float64)
            labels_ref = part_labels
        else:
            if not np.array_equal(labels_ref, part_labels):
                raise ValueError(f"Label mismatch while combining {spec.key}")
            combined += weight * part_probs.astype(np.float64)

    return np.asarray(ordered_ids, dtype=str), combined.astype(np.float32), labels_ref.astype(np.int64), id2class_ref


def write_inputs_csv(path: Path, specs: Sequence[RunSpec], weights: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    weights_arr = normalize_weights(weights)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "exp", "config", "rank", "hf1", "weight"])
        writer.writeheader()
        for spec, weight in zip(specs, weights_arr):
            writer.writerow({
                "model": spec.model,
                "exp": spec.exp,
                "config": spec.config,
                "rank": spec.rank,
                "hf1": spec.hf1,
                "weight": float(weight),
            })


def write_predictions_csv(path: Path, sound_ids: np.ndarray, labels: np.ndarray, probs: np.ndarray, id2class: Dict[int, str]) -> None:
    load_numpy()
    preds = probs.argmax(axis=1)
    scores = probs.max(axis=1)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sound_id", "ground_truth", "prediction", "prediction_score", "is_correct"])
        writer.writeheader()
        for sid, gt, pred, score in zip(sound_ids, labels, preds, scores):
            writer.writerow({
                "sound_id": sid,
                "ground_truth": id2class[int(gt)],
                "prediction": id2class[int(pred)],
                "prediction_score": round(float(score), 6),
                "is_correct": bool(int(gt) == int(pred)),
            })


def run_ensemble(
    output_dir: Path,
    exp_name: str,
    specs: Sequence[RunSpec],
    weights: Sequence[float],
    dry_run: bool,
):
    result_dir = output_dir / exp_name
    if dry_run:
        print(f"[DRY ensemble] {exp_name}: {len(specs)} inputs")
        for spec, weight in zip(specs, normalize_weights(weights)):
            print(f"  {weight:.5f}  {spec.key}")
        return None

    result_dir.mkdir(parents=True, exist_ok=True)
    sound_ids, probs, labels, id2class = combine_specs(output_dir, specs, weights)
    metrics = compute_metrics(labels, probs, id2class)

    np.savez_compressed(result_dir / "oof_softmax.npz", sound_ids=sound_ids, probs=probs, labels=labels)
    write_inputs_csv(result_dir / "inputs.csv", specs, weights)
    write_predictions_csv(result_dir / "predictions.csv", sound_ids, labels, probs, id2class)

    result = {
        "experiment": exp_name,
        "num_inputs": len(specs),
        "metrics": metrics,
        "oof_policy": "ensemble averages cached OOF softmax only; no trainable meta-model",
        "duplicate_sound_id_policy": "average duplicate fold predictions before combining models",
    }
    with (result_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2)
    print(f"[RESULT] {exp_name}: hF1={metrics['hierarchical_f1']:.2f}% acc={metrics['accuracy']:.2f}% inputs={len(specs)}")
    return result


def rank_weights(specs: Sequence[RunSpec]) -> List[float]:
    ordered = sorted(specs, key=lambda s: (-s.hf1, s.rank))
    score_by_key = {spec.key: len(specs) - i for i, spec in enumerate(ordered)}
    return [float(score_by_key[spec.key]) for spec in specs]


def hf1_weights(specs: Sequence[RunSpec]) -> List[float]:
    return [max(spec.hf1, 0.0) for spec in specs]


def rank_decay_weights(specs: Sequence[RunSpec], by_model: Dict[str, List[RunSpec]]) -> List[float]:
    weights = []
    decay = {1: 1.0, 2: 0.7, 3: 0.5}
    local_rank_by_key = {}
    for model, runs in by_model.items():
        for i, spec in enumerate(runs[:3], 1):
            local_rank_by_key[spec.key] = i
    for spec in specs:
        weights.append(decay.get(local_rank_by_key.get(spec.key, 3), 0.5))
    return weights


def model_uniform_weights(specs: Sequence[RunSpec]) -> List[float]:
    counts = defaultdict(int)
    for spec in specs:
        counts[spec.model] += 1
    return [1.0 / counts[spec.model] for spec in specs]


def selected_experiments(by_model: Dict[str, List[RunSpec]], names: Sequence[str]):
    experiments = []

    def enabled(name):
        return "all" in names or name in names

    if enabled("same_top8"):
        for model in MODELS:
            specs = by_model[model][:8]
            experiments.append((f"same_top8/{model}/uniform", specs, [1.0] * len(specs)))

    if enabled("same_top20"):
        for model in MODELS:
            specs = by_model[model][:20]
            experiments.append((f"same_top20/{model}/uniform", specs, [1.0] * len(specs)))

    if enabled("cross_top1"):
        specs = [by_model[model][0] for model in MODELS]
        experiments.append(("cross_top1/uniform", specs, [1.0] * len(specs)))
        experiments.append(("cross_top1/rank_weight", specs, rank_weights(specs)))
        experiments.append(("cross_top1/hf1_weight", specs, hf1_weights(specs)))

    if enabled("cross_top3"):
        specs = [spec for model in MODELS for spec in by_model[model][:3]]
        experiments.append(("cross_top3/uniform", specs, [1.0] * len(specs)))
        experiments.append(("cross_top3/model_uniform", specs, model_uniform_weights(specs)))
        experiments.append(("cross_top3/rank_decay", specs, rank_decay_weights(specs, by_model)))
        experiments.append(("cross_top3/hf1_weight", specs, hf1_weights(specs)))

    return experiments


def write_summary(output_dir: Path, results: Sequence[dict]) -> None:
    rows = []
    for result in results:
        if result is None:
            continue
        metrics = result["metrics"]
        rows.append({
            "experiment": result["experiment"],
            "num_inputs": result["num_inputs"],
            "hierarchical_f1": metrics["hierarchical_f1"],
            "accuracy": metrics["accuracy"],
            "top_accuracy": metrics["top_accuracy"],
            "macro_accuracy": metrics["macro_accuracy"],
            "macro_top_accuracy": metrics["macro_top_accuracy"],
            "num_samples": metrics["num_samples"],
        })
    rows.sort(key=lambda r: r["hierarchical_f1"], reverse=True)
    path = output_dir / "summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment", "num_inputs", "hierarchical_f1", "accuracy", "top_accuracy",
            "macro_accuracy", "macro_top_accuracy", "num_samples",
        ])
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"[SUMMARY] {path}")


def main():
    parser = argparse.ArgumentParser(description="Run ensemble-only OOF softmax experiments")
    parser.add_argument("--summary-csv", default=str(PROJECT_ROOT / "results_summary_all.csv"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output_ensemble"))
    parser.add_argument("--output-roots", nargs="+", default=[
        str(PROJECT_ROOT / "output_final"),
        str(PROJECT_ROOT / "output_final_ensemble"),
    ])
    parser.add_argument("--experiments", nargs="+", default=["all"],
                        choices=["all", "same_top8", "same_top20", "cross_top1", "cross_top3"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_roots = [Path(p).resolve() for p in args.output_roots]
    summary_csv = Path(args.summary_csv).resolve()

    if args.dry_run:
        device = "dry-run"
    else:
        load_runtime_imports()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    by_model = read_ranked_runs(summary_csv, MODELS)
    experiments = selected_experiments(by_model, args.experiments)

    print("========================================")
    print("[Ensemble-only experiments]")
    print(f"  output_dir : {output_dir}")
    print(f"  roots      : {', '.join(str(r) for r in output_roots)}")
    print(f"  device     : {device}")
    print(f"  experiments: {len(experiments)}")
    print("  RND policy : fold_k checkpoint -> fold_k test split only")
    print("========================================")

    needed_specs = []
    seen = set()
    for _, specs, _ in experiments:
        for spec in specs:
            if spec.key not in seen:
                needed_specs.append(spec)
                seen.add(spec.key)
    print(f"[CACHE] required base runs: {len(needed_specs)}")

    for spec in needed_specs:
        ensure_oof_cache(
            spec=spec,
            output_roots=output_roots,
            output_dir=output_dir,
            device=device,
            batch_size=args.batch_size,
            force=args.force_cache,
            dry_run=args.dry_run,
        )

    results = []
    for exp_name, specs, weights in experiments:
        results.append(run_ensemble(output_dir, exp_name, specs, weights, args.dry_run))

    if not args.dry_run:
        write_summary(output_dir, results)


if __name__ == "__main__":
    main()
