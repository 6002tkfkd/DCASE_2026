#!/usr/bin/env python3
"""
best-combo(4모델 OOF-stacking) 재현:
  1. 메타분류기(LogisticRegression)를 학습/영속화 (ensemble_output/weights/*.joblib,*.json)
  2. 10k(BSD10k-v1.2 test, 캐시 기반) / eval(DCASE2026 Task1 평가셋) 예측 CSV 출력

dcase_ensemble의 scripts/export_oof_predictions.py를 일반화한 버전. 원본의 하드코딩
절대경로(/home/islp/embeddings/..., /home/islp/dcase_real/...)는 제거했고, 모든 경로는
best_combo.yaml의 legacy_data_root 상대경로로 해석한다.

35k export는 포함하지 않음 — 원본 스크립트의 35k 입력 경로가 dcase_ensemble/MESH
양쪽 모두에 속하지 않는 별도 스크래치 위치였기 때문에, 지금 범위(10k 재현 검증 +
eval 제출)에서는 제외했다. 필요해지면 config에 `pretrain_35k:` 블록을 추가하고
이 스크립트에 동일 패턴으로 분기를 추가하면 된다.

Usage:
    python scripts/sk/export_best_combo.py --config scripts/sk/configs/best_combo.yaml
    python scripts/sk/export_best_combo.py --config scripts/sk/configs/best_combo.yaml --modes 10k
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import softmax as _softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sk import paths as sk_paths
from src.sk.module1_cache_extractor import _load_model
from src.sk.module2_combo_manager import ExperimentJob, _make_job_id_b
from src.sk.module3_ensemble_engine._shared import load_npz_aligned
from src.sk.module3_ensemble_engine.option_b import _run_oof_stacking

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def _backbone_of(model_name: str) -> str:
    return model_name.split("__", 1)[0]


def _build_hyperparams(cfg: dict) -> dict:
    bc = cfg["best_combo"]
    ob = cfg.get("search", {}).get("option_b", {})
    return {
        "model_type": bc.get("model_type", "oof_stacking"),
        "hidden_size": (ob.get("hidden_sizes") or [64])[0],
        "fusion_type": (ob.get("fusion_types") or ["softmax_gate"])[0],
        "dropout_rate": (ob.get("dropout_rates") or [0.3])[0],
        "num_epochs": int(ob.get("num_epochs", 100)),
        "lr": float(ob.get("lr", 0.001)),
        "patience": int(ob.get("patience", 15)),
    }


def _load_class_idx_to_name(exp_dir: Path) -> dict[int, str]:
    df = pd.read_csv(exp_dir / "processed_dataset.csv", usecols=["class", "class_idx"])
    return {
        int(row["class_idx"]): str(row["class"])
        for _, row in df.drop_duplicates("class_idx").iterrows()
    }


def _build_tensors(
    sound_ids: list[str], audio_dir: Path, text_dir: Path
) -> tuple[torch.Tensor, torch.Tensor]:
    audio = torch.stack(
        [torch.tensor(np.load(audio_dir / f"{sid}.npy"), dtype=torch.float32) for sid in sound_ids]
    )
    text = torch.stack(
        [torch.tensor(np.load(text_dir / f"{sid}.npy"), dtype=torch.float32) for sid in sound_ids]
    )
    return audio, text


def _infer_branch_probs(
    exp_dir: Path,
    audio_t: torch.Tensor,
    text_t: torch.Tensor,
    device: torch.device,
    k_folds: int,
    batch_size: int = 256,
) -> np.ndarray:
    """fold 0~k_folds-1 체크포인트의 softmax 확률 평균 (N, C)."""
    n = audio_t.shape[0]
    fold_probs = []
    for fold in range(k_folds):
        ckpt_path = exp_dir / "both" / f"fold_{fold}" / "best_model.pth"
        model, _mode = _load_model(ckpt_path, device)
        parts = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                a = audio_t[i : i + batch_size].to(device)
                t = text_t[i : i + batch_size].to(device)
                out = model(audio_emb=a, text_emb=t)
                parts.append(_softmax(out["logits"].cpu().numpy(), axis=1))
        fold_probs.append(np.concatenate(parts, axis=0))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.mean(fold_probs, axis=0)


class BestComboExporter:
    def __init__(self, config_path: str) -> None:
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.legacy_root: str = self.cfg["meta"]["legacy_data_root"]
        self.cache_root = Path(self.cfg["meta"]["cache_root"])
        self.output_root = Path(self.cfg["meta"]["output_root"])
        self.weights_dir = self.output_root / "weights"
        self.export_dir = self.output_root / "exports" / "best_oof_combo"
        self.k_folds = int(self.cfg["search"].get("k_folds", 5))

        pool = {m.get("short_name", m["name"]): m for m in self.cfg["model_pool"]}
        short_order = self.cfg["best_combo"]["models"]
        self.model_specs = [pool[s] for s in short_order]
        self.model_names = [m["name"] for m in self.model_specs]
        self.exp_dirs = {
            m["name"]: sk_paths.resolve_legacy(m["experiment_dir"], self.legacy_root)
            for m in self.model_specs
        }
        self.hyperparams = _build_hyperparams(self.cfg)
        self.eval_cfg = self.cfg.get("eval", {})

    def _base_job_id(self) -> str:
        short_names = [m.get("short_name", m["name"]) for m in self.model_specs]
        hp = self.hyperparams
        return (
            _make_job_id_b(short_names, hp["hidden_size"], hp["fusion_type"], hp["dropout_rate"], fold_id=0)
            + "_oof"
        )

    def step1_fit_and_save(self) -> tuple[Any, dict]:
        base_jid = self._base_job_id()
        save_path = self.weights_dir / base_jid
        if save_path.with_suffix(".joblib").exists():
            logger.info("[1/3] 기존 아티팩트 재사용: %s.joblib", save_path)
            import joblib

            clf = joblib.load(save_path.with_suffix(".joblib"))
            meta = json.load(save_path.with_suffix(".json").open())
            return clf, meta

        job = ExperimentJob(
            job_id=save_path.name,
            models=[m.get("short_name", m["name"]) for m in self.model_specs],
            model_names=self.model_names,
            option="B",
            fold_id=0,
            hyperparams=self.hyperparams,
        )
        metrics = _run_oof_stacking(job, self.cache_root, self.k_folds, save_path=save_path)
        logger.info("[1/3] 메타분류기 학습+저장 완료: %s", metrics)

        import joblib

        clf = joblib.load(save_path.with_suffix(".joblib"))
        meta = json.load(save_path.with_suffix(".json").open())
        return clf, meta

    def step2_export_10k(self, clf: Any, idx_to_name: dict[int, str]) -> None:
        logger.info("[2/3] 10k export 시작")
        fold_probs = []
        for fold in range(self.k_folds):
            test_data = load_npz_aligned(self.cache_root, self.model_names, fold, "test")
            X = np.concatenate([_softmax(d["logits"], axis=1) for d in test_data], axis=1)
            fold_probs.append(clf.predict_proba(X))
        probs = np.mean(fold_probs, axis=0)

        test_f0 = load_npz_aligned(self.cache_root, self.model_names, 0, "test")
        sound_ids = test_f0[0]["sound_ids"]
        labels = test_f0[0]["labels"]

        pred_idx = clf.classes_[probs.argmax(axis=1)]
        score = probs.max(axis=1)

        df = pd.DataFrame(
            {
                "sound_id": sound_ids,
                "ground_truth": [idx_to_name[int(l)] for l in labels],
                "prediction": [idx_to_name[int(c)] for c in pred_idx],
                "prediction_score": score,
            }
        )
        df["is_correct"] = df["ground_truth"] == df["prediction"]
        self.export_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.export_dir / "predictions_10k.csv"
        df.to_csv(out_path, index=False)
        logger.info("[2/3] 10k 저장: %s (n=%d, acc=%.4f)", out_path, len(df), df["is_correct"].mean())

    def step3_export_eval(self, clf: Any, idx_to_name: dict[int, str], device: torch.device) -> None:
        logger.info("[3/3] eval export 시작")
        audio_dirs_cfg = self.eval_cfg.get("audio_dirs", {})
        text_dir_cfg = self.eval_cfg.get("text_dir")
        if not audio_dirs_cfg or not text_dir_cfg:
            logger.warning("[3/3] eval 설정 없음 (config의 eval: 블록 확인) → 건너뜀")
            return

        audio_dirs = {
            backbone: sk_paths.resolve_legacy(rel, self.legacy_root)
            for backbone, rel in audio_dirs_cfg.items()
        }
        text_dir = sk_paths.resolve_legacy(text_dir_cfg, self.legacy_root)

        backbones = sorted({_backbone_of(n) for n in self.model_names})
        missing = [b for b in backbones if b not in audio_dirs]
        if missing:
            logger.warning("[3/3] eval.audio_dirs에 backbone 누락: %s → 건너뜀", missing)
            return

        audio_stems = [{p.stem for p in audio_dirs[b].glob("*.npy")} for b in backbones]
        text_stems = {p.stem for p in text_dir.glob("*.npy")}
        eval_ids = sorted(set.intersection(*audio_stems, text_stems))
        logger.info("[3/3] eval 샘플 수: %d", len(eval_ids))

        backbone_tensors = {
            b: _build_tensors(eval_ids, audio_dirs[b], text_dir) for b in backbones
        }

        branch_probs = []
        for name in self.model_names:
            audio_t, text_t = backbone_tensors[_backbone_of(name)]
            branch_probs.append(
                _infer_branch_probs(self.exp_dirs[name], audio_t, text_t, device, self.k_folds)
            )

        X = np.concatenate(branch_probs, axis=1)
        final_probs = clf.predict_proba(X)
        pred_idx = clf.classes_[final_probs.argmax(axis=1)]
        score = final_probs.max(axis=1)

        # eval_ids는 metadata.csv의 anonymous_id(전체 UUID)와 동일한 .npy stem이므로
        # 그대로 제출 id로 사용한다 (8자로 자르지 않음 — DCASE 제출 형식은 전체 UUID).
        df = pd.DataFrame(
            {
                "id": eval_ids,
                "predicted_bst_second_level_class": [idx_to_name[int(c)] for c in pred_idx],
                "prediction_score": score,
            }
        )
        out_path = self.export_dir / "predictions_eval.csv"
        df.to_csv(out_path, index=False)
        logger.info("[3/3] eval 저장: %s (n=%d)", out_path, len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="best-combo OOF-stacking 재현 (10k + eval)")
    parser.add_argument("--config", default="scripts/sk/configs/best_combo.yaml")
    parser.add_argument("--modes", nargs="+", default=["10k", "eval"], choices=["10k", "eval"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    exporter = BestComboExporter(args.config)
    idx_to_name = _load_class_idx_to_name(exporter.exp_dirs[exporter.model_names[0]])

    clf, meta = exporter.step1_fit_and_save()
    logger.info("[1/3] meta metrics=%s", meta.get("metrics"))

    if "10k" in args.modes:
        exporter.step2_export_10k(clf, idx_to_name)
    if "eval" in args.modes:
        exporter.step3_export_eval(clf, idx_to_name, device)

    logger.info("완료. 출력 디렉토리: %s", exporter.export_dir.resolve())


if __name__ == "__main__":
    main()
