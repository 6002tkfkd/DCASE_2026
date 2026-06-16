"""
Module 5 — InferenceEngine

연구 단계(splits.csv, train.npz 등)와 독립적인 DCASE 제출용 추론 엔진.

dcase_ensemble 체크포인트를 심링크 없이 legacy_data_root 상대경로로 참조한다
(src.sk.paths 참고). eval 임베딩 경로도 "학습 경로에서 유도"하지 않고 search_config의
`eval:` 블록에 명시된 경로(legacy_data_root 기준 상대경로)를 직접 사용한다 — 실제 eval
임베딩이 학습 임베딩과 다른 디렉토리 명명규칙을 쓰기 때문.

OOM 방지 전략:
    Branch 1 모델을 하나씩 순차적으로 GPU에 로드 → 전체 eval z/logits 추출
    → del + cuda.empty_cache() → 다음 모델 로드.
    마지막에 초경량 GatingNetwork(또는 OOF-stacking LogisticRegression)만 적용.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.sk import paths as sk_paths
from src.sk.module1_cache_extractor import _load_model
from src.sk.module3_ensemble_engine._shared import try_cuda_empty_cache
from src.sk.module3_ensemble_engine.gating_network import GatingNetwork
from src.sk.module3_ensemble_engine.option_a import _learn_weights

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    DCASE 제출용 추론 엔진.

    연구용 데이터(splits.csv, train.npz)와 완전히 독립.
    의존: 새로운 .wav 파일, 학습된 가중치(GatingNetwork .pt 또는 OOF-stacking .joblib),
    search_config.yaml.
    """

    def __init__(
        self,
        config_path: str | Path,
        summary_csv: str | Path,
        output_root: str | Path,
    ) -> None:
        self.config_path = Path(config_path)
        self.summary_csv = Path(summary_csv)
        self.output_root = Path(output_root)
        self.weights_dir = self.output_root / "weights"

        with open(self.config_path) as f:
            self._config = yaml.safe_load(f)

        self.legacy_data_root: str = self._config["meta"]["legacy_data_root"]

        pool = self._config.get("model_pool", [])
        self._exp_dirs: dict[str, Path] = {
            m["name"]: sk_paths.resolve_legacy(m["experiment_dir"], self.legacy_data_root)
            for m in pool
        }
        # short_name → 실제 모델명 (models_used 컬럼 역변환)
        self._short_to_full: dict[str, str] = {
            m.get("short_name", m["name"]): m["name"] for m in pool
        }
        self._cache_root = Path(self._config["meta"]["cache_root"])
        self._eval_config: dict = self._config.get("eval", {})

    # ─────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────

    def run(
        self,
        job_id: str,
        audio_dir: str | Path,
        output_csv: str | Path,
        eval_split: str = "eval",
        batch_size: int = 256,
        device: str = "auto",
    ) -> pd.DataFrame:
        """
        eval 오디오 추론 → submission.csv 저장.

        Args:
            job_id:      실행할 Job ID (summary.csv에 기록된 값).
            audio_dir:   평가용 .wav 파일 디렉토리 (파일 수 검증용; 실제 임베딩은
                         eval_config 경로에서 읽음).
            output_csv:  제출 CSV 저장 경로.
            eval_split:  사용하지 않음 (호환성 유지를 위해 인자만 남김; 실제 eval
                         임베딩 위치는 search_config.yaml의 `eval:` 블록으로 결정).
            batch_size:  Branch 1 forward pass 배치 크기.
            device:      "auto" | "cpu" | "cuda" 등.

        Returns:
            submission DataFrame: [filename, scene_label].
        """
        _device = _resolve_device(device)
        audio_dir = Path(audio_dir)

        # ── Job 메타 로드 ─────────────────────────────────────────────
        job_row     = self._load_job_row(job_id)
        option      = str(job_row["option"])
        fold_id     = int(job_row["fold_id"])
        hyperparams = json.loads(str(job_row["hyperparams"]))
        short_names = [s.strip() for s in str(job_row["models_used"]).split(",")]
        model_names = self._resolve_model_names(short_names)

        logger.info(
            "[M5] job=%s  option=%s  models=%s  fold=%d  device=%s",
            job_id, option, short_names, fold_id, _device,
        )

        # ── eval .wav 목록 ────────────────────────────────────────────
        wav_files = sorted(audio_dir.glob("*.wav"))
        if not wav_files:
            raise FileNotFoundError(f"eval .wav 파일이 없습니다: {audio_dir}")
        sound_ids = [f.stem for f in wav_files]
        logger.info("[M5] eval 파일 수: %d", len(sound_ids))

        # ── [OOM 방지] Branch 1 모델별 순차 추출 ──────────────────────────
        # 모델 1개 로드 → 전체 eval 추출 → 즉시 VRAM 해제 → 다음 모델
        features: dict[str, dict[str, np.ndarray]] = {}
        for model_name in model_names:
            if model_name not in self._exp_dirs:
                raise ValueError(
                    f"'{model_name}'이 search_config.yaml model_pool에 없습니다."
                )
            features[model_name] = self._extract_one_model(
                model_name=model_name,
                sound_ids=sound_ids,
                exp_dir=self._exp_dirs[model_name],
                fold_id=fold_id,
                batch_size=batch_size,
                device=_device,
            )

        # ── class_idx → class_name 매핑 ──────────────────────────────
        class_map = _load_class_map(self._exp_dirs[model_names[0]])

        # ── 앙상블 ───────────────────────────────────────────────────
        if option == "A":
            pred_indices = self._ensemble_a(features, model_names, hyperparams, fold_id)
        elif option == "B":
            pred_indices = self._ensemble_b(features, model_names, job_id, fold_id, hyperparams, _device)
        else:
            raise ValueError(f"알 수 없는 option: '{option}'")

        # ── submission.csv ────────────────────────────────────────────
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        submission = pd.DataFrame({
            "filename":    [f.name for f in wav_files],
            "scene_label": [class_map.get(int(idx), f"cls_{idx}") for idx in pred_indices],
        })
        submission.to_csv(output_csv, index=False)
        logger.info("[M5] 저장: %s  (%d 샘플)", output_csv, len(submission))

        return submission

    # ─────────────────────────────────────────────────────────────
    # OOM 방지 핵심: Branch 1 모델 하나씩 순차 처리
    # ─────────────────────────────────────────────────────────────

    def _extract_one_model(
        self,
        model_name: str,
        sound_ids: list[str],
        exp_dir: Path,
        fold_id: int,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, np.ndarray]:
        """
        Branch 1 모델 하나를 VRAM에 올려 eval z/logits 추출 후 즉시 해제.

        Returns:
            {"z": (N, z_dim), "logits": (N, C)} — CPU numpy 배열.
        """
        checkpoint_path = exp_dir / "both" / f"fold_{fold_id}" / "best_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트 없음: {checkpoint_path}")

        logger.info("[M5] 로딩: %s / fold_%d ...", model_name, fold_id)
        model, mode = _load_model(checkpoint_path, device)

        eval_df = _build_eval_dataframe(
            backbone=_backbone_of(model_name),
            sound_ids=sound_ids,
            mode=mode,
            eval_config=self._eval_config,
            legacy_data_root=self.legacy_data_root,
        )
        loader = DataLoader(
            _EvalEmbeddingDataset(eval_df, mode),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        z_parts, logits_parts = [], []
        try:
            with torch.no_grad():
                for batch in loader:
                    audio_emb = batch["audio_emb"].to(device) if "audio_emb" in batch else None
                    text_emb  = batch["text_emb"].to(device)  if "text_emb"  in batch else None
                    out = model(audio_emb=audio_emb, text_emb=text_emb)
                    z_parts.append(out["z"].cpu().numpy())
                    logits_parts.append(out["logits"].cpu().numpy())
        finally:
            del model
            gc.collect()
            try_cuda_empty_cache()
            logger.info("[M5] VRAM 해제: %s", model_name)

        return {
            "z":      np.concatenate(z_parts,     axis=0),
            "logits": np.concatenate(logits_parts, axis=0),
        }

    # ─────────────────────────────────────────────────────────────
    # 앙상블 적용
    # ─────────────────────────────────────────────────────────────

    def _ensemble_a(
        self,
        features: dict[str, dict],
        model_names: list[str],
        hyperparams: dict,
        fold_id: int,
    ) -> np.ndarray:
        """Option A: logits 가중 평균 → class_idx 예측."""
        strategy     = hyperparams.get("strategy", "simple_average")
        logits_stack = np.stack(
            [features[m]["logits"] for m in model_names], axis=0
        )  # (M, N, C)

        if strategy == "simple_average":
            return np.argmax(logits_stack.mean(axis=0), axis=1)

        if strategy == "weighted_average":
            weights = self._relearn_weights_from_val(model_names, fold_id)
            if weights is None:
                logger.warning("[M5] val 캐시 없음 → simple_average로 자동 대체")
                return np.argmax(logits_stack.mean(axis=0), axis=1)
            logger.info("[M5] 재학습 가중치: %s", np.round(weights, 4).tolist())
            ensemble = (logits_stack * weights[:, None, None]).sum(axis=0)
            return np.argmax(ensemble, axis=1)

        raise ValueError(f"알 수 없는 strategy: '{strategy}'")

    def _relearn_weights_from_val(
        self,
        model_names: list[str],
        fold_id: int,
    ) -> np.ndarray | None:
        """
        val.npz에서 weighted_average 가중치 재학습.
        val은 Branch 1 학습에 사용되지 않으므로 leakage 없음.
        캐시 없으면 None 반환.
        """
        val_logits_list: list[np.ndarray] = []
        val_labels: np.ndarray | None = None

        for m in model_names:
            val_path = self._cache_root / m / f"fold_{fold_id}" / "val.npz"
            if not val_path.exists():
                return None
            d = dict(np.load(val_path, allow_pickle=False))
            val_logits_list.append(d["logits"])
            if val_labels is None:
                val_labels = d["labels"]

        val_stack = np.stack(val_logits_list, axis=0)  # (M, N_val, C)
        return _learn_weights(val_stack, val_labels)

    def _ensemble_b(
        self,
        features: dict[str, dict],
        model_names: list[str],
        job_id: str,
        fold_id: int,
        hyperparams: dict,
        device: torch.device,
    ) -> np.ndarray:
        """
        Option B: model_type에 따라 분기.
            - "oof_stacking": LogisticRegression(.joblib) 메타분류기 적용.
            - 그 외 ("gating"/"stacking"): 학습된 GatingNetwork(.pt) 로드해 z 융합.
        """
        if hyperparams.get("model_type") == "oof_stacking":
            return self._ensemble_b_oof_stacking(features, model_names, job_id, fold_id)
        return self._ensemble_b_gating(features, model_names, job_id, device)

    def _ensemble_b_oof_stacking(
        self,
        features: dict[str, dict],
        model_names: list[str],
        job_id: str,
        fold_id: int,
    ) -> np.ndarray:
        """
        Option B(oof_stacking): joblib LogisticRegression으로 softmax(logits) concat 분류.

        저장된 아티팩트 파일명은 base_job_id(.joblib)이고, summary.csv에는 fold별로
        "{base_job_id}_fold{k}" 행이 각각 기록되어 있다 (module2/greedy_search의
        oof_stacking 기록 방식 — fold별 메트릭이 동일한 단일 메타분류기 결과를 가리킴).
        job_id에서 그 "_fold{k}" 접미사를 떼어 실제 파일명을 복원한다.
        """
        import joblib
        from scipy.special import softmax as _softmax

        suffix = f"_fold{fold_id}"
        base_job_id = job_id[: -len(suffix)] if job_id.endswith(suffix) else job_id

        weights_path = self.weights_dir / f"{base_job_id}.joblib"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"OOF-stacking 분류기 없음: {weights_path}\n"
                f"  scripts/sk/export_best_combo.py 또는 run_ensemble_search.py를 먼저 실행하세요."
            )
        clf = joblib.load(weights_path)

        X = np.concatenate(
            [_softmax(features[m]["logits"], axis=1) for m in model_names], axis=1
        )
        probs = clf.predict_proba(X)
        return clf.classes_[probs.argmax(axis=1)]

    def _ensemble_b_gating(
        self,
        features: dict[str, dict],
        model_names: list[str],
        job_id: str,
        device: torch.device,
    ) -> np.ndarray:
        """Option B(gating/stacking): 학습된 GatingNetwork 로드 → z 융합 → class_idx 예측."""
        weights_path = self.weights_dir / f"{job_id}.pt"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"GatingNetwork 가중치 없음: {weights_path}\n"
                f"  scripts/sk/run_ensemble_search.py를 먼저 실행하세요."
            )

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        hp   = ckpt["hyperparams"]

        # 학습 시 model_names 순서와 불일치 경고
        ckpt_models = ckpt.get("model_names")
        if ckpt_models and ckpt_models != model_names:
            logger.warning(
                "[M5] model_names 순서 불일치!\n  저장 시: %s\n  현재: %s",
                ckpt_models, model_names,
            )

        gate = GatingNetwork(
            z_dim       = ckpt["z_dim"],
            num_sources = ckpt["num_sources"],
            hidden_size = hp["hidden_size"],
            num_classes = ckpt["num_classes"],
            fusion_type = hp["fusion_type"],
            dropout     = hp["dropout_rate"],
        ).to(device)
        gate.load_state_dict(ckpt["state_dict"])
        gate.eval()

        # z 전체를 한 번에 GPU에 올림 (GatingNetwork는 초경량)
        z_gpu = [
            torch.tensor(features[m]["z"], dtype=torch.float32, device=device)
            for m in model_names
        ]

        try:
            with torch.no_grad():
                logits, _ = gate(z_gpu)
            pred_indices = logits.argmax(dim=1).cpu().numpy()
        finally:
            del gate, z_gpu
            gc.collect()
            try_cuda_empty_cache()

        return pred_indices

    # ─────────────────────────────────────────────────────────────
    # 유틸
    # ─────────────────────────────────────────────────────────────

    def _load_job_row(self, job_id: str) -> pd.Series:
        if not self.summary_csv.exists():
            raise FileNotFoundError(f"summary.csv 없음: {self.summary_csv}")
        df   = pd.read_csv(self.summary_csv)
        rows = df[df["job_id"] == job_id]
        if rows.empty:
            available = df["job_id"].tolist()[:10]
            raise ValueError(
                f"job_id '{job_id}'를 summary.csv에서 찾을 수 없습니다.\n"
                f"  사용 가능한 job_id (상위 10개): {available}"
            )
        return rows.iloc[0]

    def _resolve_model_names(self, short_names: list[str]) -> list[str]:
        missing = [s for s in short_names if s not in self._short_to_full]
        if missing:
            raise ValueError(
                f"다음 short_name을 model_pool에서 찾을 수 없습니다: {missing}\n"
                f"  등록된 short_name: {list(self._short_to_full.keys())}"
            )
        return [self._short_to_full[s] for s in short_names]


# ─────────────────────────────────────────────────────────────────────────────
# 모듈 수준 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _backbone_of(model_name: str) -> str:
    return model_name.split("__", 1)[0]


def _build_eval_dataframe(
    backbone: str,
    sound_ids: list[str],
    mode: str,
    eval_config: dict,
    legacy_data_root: str,
) -> pd.DataFrame:
    """
    search_config.yaml의 `eval:` 블록(legacy_data_root 기준 상대경로)으로
    eval 임베딩 디렉토리를 직접 결정한다. (학습 경로에서 유도하지 않음 — 실제 eval
    임베딩은 다른 디렉토리 명명규칙을 쓰기 때문.)
    """
    audio_eval_dir: Path | None = None
    text_eval_dir:  Path | None = None

    if mode in ("audio", "both"):
        audio_dirs = eval_config.get("audio_dirs", {})
        if backbone not in audio_dirs:
            raise KeyError(
                f"eval.audio_dirs에 backbone '{backbone}' 설정이 없습니다. "
                f"search_config.yaml의 eval 블록을 확인하세요."
            )
        audio_eval_dir = sk_paths.resolve_legacy(audio_dirs[backbone], legacy_data_root)
    if mode in ("text", "both"):
        text_dir = eval_config.get("text_dir")
        if not text_dir:
            raise KeyError("eval.text_dir 설정이 없습니다. search_config.yaml의 eval 블록을 확인하세요.")
        text_eval_dir = sk_paths.resolve_legacy(text_dir, legacy_data_root)

    rows = []
    for sid in sound_ids:
        row: dict = {"sound_id": sid}

        if audio_eval_dir is not None:
            p = audio_eval_dir / f"{sid}.npy"
            if not p.exists():
                raise FileNotFoundError(
                    f"eval audio embedding 없음: {p}\n"
                    f"  경로 패턴: {audio_eval_dir}/{{sound_id}}.npy"
                )
            row["audio_emb_filepath"] = str(p)

        if text_eval_dir is not None:
            p = text_eval_dir / f"{sid}.npy"
            if not p.exists():
                raise FileNotFoundError(
                    f"eval text embedding 없음: {p}\n"
                    f"  경로 패턴: {text_eval_dir}/{{sound_id}}.npy"
                )
            row["text_emb_filepath"] = str(p)

        rows.append(row)

    return pd.DataFrame(rows)


def _load_class_map(exp_dir: Path) -> dict[int, str]:
    """processed_dataset.csv에서 class_idx → class_name 매핑 구성."""
    df = pd.read_csv(exp_dir / "processed_dataset.csv", usecols=["class", "class_idx"])
    return {
        int(row["class_idx"]): str(row["class"])
        for _, row in df.drop_duplicates("class_idx").iterrows()
    }


class _EvalEmbeddingDataset(Dataset):
    """eval DataFrame 기반 Dataset (labels 없음; 추론 전용)."""

    def __init__(self, df: pd.DataFrame, mode: str) -> None:
        self.df   = df.reset_index(drop=True)
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row  = self.df.iloc[idx]
        item: dict = {"sound_id": str(row["sound_id"])}
        if self.mode in ("audio", "both"):
            item["audio_emb"] = torch.tensor(
                np.load(str(row["audio_emb_filepath"])), dtype=torch.float32
            )
        if self.mode in ("text", "both"):
            item["text_emb"] = torch.tensor(
                np.load(str(row["text_emb_filepath"])), dtype=torch.float32
            )
        return item


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
