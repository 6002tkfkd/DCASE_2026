import json
import os
from typing import Dict, List, Optional

import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from src.datasets.prep import build_single_dataset, load_processed_dataset
from src.filtering.pipeline import make_filtering_config_by_stage, apply_filtering_df, build_report_path
from src.datasets.hatr_dataset import HATRDataset
from src.utils.core_utils import set_seed, build_class_to_topclass_mapping


class HATRDataManager:
    def __init__(self, config: dict):
        self.config = config if isinstance(config, dict) else {}
        repro_cfg = self.config.get("repro", {}) if isinstance(self.config.get("repro"), dict) else {}
        seed = self.config.get("seed", repro_cfg.get("seed", 1821))
        self.seed = set_seed(int(seed))
        self._cached_full_df = {}
        self._cached_stage1_excluded = {}
        self._cached_active_dataset = {}
        self._master_split_filename = "master_dataset_splits.csv"

    @staticmethod
    def _join_path(*parts):
        valid_parts = [str(p) for p in parts if p is not None and str(p) != ""]
        if not valid_parts:
            return ""
        return os.path.join(*valid_parts)

    @staticmethod
    def _normalize_dataset_source(dataset_name: Optional[str], mode: str) -> str:
        dataset_name = str(dataset_name or "").lower()
        if "35k" in dataset_name or mode == "pretrain":
            return "35k"
        return "10k"

    @staticmethod
    def _to_sound_ids(values) -> List[str]:
        return [str(v) for v in values]

    @staticmethod
    def _build_split_rows(
        sound_ids,
        dataset_source: str,
        mode: str,
        fold_id: str,
        split: str,
        class_map: dict | None = None,
    ) -> List[Dict[str, str]]:
        return [
            {
                "sound_id": str(sound_id),
                "dataset_source": dataset_source,
                "mode": mode,
                "fold_id": str(fold_id),
                "split": split,
                "class": str(class_map.get(str(sound_id), "")) if class_map else "",
            }
            for sound_id in sound_ids
        ]

    def _master_split_path(self) -> str:
        return os.path.join(self._resolve_output_path(), self._master_split_filename)

    def _write_split_table(self, rows: List[Dict[str, str]], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # include class column if present in rows
        cols = ["sound_id", "dataset_source", "mode", "fold_id", "split"]
        if rows and "class" in rows[0]:
            cols.append("class")
        pd.DataFrame(rows, columns=cols).to_csv(path, index=False)

    def _append_master_split_rows(self, rows: List[Dict[str, str]]) -> None:
        if not rows:
            return

        master_path = self._master_split_path()
        cols = ["sound_id", "dataset_source", "mode", "fold_id", "split"]
        if rows and "class" in rows[0]:
            cols.append("class")
        new_df = pd.DataFrame(rows, columns=cols)

        if os.path.exists(master_path):
            existing_df = pd.read_csv(master_path, dtype=str)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        # ensure 'class' column exists
        if "class" not in combined_df.columns:
            combined_df["class"] = ""

        # Fill missing class values using any cached full_df we have for the corresponding mode
        for mode_key, full_df in self._cached_full_df.items():
            if full_df is None or "index" not in full_df.columns or "class" not in full_df.columns:
                continue
            class_map_local = dict(zip(full_df["index"].astype(str).tolist(), full_df["class"].astype(str).tolist()))
            mask = (combined_df.get("mode", "") == str(mode_key)) & (combined_df["class"].fillna("").astype(str) == "")
            if mask.any():
                combined_df.loc[mask, "class"] = combined_df.loc[mask, "sound_id"].map(class_map_local).fillna("")

        # dedupe on identifying keys; preserve class when present
        dedupe_keys = [k for k in ["sound_id", "dataset_source", "mode", "fold_id", "split"] if k in combined_df.columns]
        combined_df = combined_df.drop_duplicates(subset=dedupe_keys)
        sort_keys = [k for k in ["mode", "dataset_source", "fold_id", "split", "sound_id"] if k in combined_df.columns]
        if sort_keys:
            combined_df = combined_df.sort_values(by=sort_keys, kind="stable")
        os.makedirs(os.path.dirname(master_path), exist_ok=True)
        combined_df.to_csv(master_path, index=False)

    def _resolve_output_path(self):
        paths_cfg = self.config.get("paths", {}) if isinstance(self.config.get("paths"), dict) else {}
        output_cfg = self.config.get("output", {}) if isinstance(self.config.get("output"), dict) else {}
        model_cfg = self.config.get("model", {}) if isinstance(self.config.get("model"), dict) else {}

        if output_cfg.get("root") and output_cfg.get("run_dir"):
            return os.path.join(
                output_cfg.get("root", "./output"),
                output_cfg.get("run_dir", "model_output"),
                model_cfg.get("name", "base_classifier"),
            )

        if self.config.get("output_path"):
            return str(self.config.get("output_path"))

        output_root = paths_cfg.get("output_root", "./output")
        model_output_dir = output_cfg.get("model_output_dir", "model_output")
        return os.path.join(output_root, model_output_dir, model_cfg.get("name", "base_classifier"))

    def _get_datasets_cfg(self):
        datasets_cfg = self.config.get("datasets", {})
        return datasets_cfg if isinstance(datasets_cfg, dict) else {}

    def _identify_aux_dataset(self):
        datasets_cfg = self._get_datasets_cfg()
        for name in datasets_cfg.keys():
            if "35k" in str(name).lower():
                return name
        raise ValueError("35k dataset not found in config!")

    def _identify_main_dataset(self):
        datasets_cfg = self._get_datasets_cfg()
        for name in datasets_cfg.keys():
            if "10k" in str(name).lower():
                return name
        if datasets_cfg:
            return next(iter(datasets_cfg.keys()))
        raise ValueError("10k dataset not found in config!")

    def _resolve_mode_block(self, mode: str):
        block = self.config.get(mode, {})
        return block if isinstance(block, dict) else {}

    def _resolve_training_cfg(self, mode: str):
        mode_block = self._resolve_mode_block(mode)
        mode_training = mode_block.get("training", {}) if isinstance(mode_block.get("training"), dict) else {}
        global_training = self.config.get("training", {}) if isinstance(self.config.get("training"), dict) else {}

        resolved = dict(global_training)
        resolved.update(mode_training)
        return resolved

    def _resolve_loader_cfg(self, mode: str):
        mode_block = self._resolve_mode_block(mode)
        training_cfg = self._resolve_training_cfg(mode)
        evaluation_cfg = mode_block.get("evaluation", {}) if isinstance(mode_block.get("evaluation"), dict) else {}
        global_evaluation = self.config.get("evaluation", {}) if isinstance(self.config.get("evaluation"), dict) else {}

        resolved = {}
        resolved.update(global_evaluation)
        resolved.update(evaluation_cfg)
        resolved.update(training_cfg)

        resolved["batch_size"] = int(resolved.get("batch_size", 128))
        resolved["num_workers"] = int(resolved.get("num_workers", 4))
        return resolved

    def _resolve_processed_csv_path(self, active_dataset: str):
        paths_cfg = self.config.get("paths", {}) if isinstance(self.config.get("paths"), dict) else {}
        processed_basename = paths_cfg.get("processed_basename", "processed_dataset.csv")
        return os.path.join(self._resolve_output_path(), processed_basename)

    def _resolve_class_artifact_paths(self):
        artifact_cfg = self.config.get("artifact_names", {}) if isinstance(self.config.get("artifact_names"), dict) else {}
        output_path = self._resolve_output_path()
        class_dict_name = artifact_cfg.get("class_dict_json", "class_dict.json")
        top_class_dict_name = artifact_cfg.get("top_class_dict_json", "top_class_dict.json")
        return (
            os.path.join(output_path, class_dict_name),
            os.path.join(output_path, top_class_dict_name),
        )

    def _prepare_dataframe(self, mode: str) -> pd.DataFrame:
        if mode in self._cached_full_df:
            return self._cached_full_df[mode]

        if mode == "pretrain":
            active_dataset = self._identify_aux_dataset()
        elif mode in ["baseline", "finetune"]:
            active_dataset = self._identify_main_dataset()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        temp_cfg = dict(self.config)
        temp_cfg["active_dataset"] = active_dataset
        build_single_dataset(config=temp_cfg)

        processed_csv_path = self._resolve_processed_csv_path(active_dataset)
        full_df = pd.read_csv(processed_csv_path)
        filtering_cfg = self.config.get("filtering", {}) if isinstance(self.config.get("filtering"), dict) else {}
        stage1_excluded_ids: List[str] = []

        if isinstance(filtering_cfg, dict) and "stage1" in filtering_cfg:
            stage1_cfg = make_filtering_config_by_stage(filtering_cfg, stage="stage1")
            if stage1_cfg.enabled:
                report_path = build_report_path(stage1_cfg, active_dataset, stage="stage1")
                filtered_df, report = apply_filtering_df(
                    full_df,
                    "index",
                    stage1_cfg,
                    report_path=report_path,
                    stage="all",
                )
                stage1_excluded_ids = [str(item.get("sample_id", "")) for item in report.get("excluded", [])]
                full_df = filtered_df

        self._cached_full_df[mode] = full_df
        self._cached_stage1_excluded[mode] = stage1_excluded_ids
        self._cached_active_dataset[mode] = active_dataset
        return full_df

    def get_class_mappings(self, mode: str):
        self._prepare_dataframe(mode)
        class_dict_path, top_class_dict_path = self._resolve_class_artifact_paths()

        with open(class_dict_path, "r", encoding="utf-8") as f:
            class_dict = json.load(f)
        with open(top_class_dict_path, "r", encoding="utf-8") as f:
            top_class_dict = json.load(f)

        class_to_topclass = build_class_to_topclass_mapping(class_dict, top_class_dict)
        return class_dict, top_class_dict, class_to_topclass

    def _apply_stage2_filtering(self, train_df: pd.DataFrame) -> pd.DataFrame:
        filtering_cfg = self.config.get("filtering", {}) if isinstance(self.config.get("filtering"), dict) else {}
        stage2_cfg = make_filtering_config_by_stage(filtering_cfg, stage="stage2")

        if stage2_cfg.enabled and stage2_cfg.apply_to == "train":
            filtered_df, _ = apply_filtering_df(train_df, "index", stage2_cfg, stage="train")
            return filtered_df

        return train_df

    def _get_stage1_excluded_ids(self, mode: str) -> List[str]:
        return self._cached_stage1_excluded.get(mode, [])

    def _write_mode_split_files(self, output_dir: str, split_rows: List[Dict[str, str]]) -> None:
        if not split_rows:
            return

        self._write_split_table(split_rows, os.path.join(output_dir, "splits.csv"))
        self._append_master_split_rows(split_rows)

    def _resolve_pretrain_validation_size(self, total_samples: int) -> int:
        pretrain_cfg = self.config.get("pretrain", {}) if isinstance(self.config.get("pretrain"), dict) else {}
        validation_split_cfg = (
            pretrain_cfg.get("validation_split", {})
            if isinstance(pretrain_cfg.get("validation_split"), dict)
            else {}
        )

        validation_size = validation_split_cfg.get("size", pretrain_cfg.get("validation_size"))
        if validation_size is not None:
            validation_size = int(validation_size)
            if validation_size >= total_samples:
                raise ValueError(
                    f"Pretrain validation size must be smaller than dataset size, got {validation_size} for {total_samples} samples"
                )
            return validation_size

        validation_ratio = validation_split_cfg.get("ratio", pretrain_cfg.get("validation_ratio"))
        if validation_ratio is not None:
            ratio = float(validation_ratio)
            if not 0.0 < ratio < 1.0:
                raise ValueError(f"pretrain.validation_ratio must be between 0 and 1, got {ratio}")
            validation_size = max(1, int(round(total_samples * ratio)))
            if validation_size >= total_samples:
                raise ValueError(
                    f"Pretrain validation size must be smaller than dataset size, got {validation_size} for {total_samples} samples"
                )
            return validation_size

        validation_size = 3500
        if validation_size >= total_samples:
            raise ValueError(
                f"Pretrain validation size must be smaller than dataset size, got {validation_size} for {total_samples} samples"
            )
        return validation_size

    def _build_dataloader(self, df: pd.DataFrame, batch_size: int, shuffle: bool, drop_last: bool, num_workers: int, aug: bool):
        # aug가 True일 때는 예전처럼 0.7을 주고, False일 때는 마스킹을 아예 안 하도록(0.0) 설정
        dataset = HATRDataset(df, aug=aug, mask_pct=0.7 if aug else 0.0)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

    def get_dataloaders(self, mode: str):
        if mode not in ["baseline", "pretrain", "finetune"]:
            raise ValueError(f"Unsupported mode: {mode}")

        loader_cfg = self._resolve_loader_cfg(mode)
        batch_size = int(loader_cfg.get("batch_size", 128))
        num_workers = int(loader_cfg.get("num_workers", 4))

        full_df = self._prepare_dataframe(mode)
        labels = full_df["class_idx"].tolist()
        # build mapping from sound_id to class name for inclusion in splits
        class_map = {}
        if "index" in full_df.columns and "class" in full_df.columns:
            class_map = dict(zip(full_df["index"].astype(str).tolist(), full_df["class"].astype(str).tolist()))

        if mode == "baseline":
            training_cfg = self._resolve_training_cfg(mode)
            k_folds = int(training_cfg.get("k_folds", 5))
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.seed)
            dataset_source = self._normalize_dataset_source(self._cached_active_dataset.get(mode), mode)
            stage1_excluded_ids = self._get_stage1_excluded_ids(mode)
            stage1_rows = self._build_split_rows(stage1_excluded_ids, dataset_source, mode, "N/A", "excluded_stage1", class_map)

            loaders = []
            for fold_id, (trainval_idx, test_idx) in enumerate(skf.split([0] * len(labels), labels)):
                trainval_df = full_df.iloc[trainval_idx].reset_index(drop=True)
                test_df = full_df.iloc[test_idx].reset_index(drop=True)

                trainval_labels = trainval_df["class_idx"].tolist()
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
                train_idx_rel, val_idx_rel = next(sss.split([0] * len(trainval_labels), trainval_labels))

                train_df = trainval_df.iloc[train_idx_rel].reset_index(drop=True)
                val_df = trainval_df.iloc[val_idx_rel].reset_index(drop=True)

                train_ids_before = self._to_sound_ids(train_df["index"].tolist())
                train_df = self._apply_stage2_filtering(train_df)
                train_ids_after = set(self._to_sound_ids(train_df["index"].tolist()))
                excluded_stage2_ids = [sid for sid in train_ids_before if sid not in train_ids_after]

                train_dl = self._build_dataloader(train_df, batch_size, True, True, num_workers, True)
                val_dl = self._build_dataloader(val_df, batch_size, False, False, num_workers, False)
                test_dl = self._build_dataloader(test_df, batch_size, False, False, num_workers, False)
                loaders.append((train_dl, val_dl, test_dl))

                fold_rows: List[Dict[str, str]] = []
                fold_rows.extend(self._build_split_rows(self._to_sound_ids(train_df["index"].tolist()), dataset_source, mode, str(fold_id), "train", class_map))
                fold_rows.extend(self._build_split_rows(self._to_sound_ids(val_df["index"].tolist()), dataset_source, mode, str(fold_id), "val", class_map))
                fold_rows.extend(self._build_split_rows(self._to_sound_ids(test_df["index"].tolist()), dataset_source, mode, str(fold_id), "test", class_map))
                fold_rows.extend(self._build_split_rows(excluded_stage2_ids, dataset_source, mode, str(fold_id), "excluded_stage2", class_map))
                fold_rows.extend(stage1_rows)
                fold_output_dir = os.path.join(self._resolve_output_path(), "both", f"fold_{fold_id}")
                self._write_mode_split_files(fold_output_dir, fold_rows)

            return loaders

        if mode == "pretrain":
            validation_size = self._resolve_pretrain_validation_size(len(full_df))
            sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=self.seed)
            train_idx, val_idx = next(sss.split([0] * len(labels), labels))

            train_df = full_df.iloc[train_idx].reset_index(drop=True)
            val_df = full_df.iloc[val_idx].reset_index(drop=True)

            dataset_source = self._normalize_dataset_source(self._cached_active_dataset.get(mode), mode)
            stage1_excluded_ids = self._get_stage1_excluded_ids(mode)
            train_ids_before = self._to_sound_ids(train_df["index"].tolist())
            train_df = self._apply_stage2_filtering(train_df)
            train_ids_after = set(self._to_sound_ids(train_df["index"].tolist()))
            excluded_stage2_ids = [sid for sid in train_ids_before if sid not in train_ids_after]

            train_dl = self._build_dataloader(train_df, batch_size, True, True, num_workers, True)
            val_dl = self._build_dataloader(val_df, batch_size, False, False, num_workers, False)

            split_rows: List[Dict[str, str]] = []
            split_rows.extend(self._build_split_rows(self._to_sound_ids(train_df["index"].tolist()), dataset_source, mode, "N/A", "train", class_map))
            split_rows.extend(self._build_split_rows(self._to_sound_ids(val_df["index"].tolist()), dataset_source, mode, "N/A", "val", class_map))
            split_rows.extend(self._build_split_rows(excluded_stage2_ids, dataset_source, mode, "N/A", "excluded_stage2", class_map))
            split_rows.extend(self._build_split_rows(stage1_excluded_ids, dataset_source, mode, "N/A", "excluded_stage1", class_map))
            pretrain_output_dir = os.path.join(self._resolve_output_path(), "pretrain")
            self._write_mode_split_files(pretrain_output_dir, split_rows)
            return train_dl, val_dl, None

        training_cfg = self._resolve_training_cfg(mode)
        k_folds = int(training_cfg.get("k_folds", 5))

        sss_holdout = StratifiedShuffleSplit(n_splits=1, test_size=2192, random_state=self.seed)
        trainval_idx, test_idx = next(sss_holdout.split([0] * len(labels), labels))

        trainval_df = full_df.iloc[trainval_idx].reset_index(drop=True)
        test_df = full_df.iloc[test_idx].reset_index(drop=True)

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.seed)
        test_dl = self._build_dataloader(test_df, batch_size, False, False, num_workers, False)
        dataset_source = self._normalize_dataset_source(self._cached_active_dataset.get(mode), mode)
        stage1_excluded_ids = self._get_stage1_excluded_ids(mode)
        stage1_rows = self._build_split_rows(stage1_excluded_ids, dataset_source, mode, "N/A", "excluded_stage1", class_map)

        loaders = []
        trainval_labels = trainval_df["class_idx"].tolist()
        for fold_id, (train_idx, val_idx) in enumerate(skf.split([0] * len(trainval_labels), trainval_labels)):
            train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
            val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

            train_ids_before = self._to_sound_ids(train_df["index"].tolist())
            train_df = self._apply_stage2_filtering(train_df)
            train_ids_after = set(self._to_sound_ids(train_df["index"].tolist()))
            excluded_stage2_ids = [sid for sid in train_ids_before if sid not in train_ids_after]

            train_dl = self._build_dataloader(train_df, batch_size, True, True, num_workers, True)
            val_dl = self._build_dataloader(val_df, batch_size, False, False, num_workers, False)
            loaders.append((train_dl, val_dl, test_dl))

            fold_rows: List[Dict[str, str]] = []
            fold_rows.extend(self._build_split_rows(self._to_sound_ids(train_df["index"].tolist()), dataset_source, mode, str(fold_id), "train", class_map))
            fold_rows.extend(self._build_split_rows(self._to_sound_ids(val_df["index"].tolist()), dataset_source, mode, str(fold_id), "val", class_map))
            fold_rows.extend(self._build_split_rows(self._to_sound_ids(test_df["index"].tolist()), dataset_source, mode, str(fold_id), "test", class_map))
            fold_rows.extend(self._build_split_rows(excluded_stage2_ids, dataset_source, mode, str(fold_id), "excluded_stage2", class_map))
            fold_rows.extend(stage1_rows)
            split_mode_dir = "both" if mode == "finetune" else mode
            fold_output_dir = os.path.join(self._resolve_output_path(), split_mode_dir, f"fold_{fold_id}")
            self._write_mode_split_files(fold_output_dir, fold_rows)

        return loaders