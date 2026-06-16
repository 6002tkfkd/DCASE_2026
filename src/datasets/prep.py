import json
import os
from typing import Dict, Optional

import pandas as pd

from src.filtering.pipeline import FilteringConfig, apply_filtering_df, build_report_path, make_filtering_config, make_filtering_config_by_stage
from src.utils.core_utils import load_config


def _resolve_runtime_config(config: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    if isinstance(config, dict):
        return config
    return load_config() or {}


def _resolve_artifact_paths(config: Dict[str, object]) -> Dict[str, str]:
    paths_cfg = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}
    output_cfg = config.get("output", {}) if isinstance(config.get("output"), dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    artifact_cfg = config.get("artifact_names", {}) if isinstance(config.get("artifact_names"), dict) else {}

    # Try new structure first (output.root + output.run_dir + model.name)
    if output_cfg.get("root") and output_cfg.get("run_dir"):
        output_path = os.path.join(
            output_cfg.get("root", "./output"),
            output_cfg.get("run_dir", "model_output"),
            model_cfg.get("name", "base_classifier"),
        )
    # Fall back to legacy output_path
    elif config.get("output_path"):
        output_path = config.get("output_path")
    # Fall back to old structure (paths.output_root + output.model_output_dir)
    else:
        output_root = paths_cfg.get("output_root", "./output")
        model_output_dir = output_cfg.get("model_output_dir", "model_output")
        output_path = os.path.join(output_root, model_output_dir, model_cfg.get("name", "base_classifier"))

    return {
        "output_path": output_path,
        "processed_dataset_csv": paths_cfg.get("processed_basename", "processed_dataset.csv"),
        "class_dict_json": artifact_cfg.get("class_dict_json", "class_dict.json"),
        "top_class_dict_json": artifact_cfg.get("top_class_dict_json", "top_class_dict.json"),
        "top_class_subclass_dict_json": artifact_cfg.get(
            "top_class_subclass_dict_json", "top_class_subclass_dict.json"
        ),
    }


def build_single_dataset(config: Optional[Dict[str, object]] = None) -> None:
    """Build dataset for a single active dataset (baseline behavior)."""
    cfg = _resolve_runtime_config(config)
    dataset_name = cfg.get("active_dataset")
    datasets_cfg = cfg.get("datasets", {}) if isinstance(cfg.get("datasets"), dict) else {}

    if not dataset_name or dataset_name not in datasets_cfg:
        raise KeyError("active_dataset is missing or not registered in datasets config")

    metadata_csv = datasets_cfg[dataset_name]["metadata_csv"]
    audio_emb_folder = datasets_cfg[dataset_name]["audio_emb_folder"]
    text_emb_folder = datasets_cfg[dataset_name]["text_emb_folder"]

    paths = _resolve_artifact_paths(cfg)

    output_path = paths["output_path"]
    os.makedirs(output_path, exist_ok=True)
    processed_dataset_csv = os.path.join(output_path, paths["processed_dataset_csv"])
    class_dict_json = os.path.join(output_path, paths["class_dict_json"])
    top_class_dict_json = os.path.join(output_path, paths["top_class_dict_json"])
    top_class_subclass_dict_json = os.path.join(
        output_path, paths["top_class_subclass_dict_json"]
    )

    df = pd.read_csv(metadata_csv)
    df["sound_id"] = df["sound_id"].astype(str).str.strip()

    print(f"Examining original data from {dataset_name}:")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique classes: {df['class'].nunique()}")

    s = df["class_idx"].astype(str)
    df = df[~((s.str.len() == 3) & (s.str.endswith('99') | s.str.endswith('00')))].copy()
    print("After filtering:", len(df))

    df["original_class_idx"] = df["class_idx"]

    original_indices = sorted(df["original_class_idx"].unique())
    index_mapping = {orig: new for new, orig in enumerate(original_indices)}
    df["class_idx"] = df["original_class_idx"].map(index_mapping)

    df["class_top"] = df["class"].apply(
        lambda x: x.split("-")[0] if isinstance(x, str) else None
    )

    df_sorted = df.sort_values("original_class_idx")
    top_classes = df_sorted["class_top"].drop_duplicates()
    class_top_dict = {cls: i for i, cls in enumerate(top_classes)}
    df["top_class_idx"] = df["class_top"].map(class_top_dict)

    class_dict = dict(zip(df["class"], df["class_idx"]))

    class_top_subclass_dict = {
        top_class: {
            subclass: idx
            for idx, subclass in enumerate(
                df[df["class_top"] == top_class]
                .sort_values("original_class_idx")["class"]
                .drop_duplicates()
            )
        }
        for top_class in class_top_dict.keys()
    }

    with open(class_dict_json, "w") as f:
        json.dump(class_dict, f, indent=4)
    print(f"Saved class dictionary to {class_dict_json}")

    with open(top_class_dict_json, "w") as f:
        json.dump(class_top_dict, f, indent=4)
    print(f"Saved top class dictionary to {top_class_dict_json}")

    with open(top_class_subclass_dict_json, "w") as f:
        json.dump(class_top_subclass_dict, f, indent=4)
    print(f"Saved top class subclass dictionary to {top_class_subclass_dict_json}")

    records = []

    df.set_index("sound_id", inplace=True)
    for sound_id in df.index:
        file = f"{sound_id}.npy"

        audio_emb_filepath = os.path.abspath(os.path.join(audio_emb_folder, file))
        text_emb_filepath = os.path.abspath(os.path.join(text_emb_folder, file))

        if not os.path.isfile(audio_emb_filepath):
            print(f"Missing audio embedding for sound_id {sound_id}")
            continue

        if not os.path.isfile(text_emb_filepath):
            print(f"Missing text embedding for sound_id {sound_id}")
            continue

        match = df.loc[sound_id]

        class_top = match["class_top"]
        class_top_idx = class_top_dict.get(class_top, -1)
        class_name = match["class"]
        class_idx = int(match["class_idx"])

        records.append(
            {
                "index": sound_id,
                "audio_emb_filepath": audio_emb_filepath,
                "text_emb_filepath": text_emb_filepath,
                "top_class": class_top,
                "top_class_idx": class_top_idx,
                "class": class_name,
                "class_idx": class_idx,
            }
        )

    db_df = pd.DataFrame(records)
    db_df.to_csv(processed_dataset_csv, index=False)
    print(f"Saved embedding dataframe to {processed_dataset_csv}")
    print(f"Dataset built with {len(db_df)} samples.")


def build_multi_dataset(config: Optional[Dict[str, object]] = None) -> None:
    """Build datasets for multiple sources with canonical class mapping (combined/two-stage)."""
    cfg = _resolve_runtime_config(config)
    datasets_cfg = cfg.get("datasets", {}) if isinstance(cfg.get("datasets"), dict) else {}

    if not datasets_cfg:
        raise KeyError("datasets config is missing")

    paths = _resolve_artifact_paths(cfg)

    output_path = paths["output_path"]
    os.makedirs(output_path, exist_ok=True)
    processed_dataset_basename = paths["processed_dataset_csv"]
    class_dict_json = os.path.join(output_path, paths["class_dict_json"])
    top_class_dict_json = os.path.join(output_path, paths["top_class_dict_json"])
    top_class_subclass_dict_json = os.path.join(
        output_path, paths["top_class_subclass_dict_json"]
    )

    all_datasets = list(datasets_cfg.keys())

    canonical_class_dict = None
    canonical_top_class_dict = None
    canonical_top_subclass = None

    for dataset_name in all_datasets:
        metadata_csv = datasets_cfg[dataset_name]["metadata_csv"]
        audio_emb_folder = datasets_cfg[dataset_name]["audio_emb_folder"]
        text_emb_folder = datasets_cfg[dataset_name]["text_emb_folder"]

        df = pd.read_csv(metadata_csv)
        df["sound_id"] = df["sound_id"].astype(str).str.strip()

        print(f"Examining original data from {dataset_name}:")
        print(f"  Total rows: {len(df)}")
        print(f"  Unique classes: {df['class'].nunique()}")

        s = df["class_idx"].astype(str)
        df = df[~((s.str.len() == 3) & (s.str.endswith('99') | s.str.endswith('00')))].copy()
        print("After filtering:", len(df))

        df["original_class_idx"] = df["class_idx"]

        if canonical_class_dict is None:
            original_indices = sorted(df["original_class_idx"].unique())
            index_mapping = {orig: new for new, orig in enumerate(original_indices)}
            df["class_idx"] = df["original_class_idx"].map(index_mapping)

            df["class_top"] = df["class"].apply(
                lambda x: x.split("-")[0] if isinstance(x, str) else None
            )
            df_sorted = df.sort_values("original_class_idx")
            top_classes = df_sorted["class_top"].drop_duplicates()
            class_top_dict = {cls: i for i, cls in enumerate(top_classes)}
            df["top_class_idx"] = df["class_top"].map(class_top_dict)

            class_dict = dict(zip(df["class"], df["class_idx"]))

            class_top_subclass_dict = {
                top_class: {
                    subclass: idx
                    for idx, subclass in enumerate(
                        df[df["class_top"] == top_class]
                        .sort_values("original_class_idx")["class"]
                        .drop_duplicates()
                    )
                }
                for top_class in class_top_dict.keys()
            }

            with open(class_dict_json, "w") as f:
                json.dump(class_dict, f, indent=4)
            print(f"Saved class dictionary to {class_dict_json}")

            with open(top_class_dict_json, "w") as f:
                json.dump(class_top_dict, f, indent=4)
            print(f"Saved top class dictionary to {top_class_dict_json}")

            with open(top_class_subclass_dict_json, "w") as f:
                json.dump(class_top_subclass_dict, f, indent=4)
            print(f"Saved top class subclass dictionary to {top_class_subclass_dict_json}")

            canonical_class_dict = class_dict
            canonical_top_class_dict = class_top_dict
            canonical_top_subclass = class_top_subclass_dict
        else:
            df["class_top"] = df["class"].apply(
                lambda x: x.split("-")[0] if isinstance(x, str) else None
            )
            df["class_idx"] = df["class"].map(canonical_class_dict).fillna(-1).astype(int)
            df["top_class_idx"] = (
                df["class_top"].map(canonical_top_class_dict).fillna(-1).astype(int)
            )

        records = []
        df.set_index("sound_id", inplace=True)
        for sound_id in df.index:
            file = f"{sound_id}.npy"

            audio_emb_filepath = os.path.abspath(os.path.join(audio_emb_folder, file))
            text_emb_filepath = os.path.abspath(os.path.join(text_emb_folder, file))

            if not os.path.isfile(audio_emb_filepath):
                continue

            if not os.path.isfile(text_emb_filepath):
                continue

            match = df.loc[sound_id]

            class_top = match.get("class_top", None)
            class_top_idx = int(match.get("top_class_idx", -1))
            class_name = match.get("class", None)
            class_idx = int(match.get("class_idx", -1))

            if class_idx < 0:
                continue

            records.append(
                {
                    "index": sound_id,
                    "audio_emb_filepath": audio_emb_filepath,
                    "text_emb_filepath": text_emb_filepath,
                    "top_class": class_top,
                    "top_class_idx": class_top_idx,
                    "class": class_name,
                    "class_idx": class_idx,
                }
            )

        db_df = pd.DataFrame(records)
        out_name = f"{os.path.splitext(processed_dataset_basename)[0]}_{dataset_name}.csv"
        out_path = os.path.join(output_path, out_name)
        db_df.to_csv(out_path, index=False)
        print(f"Saved embedding dataframe to {out_path}")
        print(f"Dataset built with {len(db_df)} samples for {dataset_name}.")


def load_processed_dataset(
    csv_path: str,
    filtering_cfg: Optional[Dict[str, object]] = None,
    report_tag: Optional[str] = None,
    sample_id_col: str = "index",
    stage: Optional[str] = None,
    output_path: Optional[str] = None,
    save_stage1_csv: bool = False,
    stage1_csv_basename: str = "processed_dataset_stage1.csv",
) -> pd.DataFrame:
    """Load processed dataset CSV and apply optional stage-based filtering.
    
    Supports two-stage filtering:
    - stage1: Apply to full dataset when loading (apply_to: all)
    - stage2: Apply to train split only (apply_to: train)
    
    Args:
        csv_path: Path to processed dataset CSV
        filtering_cfg: Top-level filtering config with 'stage1' and 'stage2' keys
        report_tag: Tag for filter report filename
        sample_id_col: Column name for sample IDs
        stage: Filter stage to apply ('stage1' or None/other for legacy)
        output_path: Directory to save filtered CSV (for stage1)
        save_stage1_csv: Whether to save stage1 filtered results to CSV
        stage1_csv_basename: Filename for stage1 filtered CSV
    
    Returns:
        Filtered or unfiltered DataFrame
    """
    df = pd.read_csv(csv_path)
    if filtering_cfg is None:
        return df

    # Handle new stage-based filtering structure
    if stage == "stage1" and isinstance(filtering_cfg, dict) and "stage1" in filtering_cfg:
        filtering = make_filtering_config_by_stage(filtering_cfg, stage="stage1")
        if not filtering.enabled:
            return df
        
        report_path = build_report_path(filtering, report_tag or "dataset", stage="stage1")
        filtered_df, report = apply_filtering_df(
            df,
            sample_id_col,
            filtering,
            report_path=report_path,
            stage="all",  # stage1 apply_to is "all"
        )
        
        # Save stage1 filtered CSV if requested
        if save_stage1_csv and output_path:
            os.makedirs(output_path, exist_ok=True)
            stage1_csv_path = os.path.join(output_path, stage1_csv_basename)
            filtered_df.to_csv(stage1_csv_path, index=False)
            print(f"Saved stage1 filtered dataset ({len(filtered_df)} samples) to {stage1_csv_path}")
        
        return filtered_df
    
    # Handle legacy single-stage filtering (backward compatibility)
    filtering = make_filtering_config(filtering_cfg)
    if not filtering.enabled:
        return df

    report_path = build_report_path(filtering, report_tag or "dataset", stage=stage or "load")
    filtered_df, _ = apply_filtering_df(
        df,
        sample_id_col,
        filtering,
        report_path=report_path,
        stage=stage,
    )
    return filtered_df
