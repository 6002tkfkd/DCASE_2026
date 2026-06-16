# Embeddings Module

**Design Principle**: Offline embedding **contract & storage layer**, decoupled from training.

## Overview

This module provides:
- **EmbeddingSpec**: Metadata (name, modality, dimension) for embeddings
- **EmbeddingRegistry**: Pluggable extractor registry for future batch extraction
- **EmbeddingPaths**: Storage conventions (separate audio/text roots, `{sound_id}.npy` files)
- **Storage Planning**: Manifest of embedding roots by modality

## Responsibilities

### ✅ What This Module Does
- Defines embedding specs and storage conventions
- Maintains extractor registry for future use
- Plans storage layout (audio root, text root)
- Acts as a **contract** for batch embedding generation (external/async process)

### ❌ What This Module Does NOT Do
- Does NOT extract embeddings (that's external)
- Does NOT run during training
- Does NOT modify trainers or evaluation
- Does NOT implement model fusion (that's `src/models/modules/fusion.py`)

## Architecture

```
External Batch Pipeline (async)
    ↓
    Generate embeddings → Store as `.npy`
    ↓
Processed CSV (populated by `src/datasets/prep.py`)
    - `audio_emb_filepath`: path to audio embedding file
    - `text_emb_filepath`: path to text embedding file
    - labels, class indices, etc.
    ↓
HATRDataset loads `.npy` files directly
    ↓
Training/Eval consume embeddings as-is
```

## File Structure

- **`base.py`**: `EmbeddingSpec` dataclass, `BaseEmbeddingExtractor` protocol
- **`config.py`**: `EmbeddingConfig` (enabled flag, list of extractors, storage dirs)
- **`registry.py`**: `EmbeddingRegistry` (pluggable extractor registry)
- **`io.py`**: `EmbeddingPaths` (storage roots), `build_storage_plan()` (manifest)
- **`pipeline.py`**: `prepare_embedding_stage()` (plan extractor outputs without executing)
- **`clap_extractor.py`**: `CLAPEmbeddingExtractor` (placeholder for future use)

## API

### EmbeddingSpec
```python
@dataclass(frozen=True)
class EmbeddingSpec:
    name: str           # e.g., "clap_audio"
    modality: str       # "audio" or "text"
    dim: Optional[int]  # e.g., 512
```

### EmbeddingConfig
```python
@dataclass(frozen=True)
class EmbeddingConfig:
    enabled: bool           # Enable embedding stage
    extractors: List[str]   # List of extractor names (registry keys)
    audio_dirname: str      # e.g., "audio_embeddings"
    text_dirname: str       # e.g., "text_embeddings"
```

### Registry
```python
registry = EmbeddingRegistry()
registry.register("clap", CLAPEmbeddingExtractor())
specs = registry.list_specs()  # All specs from all extractors
```

### Storage Plan
```python
paths = EmbeddingPaths(
    root_dir="/output",
    audio_dirname="audio_embeddings",
    text_dirname="text_embeddings",
)
plan = build_storage_plan(paths, specs)
# {
#   "clap_audio": "/output/audio_embeddings",
#   "clap_text": "/output/text_embeddings",
# }
```

## Future Extension: Multi-Embedding Support

When supporting multiple embedding models:
1. Keep trainers **unchanged**
2. Extend `processed_dataset.csv` with additional columns:
   - `audio_emb_filepath_v2`, `text_emb_filepath_v2` (new models)
3. Update `HATRDataset` to select which embedding columns to load
4. Batch extraction remains **external** (same pattern)

**Key**: Change CSV selection **before** changing trainers.
