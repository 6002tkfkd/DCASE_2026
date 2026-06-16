from typing import Dict, Iterable, List, Optional

from .base import BaseEmbeddingExtractor, EmbeddingSpec
from .config import EmbeddingConfig
from .io import EmbeddingPaths, build_storage_plan
from .registry import EmbeddingRegistry


def prepare_embedding_stage(
    config: EmbeddingConfig, registry: EmbeddingRegistry, root_dir: str
) -> Dict[str, str]:
    """Build storage plan for embeddings without executing extraction."""
    if not config.enabled:
        return {}

    paths = EmbeddingPaths(
        root_dir=root_dir,
        audio_dirname=config.audio_dirname,
        text_dirname=config.text_dirname,
    )

    specs: List[EmbeddingSpec] = []
    for name in config.extractors:
        extractor = registry.get(name)
        specs.extend(list(extractor.specs()))

    return build_storage_plan(paths, specs)
