from .registry import EmbeddingRegistry
from .clap_extractor import CLAPEmbeddingExtractor


def build_default_registry() -> EmbeddingRegistry:
    registry = EmbeddingRegistry()
    registry.register("clap", CLAPEmbeddingExtractor())
    return registry
