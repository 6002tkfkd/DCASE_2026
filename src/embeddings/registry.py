from typing import Dict, Iterable

from .base import BaseEmbeddingExtractor, EmbeddingSpec


class EmbeddingRegistry:
    """Registry for embedding extractors."""

    def __init__(self) -> None:
        self._extractors: Dict[str, BaseEmbeddingExtractor] = {}

    def register(self, name: str, extractor: BaseEmbeddingExtractor) -> None:
        self._extractors[name] = extractor

    def get(self, name: str) -> BaseEmbeddingExtractor:
        return self._extractors[name]

    def list_specs(self) -> Iterable[EmbeddingSpec]:
        for extractor in self._extractors.values():
            for spec in extractor.specs():
                yield spec
