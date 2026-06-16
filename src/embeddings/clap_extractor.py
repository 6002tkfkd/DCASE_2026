from typing import Dict, Iterable

from .base import BaseEmbeddingExtractor, EmbeddingSpec


class CLAPEmbeddingExtractor:
    """Placeholder CLAP extractor interface (implementation deferred)."""

    def specs(self) -> Iterable[EmbeddingSpec]:
        return [
            EmbeddingSpec(name="clap_audio", modality="audio", dim=512),
            EmbeddingSpec(name="clap_text", modality="text", dim=512),
        ]

    def extract_audio(self, input_path: str) -> Dict[str, object]:
        raise NotImplementedError("CLAP audio extraction is not implemented yet.")

    def extract_text(self, input_path: str) -> Dict[str, object]:
        raise NotImplementedError("CLAP text extraction is not implemented yet.")
