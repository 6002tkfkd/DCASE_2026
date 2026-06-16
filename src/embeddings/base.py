from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Protocol


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    modality: str  # "audio" or "text"
    dim: Optional[int] = None


class BaseEmbeddingExtractor(Protocol):
    """Abstract interface for embedding extraction."""

    def specs(self) -> Iterable[EmbeddingSpec]:
        ...

    def extract_audio(self, input_path: str) -> Dict[str, object]:
        ...

    def extract_text(self, input_path: str) -> Dict[str, object]:
        ...
