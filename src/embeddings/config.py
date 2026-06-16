from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class EmbeddingConfig:
    enabled: bool
    extractors: List[str]
    audio_dirname: str
    text_dirname: str
