from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class EmbeddingPaths:
    root_dir: str
    audio_dirname: str
    text_dirname: str

    def audio_root(self) -> str:
        return f"{self.root_dir}/{self.audio_dirname}"

    def text_root(self) -> str:
        return f"{self.root_dir}/{self.text_dirname}"


def build_storage_plan(paths: EmbeddingPaths, specs) -> Dict[str, str]:
    """Return storage roots keyed by embedding spec name."""
    plan: Dict[str, str] = {}
    for spec in specs:
        if spec.modality == "audio":
            plan[spec.name] = paths.audio_root()
        elif spec.modality == "text":
            plan[spec.name] = paths.text_root()
    return plan
