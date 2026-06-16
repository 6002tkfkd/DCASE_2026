import os
import shutil
from typing import Any, Dict, Optional

from src.utils.config_utils import load_config
from src.trainers.pretrain_trainer import PretrainTrainer
from src.trainers.finetune_trainer import FinetuneTrainer


def _prepare_finetune_checkpoints() -> None:
    """Expose the shared pretrain checkpoint at the paths finetune already expects."""
    source_path = os.path.join(".", "model_output_pretrain", "pretrained_model.pth")
    target_paths = [
        os.path.join(".", "model_output_pretrain_both", "pretrained_model.pth"),
        os.path.join(".", "model_output_pretrain_audio", "pretrained_model.pth"),
    ]

    if not os.path.exists(source_path):
        return

    for target_path in target_paths:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(source_path, target_path)


def run(config: Optional[Dict[str, Any]] = None) -> None:
    """Two-stage pipeline entrypoint (pretrain + finetune)."""
    if config is None:
        base_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "base.yaml")
        )
        strategy_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "configs",
                "strategy",
                "two_stage.yaml",
            )
        )
        config = load_config(base_path, strategy_path)

    pretrain = PretrainTrainer(config)
    pretrain.run()
    _prepare_finetune_checkpoints()

    finetune = FinetuneTrainer(config)
    finetune.run()
