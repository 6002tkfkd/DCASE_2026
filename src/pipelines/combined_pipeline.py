import os
from typing import Any, Dict, Optional

from src.utils.config_utils import load_config
from src.trainers.combined_trainer import CombinedTrainer


def run(config: Optional[Dict[str, Any]] = None) -> None:
    """Combined pipeline entrypoint (combined-only migration)."""
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
                "combined.yaml",
            )
        )
        config = load_config(base_path, strategy_path)

    trainer = CombinedTrainer(config)
    trainer.run()
