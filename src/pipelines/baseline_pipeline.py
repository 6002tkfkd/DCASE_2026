import os
from typing import Any, Dict, Optional

from src.utils.config_utils import load_config
from src.trainers.baseline_trainer import BaselineTrainer


def run(config: Optional[Dict[str, Any]] = None) -> None:
    """Baseline pipeline entrypoint (baseline-only migration)."""
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
                "baseline.yaml",
            )
        )
        config = load_config(base_path, strategy_path)

    trainer = BaselineTrainer(config)
    trainer.run()
