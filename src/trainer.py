from __future__ import annotations

from typing import Any


class Trainer:
    """Placeholder trainer for future supervised fine-tuning workflows."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def train(self) -> None:
        # This project currently focuses on inference-time KV retention evaluation.
        # Keep the trainer entry so the project layout remains extensible.
        print("Trainer placeholder: no training pipeline is implemented yet.")
