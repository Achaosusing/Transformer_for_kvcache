from __future__ import annotations

from typing import Protocol

import numpy as np


class RetentionPolicy(Protocol):
    def select_keep_indices(
        self,
        total_tokens: int,
        cumulative_scores: np.ndarray,
    ) -> list[int]:
        ...
