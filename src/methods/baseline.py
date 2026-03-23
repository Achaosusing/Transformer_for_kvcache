from __future__ import annotations

import numpy as np


class BaselineFullAttentionPolicy:
    name = "baseline"

    def select_keep_indices(
        self,
        total_tokens: int,
        cumulative_scores: np.ndarray,
    ) -> list[int]:
        if total_tokens <= 0:
            return []
        return list(range(total_tokens))
