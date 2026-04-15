from __future__ import annotations

import torch

from ..chat_format import ROLE_SYSTEM
from .h2o import H2OPolicy


class DTAH2OPolicy(H2OPolicy):
    """Dynamic Temporal-Aware H2O with tiered eviction and bounded system anchors."""

    name = "dta_h2o"

    def __init__(
        self,
        sink_size: int,
        local_window_size: int,
        heavy_hitter_size: int,
        current_turn_ratio: float = 0.6,
        system_anchor: bool = True,
        ghost_buffer_size: int = 32,
    ) -> None:
        super().__init__(sink_size, local_window_size, heavy_hitter_size)
        if not (0.0 <= current_turn_ratio <= 1.0):
            raise ValueError("current_turn_ratio must be in [0, 1]")
        self.current_turn_ratio = current_turn_ratio
        self.system_anchor = system_anchor
        self.ghost_buffer_size = ghost_buffer_size

    def select_keep_tensor_tiered(
        self,
        total_tokens: int,
        cumulative_scores: torch.Tensor,
        role_tags: torch.Tensor,
        turn_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Tiered eviction with strict cache-budget enforcement."""
        device = cumulative_scores.device
        if total_tokens <= 0:
            return torch.empty(0, device=device, dtype=torch.long)
        if total_tokens <= self.cache_budget:
            return torch.arange(total_tokens, device=device, dtype=torch.long)

        indices = torch.arange(total_tokens, device=device, dtype=torch.long)

        # 1. Build the always-kept mask: sink | recent.
        sink_end = min(self.sink_size, total_tokens)
        recent_start = max(sink_end, total_tokens - self.local_window_size)

        protected = torch.zeros(total_tokens, device=device, dtype=torch.bool)
        if sink_end > 0:
            protected[:sink_end] = True
        if total_tokens > recent_start:
            protected[recent_start:] = True

        protected_indices = indices[protected]
        remaining_budget = max(0, self.cache_budget - protected_indices.numel())
        if remaining_budget == 0:
            return protected_indices.sort().values

        # 2. Reserve budget for system anchors before generic HH selection.
        anchor_indices = torch.empty(0, device=device, dtype=torch.long)
        if self.system_anchor and role_tags is not None:
            anchor_mask = (role_tags == ROLE_SYSTEM) & ~protected
            anchor_indices = indices[anchor_mask]

        selected_anchor_indices = torch.empty(0, device=device, dtype=torch.long)
        if anchor_indices.numel() > 0:
            if anchor_indices.numel() <= remaining_budget:
                selected_anchor_indices = anchor_indices
                remaining_budget -= anchor_indices.numel()
            else:
                anchor_scores = cumulative_scores[anchor_indices]
                topk_rel = self._topk_with_recent_tiebreak_torch(
                    anchor_scores,
                    remaining_budget,
                )
                selected_anchor_indices = anchor_indices[topk_rel]
                return torch.cat([protected_indices, selected_anchor_indices]).sort().values

        # 3. Candidate tokens = not protected and not already reserved as anchors.
        candidate_mask = ~protected
        if selected_anchor_indices.numel() > 0:
            candidate_mask[selected_anchor_indices] = False
        candidate_indices = indices[candidate_mask]
        if candidate_indices.numel() == 0:
            return torch.cat([protected_indices, selected_anchor_indices]).sort().values
        if candidate_indices.numel() <= remaining_budget:
            return torch.cat(
                [protected_indices, selected_anchor_indices, candidate_indices]
            ).sort().values

        # 4. Split candidates by current turn vs history.
        current_turn = turn_ids.max()
        cand_turn_ids = turn_ids[candidate_indices]
        is_current = cand_turn_ids == current_turn

        current_cand_indices = candidate_indices[is_current]
        history_cand_indices = candidate_indices[~is_current]

        # 5. Compute tiered budgets with overflow.
        current_budget = int(remaining_budget * self.current_turn_ratio)
        history_budget = remaining_budget - current_budget

        actual_current_k = min(current_budget, current_cand_indices.numel())
        actual_history_k = min(history_budget, history_cand_indices.numel())

        # Redistribute overflow
        current_overflow = current_budget - actual_current_k
        history_overflow = history_budget - actual_history_k
        if current_overflow > 0 and history_cand_indices.numel() > actual_history_k:
            actual_history_k = min(
                actual_history_k + current_overflow,
                history_cand_indices.numel(),
            )
        if history_overflow > 0 and current_cand_indices.numel() > actual_current_k:
            actual_current_k = min(
                actual_current_k + history_overflow,
                current_cand_indices.numel(),
            )

        # 6. Top-k selection per tier.
        parts: list[torch.Tensor] = [protected_indices]
        if selected_anchor_indices.numel() > 0:
            parts.append(selected_anchor_indices)

        if actual_current_k > 0 and current_cand_indices.numel() > 0:
            if actual_current_k >= current_cand_indices.numel():
                parts.append(current_cand_indices)
            else:
                cand_scores = cumulative_scores[current_cand_indices]
                topk_rel = self._topk_with_recent_tiebreak_torch(cand_scores, actual_current_k)
                parts.append(current_cand_indices[topk_rel])

        if actual_history_k > 0 and history_cand_indices.numel() > 0:
            if actual_history_k >= history_cand_indices.numel():
                parts.append(history_cand_indices)
            else:
                cand_scores = cumulative_scores[history_cand_indices]
                topk_rel = self._topk_with_recent_tiebreak_torch(cand_scores, actual_history_k)
                parts.append(history_cand_indices[topk_rel])

        return torch.cat(parts).sort().values
