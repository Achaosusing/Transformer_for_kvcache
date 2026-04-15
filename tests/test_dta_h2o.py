"""Tests for DTA-H2O components that don't require model/transformers."""
from __future__ import annotations

import sys
import types

# Stub out heavy dependencies so we can import src.api without transformers.
_stub_transformers = types.ModuleType("transformers")
_stub_transformers.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})
_stub_transformers.AutoTokenizer = type("AutoTokenizer", (), {})
sys.modules.setdefault("transformers", _stub_transformers)

import torch

from src.chat_format import ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_USER
from src.methods.dta_h2o import DTAH2OPolicy


# We import GhostBuffer after stubbing transformers.
from src.api import GhostBuffer, OracleKVProjectAPI


# ---------------------------------------------------------------------------
# 1. Temporal decay accumulation
# ---------------------------------------------------------------------------

def test_dta_accumulate_exponential_decay() -> None:
    """After N accumulation steps with gamma=0.9, verify geometric decay."""
    scores = torch.zeros(4, dtype=torch.float32)
    gamma = 0.9

    # Step 0: s = [1, 0, 0, 0]
    attn_0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    OracleKVProjectAPI._accumulate_dta_h2o_scores(scores, attn_0, gamma)
    assert torch.allclose(scores, torch.tensor([1.0, 0.0, 0.0, 0.0]))

    # Step 1: s = gamma * [1,0,0,0] + [0,1,0,0] = [0.9, 1.0, 0, 0]
    attn_1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    OracleKVProjectAPI._accumulate_dta_h2o_scores(scores, attn_1, gamma)
    assert torch.allclose(scores, torch.tensor([0.9, 1.0, 0.0, 0.0]))

    # Step 2: s = gamma * [0.9, 1, 0, 0] + [0, 0, 1, 0] = [0.81, 0.9, 1, 0]
    attn_2 = torch.tensor([0.0, 0.0, 1.0, 0.0])
    OracleKVProjectAPI._accumulate_dta_h2o_scores(scores, attn_2, gamma)
    assert torch.allclose(scores, torch.tensor([0.81, 0.9, 1.0, 0.0]))


def test_dta_gamma_one_equals_vanilla_accumulation() -> None:
    """gamma=1.0 should be equivalent to vanilla H2O accumulation."""
    scores_dta = torch.zeros(3, dtype=torch.float32)
    scores_vanilla = torch.zeros(3, dtype=torch.float32)

    attn = torch.tensor([0.5, 0.3, 0.2])
    for _ in range(5):
        OracleKVProjectAPI._accumulate_dta_h2o_scores(scores_dta, attn, gamma=1.0)
        OracleKVProjectAPI._accumulate_h2o_scores(scores_vanilla, attn)

    assert torch.allclose(scores_dta, scores_vanilla)


# ---------------------------------------------------------------------------
# 2. System anchor protection
# ---------------------------------------------------------------------------

def test_system_anchor_tokens_always_kept() -> None:
    """System-role tokens should be prioritised and kept when they fit in budget."""
    # budget = 2 + 2 + 3 = 7
    # protected = sink[0,1] + recent[8,9] = 4  →  remaining_budget = 3
    # system anchors = tokens [2,3,4] = 3  ≤  remaining_budget = 3  → all fit
    policy = DTAH2OPolicy(
        sink_size=2,
        local_window_size=2,
        heavy_hitter_size=3,
        current_turn_ratio=0.5,
        system_anchor=True,
    )
    total_tokens = 10
    scores = torch.rand(total_tokens)
    # Mark tokens 2,3,4 as ROLE_SYSTEM (outside sink and recent window)
    role_tags = torch.tensor([ROLE_USER] * total_tokens, dtype=torch.int8)
    role_tags[2] = ROLE_SYSTEM
    role_tags[3] = ROLE_SYSTEM
    role_tags[4] = ROLE_SYSTEM
    turn_ids = torch.zeros(total_tokens, dtype=torch.int16)
    turn_ids[5:] = 1  # split into two turns

    keep = policy.select_keep_tensor_tiered(total_tokens, scores, role_tags, turn_ids)
    kept_set = set(keep.tolist())

    # All three system tokens must be kept (they fit within the remaining budget)
    assert 2 in kept_set
    assert 3 in kept_set
    assert 4 in kept_set
    # Sink tokens 0, 1 must be kept
    assert 0 in kept_set
    assert 1 in kept_set
    # Recent tokens 8, 9 must be kept
    assert 8 in kept_set
    assert 9 in kept_set


def test_system_anchor_disabled() -> None:
    """When system_anchor=False, system tokens can be evicted."""
    policy = DTAH2OPolicy(
        sink_size=1,
        local_window_size=1,
        heavy_hitter_size=1,
        current_turn_ratio=1.0,
        system_anchor=False,
    )
    # Budget = 3, total = 6
    total_tokens = 6
    scores = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    role_tags = torch.tensor([ROLE_SYSTEM] * total_tokens, dtype=torch.int8)
    turn_ids = torch.ones(total_tokens, dtype=torch.int16)

    keep = policy.select_keep_tensor_tiered(total_tokens, scores, role_tags, turn_ids)
    # Only 3 tokens should be kept (budget=3)
    assert keep.numel() == 3


def test_system_anchor_respects_cache_budget_when_system_tokens_overflow() -> None:
    """System anchors should stay best-effort and never exceed cache budget."""
    policy = DTAH2OPolicy(
        sink_size=1,
        local_window_size=1,
        heavy_hitter_size=1,
        current_turn_ratio=1.0,
        system_anchor=True,
    )
    total_tokens = 8
    scores = torch.arange(total_tokens, dtype=torch.float32)
    role_tags = torch.tensor(
        [ROLE_USER, ROLE_SYSTEM, ROLE_SYSTEM, ROLE_SYSTEM, ROLE_SYSTEM, ROLE_SYSTEM, ROLE_SYSTEM, ROLE_USER],
        dtype=torch.int8,
    )
    turn_ids = torch.ones(total_tokens, dtype=torch.int16)

    keep = policy.select_keep_tensor_tiered(total_tokens, scores, role_tags, turn_ids)
    kept = set(keep.tolist())

    assert keep.numel() == policy.cache_budget
    assert 0 in kept  # sink
    assert 7 in kept  # recent
    assert 6 in kept  # highest-scoring system anchor wins the final slot


# ---------------------------------------------------------------------------
# 3. Tiered budget split
# ---------------------------------------------------------------------------

def test_tiered_budget_split_respects_ratio() -> None:
    """Verify current_turn_ratio splits HH budget correctly."""
    policy = DTAH2OPolicy(
        sink_size=1,
        local_window_size=1,
        heavy_hitter_size=10,
        current_turn_ratio=0.7,
        system_anchor=False,
    )
    # budget = 1 + 1 + 10 = 12
    # 22 tokens: 11 from turn 1, 11 from turn 2
    # protected: sink (1) + recent (1) = 2
    # remaining_budget = 12 - 2 = 10
    total_tokens = 22
    scores = torch.rand(total_tokens)
    role_tags = torch.tensor([ROLE_USER] * total_tokens, dtype=torch.int8)
    turn_ids = torch.tensor([1] * 11 + [2] * 11, dtype=torch.int16)

    keep = policy.select_keep_tensor_tiered(total_tokens, scores, role_tags, turn_ids)
    assert keep.numel() == 12  # budget is 12

    # Protected: token 0 (sink), token 21 (recent)
    # Candidates: tokens 1..20 (20 candidates)
    # Turn 1 candidates: 1..10 (10), Turn 2 candidates: 11..20 (10)
    # current_budget = floor(10 * 0.7) = 7, history_budget = 3
    kept_turns = turn_ids[keep]
    current_count = (kept_turns == 2).sum().item()
    history_count = (kept_turns == 1).sum().item()
    # Current turn 2: 7 from candidates + 1 recent = 8
    # History turn 1: 3 from candidates + 1 sink = 4
    assert current_count == 8
    assert history_count == 4


def test_tiered_budget_overflow_redistribution() -> None:
    """If one tier has fewer candidates than its budget, overflow to other."""
    policy = DTAH2OPolicy(
        sink_size=1,
        local_window_size=1,
        heavy_hitter_size=10,
        current_turn_ratio=0.8,
        system_anchor=False,
    )
    # budget = 12, 17 tokens: 12 from turn 1, 5 from turn 2 (current)
    # protected: token 0 (sink, turn 1), token 16 (recent, turn 2)
    # candidates: 1..15 (15), remaining_budget = 10
    # turn 1 candidates: 1..11 (11), turn 2 candidates: 12..15 (4)
    # current_budget = floor(10*0.8) = 8, but only 4 current candidates
    # overflow = 4, actual_history = min(2+4, 11) = 6
    total_tokens = 17
    scores = torch.rand(total_tokens)
    role_tags = torch.tensor([ROLE_USER] * total_tokens, dtype=torch.int8)
    turn_ids = torch.tensor([1] * 12 + [2] * 5, dtype=torch.int16)

    keep = policy.select_keep_tensor_tiered(total_tokens, scores, role_tags, turn_ids)
    assert keep.numel() == 12  # budget

    kept_turns = turn_ids[keep]
    current_count = (kept_turns == 2).sum().item()
    history_count = (kept_turns == 1).sum().item()
    # Current: 4 candidates + 1 recent = 5
    # History: 6 candidates + 1 sink = 7
    assert current_count == 5
    assert history_count == 7


# ---------------------------------------------------------------------------
# 4. Ghost buffer
# ---------------------------------------------------------------------------

def test_ghost_buffer_ring_semantics() -> None:
    """Ghost buffer should work as a FIFO ring of fixed capacity."""
    buf = GhostBuffer(capacity=3)
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    role_tags = torch.tensor([0, 1, 2, 1, 2], dtype=torch.int8)
    turn_ids = torch.tensor([1, 1, 2, 2, 2], dtype=torch.int16)

    # Evict tokens 0, 1, 2
    buf.record_eviction(scores, role_tags, turn_ids, [0, 1, 2])
    assert len(buf.entries) == 3
    assert buf.entries[0].score == 1.0
    assert buf.entries[2].turn_id == 2

    # Evict token 3 - should overwrite oldest entry (index 0)
    buf.record_eviction(scores, role_tags, turn_ids, [3])
    assert len(buf.entries) == 3
    assert buf.entries[0].score == 4.0  # overwritten


def test_ghost_buffer_anti_cascade_boost() -> None:
    """Ghost buffer should boost tokens from endangered turns."""
    buf = GhostBuffer(capacity=4)
    # Fill buffer with evictions from turn 1 (3 out of 4 = 75% > 25% threshold)
    scores_evicted = torch.tensor([1.0, 1.0, 1.0, 1.0])
    role_tags_evicted = torch.tensor([1, 1, 1, 1], dtype=torch.int8)
    turn_ids_evicted = torch.tensor([1, 1, 1, 2], dtype=torch.int16)
    buf.record_eviction(scores_evicted, role_tags_evicted, turn_ids_evicted, [0, 1, 2, 3])

    # Now check boost for surviving tokens
    current_scores = torch.tensor([1.0, 1.0, 1.0, 1.0])
    current_turns = torch.tensor([1, 1, 2, 2], dtype=torch.int16)
    current_turn = 2

    boost = buf.get_anti_cascade_boost(current_scores, current_turns, current_turn)
    assert boost is not None
    # Turn 1 tokens (indices 0, 1) should get a boost
    assert boost[0].item() > 0
    assert boost[1].item() > 0
    # Turn 2 tokens (current turn) should not get a boost
    assert boost[2].item() == 0
    assert boost[3].item() == 0


def test_ghost_buffer_no_boost_when_empty() -> None:
    """Empty ghost buffer should return None."""
    buf = GhostBuffer(capacity=4)
    scores = torch.tensor([1.0, 2.0])
    turns = torch.tensor([1, 2], dtype=torch.int16)
    assert buf.get_anti_cascade_boost(scores, turns, 2) is None


def test_ghost_buffer_clear() -> None:
    """Clear should reset the buffer."""
    buf = GhostBuffer(capacity=4)
    buf.record_eviction(
        torch.tensor([1.0]),
        torch.tensor([0], dtype=torch.int8),
        torch.tensor([1], dtype=torch.int16),
        [0],
    )
    assert len(buf.entries) == 1
    buf.clear()
    assert len(buf.entries) == 0
    assert buf._ptr == 0


# ---------------------------------------------------------------------------
# 5. Backward compatibility: vanilla H2O behavior
# ---------------------------------------------------------------------------

def test_dta_policy_fallback_to_vanilla_h2o() -> None:
    """With current_turn_ratio=1.0 and all same turn, matches vanilla H2O."""
    from src.methods.h2o import H2OPolicy

    vanilla = H2OPolicy(sink_size=2, local_window_size=3, heavy_hitter_size=5)
    dta = DTAH2OPolicy(
        sink_size=2,
        local_window_size=3,
        heavy_hitter_size=5,
        current_turn_ratio=1.0,
        system_anchor=False,
    )

    total_tokens = 20
    scores = torch.rand(total_tokens)
    role_tags = torch.tensor([ROLE_USER] * total_tokens, dtype=torch.int8)
    # All same turn -> all budget goes to current turn -> same as vanilla
    turn_ids = torch.ones(total_tokens, dtype=torch.int16)

    keep_vanilla = vanilla.select_keep_tensor(total_tokens, scores)
    keep_dta = dta.select_keep_tensor_tiered(total_tokens, scores, role_tags, turn_ids)

    assert keep_vanilla.numel() == keep_dta.numel()
    assert torch.equal(keep_vanilla, keep_dta)


# ---------------------------------------------------------------------------
# 6. Policy under budget (no eviction needed)
# ---------------------------------------------------------------------------

def test_tiered_no_eviction_when_under_budget() -> None:
    """When total_tokens <= cache_budget, keep everything."""
    policy = DTAH2OPolicy(
        sink_size=2, local_window_size=3, heavy_hitter_size=5)
    total_tokens = 8  # <= 10 budget
    scores = torch.rand(total_tokens)
    role_tags = torch.tensor([ROLE_USER] * total_tokens, dtype=torch.int8)
    turn_ids = torch.ones(total_tokens, dtype=torch.int16)

    keep = policy.select_keep_tensor_tiered(total_tokens, scores, role_tags, turn_ids)
    assert keep.numel() == total_tokens
    assert torch.equal(keep, torch.arange(total_tokens))


def test_tiered_empty_tokens() -> None:
    """Edge case: 0 tokens."""
    policy = DTAH2OPolicy(sink_size=2, local_window_size=3, heavy_hitter_size=5)
    scores = torch.tensor([])
    role_tags = torch.tensor([], dtype=torch.int8)
    turn_ids = torch.tensor([], dtype=torch.int16)

    keep = policy.select_keep_tensor_tiered(0, scores, role_tags, turn_ids)
    assert keep.numel() == 0
