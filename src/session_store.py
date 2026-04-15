"""LRU session store and signature helpers for H2O multi-turn sessions."""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .api import H2ORuntimeState
from .chat_format import ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_TOOL, ROLE_USER


@dataclass
class H2OChatSessionSnapshot:
    session_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    history_token_ids: list[int]
    runtime_state: H2ORuntimeState
    signature: tuple[Any, ...]


class LRUSessionStore:
    """Bounded LRU store for H2O session snapshots with automatic prefix matching.

    Supports two lookup modes:
    1. Exact key lookup via ``get(key)`` — used when the client provides a
       ``session_id``.
    2. Automatic token-prefix lookup via ``find_by_prefix(prompt_ids, signature)``
       — scans all snapshots and returns the one whose ``history_token_ids`` is
       the longest prefix of *prompt_ids* (with matching signature).  This makes
       multi-turn session reuse work transparently for any OpenAI-compatible
       client that does NOT send ``session_id``.
    """

    _PREFIX_HASH_LEN = 64  # number of leading tokens used for the fast hash index

    def __init__(self, max_size: int) -> None:
        self._store: OrderedDict[str, H2OChatSessionSnapshot] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()
        # Fast index: hash of the first _PREFIX_HASH_LEN token ids -> set of store keys.
        self._prefix_index: dict[str, set[str]] = {}

    @staticmethod
    def _hash_token_prefix(token_ids: list[int], length: int = 64) -> str:
        """Deterministic hash of the first *length* token ids."""
        prefix = token_ids[:length]
        raw = ",".join(map(str, prefix))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, key: str) -> H2OChatSessionSnapshot | None:
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return self._store[key]

    def set(self, key: str, value: H2OChatSessionSnapshot) -> None:
        with self._lock:
            # Remove old index entry if updating an existing key.
            if key in self._store:
                old_snap = self._store[key]
                old_h = self._hash_token_prefix(
                    old_snap.history_token_ids, self._PREFIX_HASH_LEN,
                )
                bucket = self._prefix_index.get(old_h)
                if bucket is not None:
                    bucket.discard(key)
                    if not bucket:
                        del self._prefix_index[old_h]
                self._store.move_to_end(key)

            self._store[key] = value

            # Add to prefix index.
            h = self._hash_token_prefix(
                value.history_token_ids, self._PREFIX_HASH_LEN,
            )
            self._prefix_index.setdefault(h, set()).add(key)

            # Evict oldest entries if over capacity.
            while len(self._store) > self._max_size:
                evicted_key, evicted_snap = self._store.popitem(last=False)
                eh = self._hash_token_prefix(
                    evicted_snap.history_token_ids, self._PREFIX_HASH_LEN,
                )
                bucket = self._prefix_index.get(eh)
                if bucket is not None:
                    bucket.discard(evicted_key)
                    if not bucket:
                        del self._prefix_index[eh]

    def find_by_prefix(
        self,
        prompt_ids: list[int],
        signature: tuple[Any, ...],
    ) -> H2OChatSessionSnapshot | None:
        """Find the best snapshot whose ``history_token_ids`` is a prefix of
        *prompt_ids* and whose signature matches.

        Uses a two-stage strategy:
        1. Hash the first *_PREFIX_HASH_LEN* tokens of *prompt_ids* and check
           the index for candidate keys (O(1) average).
        2. Among candidates (or all entries on hash miss), verify full prefix
           match and pick the longest.

        Returns ``None`` if no matching snapshot is found.
        """
        with self._lock:
            if not self._store:
                return None

            prompt_len = len(prompt_ids)
            h = self._hash_token_prefix(prompt_ids, self._PREFIX_HASH_LEN)
            candidate_keys = self._prefix_index.get(h)

            if candidate_keys:
                candidates = [
                    (k, self._store[k])
                    for k in candidate_keys
                    if k in self._store
                ]
            else:
                # Fallback: scan all entries (handles cases where prompt is
                # shorter than _PREFIX_HASH_LEN or hash bucket was pruned).
                candidates = list(self._store.items())

            best: H2OChatSessionSnapshot | None = None
            best_len = 0

            for _key, snap in candidates:
                if snap.signature != signature:
                    continue
                hist_len = len(snap.history_token_ids)
                if hist_len > prompt_len:
                    continue
                if hist_len <= best_len:
                    continue
                if prompt_ids[:hist_len] == snap.history_token_ids:
                    best = snap
                    best_len = hist_len

            if best is not None:
                self._store.move_to_end(best.session_id)

            return best


def _freeze_signature_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_signature_value(val))
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, list):
        return tuple(_freeze_signature_value(item) for item in value)
    return value


def hash_signature(signature: tuple[Any, ...]) -> str:
    return hashlib.sha256(repr(signature).encode()).hexdigest()[:12]


def build_h2o_session_signature(
    method_key: str,
    method_cfg: dict[str, Any],
    *,
    evict_period: int,
    collect_period: int,
    alpha: float,
    role_alphas: dict[int, float],
) -> tuple[Any, ...]:
    resolved_cfg: dict[str, Any] = {
        **method_cfg,
        "sink_size": int(method_cfg.get("sink_size", 4)),
        "local_window_size": int(method_cfg.get("local_window_size", 256)),
        "heavy_hitter_size": int(method_cfg.get("heavy_hitter_size", 128)),
        "evict_period": int(evict_period),
        "collect_period": int(collect_period),
        "session_score_alpha": float(alpha),
        "role_alpha_system": float(role_alphas[ROLE_SYSTEM]),
        "role_alpha_user": float(role_alphas[ROLE_USER]),
        "role_alpha_assistant": float(role_alphas[ROLE_ASSISTANT]),
        "role_alpha_tool": float(role_alphas[ROLE_TOOL]),
    }
    if method_key == "dta_h2o":
        resolved_cfg.update(
            {
                "dta_gamma": float(method_cfg.get("dta_gamma", 0.95)),
                "current_turn_ratio": float(
                    method_cfg.get("current_turn_ratio", 0.6),
                ),
                "system_anchor": bool(method_cfg.get("system_anchor", True)),
                "ghost_buffer_size": int(
                    method_cfg.get("ghost_buffer_size", 32),
                ),
            }
        )
    return (
        method_key,
        tuple(
            (str(key), _freeze_signature_value(value))
            for key, value in sorted(
                resolved_cfg.items(), key=lambda item: str(item[0]),
            )
        ),
    )
