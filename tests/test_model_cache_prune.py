from __future__ import annotations

import torch

from src.model import LocalTransformerModel


class _DummyModel:
    device = "cpu"


class _CacheWithLists:
    def __init__(self, key_cache, value_cache) -> None:
        self.key_cache = key_cache
        self.value_cache = value_cache


def _make_model() -> LocalTransformerModel:
    model = LocalTransformerModel.__new__(LocalTransformerModel)
    model.model = _DummyModel()
    return model


def test_prune_dynamic_cache_skips_none_entries() -> None:
    model = _make_model()

    key0 = torch.randn(1, 2, 5, 4)
    val0 = torch.randn(1, 2, 5, 4)
    cache = _CacheWithLists(
        key_cache=[key0.clone(), None],
        value_cache=[val0.clone(), None],
    )

    pruned = model.prune_past_key_values(cache, [0, 2, 4])

    assert pruned.key_cache[0].shape[2] == 3
    assert pruned.value_cache[0].shape[2] == 3
    assert pruned.key_cache[1] is None
    assert pruned.value_cache[1] is None


def test_prune_dynamic_cache_filters_out_of_range_indices() -> None:
    model = _make_model()

    key0 = torch.randn(1, 2, 4, 4)
    val0 = torch.randn(1, 2, 4, 4)
    cache = _CacheWithLists(
        key_cache=[key0.clone()],
        value_cache=[val0.clone()],
    )

    pruned = model.prune_past_key_values(cache, [0, 7, 8])

    assert pruned.key_cache[0].shape[2] == 1
    assert pruned.value_cache[0].shape[2] == 1


def test_prune_legacy_cache_supports_none_entries() -> None:
    model = _make_model()

    key0 = torch.randn(1, 2, 6, 4)
    legacy_cache = ((key0.clone(), None, "meta"),)

    pruned = model.prune_past_key_values(legacy_cache, [1, 3, 5])

    assert isinstance(pruned, tuple)
    assert pruned[0][0].shape[2] == 3
    assert pruned[0][1] is None
    assert pruned[0][2] == "meta"
