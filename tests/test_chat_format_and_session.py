from __future__ import annotations

import torch

from api_server import _build_response_tool_calls
from src.api import OracleKVProjectAPI
from src.chat_format import extract_tool_calls_from_text, format_canonical_chat


def test_canonical_chat_keeps_history_prefix_stable() -> None:
    base_messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Hello"},
    ]
    assistant_message = {"role": "assistant", "content": "Hi there."}
    next_user = {"role": "user", "content": "Please continue."}

    prompt_text = format_canonical_chat(
        base_messages,
        add_generation_prompt=True,
    )
    history_text = format_canonical_chat(
        base_messages + [assistant_message],
        add_generation_prompt=False,
    )
    next_prompt_text = format_canonical_chat(
        base_messages + [assistant_message, next_user],
        add_generation_prompt=True,
    )

    assert history_text.startswith(prompt_text + "Hi there.")
    assert next_prompt_text.startswith(
        history_text + "<|im_start|>user\nPlease continue.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_extract_tool_calls_and_build_openai_response_shape() -> None:
    tool_text = (
        "I will check that.\n\n"
        "<tool_call>\n"
        "<function=lookup_order>\n"
        "<parameter=order_id>\n123\n</parameter>\n"
        "<parameter=payload>\n{\"region\": \"us\"}\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    content, tool_calls = extract_tool_calls_from_text(tool_text)
    response_tool_calls = _build_response_tool_calls(tool_calls)

    assert content == "I will check that."
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "lookup_order"
    assert tool_calls[0]["function"]["arguments"] == {
        "order_id": 123,
        "payload": {"region": "us"},
    }
    assert response_tool_calls is not None
    assert response_tool_calls[0]["type"] == "function"
    assert response_tool_calls[0]["function"]["name"] == "lookup_order"
    assert response_tool_calls[0]["function"]["arguments"] == (
        '{"order_id": 123, "payload": {"region": "us"}}'
    )


def test_apply_max_normalized_h2o_decay() -> None:
    scores = torch.tensor([2.0, 4.0, 1.0], dtype=torch.float32)

    decayed = OracleKVProjectAPI.apply_max_normalized_h2o_decay(scores, 0.5)

    assert torch.allclose(
        decayed,
        torch.tensor([0.25, 0.5, 0.125], dtype=torch.float32),
    )


def test_apply_max_normalized_h2o_decay_handles_all_zero_scores() -> None:
    scores = torch.zeros(3, dtype=torch.float32)

    decayed = OracleKVProjectAPI.apply_max_normalized_h2o_decay(scores, 0.5)

    assert torch.equal(decayed, scores)