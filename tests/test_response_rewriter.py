# SPDX-License-Identifier: Apache-2.0
"""Unit tests for response_rewriter.rewrite_request.

Pure-function tests — no aiohttp, no proxy, no vLLM. The proxy plumbing
is exercised by integration tests; this file pins the request-side
budget-rescue rule that prevents the OpenCode/Nemotron monologue regression.

Failure mode being prevented (see docs/`vllm-swift OpenCode max_tokens Trap`):
when a reasoning model receives `max_tokens=8192` from a client like
OpenCode, the `<think>` block can eat the whole budget; vLLM truncates,
`</think>` never closes, and the parser dumps raw thinking into `content`.
The rewriter silently bumps `max_tokens` to a reasoning-safe floor so
the model has headroom for a real answer / tool_call.
"""
from __future__ import annotations

import asyncio

import pytest

from vllm_swift.response_rewriter import (
    _REASONING_MAX_TOKENS_BUMP,
    _REASONING_MAX_TOKENS_FLOOR,
    rewrite_request,
    stream_rewriter,
)

REASONING_PARSER = "nemotron_v3"
NON_REASONING_PARSER = ""


def test_bumps_starved_max_tokens_for_reasoning_parser():
    body = {"max_tokens": 8192, "messages": []}
    out = rewrite_request(body, arch="NemotronHForCausalLM",
                          reasoning_parser=REASONING_PARSER)
    assert out["max_tokens"] == _REASONING_MAX_TOKENS_BUMP


def test_leaves_generous_max_tokens_alone():
    body = {"max_tokens": 65536, "messages": []}
    out = rewrite_request(body, arch="NemotronHForCausalLM",
                          reasoning_parser=REASONING_PARSER)
    assert out["max_tokens"] == 65536


def test_leaves_max_tokens_at_floor_alone():
    """Boundary: floor itself is considered acceptable, not starved."""
    body = {"max_tokens": _REASONING_MAX_TOKENS_FLOOR, "messages": []}
    out = rewrite_request(body, arch="NemotronHForCausalLM",
                          reasoning_parser=REASONING_PARSER)
    assert out["max_tokens"] == _REASONING_MAX_TOKENS_FLOOR


def test_no_bump_when_max_tokens_absent():
    body = {"messages": []}
    out = rewrite_request(body, arch="NemotronHForCausalLM",
                          reasoning_parser=REASONING_PARSER)
    assert "max_tokens" not in out


def test_no_bump_for_non_reasoning_parser():
    """Tight budget on a non-reasoning model is the client's choice."""
    body = {"max_tokens": 256, "messages": []}
    out = rewrite_request(body, arch="LlamaForCausalLM",
                          reasoning_parser=NON_REASONING_PARSER)
    assert out["max_tokens"] == 256


def test_no_bump_for_unknown_reasoning_parser_name():
    body = {"max_tokens": 256, "messages": []}
    out = rewrite_request(body, arch="WeirdoForCausalLM",
                          reasoning_parser="not_a_real_parser")
    assert out["max_tokens"] == 256


@pytest.mark.parametrize("parser", [
    "nemotron_v3", "qwen3", "deepseek_r1", "deepseek_v3",
    "openai_gptoss", "gemma4", "granite", "minimax_m2",
])
def test_all_known_reasoning_parsers_trigger_bump(parser: str):
    body = {"max_tokens": 8192}
    out = rewrite_request(body, arch="", reasoning_parser=parser)
    assert out["max_tokens"] == _REASONING_MAX_TOKENS_BUMP, (
        f"reasoning parser {parser!r} should trigger budget bump"
    )


def test_zero_max_tokens_left_alone():
    """max_tokens=0 is meaningless; don't paper over it, let vLLM reject."""
    body = {"max_tokens": 0}
    out = rewrite_request(body, arch="", reasoning_parser=REASONING_PARSER)
    assert out["max_tokens"] == 0


def test_negative_max_tokens_left_alone():
    body = {"max_tokens": -1}
    out = rewrite_request(body, arch="", reasoning_parser=REASONING_PARSER)
    assert out["max_tokens"] == -1


# ---------------------------------------------------------------------------
# Streaming rewriter — usage-chunk preservation
# ---------------------------------------------------------------------------


def _ssestream(events: list[str]) -> bytes:
    """Encode a list of SSE event payloads into a single byte stream."""
    return b"".join(f"data: {e}\n\n".encode() for e in events)


def _drive_rewriter(blob: bytes, arch: str) -> bytes:
    """Run stream_rewriter end-to-end on `blob`, return the joined output.

    Wraps the async machinery in `asyncio.run` so this stays a plain
    synchronous test — no pytest-asyncio dependency.
    """
    async def _async_iter_bytes():
        yield blob

    async def _collect():
        out = []
        async for chunk in stream_rewriter(_async_iter_bytes(), arch):
            out.append(chunk)
        return b"".join(out)

    return asyncio.run(_collect())


def test_stream_rewriter_preserves_usage_chunk():
    """Regression: vLLM emits the final `usage` block in a chunk with
    `choices: []`. Earlier rewriter versions silently dropped these
    metadata-only chunks because the per-choice loop never ran and the
    `if new_choices:` guard skipped the yield. Without this fix, Hermes'
    context counter never advances because token totals never reach the
    client.
    """
    content_chunk = (
        '{"id":"x","object":"chat.completion.chunk","created":1,'
        '"model":"m","choices":[{"index":0,"delta":{"content":"hi"},'
        '"finish_reason":"stop"}]}'
    )
    usage_chunk = (
        '{"id":"x","object":"chat.completion.chunk","created":1,'
        '"model":"m","choices":[],'
        '"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}'
    )
    blob = _ssestream([content_chunk, usage_chunk, "[DONE]"])
    joined = _drive_rewriter(blob, arch="NemotronH").decode()
    assert "usage" in joined, "usage chunk dropped by rewriter"
    assert '"prompt_tokens":10' in joined
    assert '"completion_tokens":2' in joined
    assert "[DONE]" in joined


def test_stream_rewriter_passes_through_for_non_rewrite_arch():
    """Sanity: non-Nemotron arches go through the fast-path passthrough
    branch (no JSON parsing at all)."""
    payload = b"data: anything goes\n\n"
    out = _drive_rewriter(payload, arch="LlamaForCausalLM")
    assert out == payload
