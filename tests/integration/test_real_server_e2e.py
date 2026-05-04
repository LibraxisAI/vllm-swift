# SPDX-License-Identifier: Apache-2.0
"""Integration: launch vllm-swift against a real model and validate the
OpenAI response shape. This is the runtime ground-truth check that closes
the 5% gap static detection can't see (template-says-it-thinks but model
doesn't, parser picked the wrong format, etc.).

Per-model timing on M5 Max:
- Qwen3-0.6B-4bit boot ~25s, request ~1s
- Llama-3.2-1B-Instruct-4bit boot ~25s, request ~1s

Skips when model not present, and the whole module skips when
`--enable-auto-tool-choice` requires a vLLM that isn't installed in the
current venv. Opt in: `pytest -m integration tests/integration/test_real_server_e2e.py`.
"""
from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request

import pytest

from tests.integration.conftest import _has_local_model


pytestmark = pytest.mark.integration


def _post_json(url: str, body: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


@pytest.fixture(scope="session", autouse=True)
def _require_vllm_swift_cli():
    if shutil.which("vllm-swift") is None:
        pytest.skip("vllm-swift CLI not on PATH; pip install -e . first")


# ---------------------------------------------------------------------------
# Tool calling round-trip — Llama-3.2 (tools, no reasoning)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "vllm_swift_server",
    [{"model": "Llama-3.2-1B-Instruct-hf"}],
    indirect=True,
)
def test_llama_tool_calling_response_shape(vllm_swift_server):
    """With auto-detected llama3_json parser, a tools-bearing request must
    return structured tool_calls — NOT raw text in content."""
    if not _has_local_model("Llama-3.2-1B-Instruct-hf"):
        pytest.skip("model missing")
    body = {
        "model": vllm_swift_server["model_id"],
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Paris? Use the tool.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 200,
    }
    resp = _post_json(vllm_swift_server["base_url"] + "/chat/completions", body)
    msg = resp["choices"][0]["message"]
    # Either tool_calls is populated (model actually called the tool), OR
    # content is non-empty (model answered without calling — both valid for
    # tool_choice=auto). What MUST NOT happen: content contains a JSON-shaped
    # tool call, which signals the parser failed to extract.
    if msg.get("tool_calls"):
        for call in msg["tool_calls"]:
            assert call["type"] == "function"
            assert call["function"]["name"]
            args = call["function"]["arguments"]
            assert isinstance(args, str), "vLLM convention: arguments is a JSON string"
            json.loads(args)  # must be parseable JSON
    else:
        content = msg.get("content") or ""
        assert "{" not in content or "function" not in content.lower(), (
            f"content looks like an unparsed tool call — parser may have failed: {content[:300]}"
        )


# ---------------------------------------------------------------------------
# Reasoning round-trip — Qwen3-0.6B (thinking + tools)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "vllm_swift_server",
    [{"model": "Qwen3-0.6B-hf"}],
    indirect=True,
)
def test_qwen3_reasoning_response_shape(vllm_swift_server):
    """With auto-detected qwen3 reasoning parser, a chat request that
    triggers chain-of-thought must put the CoT in `reasoning_content`,
    not leaked into `content`."""
    if not _has_local_model("Qwen3-0.6B-hf"):
        pytest.skip("model missing")
    body = {
        "model": vllm_swift_server["model_id"],
        "messages": [
            {
                "role": "user",
                "content": "Think step by step: what is 17 × 23?",
            }
        ],
        "max_tokens": 400,
    }
    resp = _post_json(vllm_swift_server["base_url"] + "/chat/completions", body)
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    # The damning failure mode: reasoning is null AND content starts with a
    # CoT preamble (the bug that started this whole feature).
    cot_leaked = (
        not reasoning
        and any(
            preamble in content[:200].lower()
            for preamble in (
                "here's a thinking process",
                "here is a thinking process",
                "<think>",
                "let me think",
            )
        )
    )
    assert not cot_leaked, (
        "reasoning_content is empty AND content starts with CoT preamble — "
        "the reasoning parser failed to extract the thinking block. "
        f"\ncontent[:300]={content[:300]!r}"
    )


# ---------------------------------------------------------------------------
# Negative: a non-thinking model must NOT have reasoning_content populated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "vllm_swift_server",
    [{"model": "Llama-3.2-1B-Instruct-hf"}],
    indirect=True,
)
def test_non_thinking_model_no_reasoning_content(vllm_swift_server):
    """Llama 3.2 isn't a thinking model. We don't auto-inject a reasoning
    parser for it. The response must NOT have a non-empty reasoning_content."""
    if not _has_local_model("Llama-3.2-1B-Instruct-hf"):
        pytest.skip("model missing")
    body = {
        "model": vllm_swift_server["model_id"],
        "messages": [{"role": "user", "content": "Say hello in five words."}],
        "max_tokens": 30,
    }
    resp = _post_json(vllm_swift_server["base_url"] + "/chat/completions", body)
    msg = resp["choices"][0]["message"]
    reasoning = msg.get("reasoning_content")
    assert not reasoning, f"unexpected reasoning_content on non-thinking model: {reasoning!r}"
    assert msg.get("content"), "content must be populated"
