"""Tests for the optional `--retrieval-endpoint` flag (longctx-svc).

The retrieval companion is OPTIONAL: when the flag is absent and the
`LONGCTX_ENDPOINT` env var is unset, vllm-swift behaves exactly as it
did before. When set, every chat-completion goes through the existing
transparent rewriter, which calls longctx-svc, splices retrieved
chunks into a system message, and forwards to vLLM.

We test:
  1. CLI flag parsing (positional, =-form, env-var fallback, removal
     from passthrough so vLLM never sees it).
  2. needs_proxy is forced True when the endpoint is set.
  3. _enrich_with_longctx splices chunks into messages on success.
  4. _enrich_with_longctx is a no-op on network failure (optional tool
     guarantee).
  5. _format_longctx_block round-trip + system-message merge ordering.
"""

from __future__ import annotations

import asyncio

from vllm_swift.cli import (
    _extract_enable_longctx,
    _extract_longctx_scope,
    _extract_retrieval_endpoint,
)
from vllm_swift.response_rewriter import (
    _enrich_with_longctx,
    _flatten_prefill,
    _format_longctx_block,
    _last_user_text,
    _splice_longctx_into_messages,
)

# ---------------------------------------------------------------------------
# CLI flag parsing
# ---------------------------------------------------------------------------


def test_extract_retrieval_endpoint_space_form(monkeypatch):
    monkeypatch.delenv("LONGCTX_ENDPOINT", raising=False)
    url, rest = _extract_retrieval_endpoint(
        [
            "--port",
            "8000",
            "--retrieval-endpoint",
            "http://localhost:8765",
            "--max-model-len",
            "4096",
        ]
    )
    assert url == "http://localhost:8765"
    assert rest == ["--port", "8000", "--max-model-len", "4096"]


def test_extract_retrieval_endpoint_equals_form(monkeypatch):
    monkeypatch.delenv("LONGCTX_ENDPOINT", raising=False)
    url, rest = _extract_retrieval_endpoint(
        [
            "--retrieval-endpoint=http://h:8765",
            "--port",
            "8000",
        ]
    )
    assert url == "http://h:8765"
    assert rest == ["--port", "8000"]


def test_extract_retrieval_endpoint_env_fallback(monkeypatch):
    monkeypatch.setenv("LONGCTX_ENDPOINT", "http://env-host:9999")
    url, rest = _extract_retrieval_endpoint(["--port", "8000"])
    assert url == "http://env-host:9999"
    assert rest == ["--port", "8000"]


def test_extract_retrieval_endpoint_missing(monkeypatch):
    """Tool optional: no flag, no env → empty url, args untouched."""
    monkeypatch.delenv("LONGCTX_ENDPOINT", raising=False)
    url, rest = _extract_retrieval_endpoint(["--port", "8000"])
    assert url == ""
    assert rest == ["--port", "8000"]


def test_extract_retrieval_endpoint_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("LONGCTX_ENDPOINT", "http://env:1")
    url, _ = _extract_retrieval_endpoint(
        [
            "--retrieval-endpoint",
            "http://flag:2",
        ]
    )
    assert url == "http://flag:2"


# ---------------------------------------------------------------------------
# --enable-longctx flag (auto-spawn sidecar)
# ---------------------------------------------------------------------------


def test_extract_enable_longctx_default_off(monkeypatch):
    """Tool optional: flag absent + env unset → False, no auto-spawn."""
    monkeypatch.delenv("LONGCTX_ENABLE", raising=False)
    enabled, rest = _extract_enable_longctx(["--port", "8000"])
    assert enabled is False
    assert rest == ["--port", "8000"]


def test_extract_enable_longctx_flag_on(monkeypatch):
    monkeypatch.delenv("LONGCTX_ENABLE", raising=False)
    enabled, rest = _extract_enable_longctx(
        [
            "--enable-longctx",
            "--port",
            "8000",
        ]
    )
    assert enabled is True
    assert rest == ["--port", "8000"]


def test_extract_enable_longctx_env(monkeypatch):
    monkeypatch.setenv("LONGCTX_ENABLE", "1")
    enabled, rest = _extract_enable_longctx(["--port", "8000"])
    assert enabled is True
    assert rest == ["--port", "8000"]


def test_extract_enable_longctx_no_disables(monkeypatch):
    """--no-enable-longctx wins over env var (explicit user opt-out)."""
    monkeypatch.setenv("LONGCTX_ENABLE", "1")
    enabled, rest = _extract_enable_longctx(
        [
            "--no-enable-longctx",
            "--port",
            "8000",
        ]
    )
    assert enabled is False
    assert rest == ["--port", "8000"]


# ---------------------------------------------------------------------------
# --longctx-scope flag (auto-fallback for tool-using agents)
# ---------------------------------------------------------------------------


def test_extract_longctx_scope_default_empty(monkeypatch):
    """No flag, no env → empty string. cli.py supplies cwd at boot when
    --enable-longctx is on; this helper just parses."""
    monkeypatch.delenv("LONGCTX_DEFAULT_SCOPE", raising=False)
    scope, rest = _extract_longctx_scope(["--port", "8000"])
    assert scope == ""
    assert rest == ["--port", "8000"]


def test_extract_longctx_scope_space_form(monkeypatch):
    monkeypatch.delenv("LONGCTX_DEFAULT_SCOPE", raising=False)
    scope, rest = _extract_longctx_scope(
        [
            "--port",
            "8000",
            "--longctx-scope",
            "/Users/x/dev/myapp",
            "--max-model-len",
            "4096",
        ]
    )
    assert scope == "/Users/x/dev/myapp"
    assert rest == ["--port", "8000", "--max-model-len", "4096"]


def test_extract_longctx_scope_equals_form(monkeypatch):
    monkeypatch.delenv("LONGCTX_DEFAULT_SCOPE", raising=False)
    scope, rest = _extract_longctx_scope(
        [
            "--longctx-scope=/abs/path",
            "--port",
            "8000",
        ]
    )
    assert scope == "/abs/path"
    assert rest == ["--port", "8000"]


def test_extract_longctx_scope_env_fallback(monkeypatch):
    monkeypatch.setenv("LONGCTX_DEFAULT_SCOPE", "/from/env")
    scope, rest = _extract_longctx_scope(["--port", "8000"])
    assert scope == "/from/env"


def test_extract_longctx_scope_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("LONGCTX_DEFAULT_SCOPE", "/from/env")
    scope, _ = _extract_longctx_scope(
        [
            "--longctx-scope",
            "/from/flag",
        ]
    )
    assert scope == "/from/flag"


# ---------------------------------------------------------------------------
# default_scope passed to /retrieve when set
# ---------------------------------------------------------------------------


def test_enrich_forwards_default_scope_when_set():
    """The fallback path: when default_scope is provided, every call
    forwards it so longctx-svc can fall back when no path is mentioned."""
    body = {
        "messages": [
            {"role": "user", "content": "what does the app do?"},  # no path
        ]
    }
    payload = {
        "chunks": [
            {
                "text": "function hi() {}",
                "file_path": "/Users/x/dev/myapp/index.ts",
                "start_line": 1,
                "end_line": 1,
                "score": 0.5,
            }
        ],
        "scope_path": "/Users/x/dev/myapp",
        "scope_status": "ready",
        "session_id": None,
    }
    sess = _FakeSession(_FakeAioResp(200, payload))
    body2, hdrs = asyncio.run(
        _enrich_with_longctx(
            body,
            endpoint="http://h:8765",
            request_headers={},
            aiohttp_session=sess,
            default_scope="/Users/x/dev/myapp",
        )
    )
    assert sess.captured["json"].get("default_scope") == "/Users/x/dev/myapp"
    msgs = body2["messages"]
    assert msgs[0]["role"] == "system"
    assert "Retrieved code context" in msgs[0]["content"]
    assert hdrs["x-longctx-chunks-used"] == "1"


def test_enrich_omits_default_scope_when_unset():
    """Default behavior: no default_scope key in the body."""
    body = {
        "messages": [
            {"role": "user", "content": "see /Users/x/auth.ts"},
        ]
    }
    payload = {"chunks": [], "scope_status": "no-scope", "session_id": None}
    sess = _FakeSession(_FakeAioResp(200, payload))
    asyncio.run(
        _enrich_with_longctx(
            body,
            endpoint="http://h:8765",
            request_headers={},
            aiohttp_session=sess,
            # default_scope omitted
        )
    )
    assert "default_scope" not in sess.captured["json"]


# ---------------------------------------------------------------------------
# Helpers: splice + format
# ---------------------------------------------------------------------------


def test_format_longctx_block_includes_path_and_lines():
    chunks = [
        {
            "text": "x = 1\n",
            "file_path": "/p/a.py",
            "start_line": 1,
            "end_line": 1,
        }
    ]
    block = _format_longctx_block(chunks)
    assert "/p/a.py:1-1" in block
    assert "x = 1" in block
    assert "## Retrieved code context" in block


def test_splice_into_existing_system_message_prepends():
    msgs = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
    ]
    out = _splice_longctx_into_messages(msgs, "BLOCK\n\n")
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "BLOCK\n\nbe helpful"
    assert out[1]["content"] == "hi"


def test_splice_inserts_system_when_absent():
    msgs = [{"role": "user", "content": "hi"}]
    out = _splice_longctx_into_messages(msgs, "BLOCK")
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "BLOCK"
    assert out[1]["content"] == "hi"


def test_splice_handles_list_content():
    """OpenAI vision-style content arrays: stay as arrays after splice."""
    msgs = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "be helpful"},
            ],
        },
        {"role": "user", "content": "hi"},
    ]
    out = _splice_longctx_into_messages(msgs, "BLOCK")
    assert isinstance(out[0]["content"], list)
    assert out[0]["content"][0] == {"type": "text", "text": "BLOCK"}


def test_last_user_text_string_form():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    assert _last_user_text(msgs) == "second"


def test_last_user_text_list_form():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url"},
            ],
        }
    ]
    assert _last_user_text(msgs) == "hello"


def test_flatten_prefill_includes_role_tags():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    out = _flatten_prefill(msgs)
    assert "[system]" in out and "sys" in out
    assert "[user]" in out and "usr" in out


# ---------------------------------------------------------------------------
# _enrich_with_longctx — optional behavior
# ---------------------------------------------------------------------------


class _FakeAioResp:
    def __init__(self, status: int, payload: dict):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, resp: _FakeAioResp | None = None, raise_exc: Exception | None = None):
        self._resp = resp
        self._raise = raise_exc
        self.captured: dict = {}

    def post(self, url, json=None, headers=None, timeout=None):
        self.captured.update({"url": url, "json": json, "headers": headers})
        if self._raise is not None:
            raise self._raise
        return self._resp


def test_enrich_returns_unchanged_on_empty_messages():
    body = {"messages": []}
    body2, hdrs = asyncio.run(
        _enrich_with_longctx(
            body,
            endpoint="http://x:1",
            request_headers={},
            aiohttp_session=_FakeSession(),
        )
    )
    assert body2 is body
    assert hdrs == {}


def test_enrich_returns_unchanged_when_no_user_msg():
    body = {"messages": [{"role": "system", "content": "be helpful"}]}
    body2, hdrs = asyncio.run(
        _enrich_with_longctx(
            body,
            endpoint="http://x:1",
            request_headers={},
            aiohttp_session=_FakeSession(),
        )
    )
    assert body2 is body
    assert hdrs == {}


def test_enrich_splices_on_success():
    body = {
        "messages": [
            {"role": "user", "content": "explain authMiddleware in /Users/tom/p/auth.ts"},
        ],
    }
    payload = {
        "chunks": [
            {
                "text": "function authMiddleware() {}",
                "file_path": "/Users/tom/p/auth.ts",
                "start_line": 1,
                "end_line": 1,
                "score": 0.9,
            }
        ],
        "scope_path": "/Users/tom/p",
        "scope_status": "ready",
        "session_id": "sess-1",
    }
    sess = _FakeSession(_FakeAioResp(200, payload))
    body2, hdrs = asyncio.run(
        _enrich_with_longctx(
            body,
            endpoint="http://h:8765",
            request_headers={"x-session-affinity": "sess-1"},
            aiohttp_session=sess,
        )
    )
    assert sess.captured["url"] == "http://h:8765/retrieve"
    assert sess.captured["json"]["query"].startswith("explain")
    assert sess.captured["headers"]["x-session-affinity"] == "sess-1"
    msgs = body2["messages"]
    assert msgs[0]["role"] == "system"
    assert "Retrieved code context" in msgs[0]["content"]
    assert "/Users/tom/p/auth.ts:1-1" in msgs[0]["content"]
    assert hdrs["x-longctx-chunks-used"] == "1"
    assert hdrs["x-longctx-scope"] == "/Users/tom/p"
    assert hdrs["x-longctx-scope-status"] == "ready"
    assert hdrs["x-longctx-session"] == "sess-1"


def test_enrich_silent_on_network_failure():
    """Tool optional: if longctx-svc is down, request flows through."""
    body = {
        "messages": [
            {"role": "user", "content": "see /Users/x/foo.py please"},
        ]
    }
    sess = _FakeSession(raise_exc=RuntimeError("connection refused"))
    body2, hdrs = asyncio.run(
        _enrich_with_longctx(
            body,
            endpoint="http://h:8765",
            request_headers={},
            aiohttp_session=sess,
        )
    )
    # Body untouched, debug header records the failure type
    assert body2["messages"] == body["messages"]
    assert hdrs.get("x-longctx-error") == "RuntimeError"


def test_enrich_silent_on_non_200():
    body = {"messages": [{"role": "user", "content": "auth /Users/x/y.py"}]}
    sess = _FakeSession(_FakeAioResp(503, {}))
    body2, hdrs = asyncio.run(
        _enrich_with_longctx(
            body,
            endpoint="http://h:8765",
            request_headers={},
            aiohttp_session=sess,
        )
    )
    assert body2["messages"] == body["messages"]
    assert hdrs == {"x-longctx-error": "status=503"}


def test_enrich_no_chunks_no_splice():
    """Empty chunks (no scope) → message list unchanged but headers
    still record the visit."""
    body = {"messages": [{"role": "user", "content": "what is 2+2?"}]}
    payload = {
        "chunks": [],
        "scope_path": None,
        "scope_status": "no-scope",
        "session_id": None,
    }
    sess = _FakeSession(_FakeAioResp(200, payload))
    body2, hdrs = asyncio.run(
        _enrich_with_longctx(
            body,
            endpoint="http://h:8765",
            request_headers={},
            aiohttp_session=sess,
        )
    )
    assert body2["messages"] == body["messages"]
    assert hdrs["x-longctx-chunks-used"] == "0"
    assert hdrs["x-longctx-scope-status"] == "no-scope"


# ---------------------------------------------------------------------------
# v0.5.2 alpha-tester regressions (bugs #6, #3, #7)
# ---------------------------------------------------------------------------
# Buddy ran v0.5.1 alpha and found these. Lock them down so they don't
# come back. Bugs #1 (vllm not declared), #2 (max_model_len docs), #5
# (decode decay) are infra/docs/Metal-side and tested elsewhere or via
# the live repro flow.


def test_enrich_filters_chunks_below_relevance_floor():
    """Bug #6: trivial query → 8 chunks of irrelevant code spliced in,
    prompt_tokens=5423 for "say hello". Floor at 0.20 drops noise."""
    body = {
        "messages": [
            {"role": "user", "content": "say hello in one short sentence"},
        ]
    }
    payload = {
        "chunks": [
            {
                "text": "irrelevant",
                "file_path": "/p/a.py",
                "start_line": 1,
                "end_line": 50,
                "score": 0.05,
            },
            {
                "text": "also irrelevant",
                "file_path": "/p/b.py",
                "start_line": 1,
                "end_line": 50,
                "score": 0.10,
            },
        ],
        "scope_path": "/p",
        "scope_status": "ready",
        "session_id": None,
    }
    sess = _FakeSession(_FakeAioResp(200, payload))
    body2, hdrs = asyncio.run(
        _enrich_with_longctx(
            body,
            endpoint="http://h:8765",
            request_headers={},
            aiohttp_session=sess,
        )
    )
    # All chunks below 0.20 floor → no splice
    assert body2["messages"] == body["messages"]
    assert hdrs["x-longctx-chunks-used"] == "0"


def test_enrich_keeps_chunks_at_or_above_floor():
    """Counter to the above — when the best chunk is decent, splice it."""
    body = {
        "messages": [
            {"role": "user", "content": "explain authMiddleware"},
        ]
    }
    payload = {
        "chunks": [
            {
                "text": "function authMiddleware() {}",
                "file_path": "/p/a.py",
                "start_line": 1,
                "end_line": 1,
                "score": 0.45,
            },
            {
                "text": "noise",
                "file_path": "/p/b.py",
                "start_line": 1,
                "end_line": 1,
                "score": 0.05,
            },
        ],
        "scope_path": "/p",
        "scope_status": "ready",
        "session_id": None,
    }
    sess = _FakeSession(_FakeAioResp(200, payload))
    body2, hdrs = asyncio.run(
        _enrich_with_longctx(
            body,
            endpoint="http://h:8765",
            request_headers={},
            aiohttp_session=sess,
        )
    )
    assert hdrs["x-longctx-chunks-used"] == "1"
    assert "authMiddleware" in body2["messages"][0]["content"]


def test_enrich_relevance_floor_overridable_by_env(monkeypatch):
    """LONGCTX_RELEVANCE_FLOOR env tunes the threshold per-deployment."""
    monkeypatch.setenv("LONGCTX_RELEVANCE_FLOOR", "0.50")
    body = {"messages": [{"role": "user", "content": "x"}]}
    payload = {
        "chunks": [
            {
                "text": "borderline",
                "file_path": "/p/a.py",
                "start_line": 1,
                "end_line": 1,
                "score": 0.40,
            },
        ],
        "scope_path": "/p",
        "scope_status": "ready",
        "session_id": None,
    }
    sess = _FakeSession(_FakeAioResp(200, payload))
    _, hdrs = asyncio.run(
        _enrich_with_longctx(
            body,
            endpoint="http://h:8765",
            request_headers={},
            aiohttp_session=sess,
        )
    )
    # 0.40 < 0.50 floor → dropped
    assert hdrs["x-longctx-chunks-used"] == "0"


# ---------------------------------------------------------------------------
# Bug #3: rewrite_request must not bump explicit small max_tokens
# ---------------------------------------------------------------------------


def test_rewrite_request_honors_explicit_small_max_tokens():
    """Buddy sent max_tokens=64, got completion_tokens=20480 because the
    reasoning bump fired regardless. Below 1024 = explicit user intent."""
    from vllm_swift.response_rewriter import rewrite_request

    body = {
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": 64,
    }
    out = rewrite_request(body, arch="qwen3", reasoning_parser="qwen3", max_model_len=40960)
    assert out["max_tokens"] == 64, "explicit small max_tokens was bumped"


def test_rewrite_request_still_bumps_default_starvation_budget():
    """The bump was added for a reason — OpenCode-style 4K-8K defaults
    starve reasoning models. Make sure that case still bumps."""
    from vllm_swift.response_rewriter import rewrite_request

    body = {
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 4096,  # OpenCode default; well below floor
    }
    out = rewrite_request(body, arch="qwen3", reasoning_parser="qwen3", max_model_len=40960)
    assert out["max_tokens"] > 4096, "OpenCode-style default should bump"


def test_rewrite_request_bypasses_when_no_reasoning_parser():
    from vllm_swift.response_rewriter import rewrite_request

    body = {
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 64,
    }
    out = rewrite_request(body, arch="qwen3", reasoning_parser="", max_model_len=40960)
    assert out["max_tokens"] == 64


# ---------------------------------------------------------------------------
# Bug #7: normalize message.reasoning → message.reasoning_content
# ---------------------------------------------------------------------------


def test_rewrite_chat_completion_normalizes_reasoning_field():
    """Some vLLM versions emit `message.reasoning` instead of the
    OpenAI-standard `message.reasoning_content`. Normalize on the way
    out so OpenAI clients (Hermes, openai-python, etc.) see the
    expected field."""
    from vllm_swift.response_rewriter import rewrite_chat_completion

    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "the answer is 42",
                    "reasoning": "thought about it. then more thinking.",
                },
                "finish_reason": "stop",
            }
        ],
    }
    rewrite_chat_completion(payload)
    msg = payload["choices"][0]["message"]
    assert msg["reasoning_content"] == "thought about it. then more thinking."
    # back-compat: original `reasoning` field preserved
    assert msg["reasoning"] == "thought about it. then more thinking."


def test_rewrite_chat_completion_leaves_reasoning_content_alone():
    """When upstream already produces the standard field, we don't
    duplicate-write or change anything."""
    from vllm_swift.response_rewriter import rewrite_chat_completion

    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "reasoning_content": "already standard",
                },
                "finish_reason": "stop",
            }
        ],
    }
    rewrite_chat_completion(payload)
    msg = payload["choices"][0]["message"]
    assert msg["reasoning_content"] == "already standard"


# ---------------------------------------------------------------------------
# Bug #2: pre-flight max_model_len > max_position_embeddings warning
# ---------------------------------------------------------------------------


def test_warn_when_max_model_len_exceeds_model_cap(tmp_path, capsys):
    """Buddy hit: --max-model-len 65536 against a model with
    max_position_embeddings=40960. vLLM rejects later — we should warn
    upfront with the actual numbers."""
    import json

    from vllm_swift.cli import _warn_if_max_model_len_exceeds_model

    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "max_position_embeddings": 40960,
            }
        )
    )
    _warn_if_max_model_len_exceeds_model(str(model_dir), 65536)
    err = capsys.readouterr().err
    assert "65536" in err
    assert "40960" in err
    assert "Recommend" in err


def test_no_warn_when_max_model_len_within_cap(tmp_path, capsys):
    import json

    from vllm_swift.cli import _warn_if_max_model_len_exceeds_model

    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "max_position_embeddings": 65536,
            }
        )
    )
    _warn_if_max_model_len_exceeds_model(str(model_dir), 32768)
    assert capsys.readouterr().err == ""


def test_no_warn_when_config_missing(tmp_path, capsys):
    """A missing config.json (HF cache layout, etc.) shouldn't crash —
    we just skip the check silently."""
    from vllm_swift.cli import _warn_if_max_model_len_exceeds_model

    _warn_if_max_model_len_exceeds_model(str(tmp_path / "no-such"), 65536)
    assert capsys.readouterr().err == ""
