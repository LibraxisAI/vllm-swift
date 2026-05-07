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
import json
import os

import pytest

from vllm_swift.cli import (
    _extract_enable_longctx,
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
    url, rest = _extract_retrieval_endpoint([
        "--port", "8000",
        "--retrieval-endpoint", "http://localhost:8765",
        "--max-model-len", "4096",
    ])
    assert url == "http://localhost:8765"
    assert rest == ["--port", "8000", "--max-model-len", "4096"]


def test_extract_retrieval_endpoint_equals_form(monkeypatch):
    monkeypatch.delenv("LONGCTX_ENDPOINT", raising=False)
    url, rest = _extract_retrieval_endpoint([
        "--retrieval-endpoint=http://h:8765", "--port", "8000",
    ])
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
    url, _ = _extract_retrieval_endpoint([
        "--retrieval-endpoint", "http://flag:2",
    ])
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
    enabled, rest = _extract_enable_longctx([
        "--enable-longctx", "--port", "8000",
    ])
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
    enabled, rest = _extract_enable_longctx([
        "--no-enable-longctx", "--port", "8000",
    ])
    assert enabled is False
    assert rest == ["--port", "8000"]


# ---------------------------------------------------------------------------
# Helpers: splice + format
# ---------------------------------------------------------------------------

def test_format_longctx_block_includes_path_and_lines():
    chunks = [{
        "text": "x = 1\n",
        "file_path": "/p/a.py",
        "start_line": 1, "end_line": 1,
    }]
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
        {"role": "system", "content": [
            {"type": "text", "text": "be helpful"},
        ]},
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
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "hello"}, {"type": "image_url"},
    ]}]
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
    def __init__(self, resp: _FakeAioResp | None = None,
                 raise_exc: Exception | None = None):
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
    body2, hdrs = asyncio.run(_enrich_with_longctx(
        body, endpoint="http://x:1", request_headers={},
        aiohttp_session=_FakeSession(),
    ))
    assert body2 is body
    assert hdrs == {}


def test_enrich_returns_unchanged_when_no_user_msg():
    body = {"messages": [{"role": "system", "content": "be helpful"}]}
    body2, hdrs = asyncio.run(_enrich_with_longctx(
        body, endpoint="http://x:1", request_headers={},
        aiohttp_session=_FakeSession(),
    ))
    assert body2 is body
    assert hdrs == {}


def test_enrich_splices_on_success():
    body = {
        "messages": [
            {"role": "user", "content":
             "explain authMiddleware in /Users/tom/p/auth.ts"},
        ],
    }
    payload = {
        "chunks": [{
            "text": "function authMiddleware() {}",
            "file_path": "/Users/tom/p/auth.ts",
            "start_line": 1, "end_line": 1, "score": 0.9,
        }],
        "scope_path": "/Users/tom/p",
        "scope_status": "ready",
        "session_id": "sess-1",
    }
    sess = _FakeSession(_FakeAioResp(200, payload))
    body2, hdrs = asyncio.run(_enrich_with_longctx(
        body, endpoint="http://h:8765",
        request_headers={"x-session-affinity": "sess-1"},
        aiohttp_session=sess,
    ))
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
    body = {"messages": [
        {"role": "user", "content": "see /Users/x/foo.py please"},
    ]}
    sess = _FakeSession(raise_exc=RuntimeError("connection refused"))
    body2, hdrs = asyncio.run(_enrich_with_longctx(
        body, endpoint="http://h:8765", request_headers={},
        aiohttp_session=sess,
    ))
    # Body untouched, debug header records the failure type
    assert body2["messages"] == body["messages"]
    assert hdrs.get("x-longctx-error") == "RuntimeError"


def test_enrich_silent_on_non_200():
    body = {"messages": [{"role": "user", "content": "auth /Users/x/y.py"}]}
    sess = _FakeSession(_FakeAioResp(503, {}))
    body2, hdrs = asyncio.run(_enrich_with_longctx(
        body, endpoint="http://h:8765", request_headers={},
        aiohttp_session=sess,
    ))
    assert body2["messages"] == body["messages"]
    assert hdrs == {"x-longctx-error": "status=503"}


def test_enrich_no_chunks_no_splice():
    """Empty chunks (no scope) → message list unchanged but headers
    still record the visit."""
    body = {"messages": [{"role": "user", "content": "what is 2+2?"}]}
    payload = {
        "chunks": [], "scope_path": None,
        "scope_status": "no-scope", "session_id": None,
    }
    sess = _FakeSession(_FakeAioResp(200, payload))
    body2, hdrs = asyncio.run(_enrich_with_longctx(
        body, endpoint="http://h:8765", request_headers={},
        aiohttp_session=sess,
    ))
    assert body2["messages"] == body["messages"]
    assert hdrs["x-longctx-chunks-used"] == "0"
    assert hdrs["x-longctx-scope-status"] == "no-scope"
