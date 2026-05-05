"""Silent response rewriter for poorly-behaved clients + models.

Two classes of fixes, both invisible to the client:

  Request-side:
    * **max_tokens starvation rescue** — when a reasoning parser is in
      play and the client hardcodes a small `max_tokens` (e.g. OpenCode's
      8192), the `<think>` block can eat the whole budget; vLLM then
      truncates mid-thought, `</think>` never closes, and the parser dumps
      raw thinking into `content`. We silently raise `max_tokens` to a
      reasoning-safe floor so the model can finish thinking and still
      have headroom for a real answer / tool_call.

  Response-side:
    * **Thinking: prefix split** — for models that ignore their own
      `<think>` contract (notably Nemotron-Cascade-2 which writes
      'Thinking:' plaintext), split that prefix out of `content` into
      `reasoning_content`. Defense-in-depth — the request-side fix is
      what actually prevents the failure mode, this just cleans up
      stragglers.

User UX is invisible: same vllm-swift CLI invocation, same port, same
OpenAI-compatible API. No retries, no warnings, no client config.
Diagnostics go to ~/.vllm-swift/debug.log only.

Architecture:
  - vllm-swift CLI launches vLLM on an internal port (user_port + 1000).
  - This proxy listens on the user-facing port and forwards every
    /v1/chat/completions request through the rewriter.
  - All other paths pass through unchanged.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import AsyncIterator

import aiohttp
from aiohttp import web

LOG_PATH = Path(os.path.expanduser("~/.vllm-swift/debug.log"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("vllm_swift.rewriter")
if not logger.handlers:
    handler = logging.FileHandler(LOG_PATH)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Architecture prefixes whose models empirically emit 'Thinking:'
# plaintext instead of `<think>` tags. Triggers the response-side
# Thinking: prefix splitter only — request-side max_tokens rescue is
# gated on the reasoning parser, not the arch.
_REWRITE_ARCH_PREFIXES: tuple[str, ...] = (
    "NemotronH",
    "Nemotron",
)


# Reasoning parsers known to expand a `<think>` block that can starve a
# small max_tokens budget. If the proxy was launched with one of these
# parsers AND the client sent a max_tokens below the safe floor, we
# silently bump it. Empirical: OpenCode hardcodes 8192, Nemotron burns
# 4–6K tokens in <think> on tool-use turns with 23K-char system prompts,
# so the </think> never closes and the parser dumps raw thinking into
# content (manifests as monologue / hallucinated tool output).
_REASONING_PARSERS_NEEDING_BUDGET: frozenset[str] = frozenset({
    "nemotron_v3",
    "qwen3",
    "deepseek_r1",
    "deepseek_v3",
    "openai_gptoss",
    "gemma4",
    "granite",
    "minimax_m2",
})

# Floor below which we consider a reasoning request budget-starved.
# Above this, trust the client. Picked to comfortably cover the typical
# 4–6K thinking burst plus 8K headroom for tool_call + final answer.
_REASONING_MAX_TOKENS_FLOOR = 16384

# What we bump to when starved. Stays well under typical 32K context for
# Apple Silicon-friendly models so we don't blow KV cache budgets.
_REASONING_MAX_TOKENS_BUMP = 32768


# Regex for the 'Thinking:' / 'Thinking ' / 'thinking:' prefix variants.
# Capture group 1 is the prefix label; rest is the actual thinking body.
_THINKING_PREFIX_RE = re.compile(r"^(?:Thinking|thinking)\s*[:\-]\s*", re.IGNORECASE)


# Transition markers that indicate the end of the thinking block. The first
# occurrence of any of these in the content (after the prefix) marks the
# boundary between reasoning and the actual answer.
_THINKING_END_MARKERS: tuple[str, ...] = (
    "\n#",  # markdown header
    "\n$",  # shell prompt / command
    "\n<",  # any XML-like tag (tool calls etc)
    "\n```",  # code block
    "\n\n",  # paragraph break
    "\nHere ",  # common answer opener
    "\nThe ",  # common answer opener
    "\nI ",  # first-person answer opener
)


def needs_rewrite(arch: str) -> bool:
    if not arch:
        return False
    return any(arch.startswith(p) for p in _REWRITE_ARCH_PREFIXES)


def split_thinking_prefix(content: str) -> tuple[str, str]:
    """Return (reasoning_content, cleaned_content).

    If `content` begins with a 'Thinking:' prefix, returns the thinking
    text and the remaining answer separately. Otherwise returns ("",
    content) unchanged.
    """
    if not content:
        return ("", content)
    match = _THINKING_PREFIX_RE.match(content)
    if not match:
        return ("", content)
    after_prefix = content[match.end():]
    earliest = len(after_prefix)
    for marker in _THINKING_END_MARKERS:
        idx = after_prefix.find(marker)
        if 0 <= idx < earliest:
            earliest = idx
    if earliest == len(after_prefix):
        return (after_prefix.strip(), "")
    reasoning = after_prefix[:earliest].strip()
    answer = after_prefix[earliest:].lstrip()
    return (reasoning, answer)


def rewrite_chat_completion(payload: dict) -> dict:
    """Rewrite a non-streaming /v1/chat/completions response in place."""
    choices = payload.get("choices", [])
    for choice in choices:
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        existing_reasoning = msg.get("reasoning_content") or ""
        if not existing_reasoning and content:
            reasoning, cleaned = split_thinking_prefix(content)
            if reasoning:
                msg["reasoning_content"] = reasoning
                msg["content"] = cleaned
                choice["message"] = msg
                logger.info("split Thinking: prefix; %d chars reasoning, %d chars content",
                            len(reasoning), len(cleaned))
    return payload


def rewrite_request(body: dict, arch: str, reasoning_parser: str = "") -> dict:
    """Rewrite an outgoing /v1/chat/completions request in place.

    Currently does one thing: silently bumps `max_tokens` when the proxy
    is fronting a reasoning parser AND the client sent a budget below
    `_REASONING_MAX_TOKENS_FLOOR`. Prevents the OpenCode/Nemotron failure
    mode where 8192 tokens starve the `<think>` block, vLLM truncates,
    `</think>` never closes, and the parser dumps raw thinking into
    `content` as a monologue.

    `arch` is unused here today; kept in the signature so the response-
    side rewriter can stay arch-gated without a second plumbing pass.
    """
    del arch  # arch-gated rewrites all live on the response side
    if reasoning_parser not in _REASONING_PARSERS_NEEDING_BUDGET:
        return body
    requested = body.get("max_tokens")
    if isinstance(requested, int) and 0 < requested < _REASONING_MAX_TOKENS_FLOOR:
        body["max_tokens"] = _REASONING_MAX_TOKENS_BUMP
        logger.info(
            "bumped max_tokens %d -> %d (reasoning_parser=%s, client-side starvation)",
            requested, _REASONING_MAX_TOKENS_BUMP, reasoning_parser,
        )
    return body


# =============================================================================
# Streaming SSE rewriter
# =============================================================================

# Streaming state per connection. We buffer the very first content delta(s)
# of each choice until we can determine whether they begin with a
# "Thinking:" prefix. Once we know, we either:
#   (a) NO prefix: forward all buffered + subsequent deltas as content.
#   (b) Prefix detected: route deltas to reasoning_content until we see
#       a transition marker, then switch back to content for the answer.

_BUFFER_THRESHOLD_CHARS = 32  # enough to disambiguate "Thinking:" prefix


class _ChoiceState:
    __slots__ = ("buffer", "decided", "in_reasoning", "transition_seen")

    def __init__(self) -> None:
        self.buffer = ""
        self.decided = False
        self.in_reasoning = False
        self.transition_seen = False


def _split_first_transition(text: str) -> tuple[str, str] | None:
    """If `text` contains a transition marker, return (before, after).
    Otherwise return None."""
    earliest = len(text)
    for marker in _THINKING_END_MARKERS:
        idx = text.find(marker)
        if 0 <= idx < earliest:
            earliest = idx
    if earliest == len(text):
        return None
    return (text[:earliest], text[earliest:].lstrip())


def _make_delta_event(idx: int, role: str | None, reasoning: str | None,
                      content: str | None, original_chunk: dict) -> str:
    """Build an SSE chunk preserving original metadata (id, created, model)."""
    new_chunk = {
        "id": original_chunk.get("id"),
        "object": original_chunk.get("object", "chat.completion.chunk"),
        "created": original_chunk.get("created"),
        "model": original_chunk.get("model"),
        "choices": [
            {
                "index": idx,
                "delta": {
                    **({"role": role} if role else {}),
                    **({"reasoning_content": reasoning} if reasoning is not None else {}),
                    **({"content": content} if content is not None else {}),
                },
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(new_chunk)}\n\n"


async def stream_rewriter(
    upstream_iter: AsyncIterator[bytes], arch: str
) -> AsyncIterator[bytes]:
    """Rewrite an SSE stream. Yields rewritten SSE chunks."""
    if not needs_rewrite(arch):
        async for chunk in upstream_iter:
            yield chunk
        return

    # Per-choice state
    states: dict[int, _ChoiceState] = {}
    pending_text = b""
    async for raw in upstream_iter:
        pending_text += raw
        # Split into SSE events on \n\n boundaries
        while b"\n\n" in pending_text:
            event_bytes, pending_text = pending_text.split(b"\n\n", 1)
            event = event_bytes.decode("utf-8", errors="replace")
            if not event.startswith("data: "):
                yield (event + "\n\n").encode()
                continue
            payload_text = event[len("data: "):]
            if payload_text.strip() == "[DONE]":
                # Flush any remaining buffered content as content (no transition seen)
                for idx, st in states.items():
                    if not st.decided and st.buffer:
                        yield _make_delta_event(
                            idx, None, None, st.buffer, {}
                        ).encode()
                yield event.encode() + b"\n\n"
                continue
            try:
                chunk = json.loads(payload_text)
            except json.JSONDecodeError:
                yield (event + "\n\n").encode()
                continue

            # Process each choice in this chunk
            new_choices = []
            for c in chunk.get("choices", []):
                idx = c.get("index", 0)
                st = states.setdefault(idx, _ChoiceState())
                delta = c.get("delta", {})
                content_delta = delta.get("content")
                role = delta.get("role")
                # If upstream already emitted reasoning_content, pass through.
                if delta.get("reasoning_content"):
                    new_choices.append(c)
                    continue
                if content_delta is None:
                    new_choices.append(c)
                    continue
                # Buffering / decision logic
                if not st.decided:
                    st.buffer += content_delta
                    if len(st.buffer) >= _BUFFER_THRESHOLD_CHARS or c.get("finish_reason"):
                        st.decided = True
                        match = _THINKING_PREFIX_RE.match(st.buffer)
                        if match:
                            st.in_reasoning = True
                            after_prefix = st.buffer[match.end():]
                            split = _split_first_transition(after_prefix)
                            if split:
                                # Boundary already in buffer
                                reasoning_part, content_part = split
                                if reasoning_part.strip():
                                    new_choices.append({
                                        "index": idx,
                                        "delta": {**({"role": role} if role else {}),
                                                  "reasoning_content": reasoning_part.strip()},
                                        "finish_reason": None,
                                    })
                                if content_part:
                                    new_choices.append({
                                        "index": idx,
                                        "delta": {"content": content_part},
                                        "finish_reason": c.get("finish_reason"),
                                    })
                                st.in_reasoning = False
                                st.transition_seen = True
                            else:
                                # All buffered text is reasoning so far
                                if after_prefix:
                                    new_choices.append({
                                        "index": idx,
                                        "delta": {**({"role": role} if role else {}),
                                                  "reasoning_content": after_prefix},
                                        "finish_reason": None,
                                    })
                            st.buffer = ""
                        else:
                            # No prefix; flush as content
                            new_choices.append({
                                "index": idx,
                                "delta": {**({"role": role} if role else {}),
                                          "content": st.buffer},
                                "finish_reason": c.get("finish_reason"),
                            })
                            st.buffer = ""
                    # else: still buffering; emit no delta this round
                else:
                    # Already decided
                    if st.in_reasoning and not st.transition_seen:
                        # Check if this delta contains a transition
                        split = _split_first_transition(content_delta)
                        if split:
                            reasoning_part, content_part = split
                            if reasoning_part:
                                new_choices.append({
                                    "index": idx,
                                    "delta": {"reasoning_content": reasoning_part},
                                    "finish_reason": None,
                                })
                            if content_part:
                                new_choices.append({
                                    "index": idx,
                                    "delta": {"content": content_part},
                                    "finish_reason": c.get("finish_reason"),
                                })
                            st.transition_seen = True
                            st.in_reasoning = False
                        else:
                            new_choices.append({
                                "index": idx,
                                "delta": {"reasoning_content": content_delta},
                                "finish_reason": c.get("finish_reason"),
                            })
                    else:
                        # In content mode
                        new_choices.append({
                            "index": idx,
                            "delta": {"content": content_delta},
                            "finish_reason": c.get("finish_reason"),
                        })

            if new_choices:
                rewritten = {**chunk, "choices": new_choices}
                yield f"data: {json.dumps(rewritten)}\n\n".encode()
    if pending_text:
        yield pending_text


# =============================================================================
# aiohttp proxy
# =============================================================================


async def _make_app(upstream_url: str, arch: str, reasoning_parser: str = "") -> web.Application:
    timeout = aiohttp.ClientTimeout(total=600)

    async def proxy(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        is_chat = "chat/completions" in request.path and request.method == "POST"
        if is_chat and body:
            try:
                req_payload = json.loads(body.decode())
                req_payload = rewrite_request(req_payload, arch, reasoning_parser)
                body = json.dumps(req_payload).encode()
            except Exception as exc:
                logger.warning("request rewrite failed, forwarding raw: %s", exc)

        url = f"{upstream_url}{request.path}"
        if request.query_string:
            url += f"?{request.query_string}"
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length")}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                request.method, url, data=body if body else None,
                headers=headers, allow_redirects=False,
            ) as upstream_resp:
                ct = upstream_resp.headers.get("content-type", "")
                if is_chat and "text/event-stream" in ct:
                    response = web.StreamResponse(
                        status=upstream_resp.status,
                        headers={"Content-Type": ct, "Cache-Control": "no-cache"},
                    )
                    await response.prepare(request)
                    async for out_chunk in stream_rewriter(
                        upstream_resp.content.iter_any(), arch
                    ):
                        await response.write(out_chunk)
                    await response.write_eof()
                    return response
                if is_chat and "application/json" in ct:
                    raw = await upstream_resp.read()
                    try:
                        payload = json.loads(raw.decode())
                        rewrite_chat_completion(payload)
                        new_body = json.dumps(payload).encode()
                        return web.Response(
                            status=upstream_resp.status,
                            body=new_body,
                            headers={"Content-Type": "application/json"},
                        )
                    except Exception as exc:
                        logger.warning("response rewrite failed, returning raw: %s", exc)
                        return web.Response(status=upstream_resp.status, body=raw,
                                            headers=dict(upstream_resp.headers))
                # Pass-through for everything else
                response = web.StreamResponse(status=upstream_resp.status,
                                              headers=dict(upstream_resp.headers))
                await response.prepare(request)
                async for ch in upstream_resp.content.iter_any():
                    await response.write(ch)
                await response.write_eof()
                return response

    app = web.Application(client_max_size=50 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


def run(user_port: int, upstream_port: int, arch: str,
        reasoning_parser: str = "") -> None:
    """Run the rewriter proxy. Blocks until killed."""
    upstream_url = f"http://127.0.0.1:{upstream_port}"
    logger.info(
        "rewriter starting: user_port=%d upstream=%s arch=%s reasoning_parser=%s",
        user_port, upstream_url, arch, reasoning_parser or "<none>",
    )
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = loop.run_until_complete(_make_app(upstream_url, arch, reasoning_parser))
    web.run_app(app, host="127.0.0.1", port=user_port,
                print=lambda *_: None, loop=loop)
