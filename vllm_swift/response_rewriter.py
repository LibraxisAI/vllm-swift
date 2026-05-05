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
_REASONING_PARSERS_NEEDING_BUDGET: frozenset[str] = frozenset(
    {
        "nemotron_v3",
        "qwen3",
        "deepseek_r1",
        "deepseek_v3",
        "openai_gptoss",
        "gemma4",
        "granite",
        "minimax_m2",
    }
)

# Floor below which we consider a reasoning request budget-starved.
# Above this, trust the client. Picked to comfortably cover the typical
# 4–6K thinking burst plus 8K headroom for tool_call + final answer.
_REASONING_MAX_TOKENS_FLOOR = 16384

# What we bump to when starved. Stays well under typical 32K context for
# Apple Silicon-friendly models so we don't blow KV cache budgets.
_REASONING_MAX_TOKENS_BUMP = 32768

# Tool parsers known to occasionally fail to extract a structured tool_call
# even when the model emitted the right shape (the call ends up as plain
# text in `message.content`). When the auto-detected tool parser is in
# this set, the proxy spawns even for non-reasoning models so the response-
# side recovery path can synthesize the structured tool_call from content.
#   - `phi4_mini_json`: Microsoft's own model card admits Phi-4-mini emits
#     `<|tool_calls|>[{...}]<|/tool_calls|>` as text. Tracked in
#     vllm-project/vllm#14682 / #14359.
_LEAKY_TOOL_PARSERS: frozenset[str] = frozenset(
    {
        "phi4_mini_json",
    }
)


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
    after_prefix = content[match.end() :]
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


# ----------------------------------------------------------------------------
# Tool-call recovery: synthesize structured `tool_calls` when the parser
# misses an extraction and the call leaks into `message.content` as text.
# ----------------------------------------------------------------------------

# Conservative threshold: the matched tool-call shape must occupy at least
# this fraction of the content for recovery to fire. Defends against
# false-positives where a model writes natural-language commentary that
# happens to mention `<tool_call>` etc.
_RECOVERY_MIN_RATIO = 0.5

# Hermes-style: `<tool_call>{json}</tool_call>` (one or more)
_HERMES_BLOCK_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

# qwen3_coder XML: `<tool_call><function=name><parameter=k>v</parameter>...
# </function></tool_call>`
_QWEN3CODER_BLOCK_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_QWEN3CODER_PARAM_RE = re.compile(
    r"<parameter=([^>]+)>(.*?)</parameter>",
    re.DOTALL,
)

# Phi-4-mini pipe-tag: `<|tool_calls|>[{...}]<|/tool_calls|>`
# Microsoft's own model card admits the model emits this as text.
_PHI4_PIPE_RE = re.compile(
    r"<\|tool_calls?\|>\s*(\[.*?\]|\{.*?\})\s*<\|/tool_calls?\|>",
    re.DOTALL,
)

# Mistral bracket: `[TOOL_CALLS][{...}]`
_MISTRAL_BRACKET_RE = re.compile(
    r"\[TOOL_CALLS\]\s*(\[.*?\])",
    re.DOTALL,
)


def _make_tool_call_id(seed: int) -> str:
    """Synthesize a tool_call id when the parser-bypass path didn't generate one."""
    return f"chatcmpl-tool-recovered-{seed:012d}"


def _validate_function_payload(name: str, args: object) -> dict | None:
    """Return a normalized function payload if `name` and `args` look valid."""
    if not name or not isinstance(name, str):
        return None
    if isinstance(args, str):
        # Some leaks already have stringified JSON; accept as-is.
        try:
            json.loads(args)  # validation only
        except json.JSONDecodeError:
            return None
        args_str = args
    elif isinstance(args, dict):
        args_str = json.dumps(args)
    else:
        return None
    return {"name": name, "arguments": args_str}


def _recover_hermes(content: str) -> tuple[list[dict], str] | None:
    """Extract one or more `<tool_call>{json}</tool_call>` blocks."""
    matches = list(_HERMES_BLOCK_RE.finditer(content))
    if not matches:
        return None
    calls = []
    for i, m in enumerate(matches):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        fn = _validate_function_payload(obj.get("name"), obj.get("arguments"))
        if fn:
            calls.append(
                {
                    "id": _make_tool_call_id(i),
                    "type": "function",
                    "function": fn,
                }
            )
    if not calls:
        return None
    residual = _strip_matches(content, matches).strip()
    matched_len = sum(m.end() - m.start() for m in matches)
    if matched_len < int(len(content) * _RECOVERY_MIN_RATIO):
        return None
    return calls, residual


def _recover_qwen3_coder(content: str) -> tuple[list[dict], str] | None:
    """Extract qwen3_coder XML tool calls."""
    matches = list(_QWEN3CODER_BLOCK_RE.finditer(content))
    if not matches:
        return None
    calls = []
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        body = m.group(2)
        params: dict[str, object] = {}
        for pm in _QWEN3CODER_PARAM_RE.finditer(body):
            params[pm.group(1).strip()] = pm.group(2).strip()
        fn = _validate_function_payload(name, params)
        if fn:
            calls.append(
                {
                    "id": _make_tool_call_id(i),
                    "type": "function",
                    "function": fn,
                }
            )
    if not calls:
        return None
    residual = _strip_matches(content, matches).strip()
    matched_len = sum(m.end() - m.start() for m in matches)
    if matched_len < int(len(content) * _RECOVERY_MIN_RATIO):
        return None
    return calls, residual


def _recover_phi4_pipe(content: str) -> tuple[list[dict], str] | None:
    """Extract phi-4-mini-style `<|tool_calls|>...<|/tool_calls|>` blocks."""
    matches = list(_PHI4_PIPE_RE.finditer(content))
    if not matches:
        return None
    calls = []
    seq = 0
    for m in matches:
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        items = payload if isinstance(payload, list) else [payload]
        for obj in items:
            if not isinstance(obj, dict):
                continue
            fn = _validate_function_payload(obj.get("name"), obj.get("arguments"))
            if fn:
                calls.append(
                    {
                        "id": _make_tool_call_id(seq),
                        "type": "function",
                        "function": fn,
                    }
                )
                seq += 1
    if not calls:
        return None
    residual = _strip_matches(content, matches).strip()
    matched_len = sum(m.end() - m.start() for m in matches)
    if matched_len < int(len(content) * _RECOVERY_MIN_RATIO):
        return None
    return calls, residual


def _recover_mistral_bracket(content: str) -> tuple[list[dict], str] | None:
    """Extract `[TOOL_CALLS][{...}]` blocks."""
    matches = list(_MISTRAL_BRACKET_RE.finditer(content))
    if not matches:
        return None
    calls = []
    seq = 0
    for m in matches:
        try:
            arr = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(arr, list):
            continue
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            fn = _validate_function_payload(obj.get("name"), obj.get("arguments"))
            if fn:
                calls.append(
                    {
                        "id": _make_tool_call_id(seq),
                        "type": "function",
                        "function": fn,
                    }
                )
                seq += 1
    if not calls:
        return None
    residual = _strip_matches(content, matches).strip()
    matched_len = sum(m.end() - m.start() for m in matches)
    if matched_len < int(len(content) * _RECOVERY_MIN_RATIO):
        return None
    return calls, residual


def _strip_matches(content: str, matches: list) -> str:
    """Remove all matched substrings from content, preserving non-matched parts."""
    out: list[str] = []
    cursor = 0
    for m in matches:
        out.append(content[cursor : m.start()])
        cursor = m.end()
    out.append(content[cursor:])
    return "".join(out)


def _recover_tool_calls_from_content(content: str) -> tuple[list[dict], str] | None:
    """Try each known leak shape in order; return first successful recovery.

    Returns (tool_calls, residual_content) or None if no shape matched
    convincingly enough (above the `_RECOVERY_MIN_RATIO` floor).
    """
    for fn in (
        _recover_hermes,
        _recover_qwen3_coder,
        _recover_phi4_pipe,
        _recover_mistral_bracket,
    ):
        result = fn(content)
        if result:
            return result
    return None


def rewrite_chat_completion(payload: dict) -> dict:
    """Rewrite a non-streaming /v1/chat/completions response in place.

    Two response-side fixes:
      1. **Thinking: prefix splitter** — for models that ignore their own
         `<think>` contract and emit "Thinking:" plaintext (notably
         Nemotron-Cascade-2). Pull that out of `content` into
         `reasoning_content`.
      2. **Tool-call recovery** — for models whose tool-call output is
         unparsed by vLLM's tool parser and ends up as plain text in
         `content` (notably Phi-4-mini emitting `<|tool_calls|>[{...}]
         <|/tool_calls|>` per Microsoft's own model card; older Mistral
         emissions; misrouted hermes/qwen3_coder shapes). Detect the
         shape, parse, synthesize structured `message.tool_calls`, and
         clear the leaked text from `content`.
    """
    choices = payload.get("choices", [])
    for choice in choices:
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        existing_reasoning = msg.get("reasoning_content") or ""

        # Pass 1: Thinking: prefix split
        if not existing_reasoning and content:
            reasoning, cleaned = split_thinking_prefix(content)
            if reasoning:
                msg["reasoning_content"] = reasoning
                content = cleaned
                msg["content"] = cleaned
                choice["message"] = msg
                logger.info(
                    "split Thinking: prefix; %d chars reasoning, %d chars content",
                    len(reasoning),
                    len(cleaned),
                )

        # Pass 2: tool-call recovery (only if model didn't already produce
        # structured tool_calls)
        if not msg.get("tool_calls") and content:
            recovered = _recover_tool_calls_from_content(content)
            if recovered:
                tool_calls, residual = recovered
                msg["tool_calls"] = tool_calls
                msg["content"] = residual or None
                choice["message"] = msg
                # Update finish_reason to tool_calls (was likely 'stop'
                # or 'length' since the parser missed the dispatch)
                if choice.get("finish_reason") in ("stop", "length", None):
                    choice["finish_reason"] = "tool_calls"
                logger.info(
                    "recovered %d tool_call(s) from content (was %d chars, residual %d chars)",
                    len(tool_calls),
                    len(content),
                    len(residual),
                )
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
            requested,
            _REASONING_MAX_TOKENS_BUMP,
            reasoning_parser,
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


def _make_delta_event(
    idx: int, role: str | None, reasoning: str | None, content: str | None, original_chunk: dict
) -> str:
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


# =============================================================================
# Streaming tool-call recovery
# =============================================================================
#
# Mirrors `_recover_tool_calls_from_content` but for SSE streams. Most clients
# (OpenCode / Pi / Hermes) use stream=true, so non-streaming-only recovery
# wouldn't actually fix Phi-4-mini against them.
#
# Design: per-choice state machine over content deltas.
#   - DECIDING: buffer first ~32 chars. Compare against known leak openers.
#       - If prefix matches a leak opener → BUFFERING
#       - If buffer past threshold without matching → PASSTHROUGH (flush + stream)
#       - Else stay DECIDING (need more data)
#   - PASSTHROUGH: forward content deltas verbatim. Streaming UX preserved.
#   - BUFFERING: accumulate content, emit nothing until finish_reason.
#       On finish: try _recover_tool_calls_from_content over the full buffer.
#         - Success → emit synthesized tool_calls delta + finish_reason=tool_calls
#         - Failure → flush buffer as a single content delta + original finish

_LEAK_OPEN_MARKERS: tuple[str, ...] = (
    "<|tool_call",  # phi4 — covers both `<|tool_calls|>` and `<|tool_call|>`
    "<tool_call>",  # hermes JSON or qwen3_coder XML
    "[TOOL_CALLS]",  # mistral
)
_LEAK_DECISION_THRESHOLD = 32  # chars to buffer before deciding passthrough


def _classify_streaming_buffer(buffer: str) -> str | None:
    """Decide whether a streamed buffer prefix is a leak opener.

    Returns:
        "leak" — definite leak shape; switch to BUFFERING.
        "no-leak" — definitely not a leak shape; switch to PASSTHROUGH.
        None — ambiguous; need more data before deciding.
    """
    stripped = buffer.lstrip()
    if any(stripped.startswith(m) for m in _LEAK_OPEN_MARKERS):
        return "leak"
    # Could still become a leak if we get more chars (e.g., buffer is `<` and
    # next delta starts with `|`). Keep deciding.
    if any(m.startswith(stripped) for m in _LEAK_OPEN_MARKERS):
        return None
    if len(buffer) >= _LEAK_DECISION_THRESHOLD:
        return "no-leak"
    return None


def _streaming_tool_call_delta(idx: int, tc: dict) -> dict:
    """Format a recovered tool_call as a single streaming delta entry."""
    return {
        "index": idx,
        "id": tc["id"],
        "type": tc.get("type", "function"),
        "function": tc["function"],
    }


class _RecoveryChoiceState:
    __slots__ = ("buffer", "decision")
    DECIDING = "deciding"
    PASSTHROUGH = "passthrough"
    BUFFERING = "buffering"

    def __init__(self) -> None:
        self.buffer = ""
        self.decision = self.DECIDING


async def stream_tool_recovery(
    upstream_iter: AsyncIterator[bytes],
    tool_parser: str,
) -> AsyncIterator[bytes]:
    """SSE wrapper that recovers structured tool_calls from leak-shaped content.

    Engaged only when `tool_parser` is in `_LEAKY_TOOL_PARSERS`. For all other
    parsers, this is a thin passthrough (no buffering overhead).
    """
    if tool_parser not in _LEAKY_TOOL_PARSERS:
        async for chunk in upstream_iter:
            yield chunk
        return

    states: dict[int, _RecoveryChoiceState] = {}
    pending = b""
    async for raw in upstream_iter:
        pending += raw
        while b"\n\n" in pending:
            event_bytes, pending = pending.split(b"\n\n", 1)
            event = event_bytes.decode("utf-8", errors="replace")
            if not event.startswith("data: "):
                yield (event + "\n\n").encode()
                continue
            payload_text = event[len("data: ") :]
            if payload_text.strip() == "[DONE]":
                # Defensive: flush any still-buffered content before terminating.
                # Normal flow flushes on finish_reason; this catches edge cases
                # where the upstream forgot to set finish_reason on a final chunk.
                for idx, st in states.items():
                    if st.buffer and st.decision != _RecoveryChoiceState.PASSTHROUGH:
                        flush = {
                            "id": "stream",
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": idx,
                                    "delta": {"content": st.buffer},
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                        yield f"data: {json.dumps(flush)}\n\n".encode()
                        st.buffer = ""
                        st.decision = _RecoveryChoiceState.PASSTHROUGH
                yield event.encode() + b"\n\n"
                continue
            try:
                chunk = json.loads(payload_text)
            except json.JSONDecodeError:
                yield (event + "\n\n").encode()
                continue
            # Pass-through metadata chunks (usage)
            if not chunk.get("choices"):
                yield (event + "\n\n").encode()
                continue

            forward_choices = []
            for c in chunk.get("choices", []):
                idx = c.get("index", 0)
                st = states.setdefault(idx, _RecoveryChoiceState())
                delta = c.get("delta", {}) or {}
                content_delta = delta.get("content")
                finish_reason = c.get("finish_reason")

                # Already-structured tool_calls in delta → flip to passthrough
                # for this choice and just forward.
                if delta.get("tool_calls"):
                    st.decision = _RecoveryChoiceState.PASSTHROUGH
                    forward_choices.append(c)
                    continue

                # PASSTHROUGH choice: forward verbatim
                if st.decision == _RecoveryChoiceState.PASSTHROUGH:
                    forward_choices.append(c)
                    continue

                # No content delta in this chunk
                if content_delta is None:
                    in_buffering = st.decision == _RecoveryChoiceState.BUFFERING
                    in_deciding = st.decision == _RecoveryChoiceState.DECIDING
                    if finish_reason and in_buffering and st.buffer:
                        # Stream end while buffering — try recovery
                        recovered = _recover_tool_calls_from_content(st.buffer)
                        if recovered:
                            tool_calls, _residual = recovered
                            forward_choices.append(
                                {
                                    "index": idx,
                                    "delta": {
                                        "tool_calls": [
                                            _streaming_tool_call_delta(i, tc)
                                            for i, tc in enumerate(tool_calls)
                                        ],
                                    },
                                    "finish_reason": "tool_calls",
                                }
                            )
                            logger.info(
                                "stream-recovered %d tool_calls from %d buffered chars",
                                len(tool_calls),
                                len(st.buffer),
                            )
                        else:
                            # Recovery failed — flush buffer as content
                            forward_choices.append(
                                {
                                    "index": idx,
                                    "delta": {"content": st.buffer},
                                    "finish_reason": finish_reason,
                                }
                            )
                        st.buffer = ""
                        st.decision = _RecoveryChoiceState.PASSTHROUGH
                    elif finish_reason and in_deciding and st.buffer:
                        # Stream end while still deciding — flush buffer as content.
                        # This handles vLLM's pattern of emitting finish_reason in
                        # a separate empty-delta chunk after the last content chunk.
                        forward_choices.append(
                            {
                                "index": idx,
                                "delta": {"content": st.buffer},
                                "finish_reason": finish_reason,
                            }
                        )
                        st.buffer = ""
                        st.decision = _RecoveryChoiceState.PASSTHROUGH
                    else:
                        forward_choices.append(c)
                    continue

                # Content delta arrived; accumulate
                st.buffer += content_delta

                if st.decision == _RecoveryChoiceState.DECIDING:
                    classification = _classify_streaming_buffer(st.buffer)
                    if classification == "leak":
                        st.decision = _RecoveryChoiceState.BUFFERING
                        # If finish_reason came along with this content delta,
                        # try recovery now.
                        if finish_reason:
                            recovered = _recover_tool_calls_from_content(st.buffer)
                            if recovered:
                                tool_calls, _residual = recovered
                                forward_choices.append(
                                    {
                                        "index": idx,
                                        "delta": {
                                            "tool_calls": [
                                                _streaming_tool_call_delta(i, tc)
                                                for i, tc in enumerate(tool_calls)
                                            ],
                                        },
                                        "finish_reason": "tool_calls",
                                    }
                                )
                                logger.info(
                                    "stream-recovered %d tool_calls from %d buffered chars",
                                    len(tool_calls),
                                    len(st.buffer),
                                )
                            else:
                                forward_choices.append(
                                    {
                                        "index": idx,
                                        "delta": {"content": st.buffer},
                                        "finish_reason": finish_reason,
                                    }
                                )
                            st.buffer = ""
                        # else: keep buffering, emit no delta this round
                    elif classification == "no-leak":
                        st.decision = _RecoveryChoiceState.PASSTHROUGH
                        # Flush accumulated buffer as a single content delta
                        forward_choices.append(
                            {
                                "index": idx,
                                "delta": {"content": st.buffer},
                                "finish_reason": finish_reason,
                            }
                        )
                        st.buffer = ""
                    else:
                        # Still ambiguous; if finish_reason arrives, flush as content
                        if finish_reason:
                            forward_choices.append(
                                {
                                    "index": idx,
                                    "delta": {"content": st.buffer},
                                    "finish_reason": finish_reason,
                                }
                            )
                            st.buffer = ""
                            st.decision = _RecoveryChoiceState.PASSTHROUGH
                        # else: hold and wait for more
                elif st.decision == _RecoveryChoiceState.BUFFERING:
                    # Keep accumulating; recover only at finish_reason
                    if finish_reason:
                        recovered = _recover_tool_calls_from_content(st.buffer)
                        if recovered:
                            tool_calls, _residual = recovered
                            forward_choices.append(
                                {
                                    "index": idx,
                                    "delta": {
                                        "tool_calls": [
                                            _streaming_tool_call_delta(i, tc)
                                            for i, tc in enumerate(tool_calls)
                                        ],
                                    },
                                    "finish_reason": "tool_calls",
                                }
                            )
                            logger.info(
                                "stream-recovered %d tool_calls from %d buffered chars",
                                len(tool_calls),
                                len(st.buffer),
                            )
                        else:
                            forward_choices.append(
                                {
                                    "index": idx,
                                    "delta": {"content": st.buffer},
                                    "finish_reason": finish_reason,
                                }
                            )
                        st.buffer = ""

            if forward_choices:
                rewritten = {**chunk, "choices": forward_choices}
                yield f"data: {json.dumps(rewritten)}\n\n".encode()

    if pending:
        yield pending


# =============================================================================
# Streaming Thinking-prefix splitter (existing)
# =============================================================================


async def stream_rewriter(upstream_iter: AsyncIterator[bytes], arch: str) -> AsyncIterator[bytes]:
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
            payload_text = event[len("data: ") :]
            if payload_text.strip() == "[DONE]":
                # Flush any remaining buffered content as content (no transition seen)
                for idx, st in states.items():
                    if not st.decided and st.buffer:
                        yield _make_delta_event(idx, None, None, st.buffer, {}).encode()
                yield event.encode() + b"\n\n"
                continue
            try:
                chunk = json.loads(payload_text)
            except json.JSONDecodeError:
                yield (event + "\n\n").encode()
                continue

            # Pass through metadata-only chunks unchanged. vLLM emits the
            # final `usage` block in a chunk with `choices: []`; if we ran
            # the choice-rewrite path on it we'd swallow it silently and
            # the client would lose all token-usage telemetry (visible as
            # Hermes' context counter never advancing).
            if not chunk.get("choices"):
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
                            after_prefix = st.buffer[match.end() :]
                            split = _split_first_transition(after_prefix)
                            if split:
                                # Boundary already in buffer
                                reasoning_part, content_part = split
                                if reasoning_part.strip():
                                    new_choices.append(
                                        {
                                            "index": idx,
                                            "delta": {
                                                **({"role": role} if role else {}),
                                                "reasoning_content": reasoning_part.strip(),
                                            },
                                            "finish_reason": None,
                                        }
                                    )
                                if content_part:
                                    new_choices.append(
                                        {
                                            "index": idx,
                                            "delta": {"content": content_part},
                                            "finish_reason": c.get("finish_reason"),
                                        }
                                    )
                                st.in_reasoning = False
                                st.transition_seen = True
                            else:
                                # All buffered text is reasoning so far
                                if after_prefix:
                                    new_choices.append(
                                        {
                                            "index": idx,
                                            "delta": {
                                                **({"role": role} if role else {}),
                                                "reasoning_content": after_prefix,
                                            },
                                            "finish_reason": None,
                                        }
                                    )
                            st.buffer = ""
                        else:
                            # No prefix; flush as content
                            new_choices.append(
                                {
                                    "index": idx,
                                    "delta": {
                                        **({"role": role} if role else {}),
                                        "content": st.buffer,
                                    },
                                    "finish_reason": c.get("finish_reason"),
                                }
                            )
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
                                new_choices.append(
                                    {
                                        "index": idx,
                                        "delta": {"reasoning_content": reasoning_part},
                                        "finish_reason": None,
                                    }
                                )
                            if content_part:
                                new_choices.append(
                                    {
                                        "index": idx,
                                        "delta": {"content": content_part},
                                        "finish_reason": c.get("finish_reason"),
                                    }
                                )
                            st.transition_seen = True
                            st.in_reasoning = False
                        else:
                            new_choices.append(
                                {
                                    "index": idx,
                                    "delta": {"reasoning_content": content_delta},
                                    "finish_reason": c.get("finish_reason"),
                                }
                            )
                    else:
                        # In content mode
                        new_choices.append(
                            {
                                "index": idx,
                                "delta": {"content": content_delta},
                                "finish_reason": c.get("finish_reason"),
                            }
                        )

            if new_choices:
                rewritten = {**chunk, "choices": new_choices}
                yield f"data: {json.dumps(rewritten)}\n\n".encode()
    if pending_text:
        yield pending_text


# =============================================================================
# aiohttp proxy
# =============================================================================


async def _make_app(
    upstream_url: str, arch: str, reasoning_parser: str = "", tool_parser: str = ""
) -> web.Application:
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
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")
        }
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                request.method,
                url,
                data=body if body else None,
                headers=headers,
                allow_redirects=False,
            ) as upstream_resp:
                ct = upstream_resp.headers.get("content-type", "")
                if is_chat and "text/event-stream" in ct:
                    response = web.StreamResponse(
                        status=upstream_resp.status,
                        headers={"Content-Type": ct, "Cache-Control": "no-cache"},
                    )
                    await response.prepare(request)
                    # Chain: thinking-split → tool-call streaming recovery
                    thinking_split = stream_rewriter(upstream_resp.content.iter_any(), arch)
                    async for out_chunk in stream_tool_recovery(thinking_split, tool_parser):
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
                        return web.Response(
                            status=upstream_resp.status,
                            body=raw,
                            headers=dict(upstream_resp.headers),
                        )
                # Pass-through for everything else
                response = web.StreamResponse(
                    status=upstream_resp.status, headers=dict(upstream_resp.headers)
                )
                await response.prepare(request)
                async for ch in upstream_resp.content.iter_any():
                    await response.write(ch)
                await response.write_eof()
                return response

    app = web.Application(client_max_size=50 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


def run(
    user_port: int,
    upstream_port: int,
    arch: str,
    reasoning_parser: str = "",
    tool_parser: str = "",
) -> None:
    """Run the rewriter proxy. Blocks until killed."""
    upstream_url = f"http://127.0.0.1:{upstream_port}"
    logger.info(
        "rewriter starting: user_port=%d upstream=%s arch=%s reasoning_parser=%s tool_parser=%s",
        user_port,
        upstream_url,
        arch,
        reasoning_parser or "<none>",
        tool_parser or "<none>",
    )
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = loop.run_until_complete(_make_app(upstream_url, arch, reasoning_parser, tool_parser))
    web.run_app(app, host="127.0.0.1", port=user_port, print=lambda *_: None, loop=loop)
