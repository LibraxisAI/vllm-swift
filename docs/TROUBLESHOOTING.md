# Troubleshooting

Known failure patterns when running vllm-swift against an MLX model on
Apple Silicon, plus the diagnostic command and the working fix for
each. Most patterns auto-resolve via the detector + rewriter; this
document covers the cases where you have to intervene manually.

For the empirical pass/fail status of specific models, see
[MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md).

## Symptom: auto-detect picked the wrong parser

You see `vllm-swift: auto-detected ... parser '<wrong>'` in the boot
output, and the model misbehaves at runtime (tool calls don't dispatch,
content has weird leakage, etc.).

**Diagnostic:**
```bash
python -m vllm_swift.detect_tool_parser /path/to/model
python -m vllm_swift.detect_reasoning_parser /path/to/model
```
Compare against what you believe the model actually wants. The
`config.json` `architectures` field, the `chat_template.jinja`
contents, and the directory name all feed the detector.

**Fix:** override at the CLI. The detector backs off when an explicit
flag is present.

```bash
vllm-swift serve /path/to/model \
    --tool-call-parser hermes \
    --reasoning-parser qwen3
```

If the override is consistently right for a model family the detector
gets wrong, that's a real auto-detect bug — open an issue with the
arch, the dirname, and the chat template excerpt.

## Symptom: tool-call XML/JSON leaks into `message.content`

You're getting raw `<tool_call><function=...><parameter=...>` or
`{"name": "...", "arguments": {...}}` showing up as plain text in the
assistant message instead of structured `message.tool_calls`. The
client renders it as visible noise; tools never dispatch.

**Diagnostic:** the tool parser's emission shape doesn't match the
model's chat-template shape. Two main families:

- **`hermes` parser** expects JSON inside `<tool_call>...</tool_call>`
- **`qwen3_coder` parser** expects XML: `<tool_call><function=name><parameter=k>v</parameter>...`

Older Qwen3 dense models ship hermes JSON. Qwen3.5+/3.6+/Next/MoE,
Qwen3-Coder, and Nemotron-Cascade-2 ship qwen3_coder XML. The
detector handles this, but if you've manually overridden or the
detector mis-routed, mismatched parser explains the leak.

**What vllm-swift now does automatically (since v0.4.0):** the response
rewriter detects four leak shapes in `message.content` and synthesizes
proper structured `message.tool_calls`, clearing the leaked text and
bumping `finish_reason` to `tool_calls`. Shapes covered:

  - `<tool_call>{"name":...,"arguments":...}</tool_call>` (hermes JSON)
  - `<tool_call><function=name><parameter=k>v</parameter>...</function></tool_call>` (qwen3_coder XML)
  - `<\|tool_calls\|>[{...}]<\|/tool_calls\|>` (phi4 pipe-tag)
  - `[TOOL_CALLS][{...}]` (mistral bracket)

Recovery runs in both non-streaming and streaming responses. For
non-leaky parsers it's pass-through with no overhead. For known-leaky
parsers (`phi4_mini_json` today) the rewriter proxy auto-spawns even
on non-reasoning models so recovery has a chance to fire. Tail
`~/.vllm-swift/debug.log` for `recovered N tool_call(s)` lines.

**Manual fix when recovery doesn't catch your case:** override to the
correct parser (see "auto-detect picked the wrong parser" above). If
you believe the routing is wrong for a model the detector picks, file
an issue with the chat template excerpt showing the actual emission
shape — and ideally a captured response showing how it leaked, so the
next person hits the auto-recovery path instead of needing a manual
override.

## Symptom: "Thinking-only response", "Empty response", or agent loop terminates after one turn

The agent client (Hermes, OpenCode, Pi, etc.) shows messages like:
- `Thinking-only response — prefilling to continue (1/2)`
- `Empty response from model — retrying (1/3)`
- `Model produced reasoning but no visible response after all retries. Returning empty.`

Or you see in `~/.vllm-swift/debug.log`:
- Multiple `bumped max_tokens` lines but the response still has empty content

**Diagnostic:** the model spent the entire turn inside `<think>` and
never emitted final content or a structured tool call. Three subcauses:

1. **Budget starvation.** Client hardcoded a small `max_tokens` (commonly 8192) and reasoning ate the whole budget before the model could close `</think>`. The vllm-swift rewriter auto-bumps to 32768 for known reasoning parsers; check `~/.vllm-swift/debug.log` for `bumped max_tokens 8192 -> 32768` lines confirming the bump fired.

2. **Reasoning + tool parser race.** Some combos (notably qwen3 reasoning + qwen3_coder tool on Qwen3-Coder builds) cause the model to emit tool calls *inside* `<think>` blocks, where the reasoning parser eats them before the tool parser can extract them. The detector suppresses reasoning for `-Coder-` directory names to mitigate this; if you hit it on a different model, see the manual workaround below.

3. **Model just thinks too much for the agentic prompt.** Long system prompts (OpenCode's 23K-char prompt is the canonical example) push reasoning models into meta-rumination. No server-side fix bridges this; see workarounds.

**Fixes (try in order):**

a. **Confirm the rewriter is firing.** Tail `~/.vllm-swift/debug.log` and look for `bumped max_tokens` entries on each request. If absent, the rewriter isn't engaged — check that auto-detect picked a reasoning parser at boot.

b. **Disable thinking globally for this serve session.** Pass the
`enable_thinking=false` chat-template kwarg through to vLLM (workaround
originally documented by [@Defilan in PR #14](https://github.com/TheTom/vllm-swift/pull/14#issuecomment-4376186794) — "the symptom that drove me toward
`--default-chat-template-kwargs '{...}'`"):

```bash
vllm-swift serve /path/to/model \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

This is the sledgehammer: model loses CoT capability for *all* turns,
not just agentic ones. Useful when the surgical mitigations don't
catch your specific model+client combo.

c. **Bump max_tokens client-side.** If the rewriter isn't firing
because you're using a non-reasoning model that's just generating long
content, increase the client's `max_tokens` directly.

## Symptom: model never dispatches tool calls regardless of prompt

Single-turn requests with `tool_choice: auto` and well-formed `tools[]`
return `finish_reason: stop`, `content` with chat-style explanation,
`tool_calls: []`. Every prompt produces narrative, never structured
calls.

**Diagnostic:** likely a model capability floor, not a parser issue.
The empirical sweep confirms sub-1B models effectively cannot do
agentic tool dispatch (Llama-3.2-1B, Qwen3-0.6B), and 2B variants are
unreliable across multi-turn.

**Fix:** use a 7B-or-larger model for agentic workloads. See
[MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md) for verified-working
models.

## Symptom: server boot fails on Gemma-4 with `video_preprocessor_config.json`

```
OSError: Can't load video processor for '/path/to/gemma-4-...'.
...make sure '...' is the correct path to a directory containing a
video_preprocessor_config.json file
```

**Diagnostic:** vLLM treats Gemma-4 as multimodal even when the MLX
4-bit build ships text-only and lacks the vision config files.

**Fix:** pass through `--limit-mm-per-prompt` to skip multimodal
profiling. vllm-swift's CLI forwards extra args to vLLM:

```bash
vllm-swift serve /path/to/gemma-4-... --limit-mm-per-prompt image=0,audio=0
```

Per the [vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html).
Tracked in [llm-compressor #1305](https://github.com/vllm-project/llm-compressor/issues/1305)
and [lmstudio bug-tracker #1741](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1741).

## Symptom: OpenCode/Pi shows "Thinking:" prefix as visible content

OpenCode UI: visible italic `Thinking: ...` blocks rendered inline as
content, not collapsed.

**Diagnostic:** this is OpenCode's rendering of the `reasoning_content`
field, not a leak. It's the **structured** reasoning output, displayed
inline by the client. Compare against MLX-LM through the same client
on the same model — both produce the same UX.

**Fix:** none needed server-side. If you want the reasoning hidden,
that's a client-side display setting (OpenCode config), not a parser
issue.

## Symptom: how do I see which parsers vllm-swift injected?

**Boot-time:** the CLI prints
`vllm-swift: auto-detected ... parser '<name>' for <model>; injecting ...`
to stderr. Capture stderr to confirm:

```bash
vllm-swift serve /path/to/model 2>&1 | tee /tmp/vllm-swift.log
```

**Runtime:** the rewriter logs to `~/.vllm-swift/debug.log` whenever
it intervenes (e.g. `max_tokens` bumps). The vLLM API server logs
incoming requests to its own stdout.

**Process args:** `ps -ef | grep vllm` shows the final flags vLLM
received, which tells you what auto-detect resolved to even after the
fact.

## Filing issues

When you hit a pattern not covered here, the most useful info to
include in an issue:

1. Output of `vllm-swift serve ... 2>&1 | head -30` (boot section)
2. Output of `python -m vllm_swift.detect_tool_parser <path>` and `python -m vllm_swift.detect_reasoning_parser <path>`
3. The model's `config.json` `architectures` field
4. The first 30 lines of `chat_template.jinja` (or `tokenizer_config.json` if no jinja file)
5. The directory name of the model
6. A minimal `curl` repro of the failing request shape

vllm-swift can't fix model-side or vLLM-upstream bugs, but it can
route around them once we know the shape.
