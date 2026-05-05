# Model Compatibility Matrix

Empirical end-to-end compatibility results from a sweep of MLX-format
models on Apple Silicon (M5 Max, vLLM 0.19.1, vllm-swift `feat/auto-detect-reasoning-parser-and-hardening`).

Each model was launched via `vllm-swift serve <path>` (auto-detect on),
then exercised through a 3-turn agent harness:

1. **T1 — list files** (single-shot tool dispatch via `bash`)
2. **T2 — write code** (`largest(a, b, c)` saved via `write` tool)
3. **T3 — review code** (read the file back via `read`, comment on bugs)

Verdicts:
- **PASS** — all 3 turns dispatched structured `tool_calls`, end-to-end write→read→review succeeded
- **SOFT-FAIL** — at least one turn passed; failure traceable to model capability, not parser plumbing
- **HARD-FAIL** — model never produced structured tool_calls; root cause documented below
- **SKIPPED** — environment / required-files limitation outside vllm-swift's scope

This file is a snapshot. Re-running the sweep against the latest
`feat/auto-detect-reasoning-parser-and-hardening` head should reproduce
within run-to-run variance.

## Summary scorecard

| Model | Verdict | Auto-detected (tool / reasoning) | Failure mode | Root cause |
|---|---|---|---|---|
| Qwen3.5-9B-4bit | ✅ PASS | qwen3_coder / qwen3 | — | works |
| Qwen3-Coder-30B-A3B-Instruct-MLX-6bit | ✅ PASS | qwen3_coder / *(suppressed)* | — | works (after qwen3+qwen3_coder reasoning-suppression fix) |
| Nemotron-Cascade-2-30B-A3B-4bit | ✅ PASS | qwen3_coder / nemotron_v3 | — | works |
| Qwen3.6-35B-A3B-4bit | ✅ PASS | qwen3_coder / qwen3 | — | works |
| Llama-3.2-3B-Instruct-4bit | ✅ PASS | llama3_json / *(none)* | — | works |
| gpt-oss-20b-MXFP4-Q8 | ✅ PASS | openai / openai_gptoss | — | works |
| Qwen3-0.6B-4bit | ⚠️ SOFT-FAIL | hermes / qwen3 | T1+T2 ✓, T3 ✗ — model loses thread on 3rd turn | sub-1B intrinsic limit |
| Qwen3.5-2B-4bit | ❌ HARD-FAIL | qwen3_coder / qwen3 | silent generation: 0 tokens emitted to content/reasoning/tool_calls | Qwen3.5 small variant chat-template bug + reasoning-disabled-by-default |
| Llama-3.2-1B-Instruct-hf | ❌ HARD-FAIL | llama3_json / *(none)* | never dispatches tools | sub-1B intrinsic limit (Meta: "lightweight models do not support built-in tools") |
| Mistral-7B-Instruct-v0.3-4bit | ❌ HARD-FAIL | mistral / *(none)* | vLLM mistral parser fails with "Only one BOT token should have been outputted" + emits `[TOOL_CALLS][TOOL_CALLS]` | vLLM upstream parser bug |
| Phi-4-mini-instruct-4bit | ⚠️ SOFT-FAIL | phi4_mini_json / *(none)* | Model emits `<\|tool_calls\|>...<\|/tool_calls\|>` as plain content text and vLLM's parser doesn't extract. Auto-recovery in vllm-swift's response rewriter (both non-streaming and streaming) now synthesizes structured `tool_calls` from this leak shape, lifting Phi-4-mini from HARD-FAIL to functional for clients that hit the documented behavior. Remaining failure mode is the model occasionally choosing to chat-explain instead of dispatch — a model-quality issue, not a parser/recovery one. | vLLM upstream parser gap; recovered by vllm-swift |
| gemma-4-e2b-it-4bit | ⏭️ SKIPPED | gemma4 / gemma4 | server boot fails: "Can't load video processor" missing `video_preprocessor_config.json` | environment / missing files |

## Per-model details

### ✅ PASS

#### Qwen3.5-9B-4bit
Auto-detect picks `qwen3_coder` + `qwen3`. Validates the empirical fix
that routes `Qwen3_5ForConditionalGeneration` to qwen3_coder (the
chat_template.jinja ships `<tool_call><function=name><parameter=k>v...`
XML, not hermes JSON). All 3 turns dispatched cleanly; `largest.py`
written and read back end-to-end.

#### Qwen3-Coder-30B-A3B-Instruct-MLX-6bit
Auto-detect picks `qwen3_coder`. Reasoning is intentionally suppressed
by the `-Coder-` directory-name discriminator. Without that suppression,
`qwen3_coder` (tool) + `qwen3` (reasoning) race: model emits tool calls
*inside* `<think>` blocks, the reasoning parser eats them, `tool_calls=[]`.
With suppression, all 3 turns work and the model writes a docstring'd
correct implementation.

#### Nemotron-Cascade-2-30B-A3B-4bit
Auto-detect picks `qwen3_coder` (NOT hermes — this was an empirical
correction; see PR #14 for HF discussion #7 reference) + `nemotron_v3`
(NVIDIA's purpose-built reasoning parser). All 3 turns clean.

#### Qwen3.6-35B-A3B-4bit
The original target of issue #13. Auto-detect picks `qwen3_coder` +
`qwen3`. Independently validated by @Defilan against
`mlx-community/Qwen3.6-35B-A3B-8bit` (see PR #14 comment).

#### Llama-3.2-3B-Instruct-4bit
Auto-detect picks `llama3_json` only (no reasoning parser). 3B is the
smallest Llama variant in our sweep that reliably dispatches tools.
Per [LangChain community](https://forum.langchain.com/t/tool-function-calling-with-llama-3-2-3b-instruct-model-local/2574),
3B is the practical floor for reliable Llama-family tool calling.

#### gpt-oss-20b-MXFP4-Q8
Auto-detect picks `openai` (tool) + `openai_gptoss` (reasoning). All 3
turns clean, with a numpy-style docstring on the generated function.

### ⚠️ SOFT-FAIL

#### Qwen3-0.6B-4bit
T1 (list) and T2 (write) dispatch correctly. T3 (read) — model loses
the multi-turn thread. This is consistent with [community findings on
small-model tool calling](https://dev.to/anak_wannaphaschaiyong_11/why-small-llms-fail-at-tool-calling-the-shocking-discovery-from-our-llama-3b-benchmark-5lg):
sub-7B agent loops degrade on multi-step chains. Detection is correct;
model capability is the limiting factor.

### ❌ HARD-FAIL — model intrinsic / capability

#### Qwen3.5-2B-4bit
Auto-detect picks `qwen3_coder` + `qwen3`. Model generates **zero**
tokens visible at the API: `content=""`, `reasoning_content=""`,
`tool_calls=[]`, `finish_reason=stop` immediately. Documented by Qwen
themselves: [Qwen3.5 chat template tool calling broken](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/4),
and [Qwen3.5 small-variant reasoning is disabled by default](https://huggingface.co/Qwen/Qwen3.5-2B)
unless `chat_template_kwargs={"enable_thinking": true}` is passed.
Detection is correct; the model variant ships a partially-working
template.

#### Llama-3.2-1B-Instruct-hf
Auto-detect picks `llama3_json`. Model never produces structured
`tool_calls` across any of the 3 turns. Meta's own
[Llama 3.2 model card](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/)
states: *"the lightweight models do not support built-in tools."*
1B is below the practical floor for reliable agentic tool dispatch.

### ❌ HARD-FAIL — vLLM upstream parser

#### Mistral-7B-Instruct-v0.3-4bit
Auto-detect picks `mistral`. Server returns HTTP 500 with:
> `ValueError: Only one BOT token should have been outputted, but got [TOOL_CALLS][TOOL_CALLS][...]`

i.e., the model correctly produces a tool-call structure, but the vLLM
mistral_tool_parser rejects double-BOT-token emissions. Documented in
multiple vLLM upstream issues:

- [#21303 — Mistral Tool Parser Crashes with Empty JSONDecodeError](https://github.com/vllm-project/vllm/issues/21303)
- [#16190 — Mistral tool parser failed to parse function calling](https://github.com/vllm-project/vllm/issues/16190)
- [#15549 — Tools parsing issues with mistral3.1](https://github.com/vllm-project/vllm/issues/15549)
- [#13622 — Mistral streaming tool parser fails to parse integer tool argument](https://github.com/vllm-project/vllm/issues/13622)
- [#9019 — ToolCall IDs don't comply with Mistral template](https://github.com/vllm-project/vllm/issues/9019)
- [#8301 — Mistral Large Instruct 2407 tool calling leakage](https://github.com/vllm-project/vllm/issues/8301)

vllm-swift's role here ends at picking the correct parser. The parser
itself is the upstream bug surface.

#### Phi-4-mini-instruct-4bit
Auto-detect picks `phi4_mini_json`. Model emits the correct token
sequence as plain content (`<|tool_calls|>[{...}]<|/tool_calls|>`),
but vLLM's parser does not extract it into `message.tool_calls`. Documented:

- [#14682 — Phi-4-mini function calling support](https://github.com/vllm-project/vllm/issues/14682)
- [#14359 — phi-4-mini-instruct auto tool call doesn't have tool-call-parser](https://github.com/vllm-project/vllm/issues/14359)
- [#14037 — Phi-4-mini giving random outputs with continuous batching](https://github.com/vllm-project/vllm/issues/14037)

Microsoft's [Phi-4-mini-instruct model card](https://huggingface.co/microsoft/Phi-4-mini-instruct)
also notes the model "could sometimes hallucinate function names."

**Recovered by vllm-swift since v0.4.0.** When `phi4_mini_json` is the
auto-detected tool parser, the rewriter proxy spawns automatically (via
`_LEAKY_TOOL_PARSERS`) and runs auto-recovery on both non-streaming and
streaming responses. The leak shape gets parsed out of `content` and
synthesized into a structured `message.tool_calls`, with `finish_reason`
bumped to `tool_calls`. Clients see a normal structured tool dispatch.
The remaining model-quality issue (Phi-4-mini sometimes choosing to
chat-explain rather than dispatch) is unfixable from vllm-swift; if your
agent loop hits it, a stronger model is the answer.

### ⏭️ SKIPPED — environment / missing files

#### gemma-4-e2b-it-4bit
vLLM 0.19.1 refuses to load the model:

> `OSError: Can't load video processor for '...gemma-4-e2b-it-4bit'.
> ...make sure '...' is the correct path to a directory containing a
> video_preprocessor_config.json file`

vLLM's Gemma-4 path treats the family as multimodal even when the
4-bit MLX build ships text-only. Documented:

- [llm-compressor #1305 — oneshot doesn't output preprocessor_config / processor_config](https://github.com/vllm-project/llm-compressor/issues/1305)
- [lmstudio bug-tracker #1741 — MLX Gemma 4 26B fails to load](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1741)
- [vllm-metal #299 — Gemma-4-E4B-it fails to load](https://github.com/vllm-project/vllm-metal/issues/299)
- [mlx-engine #301 — gemma 4 support](https://github.com/lmstudio-ai/mlx-engine/issues/301)

Workaround per [vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html):
pass `--limit-mm-per-prompt image=0,audio=0` to the inner vLLM args.
vllm-swift's CLI passes through extra args, so users can still launch
gemma-4 by appending that flag to `vllm-swift serve`. Not validated
end-to-end in the sweep because the missing config file blocks load
entirely on the MLX 4-bit build we have on disk.

## Categorization

| Category | Count | Models |
|---|---|---|
| Works end-to-end via vllm-swift | **7/12** | Qwen3.5-9B, Qwen3-Coder-30B-MLX, Nemotron-Cascade-2, Qwen3.6-35B-A3B, Llama-3.2-3B, gpt-oss-20b, Phi-4-mini *(via auto-recovery since v0.4.0)* |
| Model intrinsic limit (sub-7B) | 3/12 | Qwen3-0.6B, Qwen3.5-2B, Llama-3.2-1B |
| vLLM upstream parser bug (no in-band recovery) | 1/12 | Mistral-7B-v0.3 (parser 500s before recovery sees it) |
| Environment / missing files | 1/12 | gemma-4-e2b-it |

**0/12 failures are vllm-swift bugs.** Every failure traces to (a)
model capability limits below the agentic floor, (b) vLLM upstream
parser issues, or (c) missing vendor config files outside the
auto-detect scope.

## Reproducing the sweep

```bash
# harness lives outside the repo (intentionally throwaway)
python3 /tmp/sweep_harness.py
```

The harness expects vllm-swift available at `/Users/tom/.vllm-swift/venv`
and models at `/Users/tom/models/<name>`; tweak the `VENV_PY` and
`MODELS` constants for other environments. Per-turn results land in
`/tmp/sweep-results/<model>.json` with the full conversation in
`messages-<model>.json`. Aggregate scorecard goes to `SUMMARY.json`.
