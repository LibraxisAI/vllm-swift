# Release History

## v0.4.1 — May 5, 2026

**Patch: clamp `max_tokens` rescue against the configured `max_model_len`.**
Empirical bug surfaced on Mac Mini M2 testing v0.4.0 against `Qwen3.5-2B-4bit` with `--max-model-len 4096`: the request rewriter bumped a client `max_tokens=256` to the static `_REASONING_MAX_TOKENS_BUMP=32768`, which vLLM then 400'd with `max_tokens cannot be greater than max_model_len=4096`. The bump was correct in spirit (small budget against a reasoning model would have starved the `<think>` block) but didn't know about the server's context ceiling.

- `rewrite_request` now takes an optional `max_model_len` and clamps the bump to `max_model_len - 256` (safety margin for prompt tokens).
- If the clamped bump is below the client's requested value, leave the request alone — never bump *down*.
- CLI extracts `--max-model-len` from passthrough args and threads it through `_serve_with_rewriter` → `run` → `_make_app` so the rewriter has the value when bumping.
- 3 new tests pin the clamp behavior at three boundaries (clamp applies, clamp is no-op when ample, clamp would bump down so skip).

`pip install vllm-swift==0.4.1` and the rebuilt Homebrew bottle for v0.4.1 carry the fix.

## v0.4.0 — May 5, 2026

**Auto-detect tool + reasoning parsers, plus an invisible self-heal layer for the rough edges.** Closes [#13](https://github.com/TheTom/vllm-swift/issues/13). The original triggering case (`mlx-community/Qwen3.6-35B-A3B-8bit` needing a manual `--tool-call-parser qwen3_coder --reasoning-parser qwen3` workaround) now Just Works, and several other footguns get caught by the same plumbing.

### Auto-detection

- Three-layer detection from a model directory: architecture-prefix mapping (40+ families), chat-template marker fallback for unknown architectures, and a directory-name discriminator that catches converted MLX/GGUF builds whose `config.json` lost the specialized arch suffix (Qwen3-Coder MLX, R1 forks, Kimi-K2.6 disguised as DeepSeekV3, etc).
- Capability gate: skip injection on models whose chat template carries no tool fragments (Phi-3-mini, Gemma 2, etc) so they don't get a parser they can't satisfy.
- Pre-flight registry validation: parser names are checked against the running vLLM's `_TOOL_PARSERS_TO_REGISTER` / `_REASONING_PARSERS_TO_REGISTER` before injection. If a name isn't registered (vLLM renamed or removed it, or our detector got ahead of upstream), the injection is skipped with a stderr warning rather than letting vLLM crash with an opaque "unknown parser" error.
- Validated against 18 real local MLX models in CI; the `mlx-community/Qwen3.6-35B-A3B-8bit` case from #13 was independently re-validated by [@Defilan](https://github.com/TheTom/vllm-swift/pull/14#issuecomment-4376186794).

### Empirical correctness fixes (versus the original detector intent)

- All `Qwen3.5+`, `Qwen3.6+`, `Qwen3Next`, and `Qwen3MoeForCausalLM` variants ship the `qwen3_coder` XML tool-call shape in their chat template, not the older `hermes` JSON. Routing fixed; older dense Qwen3 / Qwen3.5-Instruct dense kept on hermes per their actual templates.
- Nemotron H / Cascade-2 routes to `qwen3_coder` (tool) + `nemotron_v3` (reasoning) per NVIDIA's HF discussion #7, not the previous Qwen3-derivative defaults.
- Qwen3-Coder gets reasoning auto-suppressed via a `-Coder-` directory-name rule; the `qwen3` reasoning parser otherwise eats tool calls emitted inside `<think>` blocks (vllm-project/vllm#39056-class race) and clients see `tool_calls=[]`.
- MiMo (Xiaomi) routes to `qwen3` reasoning + `qwen3_xml` tool per the official MiMo-V2-Flash vLLM recipe (the `mimo` parser name our detector previously emitted is not registered in vLLM 0.19.1's parser set and would fail server startup).
- GLM-4.7 routes to `glm45` reasoning as a workaround until vLLM ships a dedicated `glm47` reasoning parser (vllm-project/vllm#33348).
- xLAM family (`Salesforce/xLAM-1b-fc-r`, `Salesforce/Llama-xLAM-2-*-fc-r`) ships as `LlamaForCausalLM` arch but uses the dedicated `xlam` parser; dirname discriminator now handles this.
- LongCat (Meituan `LongCat-Flash-*`) routes to the dedicated `longcat` parser.

### Invisible self-heal layer (response_rewriter)

A transparent proxy that fronts vLLM on the user-facing port and applies these rewrites only when needed (no-op for non-reasoning, non-leaky-parser models — zero overhead path):

- **`max_tokens` rescue.** When a reasoning parser is in play and the client sent `max_tokens` below a reasoning-safe floor (16384), bump it to 32768 in-flight. Prevents the OpenCode/Pi pattern where a hardcoded 8192 budget gets eaten by `<think>`, vLLM truncates, `</think>` never closes, and the parser dumps raw thinking into `content` as a monologue. Empirically validated: takes Nemotron-Cascade-2 + OpenCode from a 4-minute wedge to 4/4-pass on the standard agent test set.
- **Auto-recovery for plaintext-JSON tool-call leaks.** Four shapes detected and re-synthesized into structured `message.tool_calls`: hermes JSON, qwen3_coder XML, phi4 pipe-tag (Microsoft's own model card admits Phi-4-mini emits this shape as text — vllm-project/vllm#14682), and mistral bracket. Both non-streaming and streaming paths covered; streaming uses a per-choice three-state machine (DECIDING / PASSTHROUGH / BUFFERING) so healthy chat traffic still streams delta-by-delta. Conservative ratio gate (≥50% of content) defends against false-positives on responses that legitimately mention tool-call shapes inline.
- **`Thinking:` prefix split.** For models that emit "Thinking:" plaintext instead of `<think>...</think>` tags (notably Nemotron-Cascade-2), the prefix is split out of `content` into `reasoning_content` so the OpenAI-shape contract holds.
- **Streaming usage-chunk preservation.** vLLM emits the final `usage` block in a chunk with `choices: []`; the rewriter passes these through verbatim instead of dropping them (had been swallowing them, visible to users as Hermes' context-token counter never advancing).

### Documentation

- New [`docs/MODEL_COMPATIBILITY.md`](docs/MODEL_COMPATIBILITY.md) — empirical pass / soft-fail / hard-fail across 12 local MLX models with root-cause classification. Updated for v0.4.0: 7/12 PASS now that Phi-4-mini gets caught by auto-recovery.
- New [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — symptom → diagnostic → fix for known failure patterns. Includes the `--default-chat-template-kwargs '{"enable_thinking": false}'` escape hatch originally surfaced by [@Defilan](https://github.com/TheTom/vllm-swift/pull/14#issuecomment-4376186794).

### Tests

460 unit + integration tests, including 8 fixture-based replay tests using anonymized snapshots of the actual agent traffic shapes that triggered the original bugs. Auto-recovery alone has 27 dedicated tests across positive / negative / boundary / replay axes. The healthy-chat false-positive replay is the strongest defense against a future over-eager regex change.

### Compatibility note

The Homebrew bash wrapper at `/opt/homebrew/bin/vllm-swift` will rebuild against this release when the bottle workflow runs against the v0.4.0 tag; the pip wheel is the authoritative path until that lands.

## v0.3.3 — May 5, 2026

**Re-release of v0.3.2 with proper wheel contents.** The 0.3.2 PyPI wheel was built before the `package_data` changes were merged to `main`, so it shipped without the bundled `libVLLMBridge.dylib` + `mlx.metallib`. PyPI release files are immutable, so 0.3.2 is yanked and 0.3.3 is the working release. No source changes vs 0.3.2.

- `pip install vllm-swift==0.3.3` now ships the dylib + metallib via wheel `package_data`.
- Homebrew bottle rebuilt against the same source tree.

## v0.3.2 — May 4, 2026

**Patch release: symlinked model dirs no longer break MLX qwen3 loader.** Reported by @defilan (LLMKube metal-agent integration): passing a symlinked model dir (e.g. `~/models/mlx-community/Qwen3.6-35B-A3B-8bit -> ~/models/Qwen3.6-35B-A3B-8bit`) crashed vllm-swift with `[vsm] Failed to load model: Unsupported model type: qwen3`. The Swift bridge handed the symlinked URL straight to MLX's `LLMModelFactory`, which derives the architecture key from a mix of path components and `config.json` — those disagree on a symlinked path and the qwen3 codepath rejects the mismatch.

- `swift/Sources/VLLMBridge/Bridge.swift` now calls `URL.resolvingSymlinksInPath()` before handing the URL to MLX. No-op on canonical paths; fixes the symlink case.
- Verified locally on M5 Max: canonical `~/.cache/huggingface/hub/.../snapshots/<commit>/` still loads; symlinked dir pointing at the same target now also loads.
- Reported by @defilan in defilantech/LLMKube#393.

## v0.3.1 — May 2, 2026

**Patch release: serve subcommand model-path bug.** The wrapper was forwarding the model path as a positional argument to `vllm.entrypoints.openai.api_server`, where vLLM 0.19.1's argparse maps a stray positional to the deprecated `model_tag` slot rather than `ModelConfig.model`. The path was silently dropped and the engine fell back to the `Qwen/Qwen3-0.6B` placeholder. Reported in #11 (Defilan), surfaced again triaging #4 and #10.

- `vllm-swift serve <path>` now passes `--model <path>` explicitly to vLLM (#12)
- Fixes silent fallback to `Qwen/Qwen3-0.6B` when serving local model directories
- No dylib changes; bottle rebuild ships only the wrapper fix

## v0.3.0 — April 28, 2026

**Stability and throughput on TurboQuant MoE.** Closes a long-standing Metal `Invalid Resource` race that hit concurrent custom-kernel workloads (TurboQuant B-path on MoE B≥8). Removing the swift-side band-aid that was only there to mask the underlying race recovers ~10% throughput on Qwen3.5-35B-A3B at B≥17.

- Metal buffer-aliasing race fixed via `CommandEncoder` retain on first-bind under `MTLResourceHazardTrackingModeUntracked` (mirrors `ml-explore/mlx#3461 / #3462`)
- Removed redundant `stopGradient + asyncEval` boundary in `compressedAttention` — Qwen3.5-35B-A3B B=17 4K decode 108.7 → 119.9 tok/s (+10%)
- TurboQuant compressed-attention path is now the default decode method (B-path) — A-path still selectable via `TURBO_COMPRESSED_ATTENTION=0`
- bf16 kernel output for TurboFlash pass2 + dim=512 instantiation for Gemma 4 31B
- Per-model `prefillStepSize` defaults via protocol (drops the stacked 3-place caller / model / fallback resolution)
- `prepareQueriesScaled` per-layer cache (saves one elementwise multiply per decode step)
- A-path rotation bypass — recovers decode tok/s and matches `--kv none` peak when `TURBO_COMPRESSED_ATTENTION=0`
- Initial foundation for DeepSeek-V4 — `model_type: deepseek_v4` dispatch wired in, weight loading works (DSV4-Flash-2bit-DQ tested on M5 Max). Forward pass not yet production-stable, follow-up in Phase 2.

## v0.2.2 — April 26, 2026

**Batched decode for hybrid models and capacity fixes.** Qwen3.5, Qwen3.6, and Qwen3Next now scale with concurrency instead of staying flat at single-request speed. Long-context high-batch workloads no longer OOM. Source installs now work without manually copying the Metal library.

- Qwen3.5 / Qwen3.6 / Qwen3Next batched decode — 16× total tok/s at B=64 on Qwen3.6-27B (was flat across B)
- Fixed OOM at high batch + long context (4B / B=64 / 8K) by releasing per-request prefill caches as they are copied into the batched cache
- Fixed crash at prompt length ≥ 2048 with batched decode (cache was sized for the wrong dimension)
- `scripts/install.sh` now builds and places `mlx.metallib`; source installs of GatedDelta / TurboFlash models work without manual steps

## v0.2.1 — April 25, 2026

**Performance recovery for small models.** Decode throughput on models with fewer KV heads (0.8B, 2B, 35B-A3B) was 40-60% slower than expected due to an overly aggressive GPU sync barrier. This release replaces it with a lightweight alternative, bringing decode speed back to within 10-17% of uncompressed baseline.

- Faster TurboQuant+ decode on small models (0.8B, 2B, 35B-A3B)
- TurboQuant+ support for NemotronH hybrid models
- Fixed a bug where compressed KV cache slots were being overwritten instead of appended
- Install script fixes for machines without MLX Python installed

## v0.2.0 — April 24, 2026

**KV cache compression and Homebrew install.** TurboQuant+ compresses the KV cache 3-5x with no measurable impact on output quality, enabling longer conversations on memory-constrained devices. Homebrew bottle means no Swift toolchain needed.

- TurboQuant+ KV cache compression (`--additional-config '{"kv_scheme": "turbo4v2"}'`)
- `brew install vllm-swift` with prebuilt bottle
- `vllm-swift update` command
- Decode and prompt logprobs
- Experimental vision-language model support

## v0.1.0 — April 22, 2026

**Initial release.** Native Swift/Metal inference backend for vLLM on Apple Silicon. Up to 2.6x faster decode than Python/MLX at low concurrency by removing Python from the inference hot path.

- OpenAI-compatible API server
- Batched concurrent decode
- Streaming responses
- Auto model download from HuggingFace
