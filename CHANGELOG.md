# Release History

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
