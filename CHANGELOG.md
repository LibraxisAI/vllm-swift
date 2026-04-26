# Release History

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
