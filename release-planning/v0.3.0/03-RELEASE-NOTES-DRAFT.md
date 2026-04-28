# v0.3.0 — Release Notes Draft

Two parallel artifacts:

1. The **CHANGELOG.md** entry — short, matches the existing terse style of v0.2.0/0.2.1/0.2.2.
2. The **GitHub Release body** — a longer narrative for the release page.

## CHANGELOG.md entry (top of file)

```markdown
## v0.3.0 — TBD, 2026

**Stability and throughput on TurboQuant MoE.** Closes a long-standing Metal `Invalid Resource` race that hit concurrent custom-kernel workloads (TurboQuant B-path on MoE B≥8). Removing the swift-side band-aid that was only there to mask the underlying race recovers ~10% throughput on Qwen3.5-35B-A3B at B≥16.

- Metal buffer-aliasing race fixed via `CommandEncoder` retain on first-bind under `MTLResourceHazardTrackingModeUntracked` (mirrors `ml-explore/mlx#3461 / #3462`)
- Removed redundant `stopGradient + asyncEval` boundary in `compressedAttention` — Qwen3.5-35B-A3B B=17 4K decode 108.7 → 119.9 tok/s (+10%)
- TurboQuant compressed-attention path is now the default decode method (B-path) — A-path still selectable via `TURBO_COMPRESSED_ATTENTION=0`
- bf16 kernel output for TurboFlash pass2 + dim=512 instantiation for Gemma 4 31B
- Per-model `prefillStepSize` defaults via protocol (drops the stacked 3-place caller / model / fallback resolution)
- `prepareQueriesScaled` per-layer cache (saves one elementwise multiply per decode step)
- A-path rotation bypass — recovers decode tok/s and matches `--kv none` peak when `TURBO_COMPRESSED_ATTENTION=0`
- Initial foundation for DeepSeek-V4 — `model_type: deepseek_v4` dispatch wired in, weight loading works (DSV4-Flash-2bit-DQ tested on M5 Max). Forward pass not yet production-stable, follow-up in Phase 2.
```

## GitHub Release body

```markdown
# v0.3.0 — Metal race fix and TurboQuant MoE throughput recovery

Two related stories in one release.

## Stability — Metal Invalid Resource race, root caused and fixed

We've been chasing an `Invalid Resource` crash that fired under concurrent custom-kernel workloads on Metal. Running TurboQuant compressed attention at MoE B≥8 (Qwen3.5-35B-A3B) reproduced it 5/5 every time. Single-stream and dense workloads never hit it. Working backwards from `cudaMallocAsync`-style allocation patterns turned up the wrong tree; the bug is older and lives in MLX itself.

`mlx/backend/metal` allocates every buffer with `MTLResourceHazardTrackingModeUntracked` and creates every command buffer with `commandBufferWithUnretainedReferences()`. Both Apple APIs require the application to keep the buffer alive until the command buffer completes. The `CommandEncoder` bind path doesn't take an explicit `retain()` — it inserts the buffer pointer into a dedup set for barrier tracking but never bumps the Metal reference count. When a Swift task drops its `MLXArray` reference between encode and CB completion (which Swift structured concurrency does aggressively), the C++ destructor calls `buf->release()` and Metal sees the buffer destroyed while the GPU is still using it.

The fix is small: retain on first sighting in the bind path, transfer the per-CB retained vector into the addCompletedHandler lambda, release on completion. 6 files, ~100 lines, including a smoke test.

This release ships the fix as a local pin in our mlx-swift-lm submodule. The same fix is also under review upstream as [`ml-explore/mlx#3462`](https://github.com/ml-explore/mlx/pull/3462) (issue: [#3461](https://github.com/ml-explore/mlx/issues/3461)).

## Performance — ~10% recovery on TurboQuant MoE

The previous release shipped a swift-side workaround that wrapped `compressedAttention` output in `stopGradient + asyncEval`. We added it because removing it crashed 5/5 at MoE B=16. Once the underlying retain race is fixed, that workaround is doing nothing useful and was costing throughput.

Ablation on M5 Max, Qwen3.5-35B-A3B turbo4v2 at `MLX_QUEUE_DEPTH=64`:

| cell | with stopGradient (prior) | without (this release) | Δ |
|---|---:|---:|---:|
| B=17 4K | 108.7 tok/s | 119.9 tok/s | +10.3% |
| B=32 4K | 111.2 tok/s | 119.5 tok/s | +7.5% |

25/25 clean across the three cells (B=16, 17, 32). The retain commit is the causal fix.

## Other improvements landing in this release

- **Compressed attention by default.** TurboQuant decode now goes through the compressed-domain Metal kernels by default (B-path). The previous A-path (raw FP16 + standard SDPA) is still selectable with `TURBO_COMPRESSED_ATTENTION=0`. The compressed path matches A-path throughput at most cells and saves the parallel FP16 dequant workspace at the prefill→decode boundary.
- **bf16 TurboFlash pass2 output + Gemma 4 dim=512 kernel.** Removes a `asType` workaround introduced earlier and unblocks Gemma 4 31B for TurboQuant.
- **`prepareQueriesScaled` cache.** Folds the SDPA scale into the rotation matrix on first lookup, hits the cache thereafter. Saves one elementwise multiply per decode step.
- **A-path rotation bypass.** Recovers the historic A/B gap when running with `TURBO_COMPRESSED_ATTENTION=0`. Matches `--kv none` peak.
- **`prefillStepSize` per-model protocol.** Replaces the three-place resolution (caller `GenerateParameters`, default `LLMModel.prepare()`, per-model `max()` clamps) with a single protocol property.
- **Initial foundation for DeepSeek-V4.** `model_type: deepseek_v4` now dispatches in the model factory. DSV4-Flash-2bit-DQ (90 GB, 284B / 21B active MoE) loads cleanly through the vllm-swift bridge on M5 Max 128GB. Phase 1 forward pass is not yet stable (single-step decode hits a GPU kernel timeout — Phase 2 follow-up will land the kernel optimizations). Source-only landing in this release; not yet wired into a recommended quant variant or default config.

## Compression numbers (refreshed)

Qwen3.5 2B at long context, ratio = `KV_fp16 / KV_compressed`:

| context | turbo4 | turbo3 | turbo2 | turbo4v2 |
|---|---:|---:|---:|---:|
| 8K | 2.94× | 3.66× | 4.61× | 3.66× |
| 32K | 3.58× | 4.64× | 6.46× | 4.64× |
| 64K | **3.74×** | **4.86×** | **6.95×** | **4.86×** |

64K is within 1–3% of the documented asymptote (3.8× / 4.9× / 7.1× / 4.9×). Codec overhead amortizes with context length.

## Throughput baseline (M5 Max 128GB)

Decode tok/s, identical-prompt mode (`scripts/bench_throughput.py`):

| | B=1 | B=8 | B=32 | B=64 |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 403 | 1,541 | 2,844 | 3,264 |
| Qwen3-4B | 156 | 482 | 1,191 | 1,493 |

Within run-to-run noise of v0.2.2 documented baselines. No regression.

## Upgrading

```bash
brew update
brew upgrade vllm-swift
vllm-swift version  # should print 0.3.0
```

If you're tracking a fork of `mlx-swift-lm` directly: `swift package update` and rebuild.

## What's next

- Upstream `ml-explore/mlx#3462` retain commit lands.
- Cap=4 default in mlx (the second commit on `ekryski/mlx#19`) we proved redundant after retain — likely dropped.
- Continued work on upstreaming TurboQuant to `ml-explore/mlx-swift-lm` (`#232`).

## Acknowledgements

- @ekryski for the bf16 TurboFlash output fix and ongoing alpha-branch maintenance.
- @sztlink, @lkaupp, @Xananthium, @dentity007, and the rest of the community testers across CUDA, ROCm, Vulkan, Metal.
- @signalnine for CUDA fattn template work in the broader fork.
- @dusterbloom for prior art on Metal centroid-LUT precision pressure (informed our debugging on the upstream long-context decode regression).
```

## X post draft (in Tom's voice — direct, lowercase, data-first, no em-dashes)

```
v0.3.0 of vllm-swift just shipped.

main thing: closed a metal race that hit turboquant moe at high concurrency. the swift-side bandaid we had for it was costing 10% throughput. removed it. qwen3.5 35b-a3b b=17 went from 108.7 to 119.9 tok/s.

retain fix is also up at ml-explore/mlx#3462 for upstream review.

brew upgrade vllm-swift.
```

## Manager / team summary (one-line)

> v0.3.0 ships a Metal stability fix that closes the remaining `Invalid Resource` race on concurrent custom-kernel workloads, recovers ~10% throughput on TurboQuant MoE at B≥16, and refreshes the bottle against the alpha-tip dependency chain.
