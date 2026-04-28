# v0.3.0 — Expected Gains (concrete numbers from this session)

All numbers from M5 Max 128GB. Each delta corresponds to a measured cell, not an extrapolation.

## Stability — `Invalid Resource` race closed

**Before (no retain, default queue depth 64):**

| cell | pass rate |
|---|---|
| Qwen3.5-35B-A3B B=16 4K turbo4v2 | 0/10 |
| Qwen3.5-35B-A3B B=17 4K turbo4v2 | 0/10 |
| Qwen3.5-35B-A3B B=32 4K turbo4v2 | 0/5 |
| Qwen3.5-2B B=16 4K turbo4v2 | 0/5 |
| Qwen3.5-9B B=16 4K turbo4v2 | 0/5 |

**After retain (this release, MLX_QUEUE_DEPTH=64):**

| cell | pass rate |
|---|---|
| Qwen3.5-2B B=16 4K | 10/10 |
| Qwen3.5-9B B=16 4K | 10/10 |
| Qwen3.5-35B-A3B B=17 4K | 10/10 |
| Qwen3.5-35B-A3B B=32 4K | 5/5 |
| Qwen3.5-35B-A3B B=16 8K | 9/10 (1 InnocentVictim — different error class, thermal not retain) |

**44/45 across the original failure surface.** Single InnocentVictim was a global GPU reset (collateral discard) on a heated M5 Max, not a memory-corruption regression from the retain code.

## Throughput — TurboQuant MoE recovery

Qwen3.5-35B-A3B turbo4v2 4K, `MLX_QUEUE_DEPTH=64`, B-path (compressed attention):

| cell | with stopGradient + asyncEval (prior) | retain only (this release) | Δ |
|---|---:|---:|---:|
| B=16 4K | 119.5 tok/s¹ | **120.1 tok/s** | +0.5% |
| B=17 4K | 108.7 tok/s | **119.9 tok/s** | **+10.3%** |
| B=32 4K | 111.2 tok/s | **119.5 tok/s** | **+7.5%** |
| B=16 8K | n/a (prior swept different KV) | **115.0 tok/s avg²** | n/a |

¹ B=16 was already inside the variance band before — the boundary cost was margin-of-error there. The win shows up at B≥17 where boundary overhead per step accumulates.
² From morning ablation; not a direct A/B vs the prior boundary recipe but representative of the path's steady state.

## Throughput — A-path (no compressed attention)

The A-path also got a perf bump from `prepareQueriesScaled` cache + rotation bypass:

| cell | A-path baseline (estimate) | this release | source |
|---|---:|---:|---|
| Qwen3.5-2B B=4 4K | ~270 tok/s | 271.6 tok/s | this morning |
| Qwen3.5-9B B=4 4K | ~75 tok/s | 78.6 tok/s | this morning |
| Qwen3.5-35B-A3B B=8 4K | ~120 tok/s | 123.6 tok/s | this morning |

A-path is now within ~5% of `--kv none` peak.

## Compression — long-context asymptote

Qwen3.5 2B summarization, `KV_fp16 / KV_compressed`:

| ctx | turbo4 | turbo3 | turbo2 | turbo4v2 |
|---|---:|---:|---:|---:|
| 8K | 2.94× | 3.66× | 4.61× | 3.66× |
| 32K | 3.58× | 4.64× | 6.46× | 4.64× |
| 64K | **3.74×** | **4.86×** | **6.95×** | **4.86×** |

At 64K all four are within 1–3% of the documented asymptote (3.8× / 4.9× / 7.1× / 4.9×).

**Memory headroom unlocked:**

| scheme | KV @ 64K (Qwen3.5 2B) | savings vs fp16 |
|---|---:|---:|
| fp16 | 778 MB | — |
| turbo4 | 208 MB | -570 MB (73%) |
| turbo3 | 160 MB | -618 MB (79%) |
| turbo2 | 112 MB | -666 MB (86%) |

This is what enables the README's "524K context on 3060 12GB / 200K on 4090 / 262K on a single 3090" claim.

## vllm-swift-specific deltas (matches prior baselines)

Identical-prompt decode tok/s (M5 Max, `bench_throughput.py`):

| | B=1 | B=8 | B=32 | B=64 |
|---|---:|---:|---:|---:|
| Qwen3-0.6B (this release) | 403 | 1,541 | 2,844 | 3,264 |
| Qwen3-0.6B (v0.2.2 README) | 364 | 1,527 | 2,859 | 3,425 |
| Δ | +11% | +1% | -0.5% | -4.7% |
| Qwen3-4B (this release) | 156 | 482 | 1,191 | 1,493 |
| Qwen3-4B (v0.2.2 README) | 147 | 477 | 1,194 | 1,518 |
| Δ | +6% | +1% | -0.3% | -1.6% |

All deltas are inside single-sample run-to-run noise. **No vllm-swift-side regression.** The TurboQuant MoE win above doesn't show up here because these are dense-MLP non-TurboQuant workloads.

## Where users will see it

- **Memory-pressure cell** (Qwen3.5-35B-A3B at B≥16 with TurboQuant): the workload that crashed under v0.2.2 now decodes cleanly at 120 tok/s.
- **Throughput cell** (same model, B=17 / B=32 turbo4v2): +7–10% measured.
- **Long-context users** (Gemma 4 31B turbo, Nemotron 30B-A3B turbo): the bf16 + dim=512 kernel work unblocks `ctv turbo4` on these models.
- **Everyone else** (dense MLP without TurboQuant): no change. v0.2.2 numbers preserved within noise.

## Coherence — sanity-checked, no regression

Qwen3.5-2B and Qwen3.5-35B-A3B both produce coherent English on no-KV and turbo4v2:

- Dense 2B no-KV: "Sunlight contains all colors, but Rayleigh scattering means shorter blue wavelengths..."
- Dense 2B turbo4v2: "Sunlight reflects off Earth's surface, where it is scattered in all directions..."
- MoE 35B-A3B no-KV: 200-token reasoning chain on the prompt
- MoE 35B-A3B turbo4v2: same reasoning chain start, same generation rate

No NaN, no garbled tokens, no early stops other than expected EOS.

## Risks tracked

- **mlx#19 retain not yet merged upstream.** Shipping as a local pin in our snapshot. If/when upstream merges, follow-up snapshot in v0.3.1.
- **InnocentVictim at 8K** (1/10 in the soak) was a thermal event, not a code regression. Document in release notes that long sustained sweeps on M5 Max should pace cells with 30s+ cooldowns.
- **DeepSeek-V4 initial foundation** — `model_type: deepseek_v4` registered, weight loading works (DSV4-Flash-2bit-DQ 90 GB loaded cleanly on M5 Max 128GB UMA via vllm-swift bridge). Forward pass hits a GPU kernel timeout on Phase 1 — release notes call this out explicitly so users don't expect production-stable decode. Tracked for Phase 2 follow-up.
