# Performance

Canonical performance reference for vllm-swift. This document covers methodology, current numbers, where vllm-swift wins and where it loses, and pointers to date-locked benchmark snapshots.

For a quick summary, the [README](../README.md) covers the headline short-context decode numbers. This document goes deeper.

## Hardware and software

All numbers in this document were measured on:

- Apple M5 Max, 128 GB unified memory
- macOS 26.x (Darwin 25.x)
- mlx-swift-lm pinned revision (recorded in `swift/Package.resolved`)
- vLLM core 0.19.x (vllm-metal comparison)

## Headline: short-context decode

Decode tok/s, prompt = 18 tokens, generation = 50 tokens, greedy (temp=0), unique prompts (no prefix-cache hits). Both engines measured via offline benchmark, no HTTP overhead.

### Qwen3-0.6B-4bit

| | Single | 8 concurrent | 32 concurrent | 64 concurrent |
|---|:---:|:---:|:---:|:---:|
| **vllm-swift** | **364** | **1,527** | **2,859** | **3,425** |
| vllm-metal (Python/MLX) | 111 | 652 | 2,047 | 2,620 |
| vllm-swift speedup | 3.30× | 2.34× | 1.40× | 1.31× |

### Qwen3-4B-4bit

| | Single | 8 concurrent | 32 concurrent | 64 concurrent |
|---|:---:|:---:|:---:|:---:|
| **vllm-swift** | **147** | **477** | **1,194** | **1,518** |
| vllm-metal (Python/MLX) | 104 | 396 | 1,065 | 1,375 |
| vllm-swift speedup | 1.41× | 1.20× | 1.12× | 1.10× |

vllm-swift is faster at every measured cell. The advantage is largest at low concurrency on small models, where Python scheduler overhead is a higher fraction of total time. At higher concurrency, the gap narrows as compute dominates.

Numbers are measured under apples-to-apples conditions: unique prompts (no prefix-cache deduplication), subprocess-per-cell isolation, identical model files for both engines.

## Long-context decode

Decode tok/s under unique prompts, no prefix cache hits. Single-sample measurements; median-of-3 publication-grade re-run pending. See [benchmarks/baseline-2026-04-26.md](../benchmarks/baseline-2026-04-26.md) for full matrix and methodology.

### Qwen3-4B-4bit, B=64

| Context | vllm-swift decode | vllm-metal decode | vllm-swift advantage |
|---|---:|---:|---:|
| 18 tokens (short) | 1,480 | 1,315 | 1.13× |
| 2K | 730 | 62 | 11.7× |
| 4K | 475 | 23 | 21.0× |
| 8K | 259 | 7 | 37× |

### Qwen3-0.6B-4bit, B=64

| Context | vllm-swift decode | vllm-metal decode | vllm-swift advantage |
|---|---:|---:|---:|
| 18 tokens (short) | 3,186 | 2,741 | 1.16× |
| 2K | 1,262 | 224 | 5.63× |
| 4K | 800 | 75 | 10.7× |
| 8K | 432 | 22 | 19.6× |

Bench harnesses that run identical prompts (a common pattern: `[prompt] * B`) let prefix-caching engines dedupe shared prefixes, producing measurements that reflect cache-hit speed rather than raw recompute. The numbers above are taken under unique prompts so both engines do the full per-request work. Under those conditions, vllm-swift's contiguous BatchedKVCache outperforms paged attention at decode time across context lengths. The gap is widest at high concurrency and long context, where per-step block-table indirection costs the most.

## Capacity

Both engines run every cell in the matrix.

### Qwen3-4B-4bit, 8K context, B=64

| Engine | Decode tok/s | Notes |
|---|---:|---|
| vllm-swift | 259 | Contiguous BatchedKVCache, eager release of per-request KVCacheSimple in init_batched copy loop |
| vllm-metal | 22 | Paged storage allocates blocks on demand |

vllm-swift's contiguous BatchedKVCache preserves the fast path (no per-slot dispatch overhead) while keeping peak memory bounded. Implementation detail in [BatchedKVCache.swift](https://github.com/TheTom/mlx-swift-lm/blob/feat/paged-attention/Libraries/MLXLMCommon/BatchedKVCache.swift).

## Methodology

### Bench scripts

- `scripts/bench_throughput.py` — vllm-swift via ctypes bridge
- `scripts/bench_vllm_metal.py` — vllm-metal via vLLM offline `LLM` API, subprocess-per-level
- `scripts/baseline_matrix.py` — orchestrator that runs both at each cell

### Required for trustworthy comparisons

1. **Unique prompts.** Identical prompts let vllm-metal's prefix caching dedupe up to N-1 of N requests, producing a measurement that is neither raw compute nor a realistic workload. The `--prompts unique` flag generates B distinct prompts.
2. **Subprocess-per-level isolation.** vLLM's `LLM` class spawns `EngineCore` as a child subprocess that does not reap on `del llm` (see [#19849](https://github.com/vllm-project/vllm/issues/19849), [#1908](https://github.com/vllm-project/vllm/issues/1908), [#17273](https://github.com/vllm-project/vllm/issues/17273), [#24885](https://github.com/vllm-project/vllm/issues/24885)). The bench script spawns a fresh subprocess per concurrency level so subsequent cells start clean.
3. **Same models, same hardware, same load.** No background GPU consumers during measurement.
4. **Multiple metrics per cell.** Both `bench_throughput.py` and `bench_vllm_metal.py` capture `prefill_ms`, `decode_elapsed`, `tps_e2e` (gen / (prefill + decode)), and `tps_decode` (gen / decode only). Comparing engines at different metrics produces meaningless results, especially at long context.

### Reproducing

```bash
# vllm-swift baseline
DYLD_LIBRARY_PATH=swift/.build/arm64-apple-macosx/release \
    python3 scripts/bench_throughput.py ~/models/Qwen3-4B-4bit \
    --tokens 50 --prompt-tokens 18 --prompts unique

# vllm-metal baseline (from the vllm-metal venv)
~/.venv-vllm-metal/bin/python3 scripts/bench_vllm_metal.py ~/models/Qwen3-4B-4bit \
    --tokens 50 --prompt-tokens 18 --prompts unique
```

## Where vllm-swift wins

- Raw decode compute at every measured cell, every concurrency level, both models tested.
- Short-context low-concurrency by the largest margin (Python overhead dominates at small workloads).
- Long-context high-B decode by 5–12× (contiguous BatchedKVCache avoids paged block-table indirection).
- Capacity at 4B/8K/B=64 (12× faster than vllm-metal at this cell).

## Where vllm-metal still wins

- **Workloads with shared prompt prefixes.** vllm-metal's prefix caching deduplicates KV across requests with shared system prompts, conversation history, or repeated tool-call patterns. vllm-swift currently has no equivalent feature; for those workloads vllm-metal's reported throughput accurately reflects production behavior. Prefix caching is on the vllm-swift roadmap.
- **Variable-length high-concurrency packing.** Paged attention packs sequences of varying lengths into shared blocks more memory-efficiently than vllm-swift's contiguous cache. vllm-swift handles this via the post-fix BatchedKVCache for fixed-size workloads but does not yet match vllm-metal's flexibility.
- **Models requiring vllm-only features.** Speculative decoding (Medusa, EAGLE), structured output via grammar-based decoding, and other vLLM-specific features are not yet ported.

## Active investigation and roadmap

- **Prefix caching feature.** Block-hash → KV slice mapping, refcounted blocks, LRU eviction. Architecture compatible with future paged storage.
- **PagedAttention foundation.** `PagedKVCache` + `BlockAllocator` mirror vllm-metal's block-based layout. Foundation committed; Metal kernel port pending.
- **Long-context median-of-3.** Single-sample numbers above will be replaced with median-of-3 measurements before any external comparison claims.

## Date-locked snapshots

- [benchmarks/baseline-2026-04-26.md](../benchmarks/baseline-2026-04-26.md) — unique-prompts apples-to-apples, post-OOM-fix capacity verification, decode-only and e2e per cell, 5-37× decode wins at long ctx
- [benchmarks/baseline-2026-04-25.md](../benchmarks/baseline-2026-04-25.md) — original short-ctx + long-ctx matrix, methodology, vs-README diff, attention microbench (long-ctx cells superseded by 2026-04-26)
- Raw JSON cell data: [benchmarks/baseline-2026-04-25-*.json](../benchmarks/)

## Related

- [PAGED_ATTENTION_TESTPLAN.md](PAGED_ATTENTION_TESTPLAN.md) — paged attention design and Phase 3 success criteria
- [MLX_SWIFT_API_SPIKE.md](MLX_SWIFT_API_SPIKE.md) — answers to MLX-Swift API capability questions for kernel work
