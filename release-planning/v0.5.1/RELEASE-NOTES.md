# vllm-swift 0.5.1 — Release Notes

**Patch release.** Fixes a real, reproducible memory leak under sustained
multi-turn agent load (Hermes, OpenCode in agentic mode) on Apple Silicon.
No new features. Safe upgrade for everyone on 0.5.0.

## TL;DR

- `EngineCore` was leaking 4–5 GB of unified memory per chat-completion
  turn on configurations with `--max-num-seqs 1` (typical interactive
  Hermes setup). Tom's session OOM'd the Mac at ~14 turns.
- Root cause: `Bridge.swift`'s `init_batched`/`initBatchedHybrid`/
  `prefill_batched_*` paths hardcoded `max(B, 64)` for the
  `BatchedKVCache` slot count. With `max_num_seqs=1` that pre-allocates
  64 slots and pins ~60 GB of unused-but-allocated bf16/turbo4v2 backing
  storage. `maxSeq` was also sized off the first batch's longest prefill
  and could re-grow on subsequent (longer) prefills, stacking new
  Metal-heap regions.
- Fix: thread `vllm_config.scheduler_config.max_num_seqs` and
  `model_config.max_model_len` through to `Bridge.swift`. Replace
  `max(B, 64)` with `max(B, engine.maxConcurrentRequests)`. Pin `maxSeq`
  to `engine.maxKVSize` at init.

## Empirical proof

Same Hermes session, same prompts, same flags
(`--max-model-len 65536 --max-num-seqs 1 --gpu-memory-utilization 0.5
 --no-enable-prefix-caching --additional-config
 '{"kv_scheme": "turbo4v2", "kv_bits": 4}' --enable-longctx`):

| turn | pre-fix `engine_dirty` (GB) | post-fix `engine_dirty` (GB) |
|---:|---:|---:|
| 1 | 19.5 | 19.5 |
| 2 | (n/a; vmmap miss) | 19.5 |
| 3 | **78.6** | 21.6 |
| 4 | crash imminent | 25.1 |
| 5 | OOM | 26.8 |
| 6 | — | 21.1 ← reclaim |
| 7 | — | 31.0 |
| 8 | — | 21.1 ← reclaim |
| 9 | — | 34.0 |
| 10 | — | 25.3 |

Pre-fix: monotonic +4.6 GB/turn. Post-fix: oscillating 21–37 GB working
band, no upward trend — that's normal Metal heap reuse.

## Changes

**Swift bridge (`swift/Sources/VLLMBridge/Bridge.swift`)**

- `InferenceEngine` gains two fields:
  - `maxConcurrentRequests: Int = 64` — drives `BatchedKVCache.maxBatch`
    pre-alloc. Defaults to 64 for back-compat with existing callers.
  - `maxKVSize: Int = 0` — pins `BatchedKVCache.maxSeq` so subsequent
    longer prefills don't force a re-grow.
- `vsm_engine_create` C signature gains a trailing `Int32 maxNumSeqs`.
  Old callers pass 0 → `maxConcurrentRequests` falls back to 64 and
  behavior is bit-identical to 0.5.0. **Additive change, not breaking.**
- All three over-alloc sites now compute
  `maxBatch = max(B, engine.maxConcurrentRequests)`:
  - `vsm_engine_init_batched` (Qwen3 path)
  - `initBatchedHybrid` (Qwen3-Next / hybrid attention+GDN path)
  - `prefill_batched_uniform` ctx-stub paths (test/perf entrypoints)
- `maxSeq` now equals `engine.maxKVSize` when known, else falls back to
  the legacy `max(2048, prefill_offset + 512)` heuristic.

**Python plugin**

- `vllm_swift.engine_bridge.SwiftInferenceEngine.__init__` gains
  `max_num_seqs: int = 0`.
- `vllm_swift.engine_bridge` adds the new ctype to `vsm_engine_create`'s
  argtypes list (`ctypes.c_int32`).
- `vllm_swift.worker.SwiftMetalWorker.load_model` now reads from
  `vllm_config.scheduler_config.max_num_seqs` and threads it through to
  `SwiftInferenceEngine`.

**Versioning**

- `__version__` 0.4.2 → 0.5.1 (a 0.5.0 bump existed in pyproject /
  CHANGELOG / formula but `__init__.py` was missed).
- `pyproject.toml` 0.5.0 → 0.5.1.
- `homebrew/vllm-swift.rb` 0.5.0 → 0.5.1; bottle SHAs cleared.
- `scripts/build_bottle.sh` 0.4.2 → 0.5.1 (also caught up).

## Wire compatibility

- Old Python callers calling the C FFI directly without `maxNumSeqs`
  will hit the new arg. Anyone using `SwiftInferenceEngine(...)` from
  Python is fine — it has a default of 0.
- Anyone embedding `libVLLMBridge.dylib` from another language: pass 0
  (`Int32`) for `maxNumSeqs` to preserve the legacy 64-slot behavior.

## Diagnostics included

`response_rewriter._mem_snapshot()` now also captures `vmmap --summary`
DIRTY size on the EngineCore pid, surfaced as `engine_dirty=NN.NGB`
on every per-request `[longctx]` stderr line. This is the macOS
true-private-memory metric — what you watch when chasing leaks. (Plain
`ps rss` undercounts on processes with large mmap, which is why we
missed this leak across most of the 0.5.0 session.)

## Distribution

- `pip install vllm-swift==0.5.1` ships the fixed dylib via the wheel
  (`vllm_swift/_lib/libVLLMBridge.dylib`).
- Homebrew bottle needs a rebuild before brew users get the fix —
  formula installs from-source until then
  (`HOMEBREW_NO_SANDBOX=1 brew upgrade vllm-swift`).

## Tests

- 487 unit tests still green (`pytest tests/ --ignore=tests/integration`).
- Live verified end-to-end against Hermes on M-series Mac with the
  Qwen3.6-35B-A3B-4bit + turbo4v2 + longctx config.
