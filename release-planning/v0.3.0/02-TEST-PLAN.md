# v0.3.0 — Pre-release Test Plan

Each row is a pass/fail gate. Captured today's runs already cover most of the matrix; remaining gaps are the integration and clean-machine cells.

## Gate summary

| gate | done today | status |
|---|---|---|
| pytest 89/89 (local) | ✓ | pass |
| GitHub Actions CI green on the release commit | — | needed (auto-runs on push to main / PR) |
| build clean (alpha + retain submodule + DeepSeek cherry-pick) | ✓ | pass |
| dense + MoE coherence on no-KV + turbo4v2 | ✓ | pass |
| compression at 8K / 32K / 64K matches documented asymptote | ✓ | pass |
| short-context throughput (0.6B / 4B B=1..64) within ±5% of README baseline | ✓ | pass |
| MoE high-B stability (35B-A3B B=16/17/32 at QD=64 + retain) | ✓ | 25/25 from morning ablation |
| OpenAI server end-to-end (`vllm-swift serve` → curl /v1/completions) | — | needed |
| OpenCode integration (real coding session, multi-turn) | — | needed |
| Hermes integration (mobile workload smoke) | — | needed |
| Soak run at sustained MoE B=16 8K, ≥30 min, no thermal-related failures | — | needed (skip if M5 thermal pressure resurfaces) |
| Bottle builds locally on M5 Max | — | needed |
| Fresh `brew install` on Mac Mini (M2) | — | needed |
| `vllm-swift version` reports 0.3.0 | — | needed (post-bump) |
| Long-ctx vs vllm-metal capacity test (4B / B=64 / 8K) | already in v0.2.2 | passes (re-verify) |

## Detailed cells

### A. Unit + lint (run from `vllm-swift/`)

```bash
python3 -m pytest tests/ --cov=vllm_swift --cov-fail-under=95 -v
ruff check vllm_swift/ tests/
ruff format --check vllm_swift/ tests/
```

Pass: 89/89, coverage ≥ 95%, no lint errors.

### A2. GitHub Actions CI

Triggered automatically on push to `main` and on PR. Defined in `.github/workflows/ci.yml`. Mirrors the local `A` gate (pytest + ruff lint + format) on `macos-15` with Python 3.12 and CPU PyTorch wheel.

Pre-release sequence:
1. Push the version-bump commit to a feature branch first (`release/v0.3.0-prep`), open a PR against `main`.
2. Wait for the CI run on the PR to go green. Watch:
   - Lint job
   - Test job (pytest must pass with `--cov-fail-under=85`)
3. Only merge to `main` once green. The merge re-triggers CI on `main` itself — wait for that to also go green before tagging.
4. Tag `v0.3.0` after the post-merge `main` CI run is green.

Pass: both the PR CI run and the post-merge `main` CI run are green. If the lint job fails after the bump, run `ruff format vllm_swift/ tests/` locally and amend the bump commit before re-pushing. If the test job fails, the failure is the gate — do not tag.

### B. Bridge build

```bash
cd swift && swift build -c release
ls -la .build/arm64-apple-macosx/release/libVLLMBridge.dylib
```

Pass: builds clean, dylib present. Already verified today.

### C. Throughput sanity (matches README baseline)

`baseline-2026-04-26.md` cells, `bench_throughput.py` identical-prompts mode.

Qwen3-0.6B-4bit decode tok/s:

| B | README expected | acceptable (±10%) |
|---:|---:|---|
| 1 | 364 | 327–400 |
| 8 | 1,527 | 1,374–1,679 |
| 32 | 2,859 | 2,573–3,144 |
| 64 | 3,425 | 3,082–3,767 |

Qwen3-4B-4bit decode tok/s:

| B | README expected | acceptable (±10%) |
|---:|---:|---|
| 1 | 147 | 132–161 |
| 8 | 477 | 429–524 |
| 32 | 1,194 | 1,074–1,313 |
| 64 | 1,518 | 1,366–1,669 |

Pass: all cells within ±10% (allows for thermal / single-sample noise).

### D. Coherence (dense + MoE, all turbo schemes)

Use `mlx-swift-lm` bench harness from `~/dev/mlx-swift-lm`:

```bash
cd ~/dev/mlx-swift-lm
for kv in none turbo4 turbo3 turbo2 turbo4v2; do
  MLX_BENCH_BATCH=1 MLX_BENCH_METHOD=simple MLX_BENCH_KV=$kv \
  MLX_BENCH_QUANT=4bit MLX_BENCH_MODEL=qwen35-2b MLX_BENCH_CONTEXT=4096 \
  MLX_BENCH_PROMPT="Explain in two sentences why the sky is blue." MLX_BENCH_MAX_TOKENS=120 \
  swift test -c release --skip-build --no-parallel --filter benchmark 2>&1 \
    | grep "^\[BENCH\] Output:"
done
```

Pass: every output is recognisable English on-topic for the prompt. No NaN / null / repeated tokens / empty strings.

### E. Compression asymptote

Already captured (this session). Reproduce as a smoke test:

```bash
for ctx in 8192 32768 65536; do
  for kv in none turbo4 turbo3 turbo2 turbo4v2; do
    # capture KV Cache MB
  done
done
```

Pass: at 64K, ratios are within 3% of (3.8×, 4.9×, 7.1×, 4.9×).

### F. Capacity (4B / p=8K / B=64)

The cell that previously OOMed on v0.2.1. v0.2.2 fixed it. Confirm v0.3.0 still passes.

```bash
cd ~/dev/vllm-swift
DYLD_LIBRARY_PATH=swift/.build/arm64-apple-macosx/release \
  python3 scripts/bench_throughput.py ~/models/Qwen3-4B-4bit
```

Pass: B=64 cell completes without OOM. Decode tok/s reported.

### G. Server end-to-end

```bash
vllm-swift serve Qwen/Qwen3-0.6B-MLX-4bit --max-model-len 4096 &
sleep 20
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B-MLX-4bit","prompt":"hello","max_tokens":20,"temperature":0}'
kill %1
```

Pass: response with non-empty `choices[0].text`, no traceback in server log.

### H. OpenCode + Hermes smoke

Manual: load each app pointing at the local `vllm-swift serve` endpoint. Confirm a few turns of chat / tool use feel normal. Watch server log for any silent errors.

Pass: subjective. Capture a screenshot or short transcript of each.

### I. Soak run (optional but recommended)

```bash
# Sustained MoE B=16 8K for 30 min, retain on, MLX_QUEUE_DEPTH=64
# Drop if thermal log shows heavy pressure — same workload that crashed M5 Max yesterday
```

Pass: no Invalid Resource, no InnocentVictim, no thermal panic. Throughput stable across the run (no degradation > 10% from start to end).

### J. Bottle build + Mac Mini M2 upgrade-path gate

The single most important gate before declaring the release shipped. Three paths must all work on `toms-mac-mini.local`:

```bash
./scripts/build_bottle.sh
# Capture bottle SHA from output
```

#### J1. Fresh install (clean machine)

```bash
ssh toms-mac-mini.local
brew uninstall vllm-swift 2>/dev/null; brew untap TheTom/tap 2>/dev/null
rm -rf $(brew --cache)/downloads/*vllm*
brew tap TheTom/tap && brew install vllm-swift
vllm-swift version                 # expect: 0.3.0
vllm-swift serve Qwen/Qwen3-0.6B-MLX-4bit --max-model-len 1024 &
sleep 20
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B-MLX-4bit","prompt":"hello","max_tokens":20}'
kill %1
```

Pass: install completes, version reports 0.3.0, server returns non-empty completion.

#### J2. Upgrade from v0.2.2

```bash
# Pre-state: ensure v0.2.2 is installed first (point tap at the v0.2.2 commit
# of the formula, install, then point tap back to current and brew upgrade).
ssh toms-mac-mini.local
# Install v0.2.2 first by checking out the formula at the prior commit
brew uninstall vllm-swift 2>/dev/null
cd $(brew --repository TheTom/tap) && git stash && git checkout <PRIOR_COMMIT_SHA>
brew install vllm-swift
vllm-swift version                 # expect: 0.2.2
# Now apply the new formula
cd $(brew --repository TheTom/tap) && git checkout main && git pull
brew upgrade vllm-swift
vllm-swift version                 # expect: 0.3.0
vllm-swift serve Qwen/Qwen3-0.6B-MLX-4bit --max-model-len 1024 &
sleep 20
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B-MLX-4bit","prompt":"hello","max_tokens":20}'
kill %1
```

Pass: upgrade completes, version reports 0.3.0, server returns non-empty completion.

#### J3. Pin behavior (v0.2.2 user opts to stay put)

```bash
ssh toms-mac-mini.local
brew uninstall vllm-swift 2>/dev/null
# Re-install v0.2.2 same as J2 setup
cd $(brew --repository TheTom/tap) && git checkout <PRIOR_COMMIT_SHA>
brew install vllm-swift
brew pin vllm-swift
# Update tap to v0.3.0 formula
cd $(brew --repository TheTom/tap) && git checkout main && git pull
brew upgrade                       # vllm-swift should be skipped due to pin
vllm-swift version                 # expect: 0.2.2 (still pinned)
brew unpin vllm-swift              # cleanup
```

Pass: `brew upgrade` skips vllm-swift, version stays at 0.2.2 until unpinned.

**Hold rules:** if any of J1/J2/J3 fail, do not publish the GitHub Release or post on X. Either patch the formula and rebuild the bottle, or roll back per the rollback plan below.

#### J4. Mac Mini M2 performance + coherence

After each successful install in J1 and J2, run a perf + coherence pair to confirm the bottle isn't just "starts" but actually working. Run on `toms-mac-mini.local`.

**Throughput sanity (Qwen3-0.6B-4bit, identical-prompt mode):**

```bash
ssh toms-mac-mini.local
cd $(brew --prefix)/Cellar/vllm-swift/0.3.0/libexec/share/vllm-swift  # or wherever bottle drops scripts
# If bottle doesn't ship bench scripts, use a one-shot via the bridge:
DYLD_LIBRARY_PATH=$(brew --prefix vllm-swift)/lib \
  python3 -c "
import ctypes, time, os
lib = ctypes.CDLL(os.environ['DYLD_LIBRARY_PATH'] + '/libVLLMBridge.dylib')
# bind subset
lib.vsm_engine_create.restype = ctypes.c_void_p
lib.vsm_engine_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32, ctypes.c_float]
# … use the same flow as scripts/bench_throughput.py but with B=1 only and a tiny prompt
"
```

Or simpler — just hit the running server with a curl-based throughput probe:

```bash
# server already running from J1/J2
START=$(date +%s.%N)
RESP=$(curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B-MLX-4bit","prompt":"Write 50 words about Apple Silicon performance:","max_tokens":50,"temperature":0}')
END=$(date +%s.%N)
echo "elapsed: $(echo $END - $START | bc)s"
echo "response: $RESP" | python3 -m json.tool
```

Pass criteria for M2 Mini:

- elapsed time for 50 tokens reasonable (M2 Mini Qwen3-0.6B-4bit baseline ≈ 50–80 tok/s at B=1, so ~0.6–1.0s for 50 decode tokens + ~150ms prefill + HTTP overhead — under 3s end-to-end is healthy)
- response text is coherent English on-topic for the prompt (no NaN tokens, no empty completion, no garbage)

**Coherence (deterministic prompt, capture text):**

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B-MLX-4bit","prompt":"In one sentence, why is the sky blue?","max_tokens":40,"temperature":0}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'])"
```

Pass: output is a recognisable answer about light scattering, Rayleigh, atmosphere, blue wavelengths, etc. No `<unused>`, no NaN, no character repetition, no truncation mid-token.

**Hold rules (extended):** if J4 fails on either J1 or J2 install path on the Mini, do not publish. Treat M2 + M5 Max as separate gates — both must pass perf and coherence.

## Rollback plan

If any post-release gate fails (E.g., bottle install crashes on M2, server times out, etc.):

1. `brew untap TheTom/tap` on affected machines.
2. Pin `homebrew-tap` formula back to v0.2.2 SHA.
3. Pull v0.3.0 tag from `vllm-swift` repo only if fundamentally broken.
4. Triage in a `v0.3.1` patch release.
