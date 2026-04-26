#!/usr/bin/env python3
"""Same-harness throughput bench for mlx-lm (Python, no vLLM, no bridge).

Provides a third datapoint alongside vllm-swift (bridge over mlx-swift-lm)
and vllm-metal (vLLM scheduler over mlx-metal backend). Isolates whether
the bridge's sequential prefill is structural to mlx-swift-lm or specific
to the vllm-swift bridge layer.

Usage:
  /Users/tom/.venv-vllm-metal/bin/python3 scripts/bench_mlx_lm.py [model_path] \
    [--prompt-tokens N] [--tokens N] [--worker B]
"""

import json
import os
import subprocess
import sys
import time

MODEL_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
    else os.path.expanduser("~/models/Qwen3-4B-4bit")
)
MAX_TOKENS = 50
CONCURRENCY_LEVELS = [1, 8, 32, 64]
PROMPT_TOKENS = 0
for i, arg in enumerate(sys.argv):
    if arg == "--prompt-tokens" and i + 1 < len(sys.argv):
        PROMPT_TOKENS = int(sys.argv[i + 1])
    if arg == "--tokens" and i + 1 < len(sys.argv):
        MAX_TOKENS = int(sys.argv[i + 1])


def _build_prompt() -> str:
    seed = (
        "Explain the theory of relativity in detail, covering both "
        "special and general relativity:"
    )
    if PROMPT_TOKENS == 0:
        return seed
    target_chars = PROMPT_TOKENS * 4
    out = []
    while sum(len(s) for s in out) < target_chars:
        out.append(seed)
    return " ".join(out)


def run_worker(B: int) -> dict:
    """One concurrency level. mlx-lm has no native concurrent generate, so
    we do batched prefill via a single forward over [B, T] then sequential
    decode (or batched decode if mlx-lm exposes it). For honest comparison
    we time prefill and decode separately."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(MODEL_PATH)

    prompt = _build_prompt()
    input_ids = tokenizer.encode(prompt)
    if PROMPT_TOKENS > 0:
        # Pad/trim same as bench_throughput.py
        while len(input_ids) < PROMPT_TOKENS:
            input_ids.extend(input_ids[: min(len(input_ids), PROMPT_TOKENS - len(input_ids))])
        input_ids = input_ids[:PROMPT_TOKENS]

    # Batched prefill: [B, T] in one forward
    batch = mx.array([input_ids] * B)  # [B, T]

    # warmup
    _ = model(batch[:1])
    mx.eval(_)

    # ---- Prefill (timed)
    t_prefill = time.perf_counter()
    logits = model(batch)  # [B, T, V]
    mx.eval(logits)
    prefill_elapsed = time.perf_counter() - t_prefill

    # First token from last position
    first_token = mx.argmax(logits[:, -1, :], axis=-1)  # [B]
    mx.eval(first_token)

    # ---- Decode loop (timed)
    # mlx-lm doesn't expose a clean batched-decode API publicly. Approximate
    # by stepping the model with the last predicted token + a growing cache.
    # For honest per-token throughput at high B we'd need a real batched cache.
    # Use mlx-lm's stream_generate for B=1; for B>1 fall back to manual stepping.
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    # Re-run prefill into cache
    _ = model(batch, cache=cache)
    mx.eval(_)

    last_tokens = first_token  # [B]
    total_tokens = B  # count first token
    t_decode = time.perf_counter()
    for _ in range(MAX_TOKENS - 1):
        step_input = last_tokens[:, None]  # [B, 1]
        step_logits = model(step_input, cache=cache)
        last_tokens = mx.argmax(step_logits[:, -1, :], axis=-1)
        mx.eval(last_tokens)
        total_tokens += B
    decode_elapsed = time.perf_counter() - t_decode

    tps_decode = total_tokens / decode_elapsed if decode_elapsed > 0 else 0
    tps_e2e = total_tokens / (prefill_elapsed + decode_elapsed) if (prefill_elapsed + decode_elapsed) > 0 else 0

    return {
        "B": B,
        "tps_e2e": tps_e2e,
        "tps_decode": tps_decode,
        "prefill_elapsed": prefill_elapsed,
        "decode_elapsed": decode_elapsed,
        "total_tokens": total_tokens,
    }


def main_worker():
    B = int(sys.argv[sys.argv.index("--worker") + 1])
    result = run_worker(B)
    print("MLX_RESULT_JSON " + json.dumps(result))


def main_orchestrator():
    print(f"Model: {MODEL_PATH}")
    print(f"Prompt tokens: {PROMPT_TOKENS}")
    print(f"Max gen tokens: {MAX_TOKENS}")
    print(f"Concurrency: {CONCURRENCY_LEVELS}")
    print()

    results = []
    for B in CONCURRENCY_LEVELS:
        print(f"--- B={B} (fresh subprocess) ---", flush=True)
        cmd = [
            sys.executable, "-u", __file__, MODEL_PATH,
            "--worker", str(B),
            "--tokens", str(MAX_TOKENS),
        ]
        if PROMPT_TOKENS > 0:
            cmd += ["--prompt-tokens", str(PROMPT_TOKENS)]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            print(f"  WORKER FAIL exit={proc.returncode}: {proc.stderr[-300:]}")
            continue
        line = next((l for l in proc.stdout.splitlines() if l.startswith("MLX_RESULT_JSON ")), None)
        if not line:
            print(f"  no JSON: {proc.stdout[-300:]}")
            continue
        data = json.loads(line.removeprefix("MLX_RESULT_JSON "))
        print(
            f"  B={data['B']:3d}: e2e={data['tps_e2e']:,.1f} decode={data['tps_decode']:,.1f} tok/s "
            f"[prefill={data['prefill_elapsed']*1000:.0f}ms, decode={data['decode_elapsed']:.2f}s]"
        )
        results.append(data)

    print()
    print(f"=== {os.path.basename(MODEL_PATH)} — mlx-lm (Python, no vLLM) ===")
    print(f"{'B':>4} {'E2E tok/s':>11} {'Decode tok/s':>13} {'Prefill ms':>11}")
    print("-" * 44)
    for r in results:
        print(f"{r['B']:>4d} {r['tps_e2e']:>11,.1f} {r['tps_decode']:>13,.1f} {r['prefill_elapsed']*1000:>11.0f}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        main_worker()
    else:
        main_orchestrator()
