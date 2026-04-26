#!/usr/bin/env python3
"""Concurrent decode throughput benchmark for vllm-metal (Python/MLX).

Spawns a fresh subprocess per concurrency level. vLLM's EngineCore runs
in a child process and `del LLM` does NOT tear it down — multiple zombie
EngineCores accumulate (~3GB each) and contaminate subsequent runs. The
only reliable cleanup is process exit. Hence: subprocess.run per level.

Usage:
  source ~/.venv-vllm-metal/bin/activate   # NOT vllm-metal/.venv-vllm-metal — that has vllm-swift
  python3 /Users/tom/dev/vllm-swift/scripts/bench_vllm_metal.py [model_path]

CRITICAL: only ~/.venv-vllm-metal has vllm-metal installed. The venv inside the
vllm-metal repo dir (/Users/tom/dev/vllm-metal/.venv-vllm-metal) has vllm-swift
installed — running through it benchmarks vllm-swift via the vLLM offline API,
NOT vllm-metal. Look for "Platform plugin metal is activated" in stderr to
confirm you're hitting the right backend.

  # Worker mode (called recursively by orchestrator):
  python3 .../bench_vllm_metal.py [model_path] --worker B
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
PROMPT_TOKENS = 0  # 0 = use short default; >0 = pad to N tokens
for i, arg in enumerate(sys.argv):
    if arg == "--prompt-tokens" and i + 1 < len(sys.argv):
        PROMPT_TOKENS = int(sys.argv[i + 1])


def _build_prompt() -> str:
    """Short default, or padded prompt of approximately PROMPT_TOKENS length.
    Returns text — vLLM does its own tokenization."""
    seed = (
        "Explain the theory of relativity in detail, covering both "
        "special and general relativity:"
    )
    if PROMPT_TOKENS == 0:
        return seed
    # ~3 chars per token rule of thumb for English. Pad with seed repetitions
    # then trim. vLLM will tokenize and truncate to whatever fits.
    target_chars = PROMPT_TOKENS * 4
    out = []
    while sum(len(s) for s in out) < target_chars:
        out.append(seed)
    return " ".join(out)


def run_worker(B: int) -> dict:
    """Inside child process: run one concurrency level and return stats.

    Returns both metrics:
      - tps_e2e:    total_tokens / wall_clock(prefill + decode) — end-to-end
      - tps_decode: total_tokens / max(per_request decode window) — pure decode rate

    The vllm-swift bridge bench (bench_throughput.py) measures decode-only
    by construction (prefill happens before t0). Without tps_decode here the
    cross-engine comparison is unfair at long context, where prefill dominates.
    """
    from vllm import LLM, SamplingParams

    prompt = _build_prompt()
    max_model_len = max(2048, PROMPT_TOKENS + MAX_TOKENS + 256)

    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        disable_log_stats=False,  # need RequestStateStats for decode window
    )

    llm.generate(["Hello"], SamplingParams(temperature=0, max_tokens=5))

    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    prompts = [prompt] * B

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps_e2e = total_tokens / elapsed if elapsed > 0 else 0.0

    # Decode window: max across requests (continuous batching → lockstep decode)
    decode_windows = []
    ttfts = []
    for o in outputs:
        m = getattr(o, "metrics", None)
        if m and m.first_token_ts and m.last_token_ts:
            decode_windows.append(m.last_token_ts - m.first_token_ts)
        if m and getattr(m, "first_token_latency", None):
            ttfts.append(m.first_token_latency)
    decode_window = max(decode_windows) if decode_windows else elapsed
    tps_decode = total_tokens / decode_window if decode_window > 0 else 0.0
    ttft = max(ttfts) if ttfts else 0.0

    return {
        "B": B,
        "tps_e2e": tps_e2e,
        "tps_decode": tps_decode,
        "elapsed": elapsed,
        "decode_window": decode_window,
        "ttft": ttft,
        "total_tokens": total_tokens,
        # back-compat alias — older parsers read "tps"
        "tps": tps_e2e,
    }


def main_worker():
    """Worker entry: print one JSON line on stdout, exit."""
    B = int(sys.argv[sys.argv.index("--worker") + 1])
    result = run_worker(B)
    print("VSM_RESULT_JSON " + json.dumps(result))


def main_orchestrator():
    print(f"Model: {MODEL_PATH}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Concurrency levels: {CONCURRENCY_LEVELS}")
    print()

    results = []
    for B in CONCURRENCY_LEVELS:
        print(f"--- B={B} (fresh subprocess) ---")
        # Spawn worker in a fresh Python interpreter. EngineCore subprocess
        # gets cleanly torn down on worker exit.
        cmd = [sys.executable, "-u", __file__, MODEL_PATH, "--worker", str(B)]
        # vllm-metal at B>=32 can take 5-15 minutes per concurrency level
        # (slow Python scheduler + uncached EngineCore startup ~30s each).
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0:
            print(f"  WORKER FAILED (exit {proc.returncode})")
            print(f"  stderr: {proc.stderr[-500:]}")
            continue

        # Find the result line
        result_line = next(
            (l for l in proc.stdout.splitlines() if l.startswith("VSM_RESULT_JSON ")),
            None,
        )
        if not result_line:
            print(f"  WORKER OUTPUT MISSING JSON: {proc.stdout[-500:]}")
            continue

        data = json.loads(result_line.removeprefix("VSM_RESULT_JSON "))
        e2e = data.get("tps_e2e", data["tps"])
        dec = data.get("tps_decode", e2e)
        ttft = data.get("ttft", 0.0)
        print(
            f"  B={data['B']:3d}: e2e={e2e:,.1f} decode={dec:,.1f} tok/s "
            f"[ttft={ttft*1000:.0f}ms, elapsed={data['elapsed']:.2f}s, "
            f"{data['total_tokens']} tokens]"
        )
        results.append(data)

    # Summary table
    print()
    print(f"=== {os.path.basename(MODEL_PATH)} — vllm-metal (Python/MLX) ===")
    print(f"{'B':>4} {'E2E tok/s':>11} {'Decode tok/s':>13} {'TTFT ms':>9}")
    print("-" * 42)
    for r in results:
        e2e = r.get("tps_e2e", r["tps"])
        dec = r.get("tps_decode", e2e)
        ttft = r.get("ttft", 0.0)
        print(f"{r['B']:>4d} {e2e:>11,.1f} {dec:>13,.1f} {ttft*1000:>9.0f}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        main_worker()
    else:
        main_orchestrator()
