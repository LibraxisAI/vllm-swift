#!/usr/bin/env python3
"""Concurrent decode throughput benchmark for vllm-metal (Python/MLX).

Spawns a fresh subprocess per concurrency level. vLLM's EngineCore runs
in a child process and `del LLM` does NOT tear it down — multiple zombie
EngineCores accumulate (~3GB each) and contaminate subsequent runs. The
only reliable cleanup is process exit. Hence: subprocess.run per level.

Usage:
  cd /Users/tom/dev/vllm-metal
  source .venv-vllm-metal/bin/activate
  python3 /Users/tom/dev/vllm-swift/scripts/bench_vllm_metal.py [model_path]

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


def run_worker(B: int) -> tuple[float, float, int]:
    """Inside child process: run one concurrency level and return stats."""
    from vllm import LLM, SamplingParams  # imported here so import cost is per-process

    prompt = (
        "Explain the theory of relativity in detail, covering both "
        "special and general relativity:"
    )

    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        disable_log_stats=True,
    )

    # Warmup (separate from timed run)
    llm.generate(["Hello"], SamplingParams(temperature=0, max_tokens=5))

    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    prompts = [prompt] * B

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / elapsed if elapsed > 0 else 0.0

    # Process exit cleans up EngineCore subprocess + GPU memory
    return tps, elapsed, total_tokens


def main_worker():
    """Worker entry: print one JSON line on stdout, exit."""
    B = int(sys.argv[sys.argv.index("--worker") + 1])
    tps, elapsed, total_tokens = run_worker(B)
    print(
        "VSM_RESULT_JSON " + json.dumps({
            "B": B,
            "tps": tps,
            "elapsed": elapsed,
            "total_tokens": total_tokens,
        })
    )


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
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
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
        per_req = data["tps"] / data["B"] if data["B"] > 0 else 0
        print(
            f"  B={data['B']:3d}: {data['tps']:,.1f} tok/s total "
            f"({per_req:,.1f} per request) "
            f"[{data['elapsed']:.2f}s, {data['total_tokens']} tokens]"
        )
        results.append(data)

    # Summary table
    print()
    print(f"=== {os.path.basename(MODEL_PATH)} — vllm-metal (Python/MLX) ===")
    print(f"{'Concurrency':>12} {'Total tok/s':>14} {'Per-request':>14}")
    print("-" * 44)
    for r in results:
        per_req = r["tps"] / r["B"] if r["B"] > 0 else 0
        print(f"{r['B']:>12d} {r['tps']:>14,.1f} {per_req:>14,.1f}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        main_worker()
    else:
        main_orchestrator()
