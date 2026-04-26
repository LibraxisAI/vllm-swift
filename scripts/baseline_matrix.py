#!/usr/bin/env python3
"""Full baseline matrix: both engines × both models × all concurrency levels × median of 3.

Drives bench_throughput.py (vllm-swift) and bench_vllm_metal.py (vllm-metal),
parses their tok/s output, takes the median per cell, writes a markdown table.

Usage:
  ./scripts/baseline_matrix.py [--quick]
    --quick: 1 run per cell instead of 3 (smoke test)
"""

import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = ["Qwen3-0.6B-4bit", "Qwen3-4B-4bit"]
CONCURRENCY = [1, 8, 32, 64]
RUNS_PER_CELL = 1 if "--quick" in sys.argv else 3

VLLM_SWIFT_LIB = REPO / "swift" / ".build" / "arm64-apple-macosx" / "release"
VLLM_METAL_PYTHON = "/Users/tom/.venv-vllm-metal/bin/python3"


def run_vllm_swift(model_path: str) -> dict[int, float]:
    """Returns {B: tok/s} for vllm-swift bench across all concurrency levels."""
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "bench_throughput.py"),
        model_path,
        "--tokens", "50",
    ]
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = str(VLLM_SWIFT_LIB)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    out = proc.stdout
    # Lines like "B=  1: 381.4 tok/s total ..."
    results = {}
    for line in out.splitlines():
        m = re.match(r"\s*B=\s*(\d+):\s*([\d,\.]+)\s+tok/s", line)
        if m:
            B = int(m.group(1))
            tps = float(m.group(2).replace(",", ""))
            results[B] = tps
    return results


def run_vllm_metal_one(model_path: str, B: int, timeout: int = 600) -> float | None:
    """Single B run for vllm-metal. Returns tps or None on failure/timeout."""
    cmd = [
        VLLM_METAL_PYTHON,
        "-u",
        str(REPO / "scripts" / "bench_vllm_metal.py"),
        model_path,
        "--worker", str(B),
    ]
    env = os.environ.copy()
    env["PATH"] = "/Users/tom/.venv-vllm-metal/bin:" + env.get("PATH", "")
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"    [B={B}] TIMEOUT after {timeout}s", flush=True)
        return None
    if proc.returncode != 0:
        print(f"    [B={B}] FAIL exit={proc.returncode}", flush=True)
        if proc.stderr:
            print(f"    stderr: {proc.stderr[-300:]}", flush=True)
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("VSM_RESULT_JSON "):
            data = json.loads(line.removeprefix("VSM_RESULT_JSON "))
            return float(data["tps"])
    print(f"    [B={B}] no JSON result line", flush=True)
    return None


def run_vllm_metal(model_path: str) -> dict[int, float]:
    """Returns {B: tok/s} for vllm-metal — one subprocess per B, prints per-B."""
    results = {}
    # Smaller B should finish in <2 min; large B can take 10+ min.
    timeouts = {1: 180, 8: 300, 32: 900, 64: 1800}
    for B in CONCURRENCY:
        t0 = time.perf_counter()
        tps = run_vllm_metal_one(model_path, B, timeout=timeouts.get(B, 1800))
        dt = time.perf_counter() - t0
        if tps is not None:
            results[B] = tps
            print(f"    [B={B}] {tps:,.1f} tok/s ({dt:.1f}s wall)", flush=True)
    return results


def main():
    print(f"=== Baseline matrix (runs/cell={RUNS_PER_CELL}) ===\n")

    grid = {}  # {(engine, model): {B: median_tps}}
    raw = {}   # {(engine, model): {B: [tps...]}}  per-run samples for partial recovery
    json_path = REPO / "benchmarks" / "baseline-2026-04-25-raw.json"

    def flush_json():
        # Snapshot after every cell so partial data survives crashes.
        snapshot = {
            "runs_per_cell": RUNS_PER_CELL,
            "concurrency": CONCURRENCY,
            "samples": {f"{e}|{m}": s for (e, m), s in raw.items()},
            "median": {f"{e}|{m}": g for (e, m), g in grid.items()},
        }
        json_path.write_text(json.dumps(snapshot, indent=2))

    for model in MODELS:
        model_path = os.path.expanduser(f"~/models/{model}")
        for engine, runner in (("vllm-swift", run_vllm_swift), ("vllm-metal", run_vllm_metal)):
            print(f"\n[{engine}] {model}")
            t0 = time.perf_counter()
            samples: dict[int, list[float]] = {B: [] for B in CONCURRENCY}
            for r in range(RUNS_PER_CELL):
                print(f"  run {r+1}/{RUNS_PER_CELL}...", flush=True)
                tr0 = time.perf_counter()
                results = runner(model_path)
                dtr = time.perf_counter() - tr0
                print(f"    elapsed {dtr:.1f}s — {results}", flush=True)
                for B, tps in results.items():
                    samples[B].append(tps)
                # Flush partial data after every individual run
                raw[(engine, model)] = samples
                flush_json()
            medians = {B: statistics.median(s) for B, s in samples.items() if s}
            grid[(engine, model)] = medians
            flush_json()
            dt = time.perf_counter() - t0
            print(f"  done in {dt:.1f}s — {medians}", flush=True)

    # Markdown output
    print("\n\n## Results\n")
    for model in MODELS:
        print(f"### {model}\n")
        print(f"| Engine | B=1 | B=8 | B=32 | B=64 |")
        print(f"|---|---:|---:|---:|---:|")
        for engine in ("vllm-swift", "vllm-metal"):
            cells = grid.get((engine, model), {})
            row = " | ".join(f"{cells[B]:,.1f}" if B in cells else "—" for B in CONCURRENCY)
            print(f"| {engine} | {row} |")
        print()


if __name__ == "__main__":
    main()
