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

# Prompt-length sweep — 0 means default short prompt (~18 tokens).
# >0 means pad to that many tokens. Long-context cells expose paged's
# memory-packing wins that short-context cells hide.
PROMPT_TOKEN_SETS = [0, 2048, 4096, 8192]
if "--short-only" in sys.argv:
    PROMPT_TOKEN_SETS = [0]
if "--long-only" in sys.argv:
    PROMPT_TOKEN_SETS = [2048, 4096, 8192]

# `unique` is the apples-to-apples mode (B distinct prompts, prefix caching off
# on vllm-metal). `identical` keeps the old behavior (prefix-cache hits) — useful
# for shared-prefix workloads but not for cross-engine compute comparisons.
PROMPTS_MODE = "unique"
for i, a in enumerate(sys.argv):
    if a == "--prompts" and i + 1 < len(sys.argv):
        PROMPTS_MODE = sys.argv[i + 1]
        assert PROMPTS_MODE in ("identical", "unique"), PROMPTS_MODE

VLLM_SWIFT_LIB = REPO / "swift" / ".build" / "arm64-apple-macosx" / "release"
VLLM_METAL_PYTHON = "/Users/tom/.venv-vllm-metal/bin/python3"


def run_vllm_swift(model_path: str, prompt_tokens: int = 0) -> dict[int, dict]:
    """Returns {B: {tps_e2e, tps_decode, prefill_ms}} for vllm-swift bench."""
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "bench_throughput.py"),
        model_path,
        "--tokens", "50",
        "--prompts", PROMPTS_MODE,
    ]
    if prompt_tokens > 0:
        cmd += ["--prompt-tokens", str(prompt_tokens)]
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = str(VLLM_SWIFT_LIB)
    timeout = 600 if prompt_tokens <= 2048 else 1800
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    out = proc.stdout
    # Lines like "B=  1: e2e=309.1 decode=309.1 tok/s [prefill=819ms, ...]"
    results = {}
    pat = re.compile(
        r"\s*B=\s*(\d+):\s*e2e=([\d,\.]+)\s+decode=([\d,\.]+)\s+tok/s\s*\[prefill=([\d,\.]+)ms"
    )
    for line in out.splitlines():
        m = pat.match(line)
        if m:
            B = int(m.group(1))
            results[B] = {
                "tps_e2e": float(m.group(2).replace(",", "")),
                "tps_decode": float(m.group(3).replace(",", "")),
                "prefill_ms": float(m.group(4).replace(",", "")),
            }
    return results


def run_vllm_metal_one(
    model_path: str, B: int, timeout: int = 600, prompt_tokens: int = 0,
) -> dict | None:
    """Single B run for vllm-metal. Returns dict with both metrics or None."""
    cmd = [
        VLLM_METAL_PYTHON,
        "-u",
        str(REPO / "scripts" / "bench_vllm_metal.py"),
        model_path,
        "--worker", str(B),
        "--prompts", PROMPTS_MODE,
    ]
    if prompt_tokens > 0:
        cmd += ["--prompt-tokens", str(prompt_tokens)]
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
            return {
                "tps_e2e": float(data.get("tps_e2e", data["tps"])),
                "tps_decode": float(data.get("tps_decode", data.get("tps_e2e", data["tps"]))),
                "prefill_ms": float(data.get("ttft", 0.0)) * 1000.0,
            }
    print(f"    [B={B}] no JSON result line", flush=True)
    return None


def run_vllm_metal(model_path: str, prompt_tokens: int = 0) -> dict[int, dict]:
    """Returns {B: {tps_e2e, tps_decode, prefill_ms}} per concurrency level."""
    results = {}
    base_timeouts = {1: 180, 8: 300, 32: 900, 64: 1800}
    multiplier = 1 if prompt_tokens <= 2048 else (3 if prompt_tokens <= 4096 else 6)
    for B in CONCURRENCY:
        t0 = time.perf_counter()
        cell = run_vllm_metal_one(
            model_path, B,
            timeout=base_timeouts.get(B, 1800) * multiplier,
            prompt_tokens=prompt_tokens,
        )
        dt = time.perf_counter() - t0
        if cell is not None:
            results[B] = cell
            print(f"    [B={B}] e2e={cell['tps_e2e']:.1f} decode={cell['tps_decode']:.1f} "
                  f"prefill={cell['prefill_ms']:.0f}ms ({dt:.1f}s wall)", flush=True)
    return results


def median_of_dicts(samples: list[dict]) -> dict:
    """Median per metric across a list of {tps_e2e, tps_decode, prefill_ms} dicts."""
    if not samples:
        return {}
    keys = samples[0].keys()
    return {k: statistics.median([s[k] for s in samples]) for k in keys}


def main():
    print(f"=== Baseline matrix (runs/cell={RUNS_PER_CELL}, ctx={PROMPT_TOKEN_SETS}, "
          f"prompts={PROMPTS_MODE}) ===\n")

    grid = {}  # {(engine, model, p): {B: median_dict}}
    raw = {}   # {(engine, model, p): {B: [sample_dicts]}}
    has_long = any(p > 0 for p in PROMPT_TOKEN_SETS)
    base = "baseline-2026-04-25"
    suffix = "-unique" if PROMPTS_MODE == "unique" else ""
    name = f"{base}-longctx{suffix}-raw.json" if has_long else f"{base}{suffix}-raw.json"
    json_path = REPO / "benchmarks" / name

    def flush_json():
        snapshot = {
            "runs_per_cell": RUNS_PER_CELL,
            "concurrency": CONCURRENCY,
            "prompt_token_sets": PROMPT_TOKEN_SETS,
            "prompts_mode": PROMPTS_MODE,
            "samples": {f"{e}|{m}|p{p}": s for (e, m, p), s in raw.items()},
            "median": {f"{e}|{m}|p{p}": g for (e, m, p), g in grid.items()},
        }
        json_path.write_text(json.dumps(snapshot, indent=2))

    for model in MODELS:
        model_path = os.path.expanduser(f"~/models/{model}")
        for prompt_tokens in PROMPT_TOKEN_SETS:
            for engine, runner in (("vllm-swift", run_vllm_swift), ("vllm-metal", run_vllm_metal)):
                tag = f"p{prompt_tokens}" if prompt_tokens > 0 else "short"
                print(f"\n[{engine}] {model} ctx={tag}")
                t0 = time.perf_counter()
                samples: dict[int, list[dict]] = {B: [] for B in CONCURRENCY}
                for r in range(RUNS_PER_CELL):
                    print(f"  run {r+1}/{RUNS_PER_CELL}...", flush=True)
                    tr0 = time.perf_counter()
                    results = runner(model_path, prompt_tokens=prompt_tokens)
                    dtr = time.perf_counter() - tr0
                    print(f"    elapsed {dtr:.1f}s — {results}", flush=True)
                    for B, cell in results.items():
                        samples[B].append(cell)
                    raw[(engine, model, prompt_tokens)] = samples
                    flush_json()
                medians = {B: median_of_dicts(s) for B, s in samples.items() if s}
                grid[(engine, model, prompt_tokens)] = medians
                flush_json()
                dt = time.perf_counter() - t0
                print(f"  done in {dt:.1f}s", flush=True)

    # Markdown output — emit both metrics
    print("\n\n## Results\n")
    for metric_label, key in [("e2e tok/s", "tps_e2e"), ("decode tok/s", "tps_decode"),
                              ("prefill ms", "prefill_ms")]:
        print(f"\n### {metric_label}\n")
        for model in MODELS:
            for prompt_tokens in PROMPT_TOKEN_SETS:
                tag = f"p={prompt_tokens}" if prompt_tokens > 0 else "short"
                print(f"\n#### {model} ({tag})\n")
                print("| Engine | B=1 | B=8 | B=32 | B=64 |")
                print("|---|---:|---:|---:|---:|")
                for engine in ("vllm-swift", "vllm-metal"):
                    cells = grid.get((engine, model, prompt_tokens), {})
                    row_cells = []
                    for B in CONCURRENCY:
                        if B in cells and key in cells[B]:
                            v = cells[B][key]
                            row_cells.append(f"{v:,.0f}" if key == "prefill_ms" else f"{v:,.1f}")
                        else:
                            row_cells.append("—")
                    print(f"| {engine} | {' | '.join(row_cells)} |")
            print()


if __name__ == "__main__":
    main()
