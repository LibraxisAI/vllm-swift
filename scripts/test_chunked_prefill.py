#!/usr/bin/env python3
"""Top-K logit overlap test for batched (chunked) prefill vs sequential.

Strict token-equivalence at greedy temp=0 is NOT a valid gate at fp16:
argmax ties flip across forward-pass shapes due to fp16 reduction-order
differences in batched matmul. Even mlx-lm Python disagrees with itself
between B=1 and B=2 on prompts whose top logits are near-tied. Use top-K
overlap as the primary correctness gate, with cosine similarity as a
secondary signal.

Both tested paths return top-K logits at the **same conceptual point in
the model's compute graph**: the last forward of the chunked prefill
pattern (`[*, T-1]` then `[*, 1]` of the last prompt token). No decode
advance afterward — comparing apples to apples.

Pass criteria:
- top-K sets identical (best) or ≥ K-1 elements overlap (acceptable)
- cosine similarity > 0.999

Usage:
  DYLD_LIBRARY_PATH=swift/.build/arm64-apple-macosx/release \
    python3 scripts/test_chunked_prefill.py [model_path] [-B 2] [-T 64] [-K 5]
"""

import ctypes
import math
import os
import sys
from pathlib import Path

MODEL_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else os.path.expanduser("~/models/Qwen3-0.6B-4bit")
)

B = 2
T = 64
K = 5
for i, arg in enumerate(sys.argv):
    if arg == "-B" and i + 1 < len(sys.argv):
        B = int(sys.argv[i + 1])
    if arg == "-T" and i + 1 < len(sys.argv):
        T = int(sys.argv[i + 1])
    if arg == "-K" and i + 1 < len(sys.argv):
        K = int(sys.argv[i + 1])

SWIFT_BUILD = Path(__file__).parent.parent / "swift" / ".build" / "arm64-apple-macosx"
LIB_PATH = str(SWIFT_BUILD / "release" / "libVLLMBridge.dylib")
assert os.path.exists(LIB_PATH), f"Missing {LIB_PATH} — build first"
lib = ctypes.CDLL(LIB_PATH)

lib.vsm_engine_create.restype = ctypes.c_void_p
lib.vsm_engine_create.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32,
    ctypes.c_char_p, ctypes.c_int32, ctypes.c_float,
]
lib.vsm_engine_destroy.restype = None
lib.vsm_engine_destroy.argtypes = [ctypes.c_void_p]

_topk_argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),  # promptTokens [B*T]
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  # B, T, K
    ctypes.POINTER(ctypes.c_int32),  # outIndices [B*K]
    ctypes.POINTER(ctypes.c_float),  # outValues  [B*K]
]
lib.vsm_engine_prefill_seq_uniform_topk.restype = ctypes.c_int32
lib.vsm_engine_prefill_seq_uniform_topk.argtypes = _topk_argtypes
lib.vsm_engine_prefill_batched_uniform_topk.restype = ctypes.c_int32
lib.vsm_engine_prefill_batched_uniform_topk.argtypes = _topk_argtypes


def make_prompt_tokens(t):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    seeds = [
        "Explain the theory of relativity in detail, covering both special and general relativity:",
        "Describe quantum mechanics, including wave-particle duality and the uncertainty principle:",
        "Walk through the proof of the Pythagorean theorem step by step:",
        "Summarize the causes and consequences of the French Revolution:",
        "Outline how a transformer language model processes a sentence end to end:",
        "Explain how photosynthesis converts sunlight into chemical energy:",
        "Compare and contrast classical conditioning and operant conditioning:",
        "Describe how plate tectonics shapes mountain ranges over geologic time:",
    ]
    out = []
    for i in range(B):
        s = seeds[i % len(seeds)]
        ids = tok.encode(s)
        while len(ids) < t:
            ids.extend(ids[: min(len(ids), t - len(ids))])
        out.append(ids[:t])
    return out


def call_topk(engine, fn, prompt_lists):
    flat = (ctypes.c_int32 * (B * T))()
    for i in range(B):
        for j in range(T):
            flat[i * T + j] = prompt_lists[i][j]
    idx_buf = (ctypes.c_int32 * (B * K))()
    val_buf = (ctypes.c_float * (B * K))()
    rc = fn(engine, flat, B, T, K, idx_buf, val_buf)
    assert rc == B, f"{fn.__name__}: returned {rc} expected {B}"
    return [
        ([int(idx_buf[s * K + j]) for j in range(K)],
         [float(val_buf[s * K + j]) for j in range(K)])
        for s in range(B)
    ]


def cos_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def mlx_python_groundtruth(prompt_lists):
    """Run mlx-lm Python with the same chunked B path. Acts as the
    independent reference: my Swift batched should match this exactly
    (both implement B-batched chunked prefill in MLX). If Swift matches
    mlx-lm but neither matches the per-request sequential output, the
    divergence is intrinsic to batched chunked execution, not a bug."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, _ = load(MODEL_PATH)
    cache = make_prompt_cache(model)
    chunk = mx.array([p[:T - 1] for p in prompt_lists])
    _ = model(chunk, cache=cache); mx.eval(_)
    last = mx.array([[p[T - 1]] for p in prompt_lists])
    out = model(last, cache=cache)  # [B, 1, V]
    out_last = out[:, -1, :]  # [B, V]
    sorted_idx = mx.argsort(out_last, axis=-1)
    vocab = out_last.shape[-1]
    top_idx = sorted_idx[:, vocab - K:]
    top_val = mx.take_along_axis(out_last, top_idx, axis=-1)
    mx.eval(top_idx, top_val)
    return [
        ([int(top_idx[s, j].item()) for j in range(K)],
         [float(top_val[s, j].item()) for j in range(K)])
        for s in range(B)
    ]


def main():
    prompt_lists = make_prompt_tokens(T)
    print(f"Prompt: B={B} T={T}, K={K}\n")

    engine = lib.vsm_engine_create(MODEL_PATH.encode(), b"float16", 0, None, 0, 0.9)
    assert engine
    print("=== Sequential (per-request prefill) ===")
    seq = call_topk(engine, lib.vsm_engine_prefill_seq_uniform_topk, prompt_lists)
    for s, (idx, _) in enumerate(seq):
        print(f"  slot {s}: top-{K} idx={idx}")
    print("\n=== Batched (single forward over [B, T]) ===")
    bat = call_topk(engine, lib.vsm_engine_prefill_batched_uniform_topk, prompt_lists)
    for s, (idx, _) in enumerate(bat):
        print(f"  slot {s}: top-{K} idx={idx}")
    lib.vsm_engine_destroy(engine)

    print("\n=== mlx-lm Python (independent reference, B-batched chunked) ===")
    py = mlx_python_groundtruth(prompt_lists)
    for s, (idx, _) in enumerate(py):
        print(f"  slot {s}: top-{K} idx={idx}")

    def compare(label, a, b, gate_topk_min, gate_cos_min):
        print(f"\n=== {label} ===")
        all_ok = True
        for s in range(B):
            a_idx, a_val = a[s]; b_idx, b_val = b[s]
            a_set, b_set = set(a_idx), set(b_idx)
            inter = len(a_set & b_set)
            cos = cos_sim(a_val, b_val)
            ok = (a_set == b_set) or (inter >= gate_topk_min and cos >= gate_cos_min)
            mark = "PASS" if ok else "FAIL"
            print(f"  slot {s}: top-{K} {inter}/{K} overlap, cos={cos:.6f} ({mark})")
            if not ok:
                print(f"    a: {sorted(a_set)}")
                print(f"    b: {sorted(b_set)}")
                all_ok = False
        return all_ok

    # Primary gate: Swift batched ↔ mlx-lm Python batched. Same algorithm,
    # different implementations. They MUST agree (modulo Metal/Python binding
    # noise) — otherwise the Swift port is broken.
    primary = compare("PRIMARY: Swift batched vs mlx-lm Python batched",
                      bat, py, gate_topk_min=K - 1, gate_cos_min=0.999)

    # Informational: Swift batched ↔ Swift sequential. Expected to diverge
    # for prompts with near-tied logits — batched chunked execution produces
    # different numerical results than per-request chunked execution. NOT a
    # bridge bug; the same divergence shows up between mlx-lm B=1 chunked and
    # mlx-lm B=2 chunked. Logged for visibility, NOT a pass/fail criterion.
    compare("INFO: Swift batched vs Swift sequential (expected divergence)",
            bat, seq, gate_topk_min=K - 1, gate_cos_min=0.99)

    if primary:
        print("\n✓ M1 PASS — Swift batched matches mlx-lm Python batched (correctness oracle)")
        return 0
    print("\n✗ M1 FAIL — Swift batched diverges from mlx-lm Python batched")
    return 1


if __name__ == "__main__":
    sys.exit(main())
