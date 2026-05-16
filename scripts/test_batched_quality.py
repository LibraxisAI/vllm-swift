#!/usr/bin/env python3
"""Verify cross-slot output correctness at B>1 vs B=1 baseline.

Generates the same prompt sequentially at B=1 to establish ground truth,
then runs the same prompts at B=N and compares per-slot output to the
B=1 baseline. Reports per-slot agreement.
"""

import ctypes
import os
import sys
from pathlib import Path

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/Qwen2.5-7B-Instruct-4bit")
MAX_TOKENS = 20
B = int(os.environ.get("B", "4"))

SWIFT_BUILD = Path(__file__).parent.parent / "swift" / ".build" / "arm64-apple-macosx"
LIB_PATH = str(SWIFT_BUILD / "release" / "libVLLMBridge.dylib")
lib = ctypes.CDLL(LIB_PATH)

lib.vsm_engine_create.restype = ctypes.c_void_p
lib.vsm_engine_create.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32,
    ctypes.c_char_p, ctypes.c_int32, ctypes.c_float,
]
lib.vsm_engine_prefill_req.restype = ctypes.c_int32
lib.vsm_engine_prefill_req.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
    ctypes.c_float, ctypes.c_float,
]
lib.vsm_engine_decode_all.restype = ctypes.c_int32
lib.vsm_engine_decode_all.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
]
lib.vsm_engine_init_batched.restype = ctypes.c_int32
lib.vsm_engine_init_batched.argtypes = [ctypes.c_void_p]
lib.vsm_engine_finish_req.restype = None
lib.vsm_engine_finish_req.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.vsm_engine_reset.restype = None
lib.vsm_engine_reset.argtypes = [ctypes.c_void_p]
lib.vsm_engine_destroy.restype = None
lib.vsm_engine_destroy.argtypes = [ctypes.c_void_p]

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_PATH)

print(f"Model: {MODEL_PATH}")
print(f"Concurrency: B={B}, tokens={MAX_TOKENS}")

engine = lib.vsm_engine_create(MODEL_PATH.encode(), b"float16", 0, None, 0, 0.9)
assert engine, "engine create failed"

SEEDS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "The chemical formula for water is",
    "The author of Hamlet is",
] * 16  # support up to B=64
SEEDS = SEEDS[:B]


def decode_step(engine, B):
    req_ids_buf = (ctypes.c_char_p * (B + 1))()
    tokens_buf = (ctypes.c_int32 * (B + 1))()
    out_tokens = [[] for _ in range(B)]
    rid_to_slot = {f"req-{i}".encode(): i for i in range(B)}
    for _ in range(MAX_TOKENS):
        n = lib.vsm_engine_decode_all(engine, req_ids_buf, tokens_buf, B + 1)
        for k in range(n):
            rid = req_ids_buf[k]
            tk = tokens_buf[k]
            if rid in rid_to_slot:
                out_tokens[rid_to_slot[rid]].append(int(tk))
    return out_tokens


# --- B=1 baseline pass: run each prompt sequentially ---
print("\n=== B=1 baseline (sequential) ===")
baseline = []
for i, prompt in enumerate(SEEDS):
    lib.vsm_engine_reset(engine)
    ids = tok.encode(prompt)
    arr = (ctypes.c_int32 * len(ids))(*ids)
    rid = b"req-0"  # always slot 0 in baseline pass
    lib.vsm_engine_prefill_req(engine, rid, arr, len(ids), 0.0, 1.0)
    lib.vsm_engine_init_batched(engine)
    one = decode_step(engine, 1)[0]
    baseline.append(one)
    print(f"  slot {i} ({prompt!r:50s}): {tok.decode(one)!r}")
    lib.vsm_engine_finish_req(engine, rid)

# --- B=N concurrent pass ---
print(f"\n=== B={B} concurrent ===")
lib.vsm_engine_reset(engine)
for i, prompt in enumerate(SEEDS):
    ids = tok.encode(prompt)
    arr = (ctypes.c_int32 * len(ids))(*ids)
    rid = f"req-{i}".encode()
    lib.vsm_engine_prefill_req(engine, rid, arr, len(ids), 0.0, 1.0)
lib.vsm_engine_init_batched(engine)
concurrent = decode_step(engine, B)

print()
matches = 0
for i in range(B):
    base_txt = tok.decode(baseline[i])
    cc_txt = tok.decode(concurrent[i])
    ok = baseline[i] == concurrent[i]
    matches += int(ok)
    flag = "✓" if ok else "✗"
    print(f"  {flag} slot {i}")
    print(f"      base: {base_txt!r}")
    print(f"      conc: {cc_txt!r}")
    if not ok:
        # Find first divergent token
        for k in range(min(len(baseline[i]), len(concurrent[i]))):
            if baseline[i][k] != concurrent[i][k]:
                print(f"      diverge@token {k}: base={baseline[i][k]} ({tok.decode([baseline[i][k]])!r}) "
                      f"conc={concurrent[i][k]} ({tok.decode([concurrent[i][k]])!r})")
                break

print(f"\n{matches}/{B} slots match B=1 baseline")
lib.vsm_engine_destroy(engine)
