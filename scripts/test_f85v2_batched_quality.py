#!/usr/bin/env python3
"""F-85 v2 quality validation — batched-sparse decode at B>1.

The serial B=1 sparse path is validated by test_sparse_quality_longctx.py.
This validates the BATCHED-SPARSE path (VSM_SPARSE_BATCHED=1) at B=2 by
running a needle-in-haystack prompt across two slots and checking both
slots recover the needle.

Pattern (per slot):
  needle → haystack-fill-to-ctx → question

Slot-0 and slot-1 use DIFFERENT needles so we catch cross-slot bleed:
  slot-0: BANANA-7
  slot-1: COCONUT-99

If either slot's decoded text contains the wrong needle (e.g. slot-1
returns BANANA-7), there's selector cross-contamination across slots.

Usage:
  DYLD_LIBRARY_PATH=swift/.build/arm64-apple-macosx/release \
    python3 scripts/test_f85v2_batched_quality.py \
      [model_path] [--ctx N] [--tokens N] [--sparse-flags "VAR1=1 VAR2=1"]
"""

import ctypes
import os
import sys
from pathlib import Path

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
    else os.path.expanduser("~/models/Qwen2.5-14B-Instruct-1M-4bit")
CTX = 32768
MAX_TOKENS = 40
SPARSE_FLAGS = "VSM_SPARSE=1 VSM_SPARSE_BATCHED=1 VSM_SPARSE_BATCHED_KERNEL=f73"

for i, arg in enumerate(sys.argv):
    if arg == "--ctx" and i + 1 < len(sys.argv):
        CTX = int(sys.argv[i + 1])
    if arg == "--tokens" and i + 1 < len(sys.argv):
        MAX_TOKENS = int(sys.argv[i + 1])
    if arg == "--sparse-flags" and i + 1 < len(sys.argv):
        SPARSE_FLAGS = sys.argv[i + 1]

SWIFT_BUILD = Path(__file__).parent.parent / "swift" / ".build" / "arm64-apple-macosx"
LIB_PATH = str(SWIFT_BUILD / "release" / "libVLLMBridge.dylib")
if not os.path.exists(LIB_PATH):
    print(f"ERROR: dylib not found at {LIB_PATH}", file=sys.stderr)
    sys.exit(1)


def build_prompt(target_tokens, needle, tokenizer):
    """Repeat haystack to fill target_tokens, with needle near start
    and question appended."""
    haystack = (
        "The history of computing spans many centuries. From the abacus "
        "to mechanical calculators to modern electronic computers, humans "
        "have continuously developed tools to perform calculations. The "
        "20th century saw the development of programmable digital "
        "computers that could be configured to perform many different "
        "tasks. The Turing machine, conceived by Alan Turing in 1936, "
        "provided a theoretical foundation for what computers could do. "
    )
    question = " Question: what is the secret code? Answer:"
    needle_ids = tokenizer.encode(needle)
    haystack_ids = tokenizer.encode(haystack)
    question_ids = tokenizer.encode(question)
    body_budget = target_tokens - len(needle_ids) - len(question_ids)
    ids = list(needle_ids)
    while len(ids) - len(needle_ids) < body_budget:
        remaining = body_budget - (len(ids) - len(needle_ids))
        ids.extend(haystack_ids[:remaining] if remaining < len(haystack_ids) else haystack_ids)
    ids.extend(question_ids)
    return ids[:target_tokens]


def main():
    print(f"Model: {MODEL_PATH}")
    print(f"Ctx: {CTX} tokens × B=2")
    print(f"Decode: {MAX_TOKENS} tokens / slot")
    print(f"Sparse flags: {SPARSE_FLAGS}")
    print()

    # Apply sparse flags BEFORE loading dylib so static-let env reads see them.
    for kv in SPARSE_FLAGS.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            os.environ[k] = v
    # Default sampling is greedy when temperature=0.0 and topP=1.0 — we use
    # those settings so the comparison is reproducible.

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
    lib.vsm_engine_destroy.restype = None
    lib.vsm_engine_destroy.argtypes = [ctypes.c_void_p]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    needle0 = "The secret code is BANANA-7. "
    needle1 = "The secret code is COCONUT-99. "
    ids0 = build_prompt(CTX, needle0, tok)
    ids1 = build_prompt(CTX, needle1, tok)
    print(f"Slot-0 prompt: {len(ids0)} tokens, needle=BANANA-7")
    print(f"Slot-1 prompt: {len(ids1)} tokens, needle=COCONUT-99")
    print()

    engine = lib.vsm_engine_create(MODEL_PATH.encode(), b"float16", 0, None, 0, 0.9)
    if not engine:
        print("FAIL engine create", file=sys.stderr)
        sys.exit(1)

    # vsm_engine_prefill_req returns the FIRST sampled token (positive int)
    # or -1 on error. Track per-slot first tokens.
    first_tokens = {}
    for rid, ids in [(b"req-0", ids0), (b"req-1", ids1)]:
        arr = (ctypes.c_int32 * len(ids))(*ids)
        rc = lib.vsm_engine_prefill_req(engine, rid, arr, len(ids), 0.0, 1.0)
        if rc < 0:
            print(f"FAIL prefill {rid} rc={rc}", file=sys.stderr)
            sys.exit(1)
        first_tokens[rid.decode()] = int(rc)
    lib.vsm_engine_init_batched(engine)

    req_ids_buf = (ctypes.c_char_p * 4)()
    tokens_buf = (ctypes.c_int32 * 4)()
    # Seed with the first sampled tokens from prefill so they're in the
    # decoded text.
    out_tokens = {rid: [tok] for rid, tok in first_tokens.items()}
    for _ in range(MAX_TOKENS):
        n = lib.vsm_engine_decode_all(engine, req_ids_buf, tokens_buf, 4)
        for k in range(n):
            rid = req_ids_buf[k].decode() if req_ids_buf[k] else ""
            if rid in out_tokens:
                out_tokens[rid].append(int(tokens_buf[k]))

    for rid in ("req-0", "req-1"):
        lib.vsm_engine_finish_req(engine, rid.encode())
    lib.vsm_engine_destroy(engine)

    print()
    print("=== results ===")
    text0 = tok.decode(out_tokens["req-0"]) if out_tokens["req-0"] else "<NONE>"
    text1 = tok.decode(out_tokens["req-1"]) if out_tokens["req-1"] else "<NONE>"
    print(f"Slot-0 (BANANA-7 prompt) → {text0!r}")
    print(f"Slot-1 (COCONUT-99 prompt) → {text1!r}")
    print()

    s0_self = "BANANA" in text0
    s0_cross = "COCONUT" in text0
    s1_self = "COCONUT" in text1
    s1_cross = "BANANA" in text1
    print(f"slot-0 self_needle={s0_self} cross_needle={s0_cross}")
    print(f"slot-1 self_needle={s1_self} cross_needle={s1_cross}")
    print()

    ok = s0_self and s1_self and not s0_cross and not s1_cross
    if ok:
        print("=== verdict: PASS — both slots retrieved own needle, no bleed ===")
        sys.exit(0)
    else:
        print("=== verdict: FAIL ===")
        if not s0_self: print("  slot-0 failed to retrieve own needle (BANANA)")
        if not s1_self: print("  slot-1 failed to retrieve own needle (COCONUT)")
        if s0_cross: print("  slot-0 leaked slot-1's needle (COCONUT)")
        if s1_cross: print("  slot-1 leaked slot-0's needle (BANANA)")
        sys.exit(1)


if __name__ == "__main__":
    main()
