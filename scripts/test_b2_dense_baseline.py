#!/usr/bin/env python3
"""B=2 dense baseline for the F-85 v2 quality test.

Runs the exact same two prompts through the dense batched-decode path
to verify B=2 prefill+decode works correctly outside the sparse path.
"""

import ctypes
import os
import sys
from pathlib import Path

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
    else os.path.expanduser("~/models/Qwen2.5-14B-Instruct-1M-4bit")
CTX = 4096
MAX_TOKENS = 30

for i, arg in enumerate(sys.argv):
    if arg == "--ctx" and i + 1 < len(sys.argv):
        CTX = int(sys.argv[i + 1])
    if arg == "--tokens" and i + 1 < len(sys.argv):
        MAX_TOKENS = int(sys.argv[i + 1])

SWIFT_BUILD = Path(__file__).parent.parent / "swift" / ".build" / "arm64-apple-macosx"
LIB_PATH = str(SWIFT_BUILD / "release" / "libVLLMBridge.dylib")


def build_prompt(target_tokens, needle, tokenizer):
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
    print(f"Ctx: {CTX} × B=2 (DENSE)")
    print()

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
    if not engine: sys.exit(1)

    first_tokens = {}
    for rid, ids in [(b"req-0", ids0), (b"req-1", ids1)]:
        arr = (ctypes.c_int32 * len(ids))(*ids)
        rc = lib.vsm_engine_prefill_req(engine, rid, arr, len(ids), 0.0, 1.0)
        if rc < 0: sys.exit(1)
        first_tokens[rid.decode()] = int(rc)
    lib.vsm_engine_init_batched(engine)

    req_ids_buf = (ctypes.c_char_p * 4)()
    tokens_buf = (ctypes.c_int32 * 4)()
    out_tokens = {rid: [tok] for rid, tok in first_tokens.items()}
    for _ in range(MAX_TOKENS):
        n = lib.vsm_engine_decode_all(engine, req_ids_buf, tokens_buf, 4)
        for k in range(n):
            rid = req_ids_buf[k].decode() if req_ids_buf[k] else ""
            if rid in out_tokens:
                out_tokens[rid].append(int(tokens_buf[k]))

    text0 = tok.decode(out_tokens["req-0"])
    text1 = tok.decode(out_tokens["req-1"])
    print(f"Slot-0 (BANANA-7) → {text0!r}")
    print(f"Slot-1 (COCONUT-99) → {text1!r}")
    s0 = "BANANA" in text0
    s1 = "COCONUT" in text1
    print(f"slot-0 needle hit: {s0}")
    print(f"slot-1 needle hit: {s1}")
    print(f"verdict: {'PASS' if s0 and s1 else 'FAIL'}")


if __name__ == "__main__":
    main()
