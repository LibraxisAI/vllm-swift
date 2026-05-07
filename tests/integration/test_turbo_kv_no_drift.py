# SPDX-License-Identifier: Apache-2.0
"""Integration: end-to-end "turbo KV scheme actually compresses, no drift"
regression. Pins the v0.5.3 fix where `--additional-config kv_scheme=
turboNvN` had been silently dropped on the floor for batched-decode caches
(Qwen3 dense + Qwen3.5/3.6 + Qwen3Next hybrid), so user-visible output on
any model that flowed through `BatchedKVCache` was running raw fp16 KV
regardless of what scheme they set.

Buddy's v0.5.1 alpha repro: Qwen3.6-35B-A3B-4bit + max-num-seqs=1 +
turbo4v2 + "Say hello in one short sentence." → "Hello." then ".2.2.2.2..."
for thousands of tokens until cap. Post-fix the same prompt produces a
clean greeting with finish_reason=stop.

Skips when the buddy-class model isn't on disk (boots take ~60-90s on M5
Max for the 35B). Opt in: `pytest -m integration tests/integration/
test_turbo_kv_no_drift.py`. Don't run on CI runners — they don't have
the model, the GPU, or the time.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Repro config — mirrors buddy's exact failing config
# ---------------------------------------------------------------------------

# Qwen3.6 is buddy's exact model; 3.5 is the same hybrid family and
# reproduces the same silent-bypass bug pre-fix. Either gates the test in.
_CANDIDATE_MODELS = [
    os.path.expanduser("~/models/Qwen3.6-35B-A3B-4bit"),
    os.path.expanduser("~/models/Qwen3.5-35B-A3B-4bit"),
]

# Drift signature: ".N" or ".2" loops, or any single non-newline char
# repeated 20+ times in a row. Coherent text never produces these.
_DRIFT_RE = re.compile(r"(.)\1{20,}", re.DOTALL)
_DOT_DIGIT_LOOP_RE = re.compile(r"(\.[0-9])\1{10,}")


def _find_model() -> str | None:
    for p in _CANDIDATE_MODELS:
        if os.path.isdir(p):
            return p
    return None


def _free_port() -> int:
    """Pick a random free port — don't collide with the user's running
    vllm-swift session on the standard 8000."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
            time.sleep(2.0)
    return False


def _post_chat(port: int, prompt: str, timeout: float = 240.0) -> dict:
    body = {
        "model": "qwen3-test",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def turbo_server():
    if shutil.which("vllm-swift") is None:
        pytest.skip("vllm-swift CLI not on PATH; pip install -e . first")

    model = _find_model()
    if model is None:
        pytest.skip(
            "Qwen3.5-35B-A3B-4bit (or 3.6) not on disk under ~/models; "
            "run `vllm-swift download mlx-community/Qwen3.5-35B-A3B-4bit` "
            "to opt this regression test in."
        )

    port = _free_port()
    log_path = "/tmp/test_turbo_kv_no_drift_server.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [
            "vllm-swift",
            "serve",
            model,
            "--served-model-name",
            "qwen3-test",
            "--port",
            str(port),
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "1",
            "--gpu-memory-utilization",
            "0.6",
            "--no-enable-prefix-caching",
            "--additional-config",
            '{"kv_scheme":"turbo4v2","kv_bits":4}',
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        if not _wait_for_server(port, timeout=180.0):
            proc.terminate()
            proc.wait(timeout=10.0)
            log.close()
            tail = ""
            try:
                with open(log_path) as f:
                    tail = "".join(f.readlines()[-40:])
            except Exception:
                pass
            pytest.fail(f"vllm-swift server never came up on :{port}\n--- log ---\n{tail}")
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
        log.close()


def test_say_hello_does_not_drift(turbo_server: int) -> None:
    """Buddy's exact repro: 'Say hello in one short sentence.' Pre-fix
    produced 'Hello.' then '.2.2.2.2...' until cap. Post-fix produces a
    clean stop'd greeting. The cache is doing real turbo4v2 compression
    instead of silently running raw fp16."""
    resp = _post_chat(turbo_server, "Say hello in one short sentence.")

    choices = resp.get("choices", [])
    assert choices, f"no choices in response: {resp}"
    msg = choices[0].get("message", {})
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    finish = choices[0].get("finish_reason")

    # Must finish coherently, not bump up against max_tokens.
    assert finish == "stop", (
        f"finish_reason={finish!r} (expected 'stop'); model failed to stop "
        f"on its own — likely drifting or starved.\ncontent={content!r}"
    )

    # Drift signature: long runs of repeated chars or `.N` loops. Empty
    # content with substantial reasoning IS allowed (reasoning models
    # sometimes empty-string the visible content) — but not vice versa.
    full = content + "\n" + reasoning
    assert not _DRIFT_RE.search(full), (
        f"drift signature in output (>=20 char repeat). content={content[:200]!r}"
    )
    assert not _DOT_DIGIT_LOOP_RE.search(full), (
        f"`.N` loop drift signature. content={content[:200]!r}"
    )

    # Must produce SOME output (content or reasoning).
    assert content or reasoning, f"empty response: msg={msg!r}"

    # Sanity on completion_tokens — the v0.5.2 max_tokens-bump caps at
    # ~16K-20K. If we used the full budget on "Say hello" something is
    # wrong (drifting, starving, or stuck in a reasoning loop).
    n_completion = resp.get("usage", {}).get("completion_tokens", 0)
    assert n_completion < 4096, (
        f"completion_tokens={n_completion} unexpectedly high for a "
        "5-word prompt — model likely drifting even if no exact "
        "drift signature matched."
    )
