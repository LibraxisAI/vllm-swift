# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for integration tests.

Integration tests run against REAL models on disk. They auto-skip when the
model directory is missing, so the suite stays runnable on machines without
the full local model library. Marker `integration` keeps them out of the
default `pytest` run; opt in with `pytest -m integration`.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from contextlib import closing
from pathlib import Path
from typing import Iterator

import pytest

MODELS_ROOT = Path(os.environ.get("VLLM_SWIFT_MODELS_ROOT", "/Users/tom/models"))


def _model_path(name: str) -> Path:
    return MODELS_ROOT / name


def _has_local_model(name: str) -> bool:
    p = _model_path(name)
    return p.is_dir() and (p / "config.json").is_file()


def require_local_model(name: str) -> Path:
    p = _model_path(name)
    if not p.is_dir() or not (p / "config.json").is_file():
        pytest.skip(f"local model not present: {p}")
    return p


@pytest.fixture(scope="session")
def models_root() -> Path:
    return MODELS_ROOT


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(port: int, timeout: float = 90.0) -> None:
    """Block until /health returns 200, or raise on timeout."""
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"vllm-swift server on port {port} did not become healthy")


def _count_engine_cores() -> int:
    """Count live `VLLM::EngineCore` processes anywhere on the box."""
    try:
        out = subprocess.check_output(["pgrep", "-f", "VLLM::EngineCore"], text=True, timeout=2)
    except subprocess.CalledProcessError:
        return 0  # pgrep returns 1 when no matches
    except Exception:  # noqa: BLE001
        return 0
    return sum(1 for line in out.splitlines() if line.strip().isdigit())


@pytest.fixture(scope="module")
def vllm_swift_server(request) -> Iterator[dict]:
    """Launch vllm-swift against a model and yield (base_url, model_id, log_path).

    Parametrize via `request.param = {"model": "<dir>", "extra_args": [...]}`.
    Auto-skips when model not present locally. Tears down on exit via
    process-group SIGTERM so the api_server's `VLLM::EngineCore` child
    gets reaped too — `proc.terminate()` alone leaves the grandchild
    orphaned holding ~all KV cache memory.
    """
    import os
    import signal

    cfg = request.param
    model_name: str = cfg["model"]
    extra: list[str] = list(cfg.get("extra_args", []))
    model = require_local_model(model_name)
    port = _free_port()
    log_path = Path(f"/tmp/vllm-swift-itest-{port}.log")
    cmd = [
        "vllm-swift",
        "serve",
        str(model),
        "--port",
        str(port),
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "2",
        *extra,
    ]
    pre_engine_cores = _count_engine_cores()
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # pgid we can later killpg()
        )
    try:
        _wait_for_health(port, timeout=180)
        yield {
            "base_url": f"http://127.0.0.1:{port}/v1",
            "model_id": str(model),
            "log_path": str(log_path),
        }
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()
        # Give the kernel a beat to reap children, then assert no
        # leftover EngineCore from THIS server (compare against the
        # snapshot we took before launching).
        time.sleep(1.0)
        post_engine_cores = _count_engine_cores()
        if post_engine_cores > pre_engine_cores:
            raise RuntimeError(
                f"VLLM::EngineCore orphan leak: pre={pre_engine_cores} "
                f"post={post_engine_cores}. The pgroup-kill in conftest "
                f"failed — server log: {log_path}"
            )


def _read_arch(model_dir: Path) -> str:
    cfg = model_dir / "config.json"
    if not cfg.is_file():
        return ""
    try:
        with open(cfg) as f:
            data = json.load(f)
    except Exception:
        return ""
    archs = data.get("architectures") or []
    return archs[0] if archs else ""


@pytest.fixture(scope="session")
def local_models_inventory() -> dict[str, str]:
    """Return {dir_name: architecture} for every local model with a config.json.
    Lets test files cherry-pick what's actually available."""
    out: dict[str, str] = {}
    if not MODELS_ROOT.is_dir():
        return out
    for entry in sorted(MODELS_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "config.json").is_file():
            continue
        out[entry.name] = _read_arch(entry)
    return out
