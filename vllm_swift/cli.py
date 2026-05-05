"""CLI entry point for the pip-installed vllm-swift package.

Mirrors the Homebrew bash wrapper's `serve / download / version` commands
so `pip install vllm-swift && vllm-swift serve <model>` produces the same
behavior as the brew installation, with the same auto-detect smart defaults
for the tool-call parser.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from vllm_swift import __version__
from vllm_swift.detect_reasoning_parser import _load_arch
from vllm_swift.detect_reasoning_parser import detect_parser as detect_reasoning_parser
from vllm_swift.detect_tool_parser import detect_parser
from vllm_swift.known_parser_issues import (
    combo_caveats,
    reasoning_parser_caveats,
    tool_parser_caveats,
)
from vllm_swift.response_rewriter import (
    _LEAKY_TOOL_PARSERS,
    _REASONING_PARSERS_NEEDING_BUDGET,
    needs_rewrite,
)


def _lib_dir() -> Path:
    """Directory containing the bundled Swift dylib + Metal library."""
    return Path(__file__).resolve().parent / "_lib"


def _prepare_dyld_env() -> dict[str, str]:
    env = os.environ.copy()
    lib = str(_lib_dir())
    existing = env.get("DYLD_LIBRARY_PATH", "")
    env["DYLD_LIBRARY_PATH"] = f"{lib}:{existing}" if existing else lib
    return env


def _has_flag(args: list[str], targets: tuple[str, ...]) -> bool:
    for arg in args:
        for t in targets:
            if arg == t or arg.startswith(t + "="):
                return True
    return False


def _has_tool_flag(args: list[str]) -> bool:
    return _has_flag(
        args,
        (
            "--tool-call-parser",
            "--enable-auto-tool-choice",
            "--no-enable-auto-tool-choice",
        ),
    )


def _has_reasoning_flag(args: list[str]) -> bool:
    return _has_flag(
        args,
        (
            "--reasoning-parser",
            "--enable-reasoning",
            "--no-enable-reasoning",
        ),
    )


def _extract_model(args: list[str]) -> str | None:
    """Find the value of --model in args."""
    prev = ""
    for arg in args:
        if arg.startswith("--model="):
            return arg.split("=", 1)[1]
        if prev == "--model":
            return arg
        prev = arg
    return None


def _extract_port(args: list[str], default: int = 8000) -> int:
    prev = ""
    for arg in args:
        if arg.startswith("--port="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return default
        if prev == "--port":
            try:
                return int(arg)
            except ValueError:
                return default
        prev = arg
    return default


def _strip_port(args: list[str]) -> list[str]:
    """Drop any --port flag from args; caller will inject its own."""
    out: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--port":
            skip_next = True
            continue
        if arg.startswith("--port="):
            continue
        out.append(arg)
    return out


def _wait_for_vllm_ready(port: int, timeout: float = 600.0) -> bool:
    """Poll the vLLM /health endpoint until it responds or we time out."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:  # noqa: S310
                if 200 <= resp.status < 500:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    return False


def _registered_tool_parsers() -> set[str]:
    """Names registered in the running vLLM's tool parser registry, or
    empty set if vLLM isn't importable (CI / dev shell). Importing
    `vllm.tool_parsers` is cheap enough — the heavy modules are lazy.
    """
    try:
        from vllm.tool_parsers import _TOOL_PARSERS_TO_REGISTER  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        return set()
    return set(_TOOL_PARSERS_TO_REGISTER)


def _registered_reasoning_parsers() -> set[str]:
    """Names registered in the running vLLM's reasoning parser registry."""
    try:
        from vllm.reasoning import _REASONING_PARSERS_TO_REGISTER  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        return set()
    return set(_REASONING_PARSERS_TO_REGISTER)


def _validate_against_registry(parser: str, registered: set[str], kind: str) -> bool:
    """True if `parser` is in vLLM's registry (or registry is unknown).

    When False, prints a stderr warning so the user sees why injection
    got skipped — defensive against future vLLM parser renames/removals
    that would otherwise crash vLLM with an opaque "unknown parser" error.
    """
    if not registered:
        # vLLM not importable — can't validate; trust the detector.
        return True
    if parser in registered:
        return True
    sys.stderr.write(
        f"vllm-swift: detected {kind} parser '{parser}' is not registered in "
        f"the running vLLM build; skipping auto-injection. Override with an "
        f"explicit flag if you know a compatible parser name.\n"
    )
    return False


def _free_port(preferred: int) -> int:
    """Return `preferred` if free, otherwise an OS-assigned ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve(args: list[str]) -> int:
    if not args:
        sys.stderr.write("Usage: vllm-swift serve <model-path-or-hf-id> [vllm args...]\n")
        return 2
    # Accept positional model as first arg (matches brew wrapper UX).
    extra_args: list[str] = []
    if not args[0].startswith("-"):
        model = args[0]
        passthrough = args[1:]
        extra_args = ["--model", model]
    else:
        model = _extract_model(args) or ""
        passthrough = args
    auto_args: list[str] = []
    short = os.path.basename(model.rstrip("/")) if model else ""
    injected_tool: str = ""
    injected_reasoning: str = ""
    tool_registry = _registered_tool_parsers()
    reasoning_registry = _registered_reasoning_parsers()
    if model and not _has_tool_flag(passthrough):
        parser = detect_parser(model)
        if parser and _validate_against_registry(parser, tool_registry, "tool"):
            print(
                f"vllm-swift: auto-detected tool parser '{parser}' for {short}; "
                f"injecting --enable-auto-tool-choice --tool-call-parser {parser}"
            )
            print(
                "  (override with explicit --tool-call-parser <name> or "
                "--no-enable-auto-tool-choice)"
            )
            for summary, url, mit in tool_parser_caveats(parser):
                print(f"  ! known issue [{parser}]: {summary} ({url})")
                if mit:
                    print(f"    mitigation: {mit}")
            auto_args += ["--enable-auto-tool-choice", "--tool-call-parser", parser]
            injected_tool = parser
    if model and not _has_reasoning_flag(passthrough):
        rparser = detect_reasoning_parser(model)
        if rparser and _validate_against_registry(rparser, reasoning_registry, "reasoning"):
            print(
                f"vllm-swift: auto-detected reasoning parser '{rparser}' for {short}; "
                f"injecting --reasoning-parser {rparser}"
            )
            print("  (override with explicit --reasoning-parser <name> or --no-enable-reasoning)")
            for summary, url, mit in reasoning_parser_caveats(rparser):
                print(f"  ! known issue [{rparser}]: {summary} ({url})")
                if mit:
                    print(f"    mitigation: {mit}")
            auto_args += ["--reasoning-parser", rparser]
            injected_reasoning = rparser
    if injected_reasoning and injected_tool:
        for summary, url, mit in combo_caveats(injected_reasoning, injected_tool):
            print(f"  ! known combo issue [{injected_reasoning}+{injected_tool}]: {summary}")
            print(f"    {url}")
            if mit:
                print(f"    mitigation: {mit}")
    arch = _load_arch(model) if model else ""
    needs_proxy = (
        injected_reasoning in _REASONING_PARSERS_NEEDING_BUDGET
        or needs_rewrite(arch)
        or injected_tool in _LEAKY_TOOL_PARSERS
    )
    env = _prepare_dyld_env()
    if not needs_proxy:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            *extra_args,
            *auto_args,
            *passthrough,
        ]
        return subprocess.call(cmd, env=env)
    return _serve_with_rewriter(
        extra_args=extra_args,
        auto_args=auto_args,
        passthrough=passthrough,
        env=env,
        arch=arch,
        reasoning_parser=injected_reasoning,
        tool_parser=injected_tool,
    )


def _serve_with_rewriter(
    *,
    extra_args: list[str],
    auto_args: list[str],
    passthrough: list[str],
    env: dict[str, str],
    arch: str,
    reasoning_parser: str,
    tool_parser: str = "",
) -> int:
    """Spawn vLLM on an internal port and the rewriter on the user port.

    Invisible self-heal: client connects to its expected port, the proxy
    silently bumps reasoning-starved `max_tokens` and splits any leaked
    'Thinking:' prefix before the response leaves our process boundary.
    """
    user_port = _extract_port(passthrough)
    internal_port = _free_port(user_port + 1000)
    inner_passthrough = _strip_port(passthrough) + ["--port", str(internal_port)]
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        *extra_args,
        *auto_args,
        *inner_passthrough,
    ]
    print(
        f"vllm-swift: launching transparent rewriter proxy on port {user_port} "
        f"(vLLM bound to internal port {internal_port})"
    )
    if reasoning_parser in _REASONING_PARSERS_NEEDING_BUDGET:
        print(
            f"  rewriter rule: bump client-supplied max_tokens to a reasoning-safe "
            f"floor (parser={reasoning_parser})"
        )
    if needs_rewrite(arch):
        print(
            f"  rewriter rule: split leaked 'Thinking:' prefix into "
            f"reasoning_content (arch={arch})"
        )
    # Recovery rule fires whenever the proxy is engaged (cheap regex pass).
    # Surface it in the banner only when the recovery is the *primary*
    # reason we spawned — i.e., a known-leaky tool parser without reasoning.
    print(
        "  rewriter rule: recover structured tool_calls from plaintext-JSON "
        "leaks in `content` (hermes / qwen3_coder / phi4-pipe / mistral shapes)"
    )
    proc = subprocess.Popen(cmd, env=env)

    def _shutdown(*_: object) -> None:
        if proc.poll() is None:
            proc.terminate()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _shutdown)
        except ValueError:
            pass

    try:
        if not _wait_for_vllm_ready(internal_port):
            sys.stderr.write("vllm-swift: vLLM never became ready; aborting.\n")
            _shutdown()
            proc.wait(timeout=10)
            return proc.returncode or 1

        from vllm_swift.response_rewriter import run as run_rewriter

        run_rewriter(
            user_port=user_port,
            upstream_port=internal_port,
            arch=arch,
            reasoning_parser=reasoning_parser,
            tool_parser=tool_parser,
        )
        return 0
    finally:
        _shutdown()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _download(args: list[str]) -> int:
    if not args:
        sys.stderr.write("Usage: vllm-swift download <hf-model-id>\n")
        return 2
    model_id = args[0]
    short = model_id.rsplit("/", 1)[-1]
    target = os.path.expanduser(f"~/models/{short}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.stderr.write("huggingface_hub not installed. Run: pip install huggingface-hub\n")
        return 1
    path = snapshot_download(model_id, local_dir=target)
    print(f"Downloaded to {path}")
    return 0


def _version() -> int:
    print(f"vllm-swift {__version__}")
    print(f"dylib: {_lib_dir() / 'libVLLMBridge.dylib'}")
    try:
        import vllm

        print(f"vLLM: {vllm.__version__}")
    except ImportError:
        print("vLLM: not installed (pip install vllm)")
    return 0


def _help() -> int:
    print("vllm-swift — Native Swift/Metal backend for vLLM on Apple Silicon")
    print()
    print("Usage:")
    print("  vllm-swift serve <model> [args]   Start OpenAI-compatible API server")
    print("  vllm-swift download <model-id>    Download model from HuggingFace")
    print("  vllm-swift version                Show version info")
    print()
    print("Examples:")
    print("  vllm-swift download mlx-community/Qwen3-4B-4bit")
    print("  vllm-swift serve ~/models/Qwen3-4B-4bit --max-model-len 4096")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        return _help()
    cmd, rest = argv[0], argv[1:]
    if cmd == "serve":
        return _serve(rest)
    if cmd == "download":
        return _download(rest)
    if cmd == "version":
        return _version()
    sys.stderr.write(f"Unknown command: {cmd}\n")
    _help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
