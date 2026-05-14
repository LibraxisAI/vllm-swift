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


def _shutdown_pgroup(proc: subprocess.Popen) -> None:
    """SIGTERM the whole process group of `proc`.

    vLLM V1 spawns `VLLM::EngineCore` as a child of the api_server. Both
    inherit the api_server's pgid (set via `start_new_session=True`).
    `killpg(pgid, SIGTERM)` reaps the api_server AND EngineCore in one
    shot. Falls back to plain `proc.terminate()` if the process is
    already gone or the pgid lookup fails.
    """
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()


def _kill_orphan_engine_cores(parent_pid: int | None) -> int:
    """Belt-and-suspenders: SIGKILL leftover `VLLM::EngineCore` processes
    whose pgid matches `parent_pid`'s pgid.

    Covers hard-crash scenarios where the api_server died without taking
    EngineCore with it. Returns the number of processes killed.
    """
    if not parent_pid:
        return 0
    try:
        own_pgid = os.getpgid(parent_pid)
    except (ProcessLookupError, PermissionError):
        return 0
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "VLLM::EngineCore"], text=True, timeout=2
        )
    except Exception:  # noqa: BLE001
        return 0
    killed = 0
    for line in out.splitlines():
        pid_str = line.strip()
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        try:
            ppgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError):
            continue
        if ppgid != own_pgid:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except (ProcessLookupError, PermissionError):
            pass
    return killed


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


def _extract_retrieval_endpoint(args: list[str]) -> tuple[str, list[str]]:
    """Pull `--retrieval-endpoint URL` out of `args`. Optional flag.

    When set, vllm-swift's transparent rewriter calls longctx-svc on every
    chat-completion to splice retrieved code chunks into the prompt before
    forwarding to vLLM. Tool is optional — flag absent → no-op.

    Returns (url_or_empty, args_with_flag_removed).
    """
    out: list[str] = []
    url = os.environ.get("LONGCTX_ENDPOINT", "")
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--retrieval-endpoint":
            if i + 1 < len(args):
                url = args[i + 1]
                skip_next = True
            continue
        if arg.startswith("--retrieval-endpoint="):
            url = arg.split("=", 1)[1]
            continue
        out.append(arg)
    return url, out


def _extract_enable_longctx(args: list[str]) -> tuple[bool, list[str]]:
    """Pull `--enable-longctx` boolean flag out of `args`. Optional.

    When set, vllm-swift auto-spawns a longctx-svc subprocess on a
    free local port and wires its own --retrieval-endpoint to that
    port. Saves users the "start two terminals" dance.

    Returns (enabled, args_with_flag_removed).
    """
    out: list[str] = []
    enabled = os.environ.get("LONGCTX_ENABLE", "").lower() in ("1", "true", "yes")
    for arg in args:
        if arg == "--enable-longctx":
            enabled = True
            continue
        if arg == "--no-enable-longctx":
            enabled = False
            continue
        out.append(arg)
    return enabled, out


def _extract_longctx_scope(args: list[str]) -> tuple[str, list[str]]:
    """Pull `--longctx-scope PATH` out of `args`. PRD §5.8.

    When set, every chat completion's /retrieve call gets `default_scope`
    set to PATH so retrieval fires even when the user message contains
    no absolute paths. Lets tool-using agents (Hermes, OpenCode agentic)
    get retrieval without needing path mentions.

    If unset and `--enable-longctx` is on, vllm-swift falls back to
    `os.getcwd()` at boot. The flag exists for users who want to pin a
    different project than where they happen to be standing.

    Returns (scope_path_or_empty, args_with_flag_removed).
    """
    out: list[str] = []
    scope = os.environ.get("LONGCTX_DEFAULT_SCOPE", "")
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--longctx-scope":
            if i + 1 < len(args):
                scope = args[i + 1]
                skip_next = True
            continue
        if arg.startswith("--longctx-scope="):
            scope = arg.split("=", 1)[1]
            continue
        out.append(arg)
    return scope, out


def _extract_max_model_len(args: list[str]) -> int | None:
    """Find --max-model-len in args, else None."""
    prev = ""
    for arg in args:
        if arg.startswith("--max-model-len="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return None
        if prev == "--max-model-len":
            try:
                return int(arg)
            except ValueError:
                return None
        prev = arg
    return None


def _model_max_position_embeddings(model_path: str) -> int | None:
    """Read `max_position_embeddings` (or `model_max_length`) from the
    model's config.json. Returns None on any error so callers can skip
    the check rather than fail-stop on a missing/odd config.

    Used by the pre-flight warn that catches the bug #2 footgun: a user
    setting `--max-model-len 65536` for a model whose config caps at
    40960. vLLM raises a not-very-specific error in that case; we'd
    rather warn upfront with the actual numbers.
    """
    if not model_path:
        return None
    expanded = os.path.expanduser(model_path)
    candidates = [
        Path(expanded) / "config.json",
        Path(expanded),  # if user already pointed at config.json
    ]
    for p in candidates:
        if not p.is_file():
            continue
        try:
            import json as _json

            cfg = _json.loads(p.read_text())
            for key in ("max_position_embeddings", "model_max_length"):
                v = cfg.get(key)
                if isinstance(v, int) and v > 0:
                    return v
        except (OSError, ValueError):
            return None
    return None


def _warn_if_max_model_len_exceeds_model(
    model_path: str,
    requested_max_model_len: int | None,
) -> None:
    """Pre-flight warn for bug #2: `--max-model-len` larger than the
    model's declared positional embedding budget. Warning, not error —
    some long-context fine-tunes legitimately extend the model's
    declared position embeddings.
    """
    if requested_max_model_len is None or requested_max_model_len <= 0:
        return
    cap = _model_max_position_embeddings(model_path)
    if cap is None or requested_max_model_len <= cap:
        return
    sys.stderr.write(
        f"vllm-swift: --max-model-len {requested_max_model_len} exceeds "
        f"this model's max_position_embeddings ({cap}). vLLM will likely "
        f"reject prompts at that length. Recommend "
        f"--max-model-len {cap} or smaller.\n"
    )


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
    # Strip --retrieval-endpoint and friends before vLLM sees them.
    retrieval_endpoint, passthrough = _extract_retrieval_endpoint(passthrough)
    enable_longctx, passthrough = _extract_enable_longctx(passthrough)
    longctx_scope_override, passthrough = _extract_longctx_scope(passthrough)
    # Bug #2 from 0.5.1 alpha: `--max-model-len 65536` against a model
    # with max_position_embeddings=40960 → vLLM rejects prompts. Warn
    # up front with the actual numbers instead of failing later.
    _warn_if_max_model_len_exceeds_model(
        model,
        _extract_max_model_len(passthrough),
    )
    longctx_sidecar = None
    if enable_longctx and not retrieval_endpoint:
        try:
            from longctx_svc.sidecar import spawn_sidecar
        except ImportError:
            sys.stderr.write(
                "vllm-swift: --enable-longctx needs the optional retrieval "
                "companion, which isn't installed yet. Install with:\n\n"
                "  vllm-swift longctx-install\n\n"
                "or directly:\n\n"
                "  pip install longctx-svc\n"
                "  # (or: pip install vllm-swift[longctx])\n"
            )
            return 2
        print("vllm-swift: --enable-longctx set, spawning longctx-svc sidecar...")
        try:
            longctx_sidecar = spawn_sidecar(boot_timeout=30.0)
        except RuntimeError as exc:
            sys.stderr.write(f"vllm-swift: failed to start longctx-svc: {exc}\n")
            return 1
        retrieval_endpoint = longctx_sidecar.url
        print(
            f"vllm-swift: longctx-svc sidecar healthy at {retrieval_endpoint} "
            f"(pid {longctx_sidecar.proc.pid}); will be torn down on shutdown"
        )
        # Tie sidecar lifecycle to this process: it dies when we die.
        import atexit

        atexit.register(longctx_sidecar.stop)
    # Resolve the default-scope: explicit flag/env wins, else os.getcwd()
    # when --enable-longctx is on. Path-in-message still overrides.
    longctx_default_scope = longctx_scope_override
    if retrieval_endpoint and not longctx_default_scope:
        longctx_default_scope = os.getcwd()
    if retrieval_endpoint:
        print(f"vllm-swift: longctx retrieval enabled (endpoint: {retrieval_endpoint})")
        if longctx_default_scope:
            print(
                f"vllm-swift: default scope = {longctx_default_scope} "
                "(used when no path is mentioned in the message; "
                "override with --longctx-scope PATH or "
                "LONGCTX_DEFAULT_SCOPE env)"
            )
        print(
            "  → To verify it's working: include an absolute file path in "
            "your chat message,\n"
            "    then watch this terminal for a `[longctx] N chunk(s) "
            "from /path ...` line\n"
            "    after each request. You can also `curl "
            f'{retrieval_endpoint}/longctx/status -H "accept: text/plain"`\n'
            "    for a live snapshot of indexed scopes."
        )
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
        or bool(retrieval_endpoint)
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
        max_model_len=_extract_max_model_len(passthrough),
        retrieval_endpoint=retrieval_endpoint,
        longctx_default_scope=longctx_default_scope,
    )


def _serve_with_rewriter(  # pragma: no cover
    *,
    extra_args: list[str],
    auto_args: list[str],
    passthrough: list[str],
    env: dict[str, str],
    arch: str,
    reasoning_parser: str,
    tool_parser: str = "",
    max_model_len: int | None = None,
    retrieval_endpoint: str = "",
    longctx_default_scope: str = "",
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
    # start_new_session=True puts vllm api_server AND its EngineCore
    # subprocess into a fresh process group, so killpg() can reap both.
    # Without this, SIGTERM on the parent leaves EngineCore orphaned —
    # the child holds ~all KV cache memory and keeps the GPU busy.
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)

    def _shutdown(*_: object) -> None:
        _shutdown_pgroup(proc)

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
            max_model_len=max_model_len,
            retrieval_endpoint=retrieval_endpoint,
            longctx_default_scope=longctx_default_scope,
        )
        return 0
    finally:
        _shutdown()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()
        _kill_orphan_engine_cores(proc.pid)


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
    print("  vllm-swift serve <model> [args]    Start OpenAI-compatible API server")
    print("  vllm-swift download <model-id>     Download model from HuggingFace")
    print("  vllm-swift longctx-status [URL]    Show live longctx-svc status")
    print("  vllm-swift version                 Show version info")
    print()
    print("Examples:")
    print("  vllm-swift download mlx-community/Qwen3-4B-4bit")
    print("  vllm-swift serve ~/models/Qwen3-4B-4bit --max-model-len 4096")
    print("  vllm-swift serve ~/models/Qwen3-4B-4bit --enable-longctx")
    return 0


def _longctx_test(rest: list[str]) -> int:
    """Self-test: spawn longctx-svc, run a synthetic /retrieve against
    a real project, and print a clear PASS/FAIL summary.

    Use:
        vllm-swift longctx-test                   # generates a tmp project
        vllm-swift longctx-test /path/to/project  # tests against your repo
    """
    import json
    import tempfile
    import urllib.request
    from pathlib import Path

    try:
        from longctx_svc.sidecar import managed_sidecar
    except ImportError:
        sys.stderr.write(
            "vllm-swift: longctx-svc isn't installed. Run `vllm-swift longctx-install` first.\n"
        )
        return 2

    if rest:
        project = Path(rest[0]).expanduser().resolve()
        if not project.is_dir():
            sys.stderr.write(f"vllm-swift: not a directory: {project}\n")
            return 2
        # Find any real source file inside the project for the prefill
        candidates = (
            list(project.rglob("*.py"))[:3]
            + list(project.rglob("*.ts"))[:3]
            + list(project.rglob("*.js"))[:3]
            + list(project.rglob("*.go"))[:3]
        )
        if not candidates:
            sys.stderr.write(f"vllm-swift: no .py/.ts/.js/.go files found in {project}\n")
            return 2
        target_path = candidates[0]
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="vllm-swift-longctx-test-"))
        project = tmp_dir / "smokeapp"
        (project / "src").mkdir(parents=True)
        (project / "package.json").write_text('{"name":"smokeapp"}\n')
        target_path = project / "src" / "auth.ts"
        target_path.write_text(
            "export function authMiddleware() {\n"
            "  // unique self-test marker UNICORN_KIWI_TEST_TOKEN\n"
            "  return true;\n"
            "}\n"
        )

    print(f"vllm-swift: longctx self-test against {project}")
    print(f"  target file: {target_path}")
    print("  spawning longctx-svc sidecar...")
    try:
        with managed_sidecar(boot_timeout=30.0) as sc:
            print(f"  sidecar healthy at {sc.url}")
            req = urllib.request.Request(
                f"{sc.url}/retrieve",
                data=json.dumps(
                    {
                        "prefill_text": f"explain the function in {target_path}",
                        "query": "what does this code do",
                        "top_k": 4,
                    }
                ).encode(),
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"vllm-swift: self-test FAILED: {exc!s}\n")
        return 1

    n_chunks = len(data.get("chunks") or [])
    scope_path = data.get("scope_path") or "<none>"
    status = data.get("scope_status") or "<none>"
    sentinel = data.get("scope_sentinel") or "<none>"
    print(f"  scope detected: {scope_path}")
    print(f"  sentinel: {sentinel}")
    print(f"  status: {status}")
    print(f"  chunks retrieved: {n_chunks}")
    if n_chunks > 0 and status in ("ready", "empty"):
        print()
        print("  ✓ PASS — longctx is working end-to-end.")
        print(
            "  When you `serve --enable-longctx`, you'll see "
            "`[longctx] N chunk(s) ...` lines per chat completion."
        )
        return 0
    print()
    sys.stderr.write(
        "  ✗ FAIL — retrieval returned 0 chunks. Is your --target path "
        "absolute and does its parent have a sentinel "
        "(.git, package.json, pyproject.toml, …)?\n"
    )
    return 1


def _longctx_status(rest: list[str]) -> int:
    """Hit the longctx-svc /longctx/status endpoint and pretty-print.

    Defaults to http://127.0.0.1:8765 (the address vllm-swift's
    --enable-longctx sidecar uses by default). Override with a positional
    URL: `vllm-swift longctx-status http://other-host:9000`.
    """
    import urllib.error
    import urllib.request

    base = rest[0].rstrip("/") if rest else "http://127.0.0.1:8765"
    try:
        req = urllib.request.Request(
            f"{base}/longctx/status",
            headers={"accept": "text/plain"},
        )
        with urllib.request.urlopen(req, timeout=5.0) as r:
            sys.stdout.write(r.read().decode("utf-8", errors="replace"))
            if not sys.stdout.isatty():
                return 0
            sys.stdout.write("\n")
        return 0
    except urllib.error.URLError as e:
        sys.stderr.write(
            f"vllm-swift: longctx-svc not reachable at {base} ({e}).\n"
            f"Is `--enable-longctx` running, or pass the URL explicitly.\n"
        )
        return 1


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
    if cmd in ("longctx-status", "longctx-stat"):
        return _longctx_status(rest)
    if cmd == "longctx-test":
        return _longctx_test(rest)
    sys.stderr.write(f"Unknown command: {cmd}\n")
    _help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
