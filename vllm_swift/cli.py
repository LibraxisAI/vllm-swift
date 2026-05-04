"""CLI entry point for the pip-installed vllm-swift package.

Mirrors the Homebrew bash wrapper's `serve / download / version` commands
so `pip install vllm-swift && vllm-swift serve <model>` produces the same
behavior as the brew installation, with the same auto-detect smart defaults
for the tool-call parser.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from vllm_swift import __version__
from vllm_swift.detect_reasoning_parser import detect_parser as detect_reasoning_parser
from vllm_swift.detect_tool_parser import detect_parser
from vllm_swift.known_parser_issues import (
    combo_caveats,
    reasoning_parser_caveats,
    tool_parser_caveats,
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
    if model and not _has_tool_flag(passthrough):
        parser = detect_parser(model)
        if parser:
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
        if rparser:
            print(
                f"vllm-swift: auto-detected reasoning parser '{rparser}' for {short}; "
                f"injecting --reasoning-parser {rparser}"
            )
            print(
                "  (override with explicit --reasoning-parser <name> or "
                "--no-enable-reasoning)"
            )
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
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        *extra_args,
        *auto_args,
        *passthrough,
    ]
    env = _prepare_dyld_env()
    return subprocess.call(cmd, env=env)


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
