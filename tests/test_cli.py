# SPDX-License-Identifier: Apache-2.0
"""Tests for the pip-installed `vllm-swift` CLI entry point."""

import signal
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vllm_swift import cli


def test_lib_dir_is_under_package():
    p = cli._lib_dir()
    assert isinstance(p, Path)
    assert p.name == "_lib"
    assert p.parent.name == "vllm_swift"


def test_prepare_dyld_env_appends_when_existing(monkeypatch):
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/tmp/foo")
    env = cli._prepare_dyld_env()
    assert env["DYLD_LIBRARY_PATH"].endswith(":/tmp/foo")
    assert str(cli._lib_dir()) in env["DYLD_LIBRARY_PATH"]


def test_prepare_dyld_env_sets_when_missing(monkeypatch):
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)
    env = cli._prepare_dyld_env()
    assert env["DYLD_LIBRARY_PATH"] == str(cli._lib_dir())


@pytest.mark.parametrize(
    "args,expected",
    [
        (["--tool-call-parser", "hermes"], True),
        (["--tool-call-parser=hermes"], True),
        (["--enable-auto-tool-choice"], True),
        (["--no-enable-auto-tool-choice"], True),
        (["--max-model-len", "4096"], False),
        ([], False),
    ],
)
def test_has_tool_flag(args, expected):
    assert cli._has_tool_flag(args) is expected


@pytest.mark.parametrize(
    "args,expected",
    [
        (["--model", "/path/to/model"], "/path/to/model"),
        (["--model=/path/to/model"], "/path/to/model"),
        (["--port", "8080"], None),
        ([], None),
    ],
)
def test_extract_model(args, expected):
    assert cli._extract_model(args) == expected


def test_serve_no_args_returns_2(capsys):
    rc = cli._serve([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Usage:" in err


def test_serve_positional_model_invokes_subprocess():
    with (
        patch("vllm_swift.cli.subprocess.call", return_value=0) as mock_call,
        patch("vllm_swift.cli.detect_parser", return_value=None),
    ):
        rc = cli._serve(["/models/Qwen3-4B-4bit", "--max-model-len", "4096"])
    assert rc == 0
    cmd = mock_call.call_args[0][0]
    assert "--model" in cmd
    assert "/models/Qwen3-4B-4bit" in cmd
    assert "--max-model-len" in cmd
    env = mock_call.call_args[1]["env"]
    assert str(cli._lib_dir()) in env["DYLD_LIBRARY_PATH"]


def test_serve_with_explicit_model_flag_skips_positional_rewrite():
    with (
        patch("vllm_swift.cli.subprocess.call", return_value=0) as mock_call,
        patch("vllm_swift.cli.detect_parser", return_value=None),
    ):
        cli._serve(["--model", "/m"])
    cmd = mock_call.call_args[0][0]
    # Should not double-inject the positional --model rewrite when the
    # user already passed --model explicitly.
    assert cmd.count("--model") == 1


def test_serve_auto_injects_tool_parser_when_detected(capsys):
    with (
        patch("vllm_swift.cli.subprocess.call", return_value=0) as mock_call,
        patch("vllm_swift.cli.detect_parser", return_value="hermes"),
    ):
        cli._serve(["/models/Qwen3-4B-4bit"])
    cmd = mock_call.call_args[0][0]
    assert "--enable-auto-tool-choice" in cmd
    assert "--tool-call-parser" in cmd
    assert "hermes" in cmd
    out = capsys.readouterr().out
    assert "auto-detected tool parser 'hermes'" in out


def test_serve_does_not_inject_when_user_passes_tool_flag():
    with (
        patch("vllm_swift.cli.subprocess.call", return_value=0) as mock_call,
        patch("vllm_swift.cli.detect_parser", return_value="hermes") as mock_detect,
    ):
        cli._serve(["/models/X", "--tool-call-parser", "llama3_json"])
    cmd = mock_call.call_args[0][0]
    # User's explicit value should be the only occurrence.
    parser_idx = cmd.index("--tool-call-parser")
    assert cmd[parser_idx + 1] == "llama3_json"
    # detect_parser shouldn't even be called (short-circuit on _has_tool_flag).
    mock_detect.assert_not_called()


def test_download_no_args_returns_2(capsys):
    rc = cli._download([])
    assert rc == 2
    assert "Usage:" in capsys.readouterr().err


def test_download_invokes_hf_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    fake_hf = type(sys)("huggingface_hub")
    fake_hf.snapshot_download = lambda repo, local_dir: local_dir
    with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
        rc = cli._download(["org/Qwen3-4B-4bit"])
    assert rc == 0


def test_download_handles_missing_huggingface_hub(monkeypatch, capsys):
    # Make `from huggingface_hub import snapshot_download` raise ImportError.
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("no hf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    rc = cli._download(["org/foo"])
    assert rc == 1
    assert "huggingface_hub not installed" in capsys.readouterr().err


def test_version_prints(capsys, monkeypatch):
    # conftest stubs `vllm` without __version__; set one here so the
    # success branch is exercised.
    import vllm as _vllm

    monkeypatch.setattr(_vllm, "__version__", "0.19.1", raising=False)
    rc = cli._version()
    assert rc == 0
    out = capsys.readouterr().out
    assert "vllm-swift" in out
    assert "dylib:" in out
    assert "vLLM: 0.19.1" in out


def test_version_handles_missing_vllm(capsys, monkeypatch):
    # Force the import vllm path to raise.
    monkeypatch.setitem(sys.modules, "vllm", None)
    rc = cli._version()
    assert rc == 0
    assert "vLLM: not installed" in capsys.readouterr().out


def test_help_prints(capsys):
    rc = cli._help()
    assert rc == 0
    out = capsys.readouterr().out
    assert "Usage:" in out
    assert "serve" in out
    assert "download" in out


def test_main_no_args_shows_help(capsys):
    rc = cli.main([])
    assert rc == 0
    assert "Usage:" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["-h", "--help", "help"])
def test_main_help_flags(flag, capsys):
    rc = cli.main([flag])
    assert rc == 0
    assert "Usage:" in capsys.readouterr().out


def test_main_dispatches_serve():
    with patch("vllm_swift.cli._serve", return_value=0) as mock_serve:
        cli.main(["serve", "model"])
    mock_serve.assert_called_once_with(["model"])


def test_main_dispatches_download():
    with patch("vllm_swift.cli._download", return_value=0) as mock_dl:
        cli.main(["download", "id"])
    mock_dl.assert_called_once_with(["id"])


def test_main_dispatches_version():
    with patch("vllm_swift.cli._version", return_value=0) as mock_v:
        cli.main(["version"])
    mock_v.assert_called_once_with()


def test_main_unknown_command_returns_2(capsys):
    rc = cli.main(["nonexistent"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Unknown command" in err


def test_main_uses_sys_argv_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["vllm-swift", "version"])
    with patch("vllm_swift.cli._version", return_value=0) as mock_v:
        cli.main()
    mock_v.assert_called_once_with()


# ---------------------------------------------------------------------------
# Port helpers + rewriter routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args,expected",
    [
        (["--port", "9000"], 9000),
        (["--port=9001"], 9001),
        (["--max-model-len", "8192"], 8000),
        ([], 8000),
        (["--port", "not-a-number"], 8000),
    ],
)
def test_extract_port(args, expected):
    assert cli._extract_port(args) == expected


@pytest.mark.parametrize(
    "args,expected",
    [
        (["--port", "9000", "--max-model-len", "8192"], ["--max-model-len", "8192"]),
        (["--port=9001", "--foo"], ["--foo"]),
        (["--max-model-len", "4096"], ["--max-model-len", "4096"]),
        ([], []),
    ],
)
def test_strip_port(args, expected):
    assert cli._strip_port(args) == expected


def test_serve_routes_through_rewriter_when_reasoning_parser_set(tmp_path):
    """Reasoning-parser auto-injection should trigger the proxy path."""
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"architectures":["NemotronHForCausalLM"]}')

    with (
        patch("vllm_swift.cli.detect_parser", return_value=""),
        patch("vllm_swift.cli.detect_reasoning_parser", return_value="nemotron_v3"),
        patch("vllm_swift.cli._serve_with_rewriter", return_value=0) as mock_proxy,
        patch("subprocess.call", return_value=0) as mock_call,
    ):
        rc = cli._serve([str(model_dir)])
    assert rc == 0
    mock_proxy.assert_called_once()
    mock_call.assert_not_called()


def test_serve_skips_rewriter_for_plain_chat_model(tmp_path):
    """Llama-style non-reasoning models should bypass the proxy."""
    model_dir = tmp_path / "llama"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"architectures":["LlamaForCausalLM"]}')

    with (
        patch("vllm_swift.cli.detect_parser", return_value=""),
        patch("vllm_swift.cli.detect_reasoning_parser", return_value=""),
        patch("vllm_swift.cli._serve_with_rewriter", return_value=0) as mock_proxy,
        patch("subprocess.call", return_value=0) as mock_call,
    ):
        rc = cli._serve([str(model_dir)])
    assert rc == 0
    mock_proxy.assert_not_called()
    mock_call.assert_called_once()


# ---------------------------------------------------------------------------
# Pre-flight parser-registry validation
# ---------------------------------------------------------------------------


def test_validate_against_registry_passes_for_registered():
    assert cli._validate_against_registry("hermes", {"hermes", "qwen3_coder"}, "tool")


def test_validate_against_registry_skips_when_unregistered(capsys):
    """Unregistered parser name → returns False + warns to stderr."""
    ok = cli._validate_against_registry("ghost_parser", {"hermes", "qwen3_coder"}, "tool")
    assert ok is False
    err = capsys.readouterr().err
    assert "ghost_parser" in err
    assert "not registered" in err
    assert "skipping auto-injection" in err


def test_validate_against_registry_passes_when_vllm_unavailable(capsys):
    """Empty registry (vLLM not importable) → trust the detector, no warning."""
    ok = cli._validate_against_registry("hermes", set(), "tool")
    assert ok is True
    assert capsys.readouterr().err == ""


def test_serve_skips_injection_for_unregistered_parser(tmp_path):
    """Detected parser not in vLLM's registry → skip injection, fall through
    to the no-proxy bypass path so vLLM still launches without crashing."""
    model_dir = tmp_path / "hypothetical-future-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"architectures":["FutureForCausalLM"]}')

    with (
        patch("vllm_swift.cli.detect_parser", return_value="future_parser_dne"),
        patch("vllm_swift.cli.detect_reasoning_parser", return_value=""),
        patch("vllm_swift.cli._registered_tool_parsers", return_value={"hermes"}),
        patch("vllm_swift.cli._registered_reasoning_parsers", return_value=set()),
        patch("vllm_swift.cli._serve_with_rewriter", return_value=0) as mock_proxy,
        patch("subprocess.call", return_value=0) as mock_call,
    ):
        rc = cli._serve([str(model_dir)])
    assert rc == 0
    # Unregistered tool parser → no proxy path, plain bypass to vLLM
    mock_proxy.assert_not_called()
    mock_call.assert_called_once()
    # The injected args should NOT include the bogus parser
    cmd = mock_call.call_args[0][0]
    assert "future_parser_dne" not in cmd
    assert "--tool-call-parser" not in cmd


# --- EngineCore process-group lifecycle ---------------------------------
#
# Regression coverage for the "leak" where SIGTERM-ing vllm-swift left a
# `VLLM::EngineCore` grandchild alive, pinning ~all KV cache memory until
# manually killed. See `cli._shutdown_pgroup` and
# `cli._kill_orphan_engine_cores`.


def _proc_stub(pid: int = 99999, poll_return=None) -> MagicMock:
    """Minimal `subprocess.Popen` stand-in for the shutdown helpers."""
    p = MagicMock()
    p.pid = pid
    p.poll.return_value = poll_return
    return p


def test_shutdown_pgroup_sigterms_whole_group():
    proc = _proc_stub(pid=12345)
    with (
        patch("vllm_swift.cli.os.getpgid", return_value=12345) as mock_getpgid,
        patch("vllm_swift.cli.os.killpg") as mock_killpg,
    ):
        cli._shutdown_pgroup(proc)
    mock_getpgid.assert_called_once_with(12345)
    mock_killpg.assert_called_once_with(12345, signal.SIGTERM)
    proc.terminate.assert_not_called()


def test_shutdown_pgroup_falls_back_to_terminate_when_pgid_lookup_fails():
    proc = _proc_stub(pid=12345)
    with (
        patch("vllm_swift.cli.os.getpgid", side_effect=ProcessLookupError),
        patch("vllm_swift.cli.os.killpg") as mock_killpg,
    ):
        cli._shutdown_pgroup(proc)
    mock_killpg.assert_not_called()
    proc.terminate.assert_called_once()


def test_shutdown_pgroup_noops_when_process_already_dead():
    proc = _proc_stub(pid=12345, poll_return=0)  # already exited
    with (
        patch("vllm_swift.cli.os.getpgid") as mock_getpgid,
        patch("vllm_swift.cli.os.killpg") as mock_killpg,
    ):
        cli._shutdown_pgroup(proc)
    mock_getpgid.assert_not_called()
    mock_killpg.assert_not_called()
    proc.terminate.assert_not_called()


def test_kill_orphan_engine_cores_returns_zero_when_no_parent_pid():
    assert cli._kill_orphan_engine_cores(None) == 0
    assert cli._kill_orphan_engine_cores(0) == 0


def test_kill_orphan_engine_cores_returns_zero_when_pgrep_empty():
    """No leftover EngineCores → 0 kills, never call os.kill."""
    with (
        patch("vllm_swift.cli.os.getpgid", return_value=12345),
        patch("vllm_swift.cli.subprocess.check_output", return_value=""),
        patch("vllm_swift.cli.os.kill") as mock_kill,
    ):
        n = cli._kill_orphan_engine_cores(12345)
    assert n == 0
    mock_kill.assert_not_called()


def test_kill_orphan_engine_cores_kills_matching_pgid_only():
    """SIGKILL only EngineCores whose pgid matches our parent's pgid.

    Critical safety property: never kill someone else's EngineCore from
    a different vllm-swift instance.
    """
    parent_pid = 100
    parent_pgid = 200

    # Two EngineCores in pgrep output: one matches our pgid, one belongs
    # to a different vllm-swift instance.
    pgrep_output = "300\n400\n"

    def fake_getpgid(pid: int) -> int:
        return {
            parent_pid: parent_pgid,
            300: parent_pgid,  # ours — should be killed
            400: 999,  # someone else's — must be spared
        }[pid]

    with (
        patch("vllm_swift.cli.os.getpgid", side_effect=fake_getpgid),
        patch("vllm_swift.cli.subprocess.check_output", return_value=pgrep_output),
        patch("vllm_swift.cli.os.kill") as mock_kill,
    ):
        n = cli._kill_orphan_engine_cores(parent_pid)

    assert n == 1
    mock_kill.assert_called_once_with(300, signal.SIGKILL)


def test_kill_orphan_engine_cores_skips_dead_pids():
    """pgrep race: pid disappears between pgrep and getpgid → skip cleanly."""
    parent_pid = 100
    parent_pgid = 200

    def fake_getpgid(pid: int) -> int:
        if pid == parent_pid:
            return parent_pgid
        raise ProcessLookupError

    with (
        patch("vllm_swift.cli.os.getpgid", side_effect=fake_getpgid),
        patch("vllm_swift.cli.subprocess.check_output", return_value="300\n"),
        patch("vllm_swift.cli.os.kill") as mock_kill,
    ):
        n = cli._kill_orphan_engine_cores(parent_pid)

    assert n == 0
    mock_kill.assert_not_called()


def test_kill_orphan_engine_cores_swallows_pgrep_failure():
    """pgrep not installed / timed out → silent no-op, never raises."""
    with (
        patch("vllm_swift.cli.os.getpgid", return_value=200),
        patch(
            "vllm_swift.cli.subprocess.check_output",
            side_effect=FileNotFoundError("pgrep"),
        ),
        patch("vllm_swift.cli.os.kill") as mock_kill,
    ):
        n = cli._kill_orphan_engine_cores(100)
    assert n == 0
    mock_kill.assert_not_called()


def test_kill_orphan_engine_cores_handles_kill_race():
    """If SIGKILL races with natural exit, count stays accurate."""
    parent_pid = 100
    parent_pgid = 200

    def fake_getpgid(pid: int) -> int:
        return {parent_pid: parent_pgid, 300: parent_pgid, 301: parent_pgid}[pid]

    def fake_kill(pid: int, sig: int) -> None:
        if pid == 301:
            raise ProcessLookupError  # exited before our SIGKILL landed

    with (
        patch("vllm_swift.cli.os.getpgid", side_effect=fake_getpgid),
        patch("vllm_swift.cli.subprocess.check_output", return_value="300\n301\n"),
        patch("vllm_swift.cli.os.kill", side_effect=fake_kill),
    ):
        n = cli._kill_orphan_engine_cores(parent_pid)

    # Only 300 was actually killed; 301 was already gone.
    assert n == 1


def test_serve_with_rewriter_uses_start_new_session_for_pgroup_kill():
    """Regression (source-level): the Popen call in `_serve_with_rewriter`
    MUST pass `start_new_session=True`, otherwise the api_server +
    EngineCore land in our pgid and SIGTERM-ing our process group nukes
    vllm-swift itself. With start_new_session=True they get a fresh
    pgid that we can later target with killpg().

    The body of `_serve_with_rewriter` is `# pragma: no cover` because
    it spawns real subprocesses and runs the rewriter event loop —
    asserting at the source level is the practical seam.
    """
    import inspect

    src = inspect.getsource(cli._serve_with_rewriter)
    # Be flexible about formatting (keyword positioning, line wraps).
    assert "start_new_session=True" in src, (
        "subprocess.Popen in _serve_with_rewriter must pass "
        "start_new_session=True — without it SIGTERM leaves "
        "VLLM::EngineCore orphaned holding all KV cache memory."
    )
    # And the shutdown path must go through _shutdown_pgroup.
    assert "_shutdown_pgroup" in src
    # And the orphan-sweeper must fire on exit.
    assert "_kill_orphan_engine_cores" in src
