# SPDX-License-Identifier: Apache-2.0
"""Tests for the pip-installed `vllm-swift` CLI entry point."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

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
