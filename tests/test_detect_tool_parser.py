# SPDX-License-Identifier: Apache-2.0
"""Tests for the architecture -> tool-parser mapping used by `serve`."""

import json
import sys

import pytest

from vllm_swift import detect_tool_parser as dtp


@pytest.mark.parametrize(
    "arch,expected",
    [
        ("", ""),
        ("Qwen3CoderForCausalLM", "qwen3_coder"),
        ("Qwen3ForCausalLM", "hermes"),
        ("Qwen2_5ForCausalLM", "hermes"),
        ("Qwen2ForCausalLM", "hermes"),
        ("NemotronHForCausalLM", "qwen3_coder"),
        ("NemotronForCausalLM", "qwen3_coder"),
        ("HermesForCausalLM", "hermes"),
        ("Llama4ForCausalLM", "llama4_json"),
        ("LlamaForCausalLM", "llama3_json"),
        ("MistralForCausalLM", "mistral"),
        ("Gemma4ForCausalLM", "gemma4"),
        ("Gemma3ForCausalLM", "gemma4"),
        ("Phi4MiniJsonForCausalLM", "phi4_mini_json"),
        ("Phi4ForCausalLM", "phi4_mini_json"),
        ("Phi3ForCausalLM", "phi4_mini_json"),
        ("Granite4ForCausalLM", "granite4"),
        ("GraniteForCausalLM", "granite"),
        ("DeepseekV32ForCausalLM", "deepseek_v32"),
        ("DeepseekV31ForCausalLM", "deepseek_v31"),
        ("DeepseekV3ForCausalLM", "deepseek_v3"),
        ("DeepseekV2ForCausalLM", "deepseek_v3"),
        ("Glm45ForCausalLM", "glm45"),
        ("Glm47ForCausalLM", "glm47"),
        ("Glm4ForCausalLM", "glm45"),
        ("MinimaxM2ForCausalLM", "minimax_m2"),
        ("MiniMaxM2ForCausalLM", "minimax_m2"),
        ("MiniMaxText", "minimax"),
        ("MinimaxText", "minimax"),
        ("KimiK2ForCausalLM", "kimi_k2"),
        ("HunyuanA13BForCausalLM", "hunyuan_a13b"),
        ("Step3ForCausalLM", "step3"),
        ("Olmo3ForCausalLM", "olmo3"),
        ("InternLMForCausalLM", "internlm"),
        ("JambaForCausalLM", "jamba"),
        ("Ernie45ForCausalLM", "ernie45"),
        ("UnknownArchForCausalLM", ""),
    ],
)
def test_arch_to_parser(arch, expected):
    assert dtp._arch_to_parser(arch) == expected


def test_load_arch_returns_empty_when_missing(tmp_path):
    assert dtp._load_arch(str(tmp_path)) == ""


def test_load_arch_returns_empty_on_invalid_json(tmp_path):
    (tmp_path / "config.json").write_text("not json")
    assert dtp._load_arch(str(tmp_path)) == ""


def test_load_arch_returns_empty_when_no_architectures(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    assert dtp._load_arch(str(tmp_path)) == ""


def test_load_arch_reads_first_architecture(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM", "OtherArch"]})
    )
    assert dtp._load_arch(str(tmp_path)) == "Qwen3ForCausalLM"


def test_has_tool_template_true_via_tokenizer_config(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        '{"chat_template": "use tools and tool_call here"}'
    )
    assert dtp._has_tool_template(str(tmp_path)) is True


def test_has_tool_template_true_via_chat_template_jinja(tmp_path):
    (tmp_path / "chat_template.jinja").write_text("{% if tools %}{{ function }}{% endif %}")
    assert dtp._has_tool_template(str(tmp_path)) is True


def test_has_tool_template_false_when_no_files(tmp_path):
    assert dtp._has_tool_template(str(tmp_path)) is False


def test_has_tool_template_false_when_no_tool_keywords(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text('{"chat_template": "plain chat only"}')
    assert dtp._has_tool_template(str(tmp_path)) is False


def test_detect_parser_empty_for_non_directory():
    assert dtp.detect_parser("") == ""
    assert dtp.detect_parser("/nonexistent/path/xyz") == ""


def test_detect_parser_empty_when_arch_unmappable(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["UnknownArch"]}))
    (tmp_path / "tokenizer_config.json").write_text('{"chat_template": "tools and tool_call"}')
    assert dtp.detect_parser(str(tmp_path)) == ""


def test_detect_parser_empty_when_no_tool_template(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    # No template files -> returns ""
    assert dtp.detect_parser(str(tmp_path)) == ""


def test_detect_parser_full_match(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    (tmp_path / "tokenizer_config.json").write_text('{"chat_template": "tools and tool_call"}')
    assert dtp.detect_parser(str(tmp_path)) == "hermes"


def test_main_no_args_returns_0(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["detect_tool_parser"])
    assert dtp.main() == 0


def test_main_writes_parser_to_stdout(tmp_path, monkeypatch, capsys):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (tmp_path / "tokenizer_config.json").write_text('{"chat_template": "tools and tool_call"}')
    monkeypatch.setattr(sys, "argv", ["detect_tool_parser", str(tmp_path)])
    rc = dtp.main()
    assert rc == 0
    assert capsys.readouterr().out == "llama3_json"


def test_main_silent_when_no_parser(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["detect_tool_parser", str(tmp_path)])
    rc = dtp.main()
    assert rc == 0
    assert capsys.readouterr().out == ""
