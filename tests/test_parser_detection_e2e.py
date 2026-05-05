# SPDX-License-Identifier: Apache-2.0
"""End-to-end mock tests for tool + reasoning parser auto-detection.

For each common model family, we materialize a minimal model directory
(config.json + chat_template.jinja + tokenizer_config.json) that mirrors
the real layout HuggingFace ships, then run both `detect_tool_parser`
and `detect_reasoning_parser` against it. Asserts the (tool, reasoning)
pair matches what `serve` should auto-inject for that model.

Coverage targets the agents we know consume vllm-swift in the wild:
Hermes, OpenCode, Droid, Aider, plus arbitrary OpenAI-compatible clients.
All of those rely on the server populating `message.tool_calls` and
`message.reasoning_content` correctly, which only happens when vLLM has
the right `--tool-call-parser` and `--reasoning-parser` configured.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from vllm_swift import detect_reasoning_parser as drp
from vllm_swift import detect_tool_parser as dtp


@dataclass(frozen=True)
class ModelFixture:
    """Minimal-viable mock of a HuggingFace model directory."""

    name: str  # short label for test reporting
    architecture: str  # value of config.json["architectures"][0]
    chat_template: str  # contents of chat_template.jinja
    expected_tool: str  # what detect_tool_parser should return ("" if none)
    expected_reasoning: str  # what detect_reasoning_parser should return ("" if none)


# Markers used in real model templates. Tool template needs `tools` plus one
# of `tool_call`/`function`. Reasoning template needs a thinking marker.
_TOOLS_FRAGMENT = "{% if tools %}{% for tool in tools %}{{ tool.function }}{% endfor %}{% endif %}"
_THINK_FRAGMENT = "{% if add_thinking %}<think>{% endif %}"
_TOOLS_AND_THINK = _TOOLS_FRAGMENT + _THINK_FRAGMENT


FIXTURES: tuple[ModelFixture, ...] = (
    # ===== Qwen family =====
    ModelFixture(
        name="qwen3-8b",
        architecture="Qwen3ForCausalLM",
        chat_template=_TOOLS_AND_THINK,
        expected_tool="hermes",
        expected_reasoning="qwen3",
    ),
    ModelFixture(
        name="qwen3-coder-30b",
        architecture="Qwen3CoderForCausalLM",
        chat_template=_TOOLS_AND_THINK,
        expected_tool="qwen3_coder",
        expected_reasoning="qwen3",
    ),
    ModelFixture(
        name="qwen3.5-moe-30b",
        architecture="Qwen3_5MoeForConditionalGeneration",
        chat_template=_TOOLS_AND_THINK,
        # MoE-style Qwen3.5/3.6 ship qwen3_coder XML in their chat template
        # (verified Nov 2026 against Qwen3.5-35B-A3B-4bit & Qwen3.6-35B-A3B-4bit).
        # The `Qwen3_5Moe` prefix catches them; dense Qwen3.5 still routes to hermes.
        expected_tool="qwen3_coder",
        expected_reasoning="qwen3",
    ),
    ModelFixture(
        name="qwen3.6-35b-a3b",
        architecture="Qwen3_6ForCausalLM",
        chat_template=_TOOLS_AND_THINK,
        # All Qwen3.5+/3.6+ ship qwen3_coder XML, dense or MoE.
        expected_tool="qwen3_coder",
        expected_reasoning="qwen3",
    ),
    ModelFixture(
        name="qwen2.5-7b",
        architecture="Qwen2_5ForCausalLM",
        chat_template=_TOOLS_FRAGMENT,
        expected_tool="hermes",
        expected_reasoning="",  # Qwen2.5 isn't a reasoning model
    ),
    # ===== Llama family =====
    ModelFixture(
        name="llama-3.1-8b",
        architecture="LlamaForCausalLM",
        chat_template=_TOOLS_FRAGMENT,
        expected_tool="llama3_json",
        expected_reasoning="",
    ),
    ModelFixture(
        name="llama-4-scout",
        architecture="Llama4ForCausalLM",
        chat_template=_TOOLS_FRAGMENT,
        expected_tool="llama4_json",
        expected_reasoning="",
    ),
    # ===== Mistral / Magistral =====
    ModelFixture(
        name="mistral-7b-instruct",
        architecture="MistralForCausalLM",
        chat_template=_TOOLS_FRAGMENT,
        expected_tool="mistral",
        # Mistral arch alone matches reasoning mapping; without thinking marker
        # the conservative detector returns "" (correct for non-thinking Mistral).
        expected_reasoning="",
    ),
    ModelFixture(
        name="magistral-medium",
        architecture="MagistralForCausalLM",
        chat_template=_TOOLS_AND_THINK,
        # Magistral is the reasoning Mistral; tool detector now maps it
        # to mistral too (cross-port from reasoning detector).
        expected_tool="mistral",
        expected_reasoning="mistral",
    ),
    # ===== Gemma 4 =====
    ModelFixture(
        name="gemma-4-27b-it",
        architecture="Gemma4ForCausalLM",
        chat_template=_TOOLS_AND_THINK,
        expected_tool="gemma4",
        expected_reasoning="gemma4",
    ),
    # ===== DeepSeek =====
    ModelFixture(
        name="deepseek-r1-distill-llama-70b",
        architecture="DeepseekR1ForCausalLM",
        chat_template=_TOOLS_AND_THINK,
        # No DeepseekR1 prefix in tool table -> ""
        expected_tool="",
        expected_reasoning="deepseek_r1",
    ),
    ModelFixture(
        name="deepseek-v3",
        architecture="DeepseekV3ForCausalLM",
        chat_template=_TOOLS_AND_THINK,
        expected_tool="deepseek_v3",
        expected_reasoning="deepseek_v3",
    ),
    ModelFixture(
        name="deepseek-v3.2",
        architecture="DeepseekV32ForCausalLM",
        chat_template=_TOOLS_AND_THINK,
        expected_tool="deepseek_v32",
        expected_reasoning="deepseek_v3",
    ),
    # ===== Nemotron-Cascade (Qwen3.6 derivative, but uses NVIDIA's parsers) =====
    # NVIDIA recommends qwen3_coder + nemotron_v3 (HF discussion #7).
    # The model emits <function=name><parameter=k>v</parameter></function>
    # which only qwen3_coder parses; hermes silently fails on this XML shape.
    ModelFixture(
        name="nemotron-cascade-2-30b",
        architecture="NemotronHForCausalLM",
        chat_template=_TOOLS_AND_THINK,
        expected_tool="qwen3_coder",
        expected_reasoning="nemotron_v3",
    ),
    # ===== Phi-4 =====
    ModelFixture(
        name="phi-4-14b",
        architecture="Phi4ForCausalLM",
        chat_template=_TOOLS_FRAGMENT,
        expected_tool="phi4_mini_json",
        expected_reasoning="",  # Phi-4 not a thinking model in the registry
    ),
    # ===== Granite 4 =====
    ModelFixture(
        name="granite-4-instruct",
        architecture="Granite4ForCausalLM",
        chat_template=_TOOLS_AND_THINK,
        expected_tool="granite4",
        expected_reasoning="granite",
    ),
    # ===== GLM 4.5 =====
    ModelFixture(
        name="glm-4.5-air",
        architecture="Glm45ForCausalLM",
        chat_template=_TOOLS_AND_THINK,
        expected_tool="glm45",
        expected_reasoning="glm45",
    ),
    # ===== Kimi K2 =====
    ModelFixture(
        name="kimi-k2",
        architecture="KimiK2ForCausalLM",
        chat_template=_TOOLS_AND_THINK,
        expected_tool="kimi_k2",
        expected_reasoning="kimi_k2",
    ),
    # ===== MiniMax M2 =====
    ModelFixture(
        name="minimax-m2",
        architecture="MiniMaxM2ForCausalLM",
        chat_template=_TOOLS_AND_THINK,
        expected_tool="minimax_m2",
        expected_reasoning="minimax_m2",
    ),
    # ===== Olmo3 =====
    ModelFixture(
        name="olmo-3-7b",
        architecture="Olmo3ForCausalLM",
        chat_template=_TOOLS_AND_THINK,
        expected_tool="olmo3",
        expected_reasoning="olmo3",
    ),
    # ===== GPT-OSS =====
    ModelFixture(
        name="gpt-oss-20b",
        architecture="GptOssForCausalLM",
        chat_template=_TOOLS_FRAGMENT + "\n<|channel|>final<|message|>",
        # GptOss now maps to vLLM's "openai" tool parser (cross-ported).
        expected_tool="openai",
        expected_reasoning="openai_gptoss",
    ),
    # ===== Step 3 =====
    ModelFixture(
        name="step3",
        architecture="Step3ForCausalLM",
        chat_template=_TOOLS_AND_THINK,
        expected_tool="step3",
        expected_reasoning="step3",
    ),
    # ===== Edge: unknown architecture with ChatML markers, layer 2 fires =====
    # Layer 2 catches unknown architectures by chat-template content. ChatML
    # markers (`<|im_start|>`) plus a tools fragment plus `<think>` should
    # produce hermes (tool) + qwen3 (reasoning), exactly as if it were a
    # known Qwen3 fine-tune with a custom architecture string.
    ModelFixture(
        name="unknown-arch-chatml-with-tools-and-think",
        architecture="UnknownExoticForCausalLM",
        chat_template="<|im_start|>system\n" + _TOOLS_AND_THINK + "<|im_end|>",
        expected_tool="hermes",
        expected_reasoning="qwen3",
    ),
    # ===== Edge: unknown architecture, no template signals =====
    # Should produce nothing — both layer 1 and layer 2 miss.
    ModelFixture(
        name="unknown-model-plain",
        architecture="UnknownExoticForCausalLM",
        chat_template="plain chat with no markers",
        expected_tool="",
        expected_reasoning="",
    ),
    # ===== Edge: unknown arch, tools but no recognizable markers =====
    # Has a tools fragment but no ChatML/Llama/Mistral/etc. signature.
    # Tool detector should remain silent (no false-positive parser pick).
    # Reasoning detector still fires from the <think> marker.
    ModelFixture(
        name="unknown-arch-tools-no-marker",
        architecture="UnknownExoticForCausalLM",
        chat_template=_TOOLS_AND_THINK,
        expected_tool="",
        expected_reasoning="qwen3",
    ),
)


def _materialize(tmp_path: Path, fixture: ModelFixture) -> Path:
    """Write a minimal HF-style model directory and return the path."""
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": [fixture.architecture]})
    )
    (tmp_path / "chat_template.jinja").write_text(fixture.chat_template)
    # Some real templates only live inside tokenizer_config.json; mirror that.
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": fixture.chat_template})
    )
    return tmp_path


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f.name)
def test_tool_parser_detection(tmp_path, fixture):
    model = _materialize(tmp_path, fixture)
    assert dtp.detect_parser(str(model)) == fixture.expected_tool


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f.name)
def test_reasoning_parser_detection(tmp_path, fixture):
    model = _materialize(tmp_path, fixture)
    assert drp.detect_parser(str(model)) == fixture.expected_reasoning


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f.name)
def test_paired_detection_matches_expectation(tmp_path, fixture):
    """Both detectors run together (mirrors what `serve` does)."""
    model = _materialize(tmp_path, fixture)
    assert dtp.detect_parser(str(model)) == fixture.expected_tool
    assert drp.detect_parser(str(model)) == fixture.expected_reasoning


def test_no_false_positive_when_template_lacks_thinking_marker(tmp_path):
    """A Qwen3 fine-tune that disables CoT in its template should NOT get
    a reasoning parser injected. Architecture matches but template doesn't."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    (tmp_path / "chat_template.jinja").write_text(_TOOLS_FRAGMENT)  # no <think> marker
    assert drp.detect_parser(str(tmp_path)) == ""
    assert dtp.detect_parser(str(tmp_path)) == "hermes"  # tools still detected


def test_no_false_positive_when_template_lacks_tools_fragment(tmp_path):
    """A Qwen3 fine-tune for chat-only (no tool calls) should NOT get a
    tool parser injected even though its architecture supports them."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    (tmp_path / "chat_template.jinja").write_text(_THINK_FRAGMENT)  # no tools markers
    assert dtp.detect_parser(str(tmp_path)) == ""
    assert drp.detect_parser(str(tmp_path)) == "qwen3"


def test_hf_model_id_returns_empty():
    """HF id (no local dir) should return '' for both detectors so users
    can still pass the flags explicitly without surprise injection."""
    fake_id = "Qwen/Qwen3-8B"
    assert dtp.detect_parser(fake_id) == ""
    assert drp.detect_parser(fake_id) == ""
