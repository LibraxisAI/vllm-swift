# SPDX-License-Identifier: Apache-2.0
"""Tests pinning the hardening patches added after auditing llama.cpp's
known auto-detection bugs (ggml-org/llama.cpp issues #20809, #21616,
#20754, #22280, #20630, #19635, #22106 and adjacent).

Each test corresponds to a specific bug class llama.cpp has hit (or
would have hit) so we don't replicate the same blind spot.
"""
from __future__ import annotations

import json

import pytest

from vllm_swift import detect_reasoning_parser as drp
from vllm_swift import detect_tool_parser as dtp


# ---------------------------------------------------------------------------
# llama.cpp #20809 / #22684 — Qwen3-Instruct-2507 false thinking detection
# ---------------------------------------------------------------------------

def test_qwen3_instruct_2507_directory_suffix_suppresses_reasoning(tmp_path):
    """Qwen3-*-Instruct-2507 ships <think> markers but is NOT a thinking model.
    Our detector must skip reasoning injection on directory names that
    carry the -Instruct-2507 suffix even though architecture maps to qwen3."""
    model_dir = tmp_path / "Qwen3.5-7B-Instruct-2507"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    (model_dir / "chat_template.jinja").write_text("uses <think>...</think>")
    assert drp.detect_parser(str(model_dir)) == ""
    # Tool parser should still fire — the suffix only suppresses reasoning.
    assert dtp.detect_parser(str(model_dir)) == ""  # no tool fragment


def test_qwen3_instruct_2507_with_tool_fragment_keeps_tool_parser(tmp_path):
    model_dir = tmp_path / "Qwen3-30B-Instruct-2507"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    (model_dir / "chat_template.jinja").write_text(
        "{% if tools %}{{ tool.function }}{% endif %}<think>x</think>"
    )
    assert drp.detect_parser(str(model_dir)) == ""
    assert dtp.detect_parser(str(model_dir)) == "hermes"


# ---------------------------------------------------------------------------
# llama.cpp #21616 — Reka Edge marker-bearing template, no actual reasoning
# ---------------------------------------------------------------------------

def test_reka_edge_arch_suppresses_reasoning(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["RekaForCausalLM"]}))
    (tmp_path / "chat_template.jinja").write_text("template with <think>blocks</think>")
    assert drp.detect_parser(str(tmp_path)) == ""


# ---------------------------------------------------------------------------
# llama.cpp #22280 — Ring 2.0 (inclusionAI) custom <role>HUMAN</role> template
# ---------------------------------------------------------------------------

def test_ring_arch_suppresses_reasoning(tmp_path):
    """Ring 2.0 templates ship reasoning markers but the standard variants
    aren't thinking models. Suppress to avoid llama.cpp's #22280 false-positive."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["RingForCausalLM"]}))
    (tmp_path / "chat_template.jinja").write_text(
        "<role>HUMAN</role>...<role>ASSISTANT</role>...<think>cot</think>"
    )
    assert drp.detect_parser(str(tmp_path)) == ""


def test_ring_template_pattern_routes_to_hermes_tools(tmp_path):
    """Even with arch suppression, the template-pattern fallback in the
    tool detector should still pick a sensible parser for Ring's role-style
    template when tools are advertised."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["RingForCausalLM"]}))
    (tmp_path / "chat_template.jinja").write_text(
        "<role>HUMAN</role><role>ASSISTANT</role>{% if tools %}{{ tool.function }}{% endif %}"
    )
    assert dtp.detect_parser(str(tmp_path)) == "hermes"


# ---------------------------------------------------------------------------
# llama.cpp #20754 — Nemotron-Nano /no_think toggle hits FORCED_OPEN
# ---------------------------------------------------------------------------

def test_nemotron_nano_no_think_directory_suffix_suppresses(tmp_path):
    """Nemotron-Nano-*-NoThink variants should not get a reasoning parser
    even though NemotronH otherwise maps to qwen3."""
    model_dir = tmp_path / "Nemotron-Nano-9B-v2-NoThink"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["NemotronHForCausalLM"]})
    )
    (model_dir / "chat_template.jinja").write_text("/no_think marker + <think>x</think>")
    assert drp.detect_parser(str(model_dir)) == ""


# ---------------------------------------------------------------------------
# llama.cpp #20630 — Granite 4 <|start_of_role|> template marker
# ---------------------------------------------------------------------------

def test_granite4_template_marker_routes_via_layer_2(tmp_path):
    """Unknown architecture but Granite 4 role marker in template -> granite4."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["UnknownArch"]}))
    (tmp_path / "chat_template.jinja").write_text(
        "<|start_of_role|>{% if tools %}{{ tool.function }}{% endif %}"
    )
    assert dtp.detect_parser(str(tmp_path)) == "granite4"


# ---------------------------------------------------------------------------
# llama.cpp Llama-4 <|header_start|> marker fallback
# ---------------------------------------------------------------------------

def test_llama4_header_marker_routes_via_layer_2(tmp_path):
    """Custom Llama-4 fork without the standard arch name still routes to
    llama4_json via the <|header_start|> template marker."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["UnknownArch"]}))
    (tmp_path / "chat_template.jinja").write_text(
        "<|header_start|>...<|header_end|>{% if tools %}{{ tool.function }}{% endif %}"
    )
    assert dtp.detect_parser(str(tmp_path)) == "llama4_json"


# ---------------------------------------------------------------------------
# llama.cpp #22106 — MiniMax M2/M2.7 <minimax:tool_call> wrapper
# ---------------------------------------------------------------------------

def test_minimax_tool_call_marker_routes_via_layer_2(tmp_path):
    """A custom MiniMax fork with <minimax:tool_call> marker but no recognized
    arch should still route to minimax_m2 via template fallback."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["UnknownArch"]}))
    (tmp_path / "chat_template.jinja").write_text(
        "<minimax:tool_call>...{% if tools %}{{ tool.function }}{% endif %}"
    )
    assert dtp.detect_parser(str(tmp_path)) == "minimax_m2"


# ---------------------------------------------------------------------------
# llama.cpp #19635 — Step-3.5-Flash routing
# ---------------------------------------------------------------------------

def test_step35_arch_resolves_to_step3p5(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Step35ForCausalLM"]}))
    (tmp_path / "chat_template.jinja").write_text(
        "{% if tools %}{{ tool.function }}{% endif %}"
    )
    assert dtp.detect_parser(str(tmp_path)) == "step3p5"


# ---------------------------------------------------------------------------
# Cross-port fixes: parsers in reasoning table that were missing from tool
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch,expected_tool,expected_reasoning", [
    ("MagistralForCausalLM", "mistral", "mistral"),
    ("GptOssForCausalLM", "openai", "openai_gptoss"),
    ("Holo2ForCausalLM", "hermes", "holo2"),
    ("MiMoForCausalLM", "hermes", "mimo"),
    ("SeedOssForCausalLM", "seed_oss", "seed_oss"),
    ("KimiK2ThinkingForCausalLM", "kimi_k2", "kimi_k2"),
    ("KimiK25ForCausalLM", "kimi_k2", "kimi_k2"),
])
def test_cross_ported_arch_resolves_in_both_detectors(
    tmp_path, arch, expected_tool, expected_reasoning
):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": [arch]}))
    (tmp_path / "chat_template.jinja").write_text(
        "{% if tools %}{{ tool.function }}{% endif %}<think>x</think>"
    )
    assert dtp.detect_parser(str(tmp_path)) == expected_tool
    assert drp.detect_parser(str(tmp_path)) == expected_reasoning


# ---------------------------------------------------------------------------
# DeepSeek V4 falls into V3 family (until vLLM ships dedicated DSv4 parser)
# ---------------------------------------------------------------------------

def test_deepseek_v4_arch_falls_into_v3_family(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["DeepseekV4ForCausalLM"]})
    )
    (tmp_path / "chat_template.jinja").write_text(
        "{% if tools %}{{ tool.function }}{% endif %}<think>x</think>"
    )
    assert dtp.detect_parser(str(tmp_path)) == "deepseek_v3"
    assert drp.detect_parser(str(tmp_path)) == "deepseek_v3"


# ---------------------------------------------------------------------------
# Phi-4-Multimodal coverage
# ---------------------------------------------------------------------------

def test_phi4_multimodal_resolves_to_phi4(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Phi4MultimodalForCausalLM"]})
    )
    (tmp_path / "tokenizer_config.json").write_text(
        '{"chat_template": "<|im_sep|>{% if tools %}{{ tool.function }}{% endif %}"}'
    )
    assert dtp.detect_parser(str(tmp_path)) == "phi4_mini_json"


# ---------------------------------------------------------------------------
# qwen3_coder pattern fallback before generic tool_call
# ---------------------------------------------------------------------------

def test_function_xml_pattern_routes_to_qwen3_coder(tmp_path):
    """Qwen3-Coder forks without the standard arch name should still route
    to qwen3_coder via the <function= template marker (more specific than
    the generic <tool_call> -> hermes default)."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["UnknownArch"]}))
    (tmp_path / "chat_template.jinja").write_text(
        "<function=foo>{% if tools %}{{ tool.function }}{% endif %}<tool_call>"
    )
    assert dtp.detect_parser(str(tmp_path)) == "qwen3_coder"


# ---------------------------------------------------------------------------
# Gemma start_of_turn marker fallback
# ---------------------------------------------------------------------------

def test_gemma_start_of_turn_marker_routes_via_layer_2(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["UnknownArch"]}))
    (tmp_path / "chat_template.jinja").write_text(
        "<start_of_turn>user...<start_of_turn>model{% if tools %}{{ tool.function }}{% endif %}"
    )
    assert dtp.detect_parser(str(tmp_path)) == "gemma4"


# ---------------------------------------------------------------------------
# Gemma 4 dense + MoE both ship Gemma4ForConditionalGeneration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", [
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Gemma4MoeForCausalLM",
    "Gemma4MoEForCausalLM",
])
def test_gemma4_dense_and_moe_both_route(tmp_path, arch):
    """Google's new Gemma 4 family ships dense + MoE variants. Some are
    `ForCausalLM`, others `ForConditionalGeneration` (VLM flavor). All
    should hit the `Gemma4` prefix and route to the gemma4 parser."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": [arch]}))
    (tmp_path / "chat_template.jinja").write_text(
        "<start_of_turn>{% if tools %}{{ tool.function }}{% endif %}<think>x</think>"
    )
    assert dtp.detect_parser(str(tmp_path)) == "gemma4"
    assert drp.detect_parser(str(tmp_path)) == "gemma4"


# ---------------------------------------------------------------------------
# DeepSeek R1 ships V3 arch — discriminator must promote to deepseek_r1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dir_name,expected", [
    ("DeepSeek-R1-Distill-Llama-70B", "deepseek_r1"),
    ("DeepSeek-R1-Distill-Qwen-32B", "deepseek_r1"),
    ("DeepSeek-R1-0528", "deepseek_r1"),
    ("deepseek-r1", "deepseek_r1"),
    ("DeepSeek-V3-Base", "deepseek_v3"),  # not R1, must stay V3
])
def test_deepseek_r1_dirname_discriminator(tmp_path, dir_name, expected):
    """DeepSeek-R1 forks ship `DeepseekV3ForCausalLM` arch but should route
    to the dedicated deepseek_r1 reasoning parser."""
    model_dir = tmp_path / dir_name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["DeepseekV3ForCausalLM"]})
    )
    (model_dir / "chat_template.jinja").write_text("uses <think>...</think>")
    assert drp.detect_parser(str(model_dir)) == expected


# ---------------------------------------------------------------------------
# Kimi K2.x ships DeepseekV3 arch — discriminator must promote to kimi_k2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dir_name", [
    "Kimi-K2.5-Instruct",
    "Kimi-K2.6-Pro",
    "moonshotai-Kimi-K2",
])
def test_kimi_k2_dirname_discriminator(tmp_path, dir_name):
    """Kimi-K2.x ships `DeepseekV3ForCausalLM` arch but should route
    to the kimi_k2 parser via dirname signal."""
    model_dir = tmp_path / dir_name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["DeepseekV3ForCausalLM"]})
    )
    (model_dir / "chat_template.jinja").write_text(
        "{% if tools %}{{ tool.function }}{% endif %}<think>x</think>"
    )
    assert dtp.detect_parser(str(model_dir)) == "kimi_k2"
    assert drp.detect_parser(str(model_dir)) == "kimi_k2"


# ---------------------------------------------------------------------------
# GLM-5.1 (GlmMoeDsa arch)
# ---------------------------------------------------------------------------

def test_glm51_arch_resolves_to_glm45_parser(tmp_path):
    """GLM-5.1 ships `GlmMoeDsaForCausalLM` arch but uses the same
    glm4_moe_tool_parser as 4.5/4.7 per vLLM #39574."""
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["GlmMoeDsaForCausalLM"]})
    )
    (tmp_path / "chat_template.jinja").write_text(
        "{% if tools %}{{ tool.function }}{% endif %}<think>x</think>"
    )
    assert dtp.detect_parser(str(tmp_path)) == "glm45"
    assert drp.detect_parser(str(tmp_path)) == "glm45"


# ---------------------------------------------------------------------------
# Hunyuan Hy3 (HYV3 arch, vLLM PR #40681)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["HYV3ForCausalLM", "HYForCausalLM"])
def test_hunyuan_hy3_arch_resolves(tmp_path, arch):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": [arch]}))
    (tmp_path / "chat_template.jinja").write_text(
        "{% if tools %}{{ tool.function }}{% endif %}<think>x</think>"
    )
    assert dtp.detect_parser(str(tmp_path)) == "hunyuan_a13b"
    assert drp.detect_parser(str(tmp_path)) == "hunyuan_a13b"


# ---------------------------------------------------------------------------
# ERNIE 4.5 underscore variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", [
    "Ernie45ForCausalLM",
    "Ernie4_5ForCausalLM",
    "Ernie4_5_MoeForCausalLM",
    "Ernie4_5_VLMoeForConditionalGeneration",
])
def test_ernie45_underscore_variants_resolve(tmp_path, arch):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": [arch]}))
    (tmp_path / "chat_template.jinja").write_text(
        "{% if tools %}{{ tool.function }}{% endif %}<think>x</think>"
    )
    assert dtp.detect_parser(str(tmp_path)) == "ernie45"
    assert drp.detect_parser(str(tmp_path)) == "ernie45"


# ---------------------------------------------------------------------------
# Bailing/Ling/Ring (inclusionAI) — all ship BailingMoe* arch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["BailingMoeV2_5ForCausalLM", "BailingMoeForCausalLM"])
def test_bailing_arch_routes_to_hermes(tmp_path, arch):
    """inclusionAI Ling/Ring/Bailing all ship BailingMoe* arch and use
    ChatML-ish templates. hermes is the safe approximation until vLLM
    ships a dedicated parser."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": [arch]}))
    (tmp_path / "chat_template.jinja").write_text(
        "{% if tools %}{{ tool.function }}{% endif %}"
    )
    assert dtp.detect_parser(str(tmp_path)) == "hermes"
