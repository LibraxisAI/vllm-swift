# SPDX-License-Identifier: Apache-2.0
"""Safety tests: every parser name our detector emits must exist in vLLM.

If our detector returns a parser name that isn't registered in vLLM's
ToolParserManager / ReasoningParserManager, the server fails to start.
Worse, the failure is at vllm-swift launch time, not at test time, so
without these guards a typo in the mapping table would only surface
when an unlucky user ran a Qwen3 / DeepSeek / etc model.

These tests pin our mapping output against a frozen copy of vLLM's
registries so any drift (renamed parser, removed parser, typo in our
table) trips immediately. The lists below are derived from:

  Tool parsers: `vllm serve --help` (subcommand `--tool-call-parser`)
  Reasoning parsers: `vllm/reasoning/__init__.py:_REASONING_PARSERS_TO_REGISTER`

Update these sets when vLLM adds / renames parsers. Don't update by
removing entries that fail — fix the detector mapping instead.
"""

from __future__ import annotations

from vllm_swift.detect_reasoning_parser import _ARCH_TO_REASONING_PARSER
from vllm_swift.detect_tool_parser import _arch_to_parser as _tool_arch_to_parser

# Source: `vllm.entrypoints.openai.api_server --help` --tool-call-parser choices.
# Pinned at vLLM 0.19.x. Update when bumping vLLM.
VLLM_TOOL_PARSERS: frozenset[str] = frozenset(
    {
        "deepseek_v3",
        "deepseek_v31",
        "deepseek_v32",
        "ernie45",
        "functiongemma",
        "gemma4",
        "gigachat3",
        "glm45",
        "glm47",
        "granite",
        "granite-20b-fc",
        "granite4",
        "hermes",
        "hunyuan_a13b",
        "internlm",
        "jamba",
        "kimi_k2",
        "llama3_json",
        "llama4_json",
        "llama4_pythonic",
        "longcat",
        "minimax",
        "minimax_m2",
        "mistral",
        "olmo3",
        "openai",
        "phi4_mini_json",
        "pythonic",
        "qwen3_coder",
        "qwen3_xml",
        "seed_oss",
        "step3",
        "step3p5",
        "xlam",
    }
)


# Source: `vllm.reasoning.__init__._REASONING_PARSERS_TO_REGISTER`.
# Pinned at vLLM 0.19.x. Update when bumping vLLM.
VLLM_REASONING_PARSERS: frozenset[str] = frozenset(
    {
        "deepseek_r1",
        "deepseek_v3",
        "ernie45",
        "gemma4",
        "glm45",
        "openai_gptoss",
        "granite",
        "holo2",
        "hunyuan_a13b",
        "kimi_k2",
        "mimo",
        "minimax_m2",
        "minimax_m2_append_think",
        "mistral",
        "nemotron_v3",
        "olmo3",
        "qwen3",
        "seed_oss",
        "step3",
        "step3p5",
    }
)


# Tool-parser arch table is a list of (prefix, parser) tuples in order; pull
# the names from a small set of representative architectures. We also walk
# the underlying `pairs` list inside `_arch_to_parser` by enumerating the
# architecture strings used by our test fixtures.
_TOOL_ARCH_PROBES: tuple[str, ...] = (
    "Qwen3CoderForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen2_5ForCausalLM",
    "Qwen2ForCausalLM",
    "NemotronHForCausalLM",
    "NemotronForCausalLM",
    "HermesForCausalLM",
    "Llama4ForCausalLM",
    "LlamaForCausalLM",
    "MagistralForCausalLM",
    "MistralForCausalLM",
    "Gemma4ForCausalLM",
    "GemmaForCausalLM",
    "Phi4MiniJsonForCausalLM",
    "Phi4MultimodalForCausalLM",
    "Phi4MMForCausalLM",
    "Phi4ForCausalLM",
    "Phi3ForCausalLM",
    "Granite4ForCausalLM",
    "GraniteForCausalLM",
    "DeepseekV4ForCausalLM",
    "DeepSeekV4ForCausalLM",
    "DeepseekV32ForCausalLM",
    "DeepseekV31ForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekV2ForCausalLM",
    "Glm45ForCausalLM",
    "Glm47ForCausalLM",
    "Glm4ForCausalLM",
    "MinimaxM2ForCausalLM",
    "MiniMaxM2ForCausalLM",
    "MiniMaxText",
    "MinimaxText",
    "KimiK2ForCausalLM",
    "KimiK2ThinkingForCausalLM",
    "KimiK25ForCausalLM",
    "KimiForCausalLM",
    "HunyuanA13BForCausalLM",
    "Step35ForCausalLM",
    "Step3p5ForCausalLM",
    "Step3ForCausalLM",
    "Olmo3ForCausalLM",
    "GptOssForCausalLM",
    "OpenaiMoeForCausalLM",
    "Holo2ForCausalLM",
    "MiMoForCausalLM",
    "SeedOssForCausalLM",
    "SeedOSSForCausalLM",
    "InternLMForCausalLM",
    "JambaForCausalLM",
    "Ernie45ForCausalLM",
)


def test_tool_parser_emits_only_registered_names():
    """Every name our tool detector can emit must be in vLLM's registry."""
    bad = []
    for arch in _TOOL_ARCH_PROBES:
        parser = _tool_arch_to_parser(arch)
        if parser and parser not in VLLM_TOOL_PARSERS:
            bad.append((arch, parser))
    assert not bad, (
        f"Tool detector returns names vLLM does not recognize: {bad}. "
        f"Either rename in detect_tool_parser._arch_to_parser or update "
        f"VLLM_TOOL_PARSERS to match the current vLLM version."
    )


def test_reasoning_parser_emits_only_registered_names():
    """Every name our reasoning detector can emit must be in vLLM's registry."""
    emitted = {parser for _prefix, parser in _ARCH_TO_REASONING_PARSER}
    unknown = emitted - VLLM_REASONING_PARSERS
    assert not unknown, (
        f"Reasoning detector returns names vLLM does not recognize: {sorted(unknown)}. "
        f"Either rename in detect_reasoning_parser._ARCH_TO_REASONING_PARSER or update "
        f"VLLM_REASONING_PARSERS to match the current vLLM version."
    )


def test_no_duplicate_prefixes_in_reasoning_table():
    """Order matters in the reasoning prefix table; duplicate prefixes
    would silently shadow later entries. Catch shadowing early."""
    prefixes = [p for p, _ in _ARCH_TO_REASONING_PARSER]
    dupes = [p for p in set(prefixes) if prefixes.count(p) > 1]
    assert not dupes, f"duplicate prefixes in reasoning table: {dupes}"


def test_no_empty_strings_in_reasoning_table():
    """An empty prefix would match every architecture and force a parser
    onto every model. Catch accidental empty entries early."""
    empty = [(p, parser) for p, parser in _ARCH_TO_REASONING_PARSER if not p or not parser]
    assert not empty, f"empty entries in reasoning table: {empty}"
