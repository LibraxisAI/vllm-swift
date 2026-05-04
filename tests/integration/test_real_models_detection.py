# SPDX-License-Identifier: Apache-2.0
"""Integration: run the detectors against REAL local model directories.

This is the static layer of integration testing — no server boot, just
"point the detector at a real HuggingFace model dir and assert it picks
the right parser pair." Catches drift between our synthetic test fixtures
and what model authors actually ship in the wild.

Skips per-model when the model isn't on disk; opt in to the whole suite
with `pytest -m integration`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vllm_swift import detect_reasoning_parser as drp
from vllm_swift import detect_tool_parser as dtp

from tests.integration.conftest import _has_local_model, _model_path


pytestmark = pytest.mark.integration


# (dir_name, expected_tool_parser, expected_reasoning_parser, comment)
# Curated based on what each model actually ships and what vLLM expects.
# Update when adding new local models or when expectations change.
EXPECTED: tuple[tuple[str, str, str, str], ...] = (
    # ===== Qwen family =====
    ("Qwen3-0.6B-4bit", "hermes", "qwen3", "vanilla Qwen3"),
    ("Qwen3-0.6B-hf", "hermes", "qwen3", "HF original"),
    ("Qwen3-4B-4bit", "hermes", "qwen3", "Qwen3-4B"),
    ("Qwen3.5-2B-4bit", "hermes", "qwen3", "Qwen3.5-2B (arch: Qwen3_5ForConditionalGeneration)"),
    ("Qwen3.5-9B-4bit", "hermes", "qwen3", "Qwen3.5-9B"),
    ("Qwen3.5-35B-A3B-4bit", "hermes", "qwen3", "Qwen3.5 MoE"),
    ("Qwen3.6-35B-A3B-4bit", "hermes", "qwen3", "Qwen3.6 MoE"),
    # MLX build of Qwen3-Coder ships `Qwen3MoeForCausalLM` (not Qwen3Coder*)
    # so detector relies on the directory-name discriminator to bump
    # hermes -> qwen3_coder. tokenizer_config.json has <think>/</think> as
    # special tokens, so reasoning detector correctly fires too.
    ("Qwen3-Coder-30B-A3B-Instruct-MLX-6bit", "qwen3_coder", "qwen3", "Qwen3-Coder MLX (arch: Qwen3MoeForCausalLM)"),
    ("Qwen2.5-3B-Instruct-4bit", "hermes", "", "Qwen2.5 (no native reasoning)"),
    # ===== Nemotron =====
    ("Nemotron-Cascade-2-30B-A3B-4bit", "hermes", "qwen3", "Nemotron-Cascade Qwen3.6 derivative"),
    # ===== Llama =====
    ("Llama-3.2-1B-Instruct-hf", "llama3_json", "", "Llama 3.2 1B"),
    ("Llama-3.2-3B-Instruct-4bit", "llama3_json", "", "Llama 3.2 3B"),
    # ===== Mistral =====
    ("Mistral-7B-Instruct-v0.3-4bit", "mistral", "", "Mistral 7B v0.3"),
    # ===== Phi =====
    ("Phi-4-mini-instruct-4bit", "phi4_mini_json", "", "Phi-4-mini (arch: Phi3ForCausalLM)"),
    # Phi-3-mini ships a chat template without tool fragments, so the gate
    # correctly suppresses the parser injection (model is not tool-capable).
    ("Phi-3-mini-4k-hf", "", "", "Phi-3-mini (no tools in template)"),
    # ===== Gemma =====
    ("gemma-4-e2b-it-4bit", "gemma4", "gemma4", "Gemma 4 E2B"),
    ("Gemma-2-2b-it-hf", "", "", "Gemma 2 — should NOT route to gemma4 parser"),
    # ===== GPT-OSS =====
    ("gpt-oss-20b-MXFP4-Q8", "openai", "openai_gptoss", "GPT-OSS 20B"),
    # ===== DeepSeek =====
    # This 2bit-DQ build's chat template uses `<｜tool▁calls▁begin｜>` (unicode
    # bar markers) without the literal "tools"/"tool_call" fragments our gate
    # checks for, so the tool injection is correctly suppressed. The template
    # DOES include <think>/</think> markers, so the reasoning detector fires.
    # TODO: extend `_has_tool_template` to recognize the unicode `<｜tool` variant.
    ("DeepSeek-V4-Flash-2bit-DQ", "", "deepseek_v3", "DeepSeek V4 2bit-DQ — unicode tool markers, standard <think>"),
)


@pytest.mark.parametrize("dir_name,expected_tool,expected_reasoning,comment", EXPECTED)
def test_real_model_detection(dir_name, expected_tool, expected_reasoning, comment):
    if not _has_local_model(dir_name):
        pytest.skip(f"model not present: {dir_name}")
    model = str(_model_path(dir_name))
    actual_tool = dtp.detect_parser(model)
    actual_reasoning = drp.detect_parser(model)
    assert actual_tool == expected_tool, (
        f"[{dir_name}] tool parser mismatch ({comment})\n"
        f"  expected: {expected_tool!r}\n"
        f"  actual:   {actual_tool!r}"
    )
    assert actual_reasoning == expected_reasoning, (
        f"[{dir_name}] reasoning parser mismatch ({comment})\n"
        f"  expected: {expected_reasoning!r}\n"
        f"  actual:   {actual_reasoning!r}"
    )


def test_no_unexpected_models_in_inventory(local_models_inventory: dict[str, str]):
    """If new models appear locally that aren't in EXPECTED, surface them so
    we can decide whether to add expectations. This is informational — does
    not fail, just prints to stdout when run with `-s`."""
    expected_names = {dir_name for dir_name, *_ in EXPECTED}
    extra = sorted(set(local_models_inventory) - expected_names)
    if extra:
        print(f"\nLocal models without integration-test expectations ({len(extra)}):")
        for n in extra:
            print(f"  {n} (arch: {local_models_inventory[n]})")
