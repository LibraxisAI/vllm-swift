# SPDX-License-Identifier: Apache-2.0
"""Tests for the known-upstream-bug annotation table.

When a vLLM parser bug listed in `known_parser_issues` is fixed upstream,
remove the corresponding entry from the table. This test file pins the
current set so removal is intentional and visible in code review.
"""
from __future__ import annotations

import pytest

from vllm_swift import known_parser_issues as kpi


def test_tool_parser_caveats_returns_empty_for_unknown():
    assert kpi.tool_parser_caveats("unknown_made_up_parser") == []


def test_reasoning_parser_caveats_returns_empty_for_unknown():
    assert kpi.reasoning_parser_caveats("unknown_made_up_parser") == []


def test_combo_caveats_returns_empty_for_safe_combo():
    assert kpi.combo_caveats("identity", "openai") == []


def test_hermes_has_known_streaming_issue():
    """vllm/issues/31871 — hermes streaming returns raw text."""
    caveats = kpi.tool_parser_caveats("hermes")
    assert any("31871" in url for _summary, url, _mit in caveats)


def test_hermes_has_known_concurrent_load_issue():
    """vllm/issues/34932 — Hermes2ProToolParser borrow panic under load."""
    caveats = kpi.tool_parser_caveats("hermes")
    assert any("34932" in url for _summary, url, _mit in caveats)


def test_gemma4_has_known_concurrent_pad_issue():
    """vllm/issues/39392 — gemma4 emits all-<pad> under concurrent requests."""
    caveats = kpi.tool_parser_caveats("gemma4")
    assert any("39392" in url for _summary, url, _mit in caveats)


def test_gemma4_has_known_streaming_corruption_issue():
    """vllm/issues/38910 — gemma4 streaming corrupts text."""
    caveats = kpi.tool_parser_caveats("gemma4")
    assert any("38910" in url for _summary, url, _mit in caveats)


def test_qwen3_coder_has_known_xml_parse_issue():
    """vllm/issues/36769 — qwen3_coder ValueError on certain XML."""
    caveats = kpi.tool_parser_caveats("qwen3_coder")
    assert any("36769" in url for _summary, url, _mit in caveats)


def test_minimax_m2_reasoning_has_known_issue():
    """vllm/issues/38212 — MiniMax M2.5 reasoning streaming broken."""
    caveats = kpi.reasoning_parser_caveats("minimax_m2")
    assert any("38212" in url for _summary, url, _mit in caveats)


@pytest.mark.parametrize("tool_parser", ["hermes", "qwen3_coder"])
def test_qwen3_reasoning_with_tool_parser_has_combo_warning(tool_parser):
    """vllm/issues/39056 — XML tool_call inside <think> block lost."""
    caveats = kpi.combo_caveats("qwen3", tool_parser)
    assert any("39056" in url for _summary, url, _mit in caveats)


def test_combo_caveats_only_returns_for_paired_combo():
    """Reasoning+tool combo warning should only fire for the specific pair,
    not for either parser individually with an unrelated counterpart."""
    assert kpi.combo_caveats("qwen3", "mistral") == []
    assert kpi.combo_caveats("deepseek_r1", "hermes") == []


def test_all_caveat_entries_have_required_fields():
    """Every entry must have non-empty parser name, summary, and url."""
    for parser, summary, url, _mit in kpi._TOOL_PARSER_ISSUES:
        assert parser, "tool parser name empty"
        assert summary, f"empty summary for {parser}"
        assert url.startswith("https://"), f"bad url for {parser}: {url}"
    for parser, summary, url, _mit in kpi._REASONING_PARSER_ISSUES:
        assert parser, "reasoning parser name empty"
        assert summary, f"empty summary for {parser}"
        assert url.startswith("https://"), f"bad url for {parser}: {url}"
    for r, t, summary, url, _mit in kpi._REASONING_TOOL_COMBO_ISSUES:
        assert r and t, "combo entry missing parser name"
        assert summary, f"empty summary for {r}+{t}"
        assert url.startswith("https://"), f"bad url for {r}+{t}: {url}"
