"""Known upstream bugs in vLLM tool-call / reasoning parsers we auto-inject.

Curated from issue tracker scans. When the wrapper auto-injects a parser
with a tracked bug, we print a one-line caveat with the link so users hit
the issue with context, not blind. Removing an entry here means we believe
the bug is fixed in current vLLM main; the registry-validation tests will
catch regressions on the parser name itself.

Format per entry: (parser_name, summary, issue_url, optional_mitigation).
"""
from __future__ import annotations

KnownIssue = tuple[str, str, str, str]


_TOOL_PARSER_ISSUES: tuple[KnownIssue, ...] = (
    (
        "hermes",
        "streaming mode returns raw text instead of parsed tool_calls",
        "https://github.com/vllm-project/vllm/issues/31871",
        "if streaming + tool_calls are required, prefer non-streaming until fixed",
    ),
    (
        "hermes",
        "Hermes2ProToolParser can panic under concurrent load (Already borrowed)",
        "https://github.com/vllm-project/vllm/issues/34932",
        "lower --max-num-seqs if you see RuntimeError under load",
    ),
    (
        "gemma4",
        "concurrent requests can produce all-<pad> outputs",
        "https://github.com/vllm-project/vllm/issues/39392",
        "set --max-num-seqs 1 until upstream fix lands",
    ),
    (
        "gemma4",
        "streaming can corrupt text after tool calls",
        "https://github.com/vllm-project/vllm/issues/38910",
        "non-streaming responses are unaffected",
    ),
    (
        "qwen3_coder",
        "ValueError when parsing certain XML function calls",
        "https://github.com/vllm-project/vllm/issues/36769",
        "fall back to --tool-call-parser hermes if errors are frequent",
    ),
)


_REASONING_TOOL_COMBO_ISSUES: tuple[tuple[str, str, str, str, str], ...] = (
    # (reasoning_parser, tool_parser, summary, issue_url, mitigation)
    (
        "qwen3",
        "hermes",
        "tool_call XML emitted inside a <think> block can be stripped before "
        "the tool parser sees it (Qwen3.5* models in particular)",
        "https://github.com/vllm-project/vllm/issues/39056",
        "if tool calls go missing on a thinking model, drop --reasoning-parser",
    ),
    (
        "qwen3",
        "qwen3_coder",
        "qwen3_coder + qwen3 reasoning has the same XML-inside-<think> race",
        "https://github.com/vllm-project/vllm/issues/39056",
        "if tool calls go missing on a thinking model, drop --reasoning-parser",
    ),
)


_REASONING_PARSER_ISSUES: tuple[KnownIssue, ...] = (
    (
        "minimax_m2",
        "extract_reasoning_streaming assumes no <think> start tag (broken on M2.5)",
        "https://github.com/vllm-project/vllm/issues/38212",
        "for MiniMax M2.5 specifically, prefer non-streaming or pin to vLLM with the fix",
    ),
)


def tool_parser_caveats(parser: str) -> list[tuple[str, str, str]]:
    """Return (summary, url, mitigation) for each known tool-parser issue."""
    return [
        (summary, url, mit)
        for name, summary, url, mit in _TOOL_PARSER_ISSUES
        if name == parser
    ]


def reasoning_parser_caveats(parser: str) -> list[tuple[str, str, str]]:
    """Return (summary, url, mitigation) for each known reasoning-parser issue."""
    return [
        (summary, url, mit)
        for name, summary, url, mit in _REASONING_PARSER_ISSUES
        if name == parser
    ]


def combo_caveats(reasoning_parser: str, tool_parser: str) -> list[tuple[str, str, str]]:
    """Return caveats specific to a reasoning+tool parser combination."""
    return [
        (summary, url, mit)
        for r, t, summary, url, mit in _REASONING_TOOL_COMBO_ISSUES
        if r == reasoning_parser and t == tool_parser
    ]
