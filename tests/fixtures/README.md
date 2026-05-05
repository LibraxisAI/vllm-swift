# Test fixtures — captured agent traffic shapes

These JSON files are anonymized snapshots of the request/response bodies
that triggered the original bugs this PR fixes. They exist so the
test suite can replay the **exact shape** that broke things — not just
hand-crafted minimal inputs that pass a regex.

If a future change accidentally regresses the bump-rescue, recovery,
or usage-chunk-preservation logic, these replay tests will fail and
point at the specific traffic shape that breaks.

| File | Source | What it pins |
|---|---|---|
| `request_opencode_nemotron.json` | OpenCode (1.14.33, ai-sdk/bun) → vllm-swift, captured 2026-05-04 | `max_tokens=8192` + reasoning parser → bump must fire to 32768 |
| `request_hermes_uncapped.json` | Hermes against Qwen3.6, captured 2026-05-04 | `max_tokens=null` + reasoning parser → bump must NOT fire (already uncapped) |
| `response_phi4_pipe_leak.json` | Phi-4-mini-instruct, the failure shape Microsoft's own model card admits (vllm-project/vllm#14682) | `<\|tool_calls\|>[{...}]<\|/tool_calls\|>` in content → recovery must extract |
| `response_qwen3_coder_xml_leak.json` | Captured Qwen3.6-35B-A3B-4bit response when routed to wrong (hermes) parser before our fix | `<tool_call><function=name><parameter=...>` XML in content → recovery must extract |
| `response_streaming_with_usage.txt` | vLLM 0.19.1 streaming output ending with the `usage` chunk (separate `choices: []` chunk before `[DONE]`) | usage chunk must be preserved through the rewriter, not dropped |
| `response_hermes_json_leak.json` | Hermes JSON tool-call leak shape (`<tool_call>{json}</tool_call>` in content) | Defense-in-depth: future-model misroute leaks should still recover |
| `response_phi4_healthy_chat.json` | Real Phi-4-mini emission on M5 Max — markdown code block + natural-language explanation, no leak shape | False-positive guard: recovery MUST NOT fire on healthy chat traffic |
| `response_truncated_leak.json` | Partial leak shape where vLLM hit `max_tokens` mid-emission (no closing tag, finish_reason=length) | Graceful degradation: malformed/truncated leak must skip recovery cleanly |

System prompts and tool definitions in the request fixtures are
trimmed from the original 23K-char OpenCode prompt to representative
bits — the size matters for the bump-rescue trigger condition (low
`max_tokens` against reasoning parser), not the content.
