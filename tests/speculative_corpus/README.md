# Speculative decode test corpus

Locked prompts for measuring N-gram speculative decode performance. Acceptance criteria gates (≥1.5x speedup on repetitive, ≤2% regression on non-repetitive) are only falsifiable against this fixed set.

## Structure

- `repetitive/` — workloads where prompt-lookup speculation should win:
  - `code_completion_shared_imports.txt` — Python with repeated module names + boilerplate
  - `chat_long_context_recall.txt` — Q&A that re-cites prior conversation
  - `tool_call_patterns.txt` — repeated JSON tool-call structure

- `nonrepetitive/` — workloads where speculation must NOT regress:
  - `creative_generation.txt` — open-ended creative writing
  - `summarization_novel.txt` — summarize new text (low n-gram match rate)
  - `translation_fresh.txt` — translate text the model hasn't seen

## Measurement protocol

1. **Decode 200 tokens** per prompt, greedy (`temperature=0`)
2. **3 runs per cell, median reported**
3. **Record:** `tok/s`, `ngramAcceptanceRate`, `ngramAcceptedCount`, `ngramProposedCount`
4. **2x2 matrix:** {plain KV, turbo4v2 KV} × {repetitive, non-repetitive}
5. **Coherence spot-check** at end of each run

## Acceptance gates

| Corpus | Gate | Floor |
|--------|------|-------|
| repetitive | ≥1.5x speedup vs non-speculative baseline | acceptance ≥40% |
| non-repetitive | ≤2% regression vs non-speculative baseline | acceptance ≥30% (else auto-disable) |

If acceptance falls below floor, speculative is net-negative (verify cost > accept cost) and the runtime should auto-disable for that prompt.

## TQ interaction

Speculative + TurboQuant requires KV cache trim on rejected drafts. If trim cost dominates, speculative wins on plain KV but regresses on TQ KV — the production path. **TQ cell is the gate, not the plain cell.**
