"""Detect appropriate vLLM tool-call parser from a model directory or HF id.

Three-layer defense in depth (better than vLLM's planned RFC #32713 and at
parity with llama.cpp's `llm_chat_detect_template` for tool calling):

  Layer 1: architecture-prefix mapping (config.json -> parser).
           Most precise. Catches mainstream and named fine-tunes.
  Layer 2: template pattern fallback (chat template content -> parser
           family). Catches unknown architectures that still ship a
           recognizable template (e.g. random Qwen3 fine-tune with a
           custom architecture name still using ChatML markers).
  Layer 3: tool-template signal gate. Suppresses any injection on models
           whose chat template has no tool-calling fragment, so a
           non-tool-capable model never gets a parser forced onto it.

Prints the parser name to stdout, or empty if no signal. Exits 0 always
(empty stdout = no auto-injection). Used by the vllm-swift wrapper to
provide smart defaults when the user runs `serve` without explicit
--tool-call-parser / --enable-auto-tool-choice flags.
"""

import json
import os
import sys


def _arch_to_parser(arch: str) -> str:
    if not arch:
        return ""
    a = arch
    # Order matters: more specific prefixes first.
    pairs = [
        # Qwen family
        ("Qwen3CoderForCausalLM", "qwen3_coder"),
        ("Qwen3Coder", "qwen3_coder"),
        ("Qwen3", "hermes"),
        ("Qwen2_5", "hermes"),
        ("Qwen2", "hermes"),
        # Nemotron family — Cascade-2 / H variants emit qwen3_coder XML shape:
        # <tool_call><function=name><parameter=k>v</parameter></function></tool_call>
        # Confirmed by chat_template.jinja line 93 in Nemotron-Cascade-2-30B-A3B
        # and HF discussion #7 (NVIDIA DongfuJiang). The hermes parser expects
        # JSON inside <tool_call> and silently fails to extract this XML shape,
        # leaking the whole block as plaintext into message.content.
        ("NemotronH", "qwen3_coder"),
        ("Nemotron", "qwen3_coder"),
        # Hermes itself
        ("HermesForCausalLM", "hermes"),
        # Llama
        ("Llama4", "llama4_json"),
        ("Llama", "llama3_json"),
        # Mistral / Magistral
        ("Magistral", "mistral"),
        ("Mistral", "mistral"),
        # Gemma
        ("Gemma4", "gemma4"),
        ("Gemma", "gemma4"),
        # Phi
        ("Phi4MiniJson", "phi4_mini_json"),
        ("Phi4Multimodal", "phi4_mini_json"),
        ("Phi4MM", "phi4_mini_json"),
        ("Phi4", "phi4_mini_json"),
        ("Phi3", "phi4_mini_json"),
        # Granite
        ("Granite4", "granite4"),
        ("Granite", "granite"),
        # DeepSeek (V4 falls through to V3 family for now until vLLM ships
        # a dedicated DSv4 parser — most templates are still V3-compatible)
        ("DeepseekV4", "deepseek_v3"),
        ("DeepSeekV4", "deepseek_v3"),
        ("DeepseekV32", "deepseek_v32"),
        ("DeepseekV31", "deepseek_v31"),
        ("DeepseekV3", "deepseek_v3"),
        ("DeepseekV2", "deepseek_v3"),
        # GLM
        ("GlmMoeDsa", "glm45"),  # GLM-5.1 — same parser per vLLM #39574
        ("Glm45", "glm45"),
        ("Glm47", "glm47"),
        ("Glm4", "glm45"),
        # MiniMax
        ("MinimaxM2", "minimax_m2"),
        ("MiniMaxM2", "minimax_m2"),
        ("MiniMax", "minimax"),
        ("Minimax", "minimax"),
        # Kimi K2 / K2.5 / K2 Thinking variants
        ("KimiK2Thinking", "kimi_k2"),
        ("KimiK25", "kimi_k2"),
        ("KimiK2", "kimi_k2"),
        ("Kimi", "kimi_k2"),
        # Hunyuan variants — Hy3 family ships HYV3* arch (vLLM PR #40681)
        ("HunyuanA13B", "hunyuan_a13b"),
        ("HYV3", "hunyuan_a13b"),
        ("HY", "hunyuan_a13b"),
        # Step
        ("Step35", "step3p5"),
        ("Step3p5", "step3p5"),
        ("Step3", "step3"),
        # Olmo
        ("Olmo3", "olmo3"),
        # GPT-OSS — vLLM uses this name for both tool and reasoning paths
        ("GptOss", "openai"),
        ("OpenaiMoe", "openai"),
        # Holo / MiMo / SeedOSS / Mimo (reasoning-side parsers exist; tool-
        # side falls through to closest format approximation)
        ("Holo2", "hermes"),
        ("MiMo", "hermes"),
        ("SeedOss", "seed_oss"),
        ("SeedOSS", "seed_oss"),
        # Misc
        ("InternLM", "internlm"),
        ("Jamba", "jamba"),
        # ERNIE 4.5 — HF ships underscore variants (Ernie4_5_VLMoe etc).
        # Keep Ernie45 first for forward-compat with future variants.
        ("Ernie45", "ernie45"),
        ("Ernie4_5_VLMoe", "ernie45"),
        ("Ernie4_5_Moe", "ernie45"),
        ("Ernie4_5", "ernie45"),
        # Bailing/Ling/Ring (inclusionAI) — all ship `BailingMoe*` arch.
        # No dedicated vLLM tool parser yet; templates use ChatML-ish
        # `<tool_call>` so hermes is the safe approximation.
        ("BailingMoeV2", "hermes"),
        ("BailingMoe", "hermes"),
    ]
    for prefix, parser in pairs:
        if a.startswith(prefix):
            return parser
    return ""


def _load_arch(model_path: str) -> str:
    cfg = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg):
        return ""
    try:
        with open(cfg) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    archs = data.get("architectures") or []
    if not archs:
        return ""
    return archs[0]


def _read_template_blob(model_path: str) -> str:
    """Concatenate all chat-template-bearing files into a single string for
    pattern scanning. Reading once and scanning many is faster than reopening
    per pattern, and matches how llama.cpp's `llm_chat_detect_template` works.
    """
    blob_parts: list[str] = []
    for fname in ("tokenizer_config.json", "chat_template.json", "chat_template.jinja"):
        path = os.path.join(model_path, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                blob_parts.append(f.read())
        except OSError:
            continue
    return "\n".join(blob_parts)


def _has_tool_template(model_path: str) -> bool:
    """Heuristic: a tool-capable model usually exposes 'tools' in chat_template."""
    blob = _read_template_blob(model_path)
    return "tools" in blob and ("tool_call" in blob or "function" in blob)


# Layer 2 fallback: template content patterns -> parser. Mirrors the approach
# llama.cpp uses in `llm_chat_detect_template` for unknown architectures.
# Order matters: more specific markers come first so e.g. `<|im_sep|>` (Phi-4)
# is checked before `<|im_start|>` (generic ChatML / Hermes).
_TEMPLATE_PATTERN_TO_PARSER: tuple[tuple[str, str], ...] = (
    # Most-specific first
    ("<|im_sep|>", "phi4_mini_json"),  # Phi-4 family
    ("[AVAILABLE_TOOLS]", "mistral"),  # Mistral V3 instruct
    ("<|tool▁calls▁begin|>", "deepseek_v3"),  # DeepSeek V3 / V3.1 / V3.2
    ("<minimax:tool_call>", "minimax_m2"),  # MiniMax M2 / M2.7 wrapper
    ("<|start_of_role|>", "granite4"),  # Granite 4 role marker
    ("<|header_start|>", "llama4_json"),  # Llama 4 fallback for forks
    ("<|header_end|>", "llama4_json"),
    ("<seed:bos>", "seed_oss"),  # Seed-OSS BOS marker
    ("<role>ASSISTANT</role>", "hermes"),  # Ring 2.0 (inclusionAI) — best-effort
    ("<role>HUMAN</role>", "hermes"),
    # function-call XML before generic tool_call so qwen3_coder forks route
    # to the coder-specific parser instead of generic hermes
    ("<function=", "qwen3_coder"),
    ("<tool_call>", "hermes"),  # Hermes / Qwen3 / Nemotron / many ChatML-tools
    ("<function_call>", "hermes"),
    ("<|begin_of_text|>", "llama3_json"),  # Llama 3.x
    ("<start_of_turn>", "gemma4"),  # Gemma uniquely uses start_of_turn
    ("<|im_start|>", "hermes"),  # Generic ChatML, default to hermes parser
)


def _pattern_fallback(model_path: str) -> str:
    """Layer 2 fallback: pick a parser by template content when architecture
    mapping returned empty. Conservative: only matches when the same template
    also contains tool-capability markers (caller already checks this)."""
    blob = _read_template_blob(model_path)
    for pattern, parser in _TEMPLATE_PATTERN_TO_PARSER:
        if pattern in blob:
            return parser
    return ""


def _name_discriminator(parser: str, model_dir_name: str) -> str:
    """Refine an architecture-derived parser using model directory name signals.

    Catches real-world cases where MLX / unsloth / GGUF converters strip the
    specialized arch suffix, leaving the bare family arch in `config.json`:

      - Qwen3-Coder-*-MLX-* ships `Qwen3MoeForCausalLM` instead of
        `Qwen3CoderForCausalLM` -> bump hermes to qwen3_coder.
      - Kimi-K2.5/K2.6 ship `DeepseekV3ForCausalLM` arch (Kimi reused
        DeepSeek's V3 model_type with their own training) -> bump
        deepseek_v3 to kimi_k2 when "kimi" or "k2" appears in dir name.
    """
    name_lower = (model_dir_name or "").lower()
    if parser == "hermes" and "coder" in name_lower and "qwen3" in name_lower:
        return "qwen3_coder"
    if parser == "deepseek_v3" and (
        "kimi" in name_lower
        or "/k2" in name_lower
        or name_lower.startswith("k2")
        or "-k2" in name_lower
    ):
        return "kimi_k2"
    return parser


def detect_parser(model: str) -> str:
    """Return parser name for the given model path/id, or '' if unmappable.

    Three-layer detection:
      1. Architecture-prefix mapping (precise, mainstream models).
      2. Template-pattern fallback (covers unknown architectures with
         recognizable markers, mirrors llama.cpp's `llm_chat_detect_template`).
      3. Tool-template gate (suppress injection when the model does not ship
         a tool-calling template fragment, regardless of arch/pattern hits).

    A name-discriminator pass refines arch-derived parsers using the model
    directory name (handles converted MLX/GGUF builds that drop the
    specialized arch suffix — Qwen3-Coder-MLX ships Qwen3MoeForCausalLM).

    HF model ids without a local directory return '' (auto-injection skipped;
    user can pass --tool-call-parser).
    """
    if not model or not os.path.isdir(model):
        return ""
    if not _has_tool_template(model):
        return ""
    arch = _load_arch(model)
    parser = _arch_to_parser(arch)
    if parser:
        return _name_discriminator(parser, os.path.basename(model.rstrip("/")))
    # Layer 2: architecture didn't map; try template-pattern fallback
    return _pattern_fallback(model)


def main() -> int:
    if len(sys.argv) < 2:
        return 0
    parser = detect_parser(sys.argv[1])
    if parser:
        sys.stdout.write(parser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
