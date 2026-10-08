"""The 14 models and their names.

    id       canonical name used in results CSVs (the hub name for open models)
    display  name in generated tables
    latex    name in the paper's LaTeX tables
    token    suffix of the judged concept-coverage file, chess_explanation_recall_<token>.csv
    closed   True for API models
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Model:
    id: str
    display: str
    latex: str
    token: str
    closed: bool


# Table order: closed models first.
MODELS = [
    Model("gpt-5.4", "GPT-5.4", "GPT-5.4", "gpt_5_4", True),
    Model("gpt-5-mini", "GPT-5-mini", "GPT-5-mini", "gpt_5_mini", True),
    Model("claude-opus-4-6", "Claude-Opus-4.6", "Claude-Opus-4.6", "opus", True),
    Model("claude-sonnet-4-6", "Claude-Sonnet-4.6", "Claude-Sonnet-4.6", "sonnet", True),
    Model("gemini-2.5-pro", "Gemini-2.5-Pro", "Gemini-2.5-Pro", "gemini_2_5_pro", True),
    Model("gemini-2.5-flash", "Gemini-2.5-Flash", "Gemini-2.5-Flash", "gemini_2_5_flash", True),
    Model("Qwen3.5-9B", "Qwen3.5-9B", "Qwen-3.5-9b", "qwen3_5_9b", False),
    Model("Qwen3-8B", "Qwen3-8B", "Qwen-3-8b", "qwen3_8b", False),
    Model("Qwen2.5-14B-Instruct", "Qwen2.5-14B", "Qwen-2.5-14b", "qwen2_5_14b_instruct", False),
    Model("Qwen2.5-7B-Instruct", "Qwen2.5-7B", "Qwen-2.5-7b", "qwen2_5_7b_instruct", False),
    Model("Ministral-3-14B-Reasoning-2512", "Ministral-3-14B", "Ministral-3-14b",
          "ministral_3_14b_reasoning_2512", False),
    Model("Mistral-7B-Instruct-v0.3", "Mistral-7B", "Mistral-7b", "mistral_7b_instruct_v0_3", False),
    Model("Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B", "Llama-3.1-8b", "meta_llama_3_1_8b", False),
    Model("Meta-Llama-3-8B-Instruct", "Llama-3-8B", "Llama-3-8b", "meta_llama_3_8b", False),
]
BY_ID = {m.id: m for m in MODELS}
BY_TOKEN = {m.token: m for m in MODELS}
MODEL_ORDER = [m.id for m in MODELS]

# Row order of the conditioned LaTeX tables.
LATEX_ORDER = [
    "Meta-Llama-3-8B-Instruct", "Meta-Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3",
    "Ministral-3-14B-Reasoning-2512", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct",
    "Qwen3-8B", "Qwen3.5-9B", "claude-sonnet-4-6", "claude-opus-4-6",
    "gemini-2.5-flash", "gemini-2.5-pro", "gpt-5.4", "gpt-5-mini",
]

# Open models whose explanations are decomposed and judged through the API;
# the other eight were judged by hand.
RERUN_MODELS = [
    "Meta-Llama-3-8B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Ministral-3-14B-Reasoning-2512",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct",
]

# Filename tokens of runs with other prompts; excluded from the tables.
NON_BASELINE_TOKENS = ("paraphrase", "enumeration_prompt")

# Longer names before names they contain.
_SUBSTRINGS = [
    ("claude-opus", "claude-opus-4-6"),
    ("claude-sonnet", "claude-sonnet-4-6"),
    ("gemini-2.5-flash", "gemini-2.5-flash"),
    ("gemini-2.5-pro", "gemini-2.5-pro"),
    ("gpt-5-mini", "gpt-5-mini"),
    ("gpt-5.4", "gpt-5.4"),
    ("Meta-Llama-3.1-8B-Instruct", "Meta-Llama-3.1-8B-Instruct"),
    ("Meta-Llama-3-8B-Instruct", "Meta-Llama-3-8B-Instruct"),
    ("Ministral-3-14B-Reasoning-2512", "Ministral-3-14B-Reasoning-2512"),
    ("Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3"),
    ("Qwen2.5-14B-Instruct", "Qwen2.5-14B-Instruct"),
    ("Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"),
    ("Qwen3.5-9B", "Qwen3.5-9B"),
    ("Qwen3-8B", "Qwen3-8B"),
]


def is_baseline_file(name) -> bool:
    name = Path(name).name
    return not any(tok in name for tok in NON_BASELINE_TOKENS)


def canonical_id(raw_name) -> str | None:
    """Model id for an output file name, or None if it is not one of the 14."""
    name = Path(raw_name).name
    for needle, model_id in _SUBSTRINGS:
        if needle in name:
            return model_id
    return None


def display(model_id: str) -> str:
    return BY_ID[model_id].display if model_id in BY_ID else model_id


def latex(model_id: str) -> str:
    return BY_ID[model_id].latex if model_id in BY_ID else model_id


def source(model_id: str | None) -> str:
    if model_id in BY_ID:
        return "api" if BY_ID[model_id].closed else "local"
    return "unknown"
