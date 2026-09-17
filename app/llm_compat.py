"""Cross-model compatibility shims (paper phase, 1.0).

WHY THIS EXISTS
---------------
`llama-3.3-70b-versatile` was decommissioned by Groq in September 2026 (404
model_not_found), invalidating every number in this repo. Its replacement, `gpt-oss-20b`,
is a REASONING model, and that is not a drop-in swap.

A reasoning model emits hidden chain-of-thought tokens before any visible content, and
those tokens are billed against `max_tokens`. Measured on Groq, gpt-oss-20b:

    max_tokens=40   -> reasoning=38,  content=''      finish_reason='length'
    max_tokens=120  -> reasoning=118, content=''      finish_reason='length'
    max_tokens=200  -> reasoning=147, content='Pleurisy'

So every tight `max_tokens` in this codebase would have returned an EMPTY STRING rather
than an error, and each call site treats empty as "no result" and moves on. The failures
would have been silent:

    app/intent_gate.py   max_tokens=2    first-layer gate  -> always empty
    app/stages/intake.py max_tokens=10   slot filling      -> always empty
    app/RAG.py           max_tokens=40   topic derivation  -> always empty
    app/intent.py        max_tokens=50   intent classify   -> always empty
    app/query_rewrite.py max_tokens=80   retrieval rewrite -> always empty
    app/RAG.py           max_tokens=320  stage answers     -> heavily truncated

`reasoning_effort` (low/medium/high) is accepted by the API but did not reduce the token
count in testing, so headroom is the only remedy.

THE ALLOWANCE
-------------
Worst case observed across a binary gate, a slot-fill and a full RAG answer with ~200
tokens of context was 111 reasoning tokens; an adversarial single-word task reached 147.
The allowance is set to 512, roughly 3.5x the worst observed, because the cost of being
wrong is asymmetric: too much headroom wastes quota, too little silently disables a
safety layer.

Non-reasoning models (Cohere command-r, llama, qwen-instruct) are untouched — the
allowance is added only when the model name matches a known reasoning family.
"""
from __future__ import annotations

# Substring match against the model id, lowercased.
REASONING_MODEL_MARKERS = (
    "gpt-oss",      # openai/gpt-oss-20b, openai/gpt-oss-120b, gpt-oss-safeguard-20b
    "o1-", "o3-", "o4-",
    "deepseek-r1",
    "qwq",
    "-thinking",
    "reasoning",
)

REASONING_TOKEN_ALLOWANCE = 512


def is_reasoning_model(model: str | None) -> bool:
    """True when `model` emits hidden reasoning tokens billed against max_tokens."""
    if not model:
        return False
    name = str(model).lower()
    return any(marker in name for marker in REASONING_MODEL_MARKERS)


def gen_max_tokens(visible_budget: int, model: str | None) -> int:
    """Convert a desired VISIBLE output budget into an API `max_tokens` value.

    Call sites should express what they actually want to see — 700 tokens of answer, or
    2 tokens of YES/NO — and let this add the hidden reasoning overhead when the backend
    needs it. That keeps the intent readable at the call site and keeps model-specific
    arithmetic in one place.

    >>> gen_max_tokens(2, "command-r-08-2024")
    2
    >>> gen_max_tokens(2, "openai/gpt-oss-20b")
    514
    """
    if visible_budget <= 0:
        return visible_budget
    return visible_budget + REASONING_TOKEN_ALLOWANCE if is_reasoning_model(model) else visible_budget
