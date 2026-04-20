"""Clarification composer — answers meta-questions against the PRIOR
assistant turn, with no retrieval.

Gated by app.meta_question.is_meta_question. Only runs when the user's
turn is asking ABOUT the previous answer ("is the above for T2D only?",
"what about children?", "why?"). Retrieval is deliberately skipped —
the corpus has nothing to add; the question is about what we already
said, not about the world.

Guarantees enforced by the system prompt:
    1. Answer ONLY what was asked. No re-summarising the prior answer.
    2. If the prior answer did NOT actually cover the scope being
       questioned, say so explicitly — do not fabricate a scope.
    3. Do not introduce new medical claims. If the meta-question
       cannot be answered from the prior answer alone, tell the user
       and suggest they rephrase as a fresh question.
    4. Short: a clarification is a sentence or two, not a lecture.

Fails closed: Groq failure → a safe fallback string pointing the user
at rephrasing. Never raises.
"""
from __future__ import annotations

from typing import Any, Optional


_SYSTEM_PROMPT = """You are a health navigator answering a CLARIFICATION \
about your previous answer. The user is asking about something you \
already said — not asking a fresh medical question.

Rules you MUST follow:
1. Read the PRIOR ANSWER and the CLARIFICATION QUESTION carefully.
2. Answer ONLY the clarification. Do NOT restate or re-summarise the \
prior answer.
3. If the prior answer did NOT actually specify the scope the user is \
asking about (e.g. user asks "only for adults?" and the prior answer \
never mentioned age), say so directly: "My previous answer did not \
specify that — please ask it as a new question and I'll look it up."
4. Do NOT introduce new medical claims, dosages, or recommendations \
that were not already in the prior answer.
5. Keep it to 1–3 short sentences.
6. Do not add disclaimers, sign-offs, or source lists — this is a \
follow-up, not a standalone answer.
"""


_FALLBACK = (
    "I can't reliably clarify my previous answer right now. "
    "Please rephrase your question as a fresh query and I'll look it up."
)


def compose_clarification(
    *,
    prior_answer: str,
    user_question: str,
    groq_client: Any,
    groq_model: str,
    max_tokens: int = 180,
) -> str:
    """One Groq call. Returns the clarification text, or _FALLBACK on
    any failure. Never raises."""
    if not prior_answer or not prior_answer.strip():
        return _FALLBACK
    if not user_question or not user_question.strip():
        return _FALLBACK
    if groq_client is None or not groq_model:
        return _FALLBACK

    user_prompt = (
        "PRIOR ANSWER (what you said before):\n"
        f"{prior_answer.strip()}\n\n"
        "CLARIFICATION QUESTION (what the user is asking now):\n"
        f"{user_question.strip()}"
    )

    try:
        resp = groq_client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
    except Exception as exc:
        print(f"[clarification] groq failed: {exc}")
        return _FALLBACK

    try:
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        return _FALLBACK

    return text or _FALLBACK
