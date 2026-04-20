"""Meta-question detector.

Catches the Failure-B shape from docs/HALLUCINATION_ZERO_PLAN.md §8b:
after the assistant has produced an answer, the user asks a clarification
ABOUT that answer — "is the above measure for T2D only or all diabetes?",
"only for adults?", "why?", "you said X, what about Y?".

Without this gate, /query treats the follow-up as a fresh retrieval
query: it re-retrieves on the surface tokens, pulls adjacent chunks,
and regurgitates a new answer that ignores the scope the user was
actually asking about. Symptom looks like a confident off-topic reply —
worse than an honest refusal because the user cannot tell it is wrong.

Design mirrors app/intent_gate.py:
  - Hard prerequisite: history must contain at least one assistant turn.
    No prior turn → nothing for a meta-question to refer to → skip.
  - Three deterministic layers, highest-precedence match wins.
  - No LLM tie-breaker. A false negative falls through to normal
    retrieval (current behaviour) — safe. A false positive would skip
    retrieval on a genuinely new query, which IS harmful, so the
    patterns bias toward false negatives.

    Layer 1  explicit back-reference phrases ("the above", "your
             previous answer", "you said/mentioned", "as above").
    Layer 2  scope-clarification shapes: demonstrative pronoun
             (this/that/it) + scope verb (for/apply/include/only/mean).
    Layer 3  ultra-short follow-ups ("why?", "how?", "really?",
             "only adults?") — when the question stands alone it
             almost always refers to the prior turn.

No side effects. No network.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional


def _normalize(q: str) -> str:
    return (q or "").strip().lower()


def _last_assistant_text(history: Optional[Iterable[Any]]) -> Optional[str]:
    """Return the content of the most recent assistant turn, or None.

    Accepts either pydantic HistoryTurn objects (with .role / .content
    attrs) or plain dicts — /query and /query/stream both route through
    HistoryTurn, but tests and internal callers may pass dicts.
    """
    if not history:
        return None
    last: Optional[str] = None
    for turn in history:
        role = getattr(turn, "role", None) if not isinstance(turn, dict) else turn.get("role")
        content = getattr(turn, "content", None) if not isinstance(turn, dict) else turn.get("content")
        if role == "assistant" and isinstance(content, str) and content.strip():
            last = content
    return last


# ---------------------------------------------------------------------------
# Layer 1 — explicit back-reference phrases. Any of these → meta.
# ---------------------------------------------------------------------------

_BACKREF_PAT = re.compile(
    r"\b("
    r"the\s+above|above\s+(answer|response|reply|measure|advice)|"
    r"previous\s+(answer|response|reply|message)|"
    r"last\s+(answer|response|reply|message)|"
    r"prior\s+(answer|response|reply|message)|"
    r"your\s+(previous|last|prior|earlier)\s+(answer|response|reply|message)|"
    r"what\s+you\s+(said|mentioned|told|wrote)|"
    r"you\s+(said|mentioned|told|wrote)|"
    r"as\s+(above|you\s+(said|mentioned))|"
    r"based\s+on\s+(that|this|what\s+you)"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Layer 2 — demonstrative-pronoun + scope-clarification shapes.
#
# Requires BOTH a demonstrative pronoun (this/that/it/these/those) AND a
# scope/clarification cue (for / apply / include / only / mean / cover /
# work for / true for). "is this a good idea" would not trigger — no
# scope verb. "only for adults?" would — "only for" is strong alone.
# ---------------------------------------------------------------------------

_DEMONSTRATIVE_PAT = re.compile(
    r"\b(this|that|it|these|those|the\s+one)\b",
    re.IGNORECASE,
)

_SCOPE_VERB_PAT = re.compile(
    r"\b("
    r"apply|applies|applicable|"
    r"include|includes|including|"
    r"cover|covers|"
    r"mean|means|"
    r"work\s+for|works\s+for|"
    r"true\s+for|"
    r"for\s+(all|only|just|both|either|everyone|anyone|adults?|children|"
    r"infants?|pregnant|women|men|diabetics?|hypertensives?|"
    r"type[-\s]?[12])|"
    r"only\s+(for|in|when|if|applies|refers)"
    r")\b",
    re.IGNORECASE,
)

# "what about X" / "what if X" — back-refs to the prior topic when short.
_WHAT_ABOUT_PAT = re.compile(
    r"^\s*(what\s+about|what\s+if|how\s+about|and\s+what\s+about)\b",
    re.IGNORECASE,
)

# "only for X?" / "only in Y?" — scope-limiting follow-ups.
_ONLY_FOR_PAT = re.compile(
    r"^\s*(only|just|specifically)\s+(for|in|to|when|if)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Layer 3 — ultra-short follow-ups. Fewer than ~6 tokens and starts with a
# WH / modal verb → almost certainly a meta-question when a prior turn
# exists.
# ---------------------------------------------------------------------------

_ULTRA_SHORT_META_PAT = re.compile(
    r"^\s*(why|how|really|seriously|always|ever|sure|correct|right|okay|"
    r"are\s+you\s+sure|is\s+that\s+(true|right|correct|sure)|"
    r"you\s+sure)\b[\s\?\.\!]*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def is_meta_question(
    question: str,
    history: Optional[Iterable[Any]] = None,
) -> bool:
    """True iff the question is a clarification about the prior assistant
    turn. Returns False when history lacks any assistant content.
    """
    q = _normalize(question)
    if not q:
        return False
    if _last_assistant_text(history) is None:
        return False

    # Layer 1.
    if _BACKREF_PAT.search(q):
        return True

    # Layer 2.
    if _ONLY_FOR_PAT.search(q):
        return True
    if _WHAT_ABOUT_PAT.search(q) and len(q.split()) <= 10:
        # "what about X" is only meta when short. "what about the symptoms
        # of dengue in children under 5" is a fresh query, not a meta-q.
        return True
    if _DEMONSTRATIVE_PAT.search(q) and _SCOPE_VERB_PAT.search(q):
        return True

    # Layer 3.
    if len(q.split()) <= 6 and _ULTRA_SHORT_META_PAT.search(q):
        return True

    return False


def last_assistant_turn(history: Optional[Iterable[Any]]) -> Optional[str]:
    """Exposed helper so /query handlers can fetch the prior-assistant
    text for the clarification composer without re-implementing the
    dict/attr dance."""
    return _last_assistant_text(history)
