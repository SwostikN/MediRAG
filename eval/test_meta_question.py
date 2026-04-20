"""Unit tests for app/meta_question.is_meta_question.

Locks in the 2026-04-20 Failure-B fix (docs/HALLUCINATION_ZERO_PLAN.md
§8b): follow-up clarifications about the prior assistant answer must
route to the clarification composer, not to fresh retrieval.

Test families:
  - positive Layer-1 (explicit back-ref) → meta
  - positive Layer-2 (demonstrative + scope verb) → meta
  - positive Layer-3 (ultra-short follow-up) → meta
  - negatives: fresh symptom reports, fresh info queries, topic shifts
    that happen to share tokens → NOT meta
  - history guard: no prior assistant turn → NOT meta regardless of
    phrasing

Run: pytest eval/test_meta_question.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.meta_question import is_meta_question, last_assistant_turn  # noqa: E402


def _hist(*turns):
    """Build a lightweight history list — dicts, since HistoryTurn is a
    pydantic model and we don't want to import app.RAG just for tests."""
    out = []
    for role, content in turns:
        out.append({"role": role, "content": content})
    return out


_PRIOR = _hist(
    ("user", "what is type 2 diabetes management"),
    (
        "assistant",
        "Type 2 diabetes management focuses on lifestyle and medications. "
        "Lifestyle: healthy eating, regular physical activity, weight loss. "
        "Medications include metformin, sulfonylureas, SGLT-2 inhibitors.",
    ),
)


# ---------------------------------------------------------------------------
# Layer 1 — explicit back-references.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "is the above measure for type-2 diabetes only or all types of diabetes?",
    "the above answer — does it apply to gestational diabetes?",
    "in your previous answer, you mentioned metformin. is that safe in pregnancy?",
    "what you said about SGLT-2 inhibitors — is that for T2D only?",
    "you said metformin. is that always first-line?",
    "you mentioned weight loss. how much is recommended?",
    "based on that, should I start with metformin?",
    "as above — does it apply to children?",
    "your previous answer was too general. can you narrow it?",
    "your last answer mentioned statins — is that for everyone?",
])
def test_layer1_backref_positive(q):
    assert is_meta_question(q, _PRIOR) is True


# ---------------------------------------------------------------------------
# Layer 2 — demonstrative + scope verb.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "does that apply to type 1 diabetes too?",
    "does this include children?",
    "is this for all diabetics or only T2D?",
    "is it only for adults?",
    "only for T2D?",
    "only in pregnancy?",
    "what about children?",
    "what about type 1 diabetes?",
    "what if I'm pregnant?",
    "how about elderly patients?",
])
def test_layer2_demonstrative_positive(q):
    assert is_meta_question(q, _PRIOR) is True


# ---------------------------------------------------------------------------
# Layer 3 — ultra-short follow-ups.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "why?",
    "how?",
    "really?",
    "are you sure?",
    "is that true?",
    "is that right?",
    "are you sure",
    "correct?",
])
def test_layer3_ultra_short_positive(q):
    assert is_meta_question(q, _PRIOR) is True


# ---------------------------------------------------------------------------
# Negatives — fresh queries that must NOT trigger meta routing.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    # Fresh symptom reports.
    "I have a headache",
    "my chest hurts",
    "i'm having trouble breathing",
    # Fresh info queries.
    "what is hypertension",
    "explain malaria",
    "what are the symptoms of dengue",
    # Topic-shift "what about" with long specific content → fresh query.
    "what about the full list of symptoms of dengue fever in children under 5",
    # Navigation / results.
    "should I go to the ER",
    "my hba1c is 7.2",
    # Fresh conditional hypothetical (no prior reference).
    "if I have fever what should I do",
])
def test_negatives_fresh_query(q):
    # Even WITH prior history, these are not clarifications — they're new.
    assert is_meta_question(q, _PRIOR) is False


# ---------------------------------------------------------------------------
# History guard — no prior assistant turn means no meta-question is
# possible, regardless of phrasing.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "is the above for T2D only?",
    "why?",
    "what about children?",
    "does that apply to pregnancy?",
])
def test_no_history_no_meta(q):
    assert is_meta_question(q, None) is False
    assert is_meta_question(q, []) is False
    # History with only user turns → still no prior assistant content.
    user_only = _hist(("user", "I have diabetes"))
    assert is_meta_question(q, user_only) is False


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------

def test_empty_question():
    assert is_meta_question("", _PRIOR) is False
    assert is_meta_question("   ", _PRIOR) is False


def test_last_assistant_turn_picks_most_recent():
    h = _hist(
        ("user", "u1"),
        ("assistant", "a1"),
        ("user", "u2"),
        ("assistant", "a2 most recent"),
    )
    assert last_assistant_turn(h) == "a2 most recent"


def test_last_assistant_turn_ignores_empty_content():
    h = _hist(
        ("assistant", "first"),
        ("assistant", "   "),  # whitespace-only, ignored
    )
    assert last_assistant_turn(h) == "first"


def test_last_assistant_turn_accepts_attribute_objects():
    class T:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    h = [T("user", "q"), T("assistant", "a")]
    assert last_assistant_turn(h) == "a"
