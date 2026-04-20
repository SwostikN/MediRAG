"""Unit tests for app/intent_gate.classify_turn.

Locks in the pre-intake intent gate behaviour added 2026-04 to fix the
"lecture me before you answer me" UX — informational questions like
"what is hypertension?" were routed through Stage 1 intake (5-slot
history-taking questions) even when the user was clearly not reporting
a symptom.

Test families mirror the four morphologies the gate has to disambiguate:
  - Symptom reports (1st-person experience) → intake
  - Informational queries (WH-fronting, definition verbs) → condition
  - Diagnosis-framed education → condition
  - Lab-results / numeric  → results
  - Explicit care-tier questions → navigation

Plus edge cases where signals overlap (e.g. "what is this rash") — these
are tested with the LLM tie-breaker OFF so we verify the safe-default
fallback is "intake".

Run: pytest eval/test_intent_gate.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.intent_gate import classify_turn  # noqa: E402


# ---------------------------------------------------------------------------
# Layer-1 hard signals.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "what does TSH 5.2 mean",
    "my hba1c is 7.2, is that bad",
    "ldl 180 mg/dl what to do",
    "creatinine 1.8 mean",
    "my report shows hemoglobin 9",
])
def test_layer1_results(q):
    assert classify_turn(q) == "results"


@pytest.mark.parametrize("q", [
    "should I go to the ER for chest pain",
    "which hospital level should I visit",
    "call 102 or wait",
    "is this an emergency",
    "can I wait till morning",
    "where can I get a TB test in Kathmandu",
])
def test_layer1_navigation(q):
    assert classify_turn(q) == "navigation"


@pytest.mark.parametrize("q", [
    "I was diagnosed with type 2 diabetes, what should I do",
    "my doctor said I have hypertension",
    "I was told I have PCOS",
    "recently diagnosed with asthma, how do I manage it",
    "living with type 1 diabetes, diet tips",
])
def test_layer1_diagnosed(q):
    assert classify_turn(q) == "condition"


# ---------------------------------------------------------------------------
# Layer-2 symptom reports → intake.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "I have a headache",
    "i'm having chest tightness since yesterday",
    "been feeling dizzy for 3 days",
    "i feel nauseous this morning",
    "my head hurts",
    "my stomach is burning",
    "my child has fever and cough",
    "fever cough runny nose",
    "i am suffering from back pain",
    "i've been having palpitations",
    "m having trouble breathing",
    "i keep getting migraines",
])
def test_layer2_symptom_intake(q):
    assert classify_turn(q) == "intake"


# ---------------------------------------------------------------------------
# Layer-3 informational → condition.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "what is hypertension",
    "what is hypertension?",
    "what are the symptoms of dengue",
    "explain type 2 diabetes",
    "tell me about malaria",
    "define hba1c",
    "causes of jaundice",
    "symptoms of tuberculosis",
    "treatment for psoriasis",
    "risk factors for stroke",
    "how does insulin work",
    "why is blood pressure important",
    "hypertension",
    "diabetes",
    "dengue",
    "what does hba1c measure",
    "what is the difference between type 1 and type 2 diabetes",
])
def test_layer3_info_condition(q):
    assert classify_turn(q) == "condition"


# ---------------------------------------------------------------------------
# Edge cases: signal overlap. With LLM tie-breaker OFF these fall back
# to intake (safe default). The gate is deliberately conservative here —
# misrouting a symptom to retrieval is more harmful than asking extra
# clarifying questions.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "what is this rash",             # demonstrative body
    "what is this pain in my chest", # WH + 1st-person body
    "why am I having headaches",     # WH + 1st-person symptom verb
    "what's wrong with my stomach",  # 1st-person body
])
def test_overlap_defaults_to_intake_without_llm(q):
    # No groq_client passed → ambiguity resolves to intake.
    assert classify_turn(q) == "intake"


# ---------------------------------------------------------------------------
# Edge cases: diagnosed + info. Diagnosed layer (L1) takes precedence
# over WH-fronting so these route to condition, not intake.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "I was diagnosed with diabetes, what is it",
    "my doctor said I have hypertension, how do I manage it",
])
def test_diagnosed_plus_info_routes_condition(q):
    assert classify_turn(q) == "condition"


# ---------------------------------------------------------------------------
# Empty / whitespace / one-word greeting → intake (safe).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "",
    "   ",
    "hi",
    "hello",
    "help",
])
def test_empty_or_greeting_defaults_to_intake(q):
    assert classify_turn(q) == "intake"


# ---------------------------------------------------------------------------
# Layer-4 LLM tie-breaker: routes to condition when the mock returns "B".
# Verifies the wiring; the prompt contract is documented inline.
# ---------------------------------------------------------------------------

class _FakeChoice:
    def __init__(self, text):
        self.message = type("M", (), {"content": text})


class _FakeResp:
    def __init__(self, text):
        self.choices = [_FakeChoice(text)]


class _FakeGroqClient:
    """Minimal Groq-compatible client stub. Lets tests assert the gate
    honours the LLM verdict without hitting the real API."""
    def __init__(self, verdict: str):
        self._verdict = verdict
        self.chat = type("Chat", (), {
            "completions": type("C", (), {
                "create": lambda _self, **kw: _FakeResp(self._verdict),
            })(),
        })()


def test_tiebreak_returns_condition_when_llm_says_b():
    client = _FakeGroqClient("B")
    # "what is this rash" hits both L2 and L3 → tie-breaker runs.
    decision = classify_turn(
        "what is this rash",
        groq_client=client,
        groq_model="mock",
    )
    assert decision == "condition"


def test_tiebreak_returns_intake_when_llm_says_a():
    client = _FakeGroqClient("A")
    decision = classify_turn(
        "what is this rash",
        groq_client=client,
        groq_model="mock",
    )
    assert decision == "intake"


def test_tiebreak_defaults_to_intake_on_llm_failure():
    class _BrokenClient:
        chat = type("Chat", (), {
            "completions": type("C", (), {
                "create": lambda _self, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
            })(),
        })()

    decision = classify_turn(
        "what is this rash",
        groq_client=_BrokenClient(),
        groq_model="mock",
    )
    assert decision == "intake"
