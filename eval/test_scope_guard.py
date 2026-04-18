"""Week 10 scope-guard classifier tests.

classify_scope sorts an answer into one of four buckets: safe,
diagnostic, prescriptive, emergency_override. Unlike the phrase-level
_FORBIDDEN_RE (which matches specific substrings), the scope classifier
looks at the whole answer and asks "what kind of clinical work is this
text trying to do?".

Run: pytest eval/test_scope_guard.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.refusal_filter import classify_scope, SCOPE_REFUSAL_TEMPLATES  # noqa: E402


# ─── safe path ───────────────────────────────────────────────────────────

def test_empty_is_safe():
    assert classify_scope("") == "safe"
    assert classify_scope("   ") == "safe"


def test_navigation_is_safe():
    answer = (
        "Based on what you described, it would be reasonable to see a "
        "clinician for a proper evaluation. A health post can do a first "
        "assessment and refer if needed."
    )
    assert classify_scope(answer) == "safe"


def test_explanation_is_safe():
    answer = (
        "Hypertension is a condition where blood pressure is persistently "
        "elevated. It is typically monitored over multiple visits."
    )
    assert classify_scope(answer) == "safe"


def test_prescriptive_cue_without_dose_unit_is_safe():
    """'Take your time' and 'use the inhaler' are navigation framing,
    not a prescription. Require a dose unit to co-occur for the
    prescriptive bucket to fire."""
    answer = "Take your time describing the symptoms. Use the inhaler if your doctor prescribed one."
    assert classify_scope(answer) == "safe"


# ─── diagnostic path ─────────────────────────────────────────────────────

def test_you_have_x_is_diagnostic():
    assert classify_scope("You have type 2 diabetes.") == "diagnostic"


def test_sounds_like_is_diagnostic():
    assert classify_scope("This sounds like asthma, but see a doctor.") == "diagnostic"


def test_most_likely_is_diagnostic():
    assert classify_scope("Most likely you have hypertension.") == "diagnostic"


def test_your_diagnosis_is_diagnostic():
    assert classify_scope("Your diagnosis is bacterial pneumonia.") == "diagnostic"


# ─── prescriptive path ───────────────────────────────────────────────────

def test_dose_recommendation_is_prescriptive():
    assert classify_scope("Take 500 mg amoxicillin three times daily.") == "prescriptive"


def test_start_taking_dose_is_prescriptive():
    assert (
        classify_scope("Start taking 25 mg losartan once a day.")
        == "prescriptive"
    )


def test_dose_without_prescriptive_verb_is_safe():
    """A dose appearing in an explanatory context — e.g. 'aspirin 81 mg
    is a typical prophylactic dose' — is not a prescription. It's
    education. The prescriptive bucket requires BOTH a prescriptive cue
    AND a dose unit."""
    answer = "Aspirin 81 mg is sometimes used for cardiovascular protection."
    assert classify_scope(answer) == "safe"


# ─── emergency override ──────────────────────────────────────────────────

def test_emergency_override_beats_diagnostic():
    """Emergency routing wins even if the text also contains a diagnostic
    cue — 'this is a stroke, call an ambulance' must not be refused by
    scope-guard because the emergency action matters more than the
    diagnosis phrasing."""
    answer = "This is a medical emergency — call an ambulance immediately."
    assert classify_scope(answer) == "emergency_override"


def test_call_102_is_emergency_override():
    assert classify_scope("Call 102 now — nearest emergency department.") == "emergency_override"


def test_red_flag_phrase_is_emergency_override():
    assert classify_scope("This is a red flag symptom. Go to the ER.") == "emergency_override"


# ─── refusal templates ───────────────────────────────────────────────────

def test_scope_refusal_templates_exist_for_both_buckets():
    assert "diagnostic" in SCOPE_REFUSAL_TEMPLATES
    assert "prescriptive" in SCOPE_REFUSAL_TEMPLATES
    assert SCOPE_REFUSAL_TEMPLATES["diagnostic"]
    assert SCOPE_REFUSAL_TEMPLATES["prescriptive"]
    # Emergency override is not a refusal — no template needed.
    assert "emergency_override" not in SCOPE_REFUSAL_TEMPLATES
