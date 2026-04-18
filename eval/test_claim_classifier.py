"""Week 10 claim-classifier tests.

The classifier decides which sentences in a generated response need NLI
entailment verification. Hand-written sentences cover the three reasons a
sentence should be checked (dose, threshold, diagnosis, duration) and the
reasons it should NOT be checked (disclaimers, navigation framing, generic
prose).

Run: pytest eval/test_claim_classifier.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.guardrails import classify_claim  # noqa: E402


# (sentence, requires_nli, tag-for-readability)
REQUIRES_NLI = [
    ("Take 500 mg paracetamol every 6 hours for up to 3 days.", "dose+duration"),
    ("Ibuprofen 400mg three times a day with food.", "dose"),
    ("Your reading of 150/90 is considered stage 1 hypertension.", "bp-threshold"),
    ("An HbA1c of 7.2% is above the diabetes diagnostic cutoff.", "percent-threshold"),
    ("Fasting glucose over 7 mmol/L suggests diabetes.", "mmol-threshold"),
    ("You have type 2 diabetes.", "diagnosis-verb"),
    ("You are having a stroke.", "diagnosis-verb"),
    ("This is a classic migraine pattern.", "diagnosis-verb"),
    ("Continue the medication for 14 days.", "duration"),
]

SHOULD_SKIP_NLI = [
    ("I can't diagnose, please see a doctor.", "disclaimer"),
    ("Based on what you described, here's what to do next.", "framing"),
    ("Call 102 for an ambulance immediately.", "escalation-template"),
    ("Drink plenty of water and rest.", "generic-advice"),
    ("Hypertension is a common long-term condition.", "generic-definition"),
    ("See a clinician if symptoms get worse.", "disclaimer"),
    ("Please consult your doctor before starting any new medicine.", "disclaimer"),
]


@pytest.mark.parametrize("sentence,tag", REQUIRES_NLI)
def test_positive_claims_require_nli(sentence: str, tag: str):
    feats = classify_claim(sentence)
    assert feats.requires_nli, f"{tag!r}: should require NLI — got {feats}"


@pytest.mark.parametrize("sentence,tag", SHOULD_SKIP_NLI)
def test_benign_sentences_skip_nli(sentence: str, tag: str):
    feats = classify_claim(sentence)
    assert not feats.requires_nli, f"{tag!r}: should skip NLI — got {feats}"


def test_empty_and_whitespace_skip():
    assert not classify_claim("").requires_nli
    assert not classify_claim("   ").requires_nli


def test_disclaimer_short_circuits_dose():
    """Disclaimer cue wins even if a dose number appears — the sentence is
    template output, not a clinical claim requiring corpus support."""
    s = "I can't diagnose — a clinician may prescribe 500 mg paracetamol."
    feats = classify_claim(s)
    assert feats.is_disclaimer
    assert feats.has_dose
    assert not feats.requires_nli


def test_feature_flags_are_independent():
    """A sentence can trigger multiple feature flags; classifier should set
    each one truthfully rather than stopping at the first match."""
    s = "You have hypertension with a reading of 160/100 and should take 5 mg amlodipine daily."
    feats = classify_claim(s)
    assert feats.has_diagnosis_verb
    assert feats.has_threshold
    assert feats.has_dose
    assert feats.requires_nli
