"""Week 10 apply_guardrails integration tests.

Tests the orchestration layer that ties classifier + verifier together
and applies the tiered action (keep / soften / redact / redact-hard-claim).
A mock verifier is injected so these run in milliseconds without the real
NLI model.

Run: pytest eval/test_apply_guardrails.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.guardrails import apply_guardrails, _split_sentences  # noqa: E402


def _chunk(content: str) -> dict:
    return {"id": "chunk-x", "content": content}


def _verifier_returning(score_map: dict) -> callable:
    """Build a fake verifier that returns a fixed score per sentence
    prefix. Looks up the longest matching prefix so short sentences can
    still be keyed without ambiguity."""
    def _verify(sentence: str, chunk_text: str) -> float:
        for prefix, score in sorted(score_map.items(), key=lambda kv: -len(kv[0])):
            if sentence.startswith(prefix):
                return score
        return 0.0
    return _verify


# ─── sentence splitting ──────────────────────────────────────────────────

def test_split_sentences_basic():
    parts = _split_sentences("Hello world. This is a test. Bye.")
    assert parts == ["Hello world.", "This is a test.", "Bye."]


def test_split_sentences_empty():
    assert _split_sentences("") == []
    assert _split_sentences("   ") == []


def test_split_sentences_single():
    assert _split_sentences("No trailing punctuation") == ["No trailing punctuation"]


# ─── keep path ────────────────────────────────────────────────────────────

def test_non_claim_sentences_are_kept_verbatim():
    answer = "Based on what you described, here's what to do next. See a doctor if symptoms get worse."
    filtered, scores = apply_guardrails(
        answer,
        [_chunk("irrelevant chunk")],
        verifier=_verifier_returning({}),
    )
    assert filtered == answer
    assert all(s["action"] == "keep_no_claim" for s in scores)
    assert all(s["requires_nli"] is False for s in scores)


def test_well_supported_claim_is_kept():
    answer = "You have type 2 diabetes."
    filtered, scores = apply_guardrails(
        answer,
        [_chunk("HbA1c results consistent with type 2 diabetes.")],
        verifier=_verifier_returning({"You have": 0.85}),
    )
    assert filtered == answer
    assert scores[0]["action"] == "keep"
    assert scores[0]["p_entail"] == 0.85


# ─── soften path ──────────────────────────────────────────────────────────

def test_weak_support_softens_non_hard_claim():
    """A threshold-only claim with weak support (0.2 <= p < 0.5) should
    be kept but annotated, not redacted."""
    answer = "Fasting glucose above 7 mmol/L suggests diabetes."
    filtered, scores = apply_guardrails(
        answer,
        [_chunk("some chunk")],
        verifier=_verifier_returning({"Fasting glucose": 0.35}),
    )
    assert "not directly stated in sources" in filtered
    assert scores[0]["action"] == "soften"
    assert scores[0]["flags"]["threshold"] is True


# ─── redact path ──────────────────────────────────────────────────────────

def test_contradicted_claim_is_redacted():
    """Below the redact threshold (0.2) regardless of hard-claim status,
    the sentence is dropped entirely."""
    answer = "You have type 2 diabetes. Drink plenty of water."
    filtered, scores = apply_guardrails(
        answer,
        [_chunk("some chunk")],
        verifier=_verifier_returning({"You have": 0.05}),
    )
    assert "type 2 diabetes" not in filtered
    assert "Drink plenty of water" in filtered
    # Action: since "You have" is also has_diagnosis_verb, it's a hard claim.
    # Hard-claim rule fires first and short-circuits to redact_hard_claim.
    assert scores[0]["action"] == "redact_hard_claim"
    assert scores[1]["action"] == "keep_no_claim"


def test_hard_claim_with_moderate_support_is_still_redacted():
    """Dose / diagnosis claims get the harder rule: they must clear 0.5,
    not 0.2. A 0.35 score would soften a threshold claim but must redact
    a dose claim. This is the 'no half-measures on dosing' rule."""
    answer = "Take 500 mg amoxicillin three times a day."
    filtered, scores = apply_guardrails(
        answer,
        [_chunk("some chunk")],
        verifier=_verifier_returning({"Take 500": 0.35}),
    )
    assert filtered == ""
    assert scores[0]["action"] == "redact_hard_claim"
    assert scores[0]["flags"]["dose"] is True


# ─── NLI failure path ─────────────────────────────────────────────────────

def test_nli_exception_softens_instead_of_failing():
    """If the NLI model raises (OOM, model download failed, …), we fail-
    soft: keep the sentence with a 'not independently verified' note and
    record the error in the score log. The response must never 500 because
    of guardrails."""
    def _broken(sentence, chunk):
        raise RuntimeError("model not available")

    answer = "You have hypertension."
    filtered, scores = apply_guardrails(
        answer, [_chunk("some chunk")], verifier=_broken,
    )
    assert "not independently verified" in filtered
    assert scores[0]["action"] == "soften_nli_error"
    assert scores[0]["p_entail"] is None
    assert scores[0]["error"] == "model not available"


# ─── total-redaction path ────────────────────────────────────────────────

def test_all_claims_redacted_returns_empty():
    """If every sentence is a hard claim and every one fails NLI, filtered
    answer is empty. The caller is responsible for falling back to the
    standard refusal message."""
    answer = "You have diabetes. Take 500 mg metformin twice daily."
    filtered, scores = apply_guardrails(
        answer,
        [_chunk("irrelevant chunk")],
        verifier=_verifier_returning({}),  # everything scores 0.0
    )
    assert filtered == ""
    assert all(s["action"].startswith("redact") for s in scores)


# ─── multi-chunk max ──────────────────────────────────────────────────────

def test_max_score_over_chunks():
    """If any retrieved chunk entails the sentence, the sentence passes —
    we don't require every chunk to agree."""
    answer = "You have type 2 diabetes."
    chunks = [
        _chunk("unrelated chunk about hypertension"),
        _chunk("HbA1c consistent with type 2 diabetes"),
        _chunk("another unrelated chunk"),
    ]

    def _fake(sentence, chunk):
        if "diabetes" in chunk:
            return 0.9
        return 0.1

    filtered, scores = apply_guardrails(answer, chunks, verifier=_fake)
    assert filtered == answer
    assert scores[0]["p_entail"] == 0.9
    assert scores[0]["action"] == "keep"
