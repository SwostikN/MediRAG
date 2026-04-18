"""Week 10 streaming-guardrail tests.

process_streaming_chunk and flush_streaming_buffer feed LLM tokens
through the classify → NLI → tiered-action pipeline one sentence at
a time, so the /query/stream endpoint never emits a sentence it would
later have to retract.

A mock verifier is injected so the tests run in milliseconds without
the 400 MB NLI model.

Run: pytest eval/test_streaming_guardrails.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.guardrails import (  # noqa: E402
    process_streaming_chunk,
    flush_streaming_buffer,
)


def _verifier_returning(score_map: dict):
    """Fake verifier that maps sentence-prefix → score (same pattern
    as test_apply_guardrails.py)."""
    def _verify(sentence: str, chunk_text: str) -> float:
        for prefix, score in sorted(score_map.items(), key=lambda kv: -len(kv[0])):
            if sentence.startswith(prefix):
                return score
        return 0.0
    return _verify


# ─── boundary detection ─────────────────────────────────────────────────

def test_partial_sentence_does_not_emit():
    """Mid-sentence tokens must buffer, not emit — otherwise the client
    would see unsafe fragments before the guardrail ran."""
    score_log: list[dict] = []
    buffer, emits = process_streaming_chunk(
        "", "Hello there",
        [], score_log,
        verifier=_verifier_returning({}),
    )
    assert emits == []
    assert buffer == "Hello there"
    assert score_log == []


def test_complete_sentence_emits():
    score_log: list[dict] = []
    buffer, emits = process_streaming_chunk(
        "", "Here is the plan. ",  # includes boundary
        [],
        score_log,
        verifier=_verifier_returning({}),
    )
    assert emits == ["Here is the plan."]
    assert buffer == ""
    assert len(score_log) == 1


def test_multi_sentence_tokens_all_emit_in_order():
    """A single token with multiple sentences (sometimes LLMs ship
    `".   Next sentence. "` in one chunk) must emit all of them."""
    score_log: list[dict] = []
    buffer, emits = process_streaming_chunk(
        "",
        "One. Two. Three. ",
        [],
        score_log,
        verifier=_verifier_returning({}),
    )
    assert emits == ["One.", "Two.", "Three."]
    assert buffer == ""


# ─── guardrail enforcement mid-stream ───────────────────────────────────

def test_hard_claim_without_support_is_redacted_midstream():
    """'You have diabetes.' with no entailment support must not reach
    the client — the emits list should drop it. This is the central
    promise of Option C streaming."""
    score_log: list[dict] = []
    buffer, emits = process_streaming_chunk(
        "",
        "You have diabetes. Drink water. ",
        [{"content": "irrelevant"}] if False else [""],
        score_log,
        verifier=_verifier_returning({}),
    )
    assert "diabetes" not in " ".join(emits)
    assert "Drink water." in emits
    assert score_log[0]["action"] == "redact_hard_claim"


def test_hard_claim_with_support_is_kept_midstream():
    score_log: list[dict] = []
    buffer, emits = process_streaming_chunk(
        "",
        "You have type 2 diabetes. ",
        ["HbA1c consistent with type 2 diabetes"],
        score_log,
        verifier=_verifier_returning({"You have": 0.9}),
    )
    assert emits == ["You have type 2 diabetes."]
    assert score_log[0]["action"] == "keep"


def test_flush_handles_trailing_sentence_without_boundary():
    """LLMs frequently drop the terminal whitespace. The flush helper
    runs the trailing buffer through the pipeline one last time so it
    isn't silently dropped."""
    score_log: list[dict] = []
    sentence = "Please consult a clinician for further evaluation"  # disclaimer → keep_no_claim
    buffer, emits = process_streaming_chunk(
        "",
        sentence,
        [],
        score_log,
        verifier=_verifier_returning({}),
    )
    assert emits == []
    assert buffer == sentence
    final = flush_streaming_buffer(
        buffer, [], score_log, verifier=_verifier_returning({}),
    )
    assert final == [sentence]
    assert score_log[0]["action"] == "keep_no_claim"


# ─── streaming = batch parity ────────────────────────────────────────────

def test_streaming_pipeline_matches_batch_pipeline():
    """Streaming is just batch with a sentence splitter — the score
    log entries must be shape-identical so the jsonb schema stays one
    schema, not two."""
    from app.guardrails import apply_guardrails

    answer = "Here is the plan. You have hypertension. See a doctor."
    chunks = [{"content": "unrelated"}]
    # Batch.
    _, batch_log = apply_guardrails(
        answer, chunks, verifier=_verifier_returning({"You have": 0.1}),
    )

    # Streaming: feed the answer in arbitrary-sized token chunks.
    stream_log: list[dict] = []
    buffer = ""
    tokens = ["Here is ", "the plan. You ", "have hyper", "tension. See a doctor."]
    all_emits: list[str] = []
    for t in tokens:
        buffer, emits = process_streaming_chunk(
            buffer, t, [c["content"] for c in chunks], stream_log,
            verifier=_verifier_returning({"You have": 0.1}),
        )
        all_emits.extend(emits)
    all_emits.extend(
        flush_streaming_buffer(
            buffer, [c["content"] for c in chunks], stream_log,
            verifier=_verifier_returning({"You have": 0.1}),
        )
    )

    assert [e["action"] for e in stream_log] == [e["action"] for e in batch_log]
    assert [e["flags"] for e in stream_log] == [e["flags"] for e in batch_log]
