"""Week 10 NLI verifier tests.

Two layers:

1. FAST tests (always run). Mock the model load and feed controllable
   logits. These verify the softmax + label-index plumbing in
   verify_entailment() without downloading ~400 MB.

2. INTEGRATION test (opt-in). Loads the real
   cross-encoder/nli-deberta-v3-base and checks a known-entailed pair
   scores > 0.7 and a known-contradicted pair scores < 0.3. Skipped
   unless RUN_NLI_INTEGRATION=1 is set.

Run fast:        pytest eval/test_nli_verifier.py -v
Run with model:  RUN_NLI_INTEGRATION=1 pytest eval/test_nli_verifier.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import guardrails  # noqa: E402


class _FakeLogits:
    """Mimics the [1, num_labels] tensor the real model returns at [0].
    verify_entailment() indexes [0] and then softmaxes, so we only need
    a 1-D thing that supports .item() on torch.softmax's output."""

    def __init__(self, values):
        import torch
        self._tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(0)

    @property
    def logits(self):
        return self._tensor


class _FakeModel:
    def __init__(self, logits_for_next_call):
        self._logits_for_next_call = logits_for_next_call
        self.config = type("C", (), {"id2label": {0: "contradiction", 1: "entailment", 2: "neutral"}})()

    def __call__(self, **kwargs):
        return _FakeLogits(self._logits_for_next_call)

    def eval(self):
        return self


class _FakeTok:
    def __call__(self, premise, hypothesis, **kwargs):
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}


def _install_fake_nli(monkeypatch, logits):
    import torch
    monkeypatch.setattr(
        guardrails,
        "_nli_state",
        {
            "tokenizer": _FakeTok(),
            "model": _FakeModel(logits),
            "entail_idx": 1,
            "torch": torch,
        },
    )


def test_empty_inputs_return_zero():
    assert guardrails.verify_entailment("", "something") == 0.0
    assert guardrails.verify_entailment("something", "") == 0.0
    assert guardrails.verify_entailment("   ", "   ") == 0.0


def test_strong_entailment_score(monkeypatch):
    # logits favouring the entailment index
    _install_fake_nli(monkeypatch, [-2.0, 4.0, -1.0])
    p = guardrails.verify_entailment(
        "Paracetamol can help with mild pain.",
        "Paracetamol is used to treat mild to moderate pain.",
    )
    assert p > 0.9, f"expected strong entailment, got {p}"


def test_contradiction_score(monkeypatch):
    # logits favouring the contradiction index
    _install_fake_nli(monkeypatch, [4.0, -2.0, -1.0])
    p = guardrails.verify_entailment(
        "Take 1000 mg paracetamol every 2 hours.",
        "Adults take 500–1000 mg every 4–6 hours, max 4 g/day.",
    )
    assert p < 0.1, f"expected low entailment on contradiction, got {p}"


def test_neutral_score(monkeypatch):
    # logits roughly equal — softmax gives ~0.33 each
    _install_fake_nli(monkeypatch, [1.0, 1.0, 1.0])
    p = guardrails.verify_entailment("Hypertension is common.", "Diabetes is common.")
    assert 0.25 < p < 0.45, f"expected ~uniform ~0.33, got {p}"


def test_entail_idx_respected(monkeypatch):
    """If a future NLI model orders labels differently (entailment at
    index 2, say), verify_entailment() must read config.id2label — not
    hardcode index 1."""
    import torch
    _fake_model = _FakeModel([1.0, -1.0, 4.0])  # entailment logit at index 2
    _fake_model.config = type("C", (), {"id2label": {0: "contradiction", 1: "neutral", 2: "entailment"}})()
    monkeypatch.setattr(
        guardrails,
        "_nli_state",
        {
            "tokenizer": _FakeTok(),
            "model": _fake_model,
            "entail_idx": 2,
            "torch": torch,
        },
    )
    p = guardrails.verify_entailment("a", "b")
    assert p > 0.9, f"expected high entailment with entail_idx=2, got {p}"


# ─── Optional integration test ──────────────────────────────────────────────

_RUN_INTEGRATION = os.environ.get("RUN_NLI_INTEGRATION") == "1"


@pytest.mark.skipif(not _RUN_INTEGRATION, reason="Set RUN_NLI_INTEGRATION=1 to download the NLI model and run this.")
def test_real_model_entailed_pair():
    p = guardrails.verify_entailment(
        "Paracetamol can help with mild pain.",
        "Paracetamol is used to treat mild to moderate pain.",
    )
    assert p > 0.7, f"real model: expected entailed pair > 0.7, got {p}"


@pytest.mark.skipif(not _RUN_INTEGRATION, reason="Set RUN_NLI_INTEGRATION=1 to download the NLI model and run this.")
def test_real_model_contradicted_pair():
    p = guardrails.verify_entailment(
        "Aspirin cures type 2 diabetes.",
        "Aspirin is a pain reliever and anti-inflammatory drug.",
    )
    assert p < 0.3, f"real model: expected contradicted pair < 0.3, got {p}"
