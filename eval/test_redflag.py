"""Week 6 Stage 0 red-flag engine tests (docs/IMPROVEMENTS.md §4.2).

Loads eval/gold/redflag.jsonl and asserts:
- Positive recall >= 99% (CI gate: at most 0 misses out of 50)
- Negative false-positive rate <= 20% (soft cap — §4.2 accepts over-firing)
- Every rule in the YAML has at least one positive gold case

Soft checks (printed, non-blocking):
- Whether the fired rule_id matches expected_rule_id. Overlapping emergencies
  legitimately match multiple rules — what matters is that something fires.

Run: pytest eval/test_redflag.py -v -s
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import redflag  # noqa: E402


GOLD_PATH = ROOT / "eval" / "gold" / "redflag.jsonl"


def _load_gold():
    cases = []
    with GOLD_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


@pytest.fixture(scope="module")
def gold():
    return _load_gold()


def test_positive_recall_at_least_99_percent(gold):
    positives = [c for c in gold if c["subtype"] == "positive"]
    assert len(positives) >= 50, f"expected >=50 positives, got {len(positives)}"
    misses = []
    for case in positives:
        hit = redflag.check(case["query"])
        if hit is None:
            misses.append((case["id"], case.get("expected_rule_id"), case["query"]))
    total = len(positives)
    recall = (total - len(misses)) / total
    print(f"\npositive recall: {total - len(misses)}/{total} = {recall:.3f}")
    for m in misses[:20]:
        print(f"  MISS: {m[0]} expected={m[1]} query={m[2]!r}")
    assert recall >= 0.99, f"positive recall {recall:.3f} below 0.99; {len(misses)} misses"


def test_negative_false_positive_rate(gold):
    negatives = [c for c in gold if c["subtype"] == "negative"]
    assert len(negatives) >= 50, f"expected >=50 negatives, got {len(negatives)}"
    fps = []
    for case in negatives:
        hit = redflag.check(case["query"])
        if hit is not None:
            fps.append((case["id"], hit.rule_id, case["query"]))
    total = len(negatives)
    fp_rate = len(fps) / total
    print(f"\nnegative false-positive rate: {len(fps)}/{total} = {fp_rate:.3f}")
    for fp in fps[:20]:
        print(f"  FP: {fp[0]} fired={fp[1]} query={fp[2]!r}")
    assert fp_rate <= 0.20, f"FP rate {fp_rate:.3f} > 0.20 — rules are too broad"


def test_positive_rule_id_match(gold):
    """Soft check — prints mismatches, does not fail.

    First-match-wins ordering can legitimately pick a different but equally-safe
    rule (e.g., seizure_active vs infant_seizure). Both route to an emergency
    template; which one is a tuning detail, not a safety bug."""
    positives = [c for c in gold if c["subtype"] == "positive"]
    mismatches = []
    for case in positives:
        expected = case.get("expected_rule_id")
        if not expected:
            continue
        hit = redflag.check(case["query"])
        if hit is None:
            continue
        if hit.rule_id != expected:
            mismatches.append((case["id"], expected, hit.rule_id))
    print(f"\nrule_id mismatches: {len(mismatches)}/{len(positives)}")
    for m in mismatches[:20]:
        print(f"  {m[0]}: expected={m[1]} fired={m[2]}")


def test_all_rules_have_positive_coverage(gold):
    positives = [c for c in gold if c["subtype"] == "positive"]
    covered = {c.get("expected_rule_id") for c in positives if c.get("expected_rule_id")}
    all_rules = set(redflag.all_rule_ids())
    uncovered = all_rules - covered
    assert not uncovered, f"rules without positive gold coverage: {sorted(uncovered)}"


def test_engine_returns_none_on_empty():
    assert redflag.check("") is None
    assert redflag.check("   ") is None


def test_engine_returns_hit_structure():
    hit = redflag.check("I have crushing chest pain radiating to my left arm.")
    assert hit is not None
    assert hit.rule_id == "chest_pain_cardiac"
    assert hit.urgency == "emergency"
    assert "102" in hit.message
    assert hit.category == "cardiac"
