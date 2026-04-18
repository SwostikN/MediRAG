"""Stage 2 (care-tier navigation) scorer.

Direct, in-process evaluator — does NOT go through the HTTP /query path.
For each gold case in eval/gold/navigation_stage2.jsonl, calls
navigation_stage.compose_recommendation() with the synthetic Stage 1
intake summary + retrieval rows from the existing pipeline, parses the
output, and scores against the gold expectations.

Why direct (not end-to-end via /query):
    Stage 2 quality is what the Week 7B care-pathway corpus changes. An
    end-to-end eval also exercises Stage 1's intake-completion logic,
    which adds noise and ~5x latency. Direct keeps the signal clean.

Metrics:
    tier_accuracy        — exact match on tier_id (parsed from "Where to go" line)
    urgency_accuracy     — bucket match on urgency band (parsed from "When" line)
    escalation_recall    — fraction of expected triggers found via token-overlap
    refusal_hygiene_rate — fraction of cases where no forbidden phrase appears

Usage:
    python eval/score_stage2.py
    python eval/score_stage2.py --no-retrieval        # skip _retrieve_ranked
    python eval/score_stage2.py --out eval/baselines/stage2_baseline.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

EVAL_DIR = Path(__file__).resolve().parent
GOLD_PATH = EVAL_DIR / "gold" / "navigation_stage2.jsonl"
BASELINES_DIR = EVAL_DIR / "baselines"

# Order matters — match most specific first. Maps a substring (case-insensitive)
# in the LLM's "Where to go" line to the canonical tier_id from
# app/nepal_care_tiers.yaml. Some labels overlap (district label contains "OPD",
# private label also contains "OPD") so order is load-bearing.
_TIER_PATTERNS: list[tuple[str, str]] = [
    ("self-care", "self_care"),
    ("self care", "self_care"),
    ("emergency department", "emergency_department"),
    ("call 102", "emergency_department"),
    (" ed ", "emergency_department"),
    ("nearest emergency", "emergency_department"),
    ("health post", "health_post"),
    ("urban health centre", "health_post"),
    ("phcc", "phcc"),
    ("primary health care", "phcc"),
    ("zonal", "zonal_central"),
    ("central hospital", "zonal_central"),
    ("tertiary hospital", "zonal_central"),
    ("private hospital", "private_opd"),
    ("private opd", "private_opd"),
    ("specialist opd", "private_opd"),
    ("district hospital", "district_hospital"),
]

# Maps phrasing in the "When" line to one of the canonical urgency bands used
# in the gold file. Order matters: specific phrasings before generic ones.
_URGENCY_PATTERNS: list[tuple[str, str]] = [
    ("go now", "now"),
    ("right away", "now"),
    ("immediately", "now"),
    ("do not wait", "now"),
    ("now.", "now"),
    ("within 24 hours", "today"),
    ("same-day", "today"),
    ("same day", "today"),
    ("today", "today"),
    ("within the day", "today"),
    ("1-2 weeks", "routine"),
    ("1–2 weeks", "routine"),
    ("next 1-2 weeks", "routine"),
    ("next 1–2 weeks", "routine"),
    ("next 2 weeks", "routine"),
    ("routine", "routine"),
    ("within the week", "this-week"),
    ("this week", "this-week"),
    ("within a week", "this-week"),
    ("next few days", "this-week"),
    ("1-7 days", "this-week"),
    ("within days", "this-week"),
    ("monitor", "monitor"),
    ("24-48 hours", "monitor"),
    ("24–48 hours", "monitor"),
    ("watchful", "monitor"),
]

# Forbidden phrases lifted from intake_templates.yaml + navigation prompt.
# Stage 2 must remain descriptive/navigational, never diagnostic.
_FORBIDDEN_RE = re.compile(
    r"\bsounds? like\b|\bmight be\b|\bprobably\b|\byou have\b|"
    r"\bit could be\b|\bmost likely\b|\bdiagnosis\s+is\b",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "with",
    "is", "are", "be", "if", "any", "than", "more", "very", "at",
}


def _tokenize(s: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(s.lower()) if t not in _STOPWORDS and len(t) > 2}


def parse_tier(answer: str) -> Optional[str]:
    """Extract canonical tier_id from the 'Where to go' line."""
    m = re.search(r"\*\*Where to go:\*\*(.+?)(?:\n|\*\*)", answer, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    line = " " + m.group(1).lower() + " "
    for needle, tier_id in _TIER_PATTERNS:
        if needle in line:
            return tier_id
    return None


def parse_urgency(answer: str) -> Optional[str]:
    """Extract canonical urgency band from the 'When' line."""
    m = re.search(r"\*\*When:\*\*(.+?)(?:\n|\*\*)", answer, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    line = m.group(1).lower()
    for needle, band in _URGENCY_PATTERNS:
        if needle in line:
            return band
    return None


def parse_escalation_block(answer: str) -> str:
    """Return the raw text of the 'Go to 102 right away if' line, or ''."""
    m = re.search(
        r"\*\*Go to 102.*?:\*\*(.+?)(?:\n\n|\Z|\*\*Sources)",
        answer,
        re.IGNORECASE | re.DOTALL,
    )
    return m.group(1).strip() if m else ""


def escalation_recall(answer: str, expected: list[str]) -> float:
    """Fraction of expected triggers covered in the escalation block.

    A trigger counts as covered if >=50% of its non-stopword tokens appear
    anywhere in the escalation line. 50% (not 60%) because triggers tend to
    be short — 'fainting' is one token after stopword stripping.
    """
    if not expected:
        return 1.0
    block_tokens = _tokenize(parse_escalation_block(answer))
    hits = 0
    for trigger in expected:
        t_tokens = _tokenize(trigger)
        if not t_tokens:
            continue
        overlap = len(t_tokens & block_tokens) / len(t_tokens)
        if overlap >= 0.5:
            hits += 1
    return hits / len(expected)


def has_forbidden_phrase(answer: str) -> bool:
    return bool(_FORBIDDEN_RE.search(answer))


@dataclass
class CaseResult:
    case_id: str
    expected_tier: str
    predicted_tier: Optional[str]
    tier_match: bool
    expected_urgency: str
    predicted_urgency: Optional[str]
    urgency_match: bool
    escalation_recall: float
    refusal_clean: bool
    n_retrieval_rows: int
    answer: str
    error: Optional[str] = None


@dataclass
class Aggregate:
    n: int = 0
    tier_correct: int = 0
    urgency_correct: int = 0
    escalation_sum: float = 0.0
    refusal_clean_count: int = 0
    errors: list[str] = field(default_factory=list)
    per_case: list[CaseResult] = field(default_factory=list)
    by_tier: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, r: CaseResult) -> None:
        self.n += 1
        if r.error:
            self.errors.append(f"{r.case_id}: {r.error}")
            return
        if r.tier_match:
            self.tier_correct += 1
        if r.urgency_match:
            self.urgency_correct += 1
        self.escalation_sum += r.escalation_recall
        if r.refusal_clean:
            self.refusal_clean_count += 1
        bucket = self.by_tier.setdefault(r.expected_tier, {"n": 0, "correct": 0})
        bucket["n"] += 1
        if r.tier_match:
            bucket["correct"] += 1
        self.per_case.append(r)

    def summary(self) -> dict[str, Any]:
        runnable = self.n - len(self.errors)
        return {
            "n_total": self.n,
            "n_runnable": runnable,
            "n_errors": len(self.errors),
            "tier_accuracy": self.tier_correct / runnable if runnable else None,
            "urgency_accuracy": self.urgency_correct / runnable if runnable else None,
            "escalation_recall": self.escalation_sum / runnable if runnable else None,
            "refusal_hygiene_rate": self.refusal_clean_count / runnable if runnable else None,
            "by_tier": {
                k: {**v, "accuracy": v["correct"] / v["n"] if v["n"] else None}
                for k, v in sorted(self.by_tier.items())
            },
            "errors": self.errors,
        }


def load_gold() -> list[dict]:
    items: list[dict] = []
    for line_no, line in enumerate(GOLD_PATH.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        try:
            items.append(json.loads(s))
        except json.JSONDecodeError as exc:
            print(f"[error] gold line {line_no}: {exc}", file=sys.stderr)
    return items


def run_one(
    case: dict,
    *,
    use_retrieval: bool,
    retrieve_fn: Any,
    compose_fn: Any,
    groq_client: Any,
    groq_model: str,
    cohere_client: Any,
    context_chunks: int,
) -> CaseResult:
    case_id = case["id"]
    intake_summary = case["intake_summary"]
    intent_bucket = case.get("intent_bucket", "other")

    nav_rows: list[dict] = []
    if use_retrieval and retrieve_fn is not None:
        try:
            full_rows = retrieve_fn(intake_summary)
            nav_rows = full_rows[:context_chunks] if full_rows else []
        except Exception as exc:
            return CaseResult(
                case_id=case_id,
                expected_tier=case["expected_tier_id"],
                predicted_tier=None,
                tier_match=False,
                expected_urgency=case["expected_urgency_band"],
                predicted_urgency=None,
                urgency_match=False,
                escalation_recall=0.0,
                refusal_clean=False,
                n_retrieval_rows=0,
                answer="",
                error=f"retrieval failed: {exc}",
            )

    try:
        answer = compose_fn(
            intake_summary=intake_summary,
            intent_bucket=intent_bucket,
            groq_client=groq_client,
            groq_model=groq_model,
            cohere_client=cohere_client,
            cohere_model="command-r-08-2024",
            retrieval_rows=nav_rows,
        )
    except Exception as exc:
        return CaseResult(
            case_id=case_id,
            expected_tier=case["expected_tier_id"],
            predicted_tier=None,
            tier_match=False,
            expected_urgency=case["expected_urgency_band"],
            predicted_urgency=None,
            urgency_match=False,
            escalation_recall=0.0,
            refusal_clean=False,
            n_retrieval_rows=len(nav_rows),
            answer="",
            error=f"compose failed: {exc}",
        )

    pred_tier = parse_tier(answer)
    pred_urgency = parse_urgency(answer)
    return CaseResult(
        case_id=case_id,
        expected_tier=case["expected_tier_id"],
        predicted_tier=pred_tier,
        tier_match=pred_tier == case["expected_tier_id"],
        expected_urgency=case["expected_urgency_band"],
        predicted_urgency=pred_urgency,
        urgency_match=pred_urgency == case["expected_urgency_band"],
        escalation_recall=escalation_recall(answer, case.get("expected_escalation_triggers", [])),
        refusal_clean=not has_forbidden_phrase(answer),
        n_retrieval_rows=len(nav_rows),
        answer=answer,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="MediRAG Stage 2 navigation scorer")
    parser.add_argument("--no-retrieval", action="store_true",
                        help="skip _retrieve_ranked; pass empty rows (faster, isolates LLM+prompt)")
    parser.add_argument("--out", default=str(BASELINES_DIR / "stage2_baseline.json"),
                        help="path to write the snapshot JSON")
    parser.add_argument("--label", default="baseline",
                        help="label embedded in the snapshot for cross-run diffing")
    parser.add_argument("--per-case-delay", type=float, default=0.0,
                        help="seconds to sleep between cases. Set to 7.0 when using a "
                             "Cohere Trial key (10 rpm) to avoid rerank fallback to RRF.")
    args = parser.parse_args()

    sys.path.insert(0, str(EVAL_DIR.parent))
    from app.RAG import (  # noqa: E402
        _retrieve_ranked,
        groq_client,
        GROQ_MODEL,
        co,
        CONTEXT_CHUNKS,
    )
    from app.stages.navigation import compose_recommendation  # noqa: E402

    items = load_gold()
    print(f"[stage2-eval] loaded {len(items)} cases from {GOLD_PATH.name}")
    print(f"[stage2-eval] retrieval={'OFF' if args.no_retrieval else 'ON'} "
          f"groq={'OFF' if groq_client is None else 'ON'}")

    agg = Aggregate()
    t0 = time.time()
    for i, case in enumerate(items, 1):
        elapsed = time.time() - t0
        print(f"[{i}/{len(items)}] {case['id']} ({elapsed:.1f}s)", flush=True)
        r = run_one(
            case,
            use_retrieval=not args.no_retrieval,
            retrieve_fn=_retrieve_ranked,
            compose_fn=compose_recommendation,
            groq_client=groq_client,
            groq_model=GROQ_MODEL,
            cohere_client=co,
            context_chunks=CONTEXT_CHUNKS,
        )
        if r.error:
            print(f"   ERROR: {r.error}")
        else:
            print(f"   tier={r.predicted_tier or '?':<22} (gold {r.expected_tier:<22}) "
                  f"{'OK' if r.tier_match else 'MISS'} | urg={r.predicted_urgency or '?':<10} "
                  f"escal_recall={r.escalation_recall:.2f}")
        agg.add(r)
        if args.per_case_delay > 0 and i < len(items):
            time.sleep(args.per_case_delay)

    summary = agg.summary()
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()
    summary["label"] = args.label
    summary["use_retrieval"] = not args.no_retrieval
    summary["gold_path"] = str(GOLD_PATH)
    summary["per_case"] = [
        {
            "id": c.case_id,
            "expected_tier": c.expected_tier,
            "predicted_tier": c.predicted_tier,
            "tier_match": c.tier_match,
            "expected_urgency": c.expected_urgency,
            "predicted_urgency": c.predicted_urgency,
            "urgency_match": c.urgency_match,
            "escalation_recall": c.escalation_recall,
            "refusal_clean": c.refusal_clean,
            "n_retrieval_rows": c.n_retrieval_rows,
            "error": c.error,
        }
        for c in agg.per_case
    ]

    print()
    print("=" * 72)
    print(f"Stage 2 eval — {args.label}")
    print("=" * 72)
    print(f"Cases:                {summary['n_runnable']}/{summary['n_total']} runnable")
    print(f"Tier accuracy:        {summary['tier_accuracy']:.3f}" if summary['tier_accuracy'] is not None else "Tier accuracy:        —")
    print(f"Urgency accuracy:     {summary['urgency_accuracy']:.3f}" if summary['urgency_accuracy'] is not None else "Urgency accuracy:     —")
    print(f"Escalation recall:    {summary['escalation_recall']:.3f}" if summary['escalation_recall'] is not None else "Escalation recall:    —")
    print(f"Refusal hygiene rate: {summary['refusal_hygiene_rate']:.3f}" if summary['refusal_hygiene_rate'] is not None else "Refusal hygiene rate: —")
    print(f"Errors: {summary['n_errors']}")
    for e in summary["errors"]:
        print(f"  - {e}")
    print()
    print("By expected tier:")
    for tier_id, stats in summary["by_tier"].items():
        acc = stats["accuracy"]
        acc_s = f"{acc:.3f}" if acc is not None else "—"
        print(f"  {tier_id:<24} n={stats['n']:>2} correct={stats['correct']:>2} acc={acc_s}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSnapshot written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
