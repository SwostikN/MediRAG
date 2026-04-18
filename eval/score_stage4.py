"""Stage 4 (lab-results explainer) scorer.

Direct, in-process evaluator — does NOT go through the HTTP /upload path.
For each gold case in eval/gold/results.jsonl, feeds the natural-language
query as 'text' to results_stage.compose_explainer() (the parser picks
out marker + value + unit from phrases like "My TSH is 8.4 mIU/L"),
then scores the composed answer against the gold expectations.

Why direct (not end-to-end via /upload):
    Stage 4 quality is what the Week 8 lab-explainer corpus + per-marker
    prompt change. An end-to-end eval adds PyMuPDF extraction + Supabase
    I/O, both of which are orthogonal to the signal we care about.

Metrics:
    marker_parse_recall — fraction of expected_markers that the parser
                          emitted (matched on canonical name substring).
    hint_coverage       — fraction of expected_output_hints that appear
                          in the answer via token-overlap (>=50%).
    refusal_hygiene     — boolean per case: the answer contains NO
                          forbidden diagnostic / dosing phrase
                          (reuses app.refusal_filter.find_forbidden_phrases,
                          which is also the runtime safety gate).
    escalation_present  — boolean per case: the deterministic escalation
                          footer appears verbatim. Appending this is
                          deterministic so it should always be 1.0; this
                          is a sanity check for regression.

Usage:
    python eval/score_stage4.py
    python eval/score_stage4.py --no-retrieval     # feed empty rows per marker
    python eval/score_stage4.py --out eval/baselines/stage4_baseline.json
    python eval/score_stage4.py --per-case-delay 7 # Cohere Trial (10 rpm)
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
GOLD_PATH = EVAL_DIR / "gold" / "results.jsonl"
BASELINES_DIR = EVAL_DIR / "baselines"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "with",
    "is", "are", "be", "if", "any", "than", "more", "very", "at", "this",
    "that", "your", "you", "it", "its",
}


def _tokenize(s: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(s.lower()) if t not in _STOPWORDS and len(t) > 2}


def hint_coverage(answer: str, expected: list[str]) -> float:
    """Fraction of hints whose non-stopword tokens overlap the answer
    by >=50%. Loose on purpose — hints are thematic phrases, not
    verbatim strings."""
    if not expected:
        return 1.0
    ans_tokens = _tokenize(answer)
    hits = 0
    for hint in expected:
        h_tokens = _tokenize(hint)
        if not h_tokens:
            continue
        overlap = len(h_tokens & ans_tokens) / len(h_tokens)
        if overlap >= 0.5:
            hits += 1
    return hits / len(expected)


def _load_alias_to_canonical() -> dict[str, str]:
    """Import the parser's own alias table so 'Free T4' / 'thyroid
    stimulating hormone' in gold cases resolve to the same canonical
    ('FT4' / 'TSH') that the parser emits. Keeps the eval in lockstep
    with the parser — if a new alias is added to results.py it's
    picked up here automatically."""
    sys.path.insert(0, str(EVAL_DIR.parent))
    from app.stages.results import _ALIAS_TO_CANONICAL  # noqa: E402
    return dict(_ALIAS_TO_CANONICAL)


def marker_parse_recall(
    parsed_markers: list[dict],
    expected: list[str],
    alias_to_canonical: Optional[dict[str, str]] = None,
) -> float:
    """Fraction of expected marker names that show up in parsed_markers.
    Resolves the gold name ('Free T4', 'fasting glucose') to the same
    canonical the parser emits ('FT4', 'FBS') via the shared alias
    table, then checks set membership."""
    if not expected:
        return 1.0
    aliases = alias_to_canonical if alias_to_canonical is not None else _load_alias_to_canonical()
    parsed_canonicals = {m.get("name", "") for m in parsed_markers}

    hits = 0
    for name in expected:
        canonical = aliases.get(name.lower())
        if canonical is None:
            # gold name may itself already be a canonical (e.g. "TSH")
            # or an alias the parser knows — fall back to case-insensitive
            # substring match against what the parser produced.
            lowered = name.lower()
            if any(lowered == p.lower() or lowered in p.lower() or p.lower() in lowered
                   for p in parsed_canonicals):
                hits += 1
            continue
        if canonical in parsed_canonicals:
            hits += 1
    return hits / len(expected)


@dataclass
class CaseResult:
    case_id: str
    expected_markers: list[str]
    parsed_marker_names: list[str]
    marker_parse_recall: float
    hint_coverage: float
    refusal_clean: bool
    forbidden_phrases: list[str]
    escalation_present: bool
    n_sources: int
    answer_chars: int
    answer: str
    error: Optional[str] = None


@dataclass
class Aggregate:
    n: int = 0
    marker_recall_sum: float = 0.0
    hint_coverage_sum: float = 0.0
    refusal_clean_count: int = 0
    escalation_present_count: int = 0
    errors: list[str] = field(default_factory=list)
    per_case: list[CaseResult] = field(default_factory=list)

    def add(self, r: CaseResult) -> None:
        self.n += 1
        if r.error:
            self.errors.append(f"{r.case_id}: {r.error}")
            return
        self.marker_recall_sum += r.marker_parse_recall
        self.hint_coverage_sum += r.hint_coverage
        if r.refusal_clean:
            self.refusal_clean_count += 1
        if r.escalation_present:
            self.escalation_present_count += 1
        self.per_case.append(r)

    def summary(self) -> dict[str, Any]:
        runnable = self.n - len(self.errors)
        return {
            "n_total": self.n,
            "n_runnable": runnable,
            "n_errors": len(self.errors),
            "marker_parse_recall": self.marker_recall_sum / runnable if runnable else None,
            "hint_coverage": self.hint_coverage_sum / runnable if runnable else None,
            "refusal_hygiene_rate": self.refusal_clean_count / runnable if runnable else None,
            "escalation_present_rate": self.escalation_present_count / runnable if runnable else None,
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
    escalation_footer: str,
    find_forbidden: Any,
) -> CaseResult:
    case_id = case["id"]
    query_text = case["query"]

    effective_retrieve = retrieve_fn if use_retrieval and retrieve_fn is not None else (lambda _q: [])

    try:
        result = compose_fn(
            query_text,
            retrieve_fn=effective_retrieve,
            groq_client=groq_client,
            groq_model=groq_model,
            cohere_client=cohere_client,
        )
    except Exception as exc:
        return CaseResult(
            case_id=case_id,
            expected_markers=case.get("expected_markers", []),
            parsed_marker_names=[],
            marker_parse_recall=0.0,
            hint_coverage=0.0,
            refusal_clean=False,
            forbidden_phrases=[],
            escalation_present=False,
            n_sources=0,
            answer_chars=0,
            answer="",
            error=f"compose failed: {exc}",
        )

    answer: str = result.get("answer", "") or ""
    parsed_markers: list[dict] = result.get("markers", []) or []
    sources: list[dict] = result.get("sources", []) or []

    forbidden = find_forbidden(answer)
    # Escalation-present check uses the first ~40 chars of the footer as a
    # landmark so whitespace / trailing edits don't break the match.
    escalation_landmark = escalation_footer.strip().splitlines()[0][:40]

    return CaseResult(
        case_id=case_id,
        expected_markers=case.get("expected_markers", []),
        parsed_marker_names=[m.get("name", "") for m in parsed_markers],
        marker_parse_recall=marker_parse_recall(parsed_markers, case.get("expected_markers", [])),
        hint_coverage=hint_coverage(answer, case.get("expected_output_hints", [])),
        refusal_clean=not forbidden,
        forbidden_phrases=forbidden,
        escalation_present=escalation_landmark in answer,
        n_sources=len(sources),
        answer_chars=len(answer),
        answer=answer,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="MediRAG Stage 4 lab-results scorer")
    parser.add_argument("--no-retrieval", action="store_true",
                        help="skip _retrieve_ranked; pass empty rows (isolates LLM+prompt)")
    parser.add_argument("--out", default=str(BASELINES_DIR / "stage4_baseline.json"),
                        help="path to write the snapshot JSON")
    parser.add_argument("--label", default="baseline",
                        help="label embedded in the snapshot for cross-run diffing")
    parser.add_argument("--per-case-delay", type=float, default=0.0,
                        help="seconds to sleep between cases. Set ~7.0 with a "
                             "Cohere Trial key (10 rpm) to avoid rerank fallback.")
    args = parser.parse_args()

    sys.path.insert(0, str(EVAL_DIR.parent))
    from app.RAG import (  # noqa: E402
        _retrieve_ranked,
        groq_client,
        GROQ_MODEL,
        co,
    )
    from app.stages.results import compose_explainer, _ESCALATION_FOOTER  # noqa: E402
    from app.refusal_filter import find_forbidden_phrases  # noqa: E402

    items = load_gold()
    print(f"[stage4-eval] loaded {len(items)} cases from {GOLD_PATH.name}")
    print(f"[stage4-eval] retrieval={'OFF' if args.no_retrieval else 'ON'} "
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
            compose_fn=compose_explainer,
            groq_client=groq_client,
            groq_model=GROQ_MODEL,
            cohere_client=co,
            escalation_footer=_ESCALATION_FOOTER,
            find_forbidden=find_forbidden_phrases,
        )
        if r.error:
            print(f"   ERROR: {r.error}")
        else:
            print(
                f"   markers parsed={r.parsed_marker_names or '[]'} "
                f"recall={r.marker_parse_recall:.2f} | "
                f"hints={r.hint_coverage:.2f} | "
                f"refusal_clean={'Y' if r.refusal_clean else 'N'} | "
                f"escal={'Y' if r.escalation_present else 'N'} | "
                f"sources={r.n_sources}"
            )
            if r.forbidden_phrases:
                print(f"   forbidden: {r.forbidden_phrases}")
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
            "expected_markers": c.expected_markers,
            "parsed_marker_names": c.parsed_marker_names,
            "marker_parse_recall": c.marker_parse_recall,
            "hint_coverage": c.hint_coverage,
            "refusal_clean": c.refusal_clean,
            "forbidden_phrases": c.forbidden_phrases,
            "escalation_present": c.escalation_present,
            "n_sources": c.n_sources,
            "answer_chars": c.answer_chars,
            "error": c.error,
        }
        for c in agg.per_case
    ]

    print()
    print("=" * 72)
    print(f"Stage 4 eval — {args.label}")
    print("=" * 72)
    print(f"Cases:                  {summary['n_runnable']}/{summary['n_total']} runnable")
    if summary["marker_parse_recall"] is not None:
        print(f"Marker parse recall:    {summary['marker_parse_recall']:.3f}")
    if summary["hint_coverage"] is not None:
        print(f"Hint coverage:          {summary['hint_coverage']:.3f}")
    if summary["refusal_hygiene_rate"] is not None:
        print(f"Refusal hygiene rate:   {summary['refusal_hygiene_rate']:.3f}")
    if summary["escalation_present_rate"] is not None:
        print(f"Escalation present:     {summary['escalation_present_rate']:.3f}")
    print(f"Errors: {summary['n_errors']}")
    for e in summary["errors"]:
        print(f"  - {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSnapshot written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
