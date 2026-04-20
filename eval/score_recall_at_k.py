"""Retrieval recall@K scorer for MediRAG (Phase 2b hallucination-reduction).

Measures whether a gold case's expected source (or expected-hint keywords)
appears in the top-K reranked results produced by app.RAG._retrieve_ranked.
Reuses the production retrieval path — does NOT implement a second one.

Gold schema handling (rows that lack ALL of the following fields are skipped
as "unscoreable" and excluded from the denominator):
  - expected_sources (list of source name strings)   — e.g. navigation.jsonl
  - expected_source  (single string; alias)           — rare, tolerated
  - expected_output_hints (list of keywords)          — intake.jsonl, redflag.jsonl
  - expected_topics       (list of keywords; alias)   — condition.jsonl

Rows with "retrieval_scoring": "disabled" are also skipped — these are the
ALL-DROP rows from the 2026-04-20 gold rewrite where every previous
expected_sources entry was Western commercial / clinician-only and had no
Nepal-appropriate substitute. Their refusal / red-flag axes stay scoreable
on the other harnesses.

Hit definition, per row:
  - If expected_sources/expected_source present: a row counts as a HIT at
    rank r if the tokenised expected string overlaps the tokenised
    retrieved source-label (doc_title + doc_source) at >= 60 % of expected
    tokens — same rule as context_recall in score_ragas_lite.py. The row
    hits overall if ANY expected source is matched within top-K.
  - If only hints/topics present: HIT at rank r if any keyword (case-
    insensitive substring) appears in retrieved chunk content OR doc_title
    at rank r.
  - If both present: sources take precedence; hints are used as a fallback
    only when expected_sources is empty.

coverage.jsonl rows have neither expected_sources nor hints — they are
excluded from scoring (n_cases counts only scoreable rows). Pass
--include-unscoreable to see them listed in per_example with
matched_at_rank_or_null=null and a "skipped": true flag.

Usage:
    # Default: coverage.jsonl, K=5 (most coverage rows will skip — see above)
    python eval/score_recall_at_k.py

    # Navigation gold — actually scoreable
    python eval/score_recall_at_k.py --gold-file eval/gold/navigation.jsonl --k 5

    # Quick smoke test
    python eval/score_recall_at_k.py --gold-file eval/gold/intake.jsonl --k 5 --limit 3

Output:
    eval/baselines/recall_at_k_<YYYY-MM-DD>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
BASELINES_DIR = EVAL_DIR / "baselines"
DEFAULT_GOLD = EVAL_DIR / "gold" / "coverage.jsonl"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "at",
    "is", "are", "was", "were", "be", "been", "being", "it", "this",
    "that", "these", "those", "with", "by", "as", "from", "info", "your",
    "you", "we", "our", "their", "not", "but",
}

SOURCE_TOKEN_OVERLAP = 0.6  # same threshold as score_ragas_lite.context_recall


def _tokenize(s: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((s or "").lower()) if t not in _STOPWORDS}


def _row_source_label(row: dict[str, Any]) -> str:
    """The string we tokenise when checking for an expected-source hit."""
    parts = [row.get("doc_title") or "", row.get("doc_source") or "", row.get("doc_source_url") or ""]
    return " ".join(p for p in parts if p)


def _row_keyword_haystack(row: dict[str, Any]) -> str:
    """The string we scan when checking for expected-hint keyword hits."""
    parts = [row.get("content") or "", row.get("doc_title") or ""]
    return " ".join(parts).lower()


def load_gold(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"[error] gold file not found: {path}", file=sys.stderr)
        sys.exit(2)
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"[error] {path}:{line_no} invalid JSON — {exc}", file=sys.stderr)
    return rows


def extract_expectations(gold_row: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (expected_sources, expected_hints). Either may be empty."""
    sources: list[str] = []
    raw_src = gold_row.get("expected_sources")
    if isinstance(raw_src, list):
        sources = [s for s in raw_src if isinstance(s, str) and s.strip()]
    elif isinstance(raw_src, str) and raw_src.strip():
        sources = [raw_src]
    # Single-key alias
    single = gold_row.get("expected_source")
    if isinstance(single, str) and single.strip():
        sources.append(single)

    hints: list[str] = []
    raw_hints = gold_row.get("expected_output_hints") or gold_row.get("expected_topics")
    if isinstance(raw_hints, list):
        hints = [h for h in raw_hints if isinstance(h, str) and h.strip()]
    return sources, hints


def source_hit_rank(ranked_rows: list[dict[str, Any]], expected: list[str], k: int) -> Optional[int]:
    """Return 1-based rank of first top-K chunk whose source-label matches
    any expected source at >= SOURCE_TOKEN_OVERLAP. None if no match."""
    if not expected:
        return None
    exp_tok_list = [(_tokenize(e), e) for e in expected]
    exp_tok_list = [(t, e) for t, e in exp_tok_list if t]
    if not exp_tok_list:
        return None
    for rank, row in enumerate(ranked_rows[:k], start=1):
        row_tok = _tokenize(_row_source_label(row))
        if not row_tok:
            continue
        for exp_tok, _exp_str in exp_tok_list:
            if len(exp_tok & row_tok) / len(exp_tok) >= SOURCE_TOKEN_OVERLAP:
                return rank
    return None


def keyword_hit_rank(ranked_rows: list[dict[str, Any]], hints: list[str], k: int) -> Optional[int]:
    """Return 1-based rank of first top-K chunk whose content/title contains
    any hint (case-insensitive substring). None if no match."""
    if not hints:
        return None
    needles = [h.lower() for h in hints if h.strip()]
    if not needles:
        return None
    for rank, row in enumerate(ranked_rows[:k], start=1):
        hay = _row_keyword_haystack(row)
        if any(n in hay for n in needles):
            return rank
    return None


def top_titles(ranked_rows: list[dict[str, Any]], k: int) -> list[str]:
    out: list[str] = []
    for r in ranked_rows[:k]:
        title = r.get("doc_title") or r.get("doc_source") or "?"
        out.append(title)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval recall@K scorer for MediRAG")
    parser.add_argument("--gold-file", default=str(DEFAULT_GOLD),
                        help="Path to gold jsonl (default: eval/gold/coverage.jsonl)")
    parser.add_argument("--k", type=int, default=5, help="Top-K to evaluate (default: 5)")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N rows (for quick runs)")
    parser.add_argument("--out", default=None, help="Override output path (default: eval/baselines/recall_at_k_<today>.json)")
    parser.add_argument("--include-unscoreable", action="store_true",
                        help="Include rows with no expected_sources/hints in per_example (flagged skipped)")
    args = parser.parse_args()

    gold_path = Path(args.gold_file).resolve()
    gold_rows = load_gold(gold_path)
    if args.limit is not None:
        gold_rows = gold_rows[: args.limit]

    # Import AFTER argparse so --help is fast and doesn't require DB creds.
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from app.RAG import _retrieve_ranked  # type: ignore
    except Exception as exc:
        print(f"[error] could not import app.RAG._retrieve_ranked: {exc}", file=sys.stderr)
        return 2

    per_example: list[dict[str, Any]] = []
    n_scoreable = 0
    n_hits = 0
    per_stage_counts: dict[str, dict[str, int]] = {}

    for i, gr in enumerate(gold_rows, start=1):
        case_id = gr.get("id") or gr.get("case_id") or f"row_{i}"
        question = gr.get("query") or gr.get("question") or gr.get("intake_summary") or ""
        stage = gr.get("stage") or gr.get("category") or "unknown"
        exp_sources, exp_hints = extract_expectations(gr)

        if gr.get("retrieval_scoring") == "disabled":
            if args.include_unscoreable:
                per_example.append({
                    "case_id": case_id,
                    "question": question,
                    "stage": stage,
                    "matched_at_rank_or_null": None,
                    "top_titles": [],
                    "skipped": True,
                    "reason": gr.get("retrieval_scoring_reason") or "retrieval_scoring=disabled",
                })
            print(f"[{i}/{len(gold_rows)}] {case_id}  SKIP (retrieval_scoring=disabled)", flush=True)
            continue

        scoreable = bool(exp_sources or exp_hints)
        if not scoreable:
            if args.include_unscoreable:
                per_example.append({
                    "case_id": case_id,
                    "question": question,
                    "stage": stage,
                    "matched_at_rank_or_null": None,
                    "top_titles": [],
                    "skipped": True,
                    "reason": "no expected_sources or expected_output_hints",
                })
            print(f"[{i}/{len(gold_rows)}] {case_id}  SKIP (no expectations)", flush=True)
            continue

        try:
            ranked = _retrieve_ranked(question) or []
        except Exception as exc:
            print(f"[{i}/{len(gold_rows)}] {case_id}  ERROR: {exc}", flush=True)
            per_example.append({
                "case_id": case_id,
                "question": question,
                "stage": stage,
                "matched_at_rank_or_null": None,
                "top_titles": [],
                "error": str(exc),
            })
            n_scoreable += 1
            bucket = per_stage_counts.setdefault(stage, {"n": 0, "hits": 0})
            bucket["n"] += 1
            continue

        # expected_sources wins when both are present — it's the stricter signal.
        match_mode = "source" if exp_sources else "hint"
        if match_mode == "source":
            rank = source_hit_rank(ranked, exp_sources, args.k)
        else:
            rank = keyword_hit_rank(ranked, exp_hints, args.k)

        n_scoreable += 1
        hit = rank is not None
        if hit:
            n_hits += 1
        bucket = per_stage_counts.setdefault(stage, {"n": 0, "hits": 0})
        bucket["n"] += 1
        if hit:
            bucket["hits"] += 1

        per_example.append({
            "case_id": case_id,
            "question": question,
            "stage": stage,
            "match_mode": match_mode,
            "expected_sources": exp_sources,
            "expected_hints": exp_hints if match_mode == "hint" else [],
            "matched_at_rank_or_null": rank,
            "top_titles": top_titles(ranked, args.k),
        })
        print(f"[{i}/{len(gold_rows)}] {case_id}  mode={match_mode}  rank={rank}", flush=True)

    recall = (n_hits / n_scoreable) if n_scoreable else None

    # Per-stage summary
    per_stage: dict[str, dict[str, Any]] = {}
    for stage, c in per_stage_counts.items():
        per_stage[stage] = {
            "n": c["n"],
            "hits": c["hits"],
            "recall_at_k": (c["hits"] / c["n"]) if c["n"] else None,
        }

    # Output
    today = date.today().isoformat()
    out_path = Path(args.out) if args.out else (BASELINES_DIR / f"recall_at_k_{today}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_date": today,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gold_file": str(gold_path),
        "K": args.k,
        "n_cases": n_scoreable,
        "n_hits": n_hits,
        "recall_at_k": recall,
        "per_stage": per_stage,
        "per_example": per_example,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Console summary
    print("\n" + "=" * 72)
    print(f"Recall@{args.k} — {gold_path.name}")
    print("=" * 72)
    print(f"scoreable cases: {n_scoreable}")
    print(f"hits:            {n_hits}")
    if recall is None:
        print("recall@K:        n/a (no scoreable rows)")
    else:
        print(f"recall@K:        {recall:.3f}")
    if per_stage:
        print("\nPer-stage:")
        for stage, s in sorted(per_stage.items()):
            rk = s["recall_at_k"]
            rk_s = "n/a" if rk is None else f"{rk:.3f}"
            print(f"  {stage:<20} n={s['n']:<3} hits={s['hits']:<3} recall@{args.k}={rk_s}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
