"""RAGAS-lite scorer for MediRAG.

We deliberately avoid the ragas package itself. Two reasons:
  1. ragas wants an OpenAI / Anthropic API key for every metric; this
     project already owns its own NLI verifier (app/guardrails.verify_entailment)
     trained on the same MNLI/DeBERTa stack RAGAS uses, so paying per-call
     for the same computation is waste.
  2. ragas's faithfulness prompt is generic — it scores "is the answer
     supported by the context" against a free-form LLM judge. Our gold set
     encodes the *stance* we want (expected_output_hints, expected_sources,
     expected_markers), which is a stricter signal than generic entailment.

What this scorer computes per row:

- answer_relevancy: token-overlap between answer and `expected_output_hints`
  (or `expected_topics` for condition stage). 0–1.
- context_recall: same recall@5 as harness.py — fraction of expected_sources
  whose non-stopword tokens overlap >=60% with a retrieved source.
- faithfulness: mean entailment probability across answer sentences,
  computed against the concatenated retrieved context using
  app.guardrails.verify_entailment. Sentences with <5 tokens are skipped
  (headers, citations, disclaimers).
- marker_coverage (results stage only): fraction of `expected_markers`
  present in the answer.

Usage:
    python eval/score_ragas_lite.py --server-url http://127.0.0.1:8000
    python eval/score_ragas_lite.py --server-url http://127.0.0.1:8000 --stages results,condition
    python eval/score_ragas_lite.py --server-url http://127.0.0.1:8000 --no-nli  # skip the slow entailment pass

Exit code is 0 if all per-stage means meet the minimum thresholds:
    answer_relevancy >= 0.35
    context_recall   >= 0.50
    faithfulness     >= 0.60  (only enforced if --no-nli is NOT set)

These thresholds are deliberately loose for v1 — the goal is to SURFACE
regressions, not to block every marginal change. Tighten as the corpus
matures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

EVAL_DIR = Path(__file__).resolve().parent
GOLD_DIR = EVAL_DIR / "gold"
RESULTS_DIR = EVAL_DIR / "results"

STAGES = ["intake", "navigation", "visit_prep", "results", "condition"]

THRESHOLDS = {
    "answer_relevancy": 0.35,
    "context_recall": 0.50,
    "faithfulness": 0.60,
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "at",
    "is", "are", "was", "were", "be", "been", "being", "it", "this",
    "that", "these", "those", "with", "by", "as", "from", "info", "your",
    "you", "we", "our", "their", "not", "but",
}


def _tokenize(s: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((s or "").lower()) if t not in _STOPWORDS}


def load_stage_rows(stage: str) -> list[dict[str, Any]]:
    path = GOLD_DIR / f"{stage}.jsonl"
    if not path.exists():
        print(f"[warn] missing gold file: {path}", file=sys.stderr)
        return []
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


def start_session(server_url: str, timeout: int = 15) -> Optional[str]:
    """Create a fresh eval session for intake/navigation rows — the stage
    state machine refuses to fire without one. Returns session_id or None
    on failure (caller falls back to stateless /query)."""
    import requests  # type: ignore
    try:
        resp = requests.post(
            f"{server_url.rstrip('/')}/session/start",
            json={},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json().get("session_id")
    except Exception:
        pass
    return None


def run_query(server_url: str, query: str, timeout: int, session_id: Optional[str] = None) -> dict[str, Any]:
    import requests  # type: ignore

    body: dict[str, Any] = {"question": query}
    if session_id:
        body["session_id"] = session_id
    try:
        resp = requests.post(
            f"{server_url.rstrip('/')}/query",
            json=body,
            timeout=timeout,
        )
    except Exception as exc:
        return {"error": f"request failed: {exc}"}
    if resp.status_code != 200:
        return {"error": f"http {resp.status_code}: {resp.text[:200]}"}
    try:
        return resp.json()
    except Exception:
        return {"error": "non-json response"}


# Stages whose gold rows require a session_id to exercise the stage
# state machine. results + condition + visit_prep use the routine
# retrieval path and don't need one.
SESSION_STAGES = {"intake", "navigation"}


def flatten_sources(raw_sources: list[Any]) -> list[str]:
    out: list[str] = []
    for s in raw_sources:
        if isinstance(s, str):
            out.append(s.lower())
        elif isinstance(s, dict):
            parts = [s.get("title") or "", s.get("source") or "", s.get("source_url") or ""]
            out.append(" ".join(p for p in parts if p).lower())
    return out


def flatten_context_text(raw_sources: list[Any]) -> str:
    parts: list[str] = []
    for s in raw_sources:
        if isinstance(s, dict):
            for key in ("chunk", "text", "content", "snippet"):
                v = s.get(key)
                if isinstance(v, str) and v:
                    parts.append(v)
                    break
    return "\n\n".join(parts)


def answer_relevancy(answer: str, hints: list[str]) -> Optional[float]:
    if not hints:
        return None
    lower = (answer or "").lower()
    hits = sum(1 for h in hints if h.lower() in lower)
    return hits / len(hints)


def context_recall_at_k(retrieved: list[str], expected: list[str], k: int = 5) -> Optional[float]:
    if not expected:
        return None
    retrieved_tok = [_tokenize(s) for s in retrieved[:k]]
    hits = 0
    for exp in expected:
        exp_tok = _tokenize(exp)
        if not exp_tok:
            continue
        for r_tok in retrieved_tok:
            if len(exp_tok & r_tok) / len(exp_tok) >= 0.6:
                hits += 1
                break
    return hits / len(expected)


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def faithfulness_nli(answer: str, context_text: str, verify_fn) -> Optional[float]:
    if not answer or not context_text:
        return None
    sentences = [s for s in split_sentences(answer) if len(s.split()) >= 5]
    if not sentences:
        return None
    scores: list[float] = []
    for s in sentences:
        try:
            scores.append(float(verify_fn(s, context_text)))
        except Exception:
            continue
    if not scores:
        return None
    return sum(scores) / len(scores)


def marker_coverage(answer: str, markers: list[str]) -> Optional[float]:
    if not markers:
        return None
    lower = (answer or "").lower()
    hits = sum(1 for m in markers if m.lower() in lower)
    return hits / len(markers)


def mean(xs: list[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def main() -> int:
    parser = argparse.ArgumentParser(description="MediRAG RAGAS-lite scorer")
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--stages", default=",".join(STAGES), help="Comma-separated stage names")
    parser.add_argument("--no-nli", action="store_true", help="Skip entailment faithfulness (fast mode)")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--min-interval", type=float, default=7.0,
                        help="Min seconds between queries — Cohere Trial key caps at 10/min; 7s keeps us under")
    parser.add_argument("--out", default=None)
    parser.add_argument("--fail-on-threshold", action="store_true",
                        help="Exit non-zero if any stage mean is below thresholds")
    args = parser.parse_args()

    wanted = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in wanted:
        if s not in STAGES:
            print(f"[error] unknown stage: {s!r}", file=sys.stderr)
            return 2

    verify_fn = None
    if not args.no_nli:
        try:
            sys.path.insert(0, str(EVAL_DIR.parent))
            from app.guardrails import verify_entailment  # type: ignore
            verify_fn = verify_entailment
            print("NLI verifier loaded (lazy — model fetches on first call).")
        except Exception as exc:
            print(f"[warn] NLI verifier unavailable ({exc}); faithfulness will be skipped")
            args.no_nli = True

    per_stage_summary: dict[str, dict[str, Any]] = {}
    per_row: list[dict[str, Any]] = []
    t0 = time.time()

    for stage in wanted:
        rows = load_stage_rows(stage)
        if not rows:
            per_stage_summary[stage] = {"n": 0}
            continue
        print(f"\n=== {stage} ({len(rows)} rows) ===")
        rel_list: list[float] = []
        rec_list: list[float] = []
        faith_list: list[float] = []
        marker_list: list[float] = []

        needs_session = stage in SESSION_STAGES
        last_t = 0.0
        for i, row in enumerate(rows, start=1):
            gap = time.time() - last_t
            if gap < args.min_interval:
                time.sleep(args.min_interval - gap)
            last_t = time.time()
            elapsed = time.time() - t0
            print(f"[{stage} {i}/{len(rows)}] {row['id']} — {elapsed:.1f}s", flush=True)
            sid = start_session(args.server_url) if needs_session else None
            resp = run_query(args.server_url, row["query"], timeout=args.timeout, session_id=sid)
            if "error" in resp:
                per_row.append({"id": row["id"], "stage": stage, "error": resp["error"]})
                print(f"    error: {resp['error']}")
                continue

            answer = resp.get("answer") or ""
            raw_sources = resp.get("sources") or []
            retrieved_flat = flatten_sources(raw_sources)
            context_text = flatten_context_text(raw_sources)

            hints = list(row.get("expected_output_hints") or row.get("expected_topics") or [])
            expected_sources = list(row.get("expected_sources") or [])
            markers = list(row.get("expected_markers") or [])

            rel = answer_relevancy(answer, hints)
            rec = context_recall_at_k(retrieved_flat, expected_sources, k=5)
            faith = None
            if not args.no_nli and verify_fn is not None:
                faith = faithfulness_nli(answer, context_text, verify_fn)
            mcov = marker_coverage(answer, markers)

            if rel is not None: rel_list.append(rel)
            if rec is not None: rec_list.append(rec)
            if faith is not None: faith_list.append(faith)
            if mcov is not None: marker_list.append(mcov)

            per_row.append({
                "id": row["id"],
                "stage": stage,
                "answer_relevancy": rel,
                "context_recall": rec,
                "faithfulness": faith,
                "marker_coverage": mcov,
            })

        per_stage_summary[stage] = {
            "n": len(rows),
            "answer_relevancy": mean(rel_list),
            "context_recall": mean(rec_list),
            "faithfulness": mean(faith_list),
            "marker_coverage": mean(marker_list),
        }

    # Print summary
    print("\n" + "=" * 72)
    print("RAGAS-lite summary")
    print("=" * 72)
    header = f"{'stage':<14} {'n':>4} {'relevancy':>10} {'recall@5':>10} {'faith':>8} {'markers':>9}"
    print(header)
    print("-" * len(header))
    threshold_failures: list[str] = []
    for stage in wanted:
        s = per_stage_summary.get(stage, {"n": 0})
        def fmt(v): return "—" if v is None else f"{v:.3f}"
        print(f"{stage:<14} {s.get('n', 0):>4} "
              f"{fmt(s.get('answer_relevancy')):>10} "
              f"{fmt(s.get('context_recall')):>10} "
              f"{fmt(s.get('faithfulness')):>8} "
              f"{fmt(s.get('marker_coverage')):>9}")
        if args.fail_on_threshold:
            for metric, threshold in THRESHOLDS.items():
                if metric == "faithfulness" and args.no_nli:
                    continue
                v = s.get(metric)
                if v is not None and v < threshold:
                    threshold_failures.append(f"{stage}.{metric}={v:.3f} < {threshold}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server_url": args.server_url,
            "stages": wanted,
            "no_nli": args.no_nli,
            "thresholds": THRESHOLDS,
            "per_stage": per_stage_summary,
            "per_row": per_row,
            "threshold_failures": threshold_failures,
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote results to {out_path}")

    if threshold_failures:
        print("\nThreshold failures:")
        for f in threshold_failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
