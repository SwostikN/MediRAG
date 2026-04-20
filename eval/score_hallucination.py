"""Hallucination-rate scorer — Phase 2a, Metric 1.

Computes: NLI-failed sentences per 1000 emitted sentences against a gold
set. This is the "deferred metric 1" from docs/HALLUCINATION_ZERO_PLAN.md
§7, now implemented.

Definition of a hallucination (per the plan + app/guardrails.py):
  A sentence fails iff
      classify_claim(s).requires_nli == True
    AND
      max_{c in top-k retrieved chunks} P(entailment | premise=c, hypothesis=s)
        < 0.5        (threshold _NLI_SOFTEN_BELOW in guardrails.py)

Rate is reported per-1000 *emitted* sentences (the standard unit in the
plan), not per classifier-eligible sentences. That matches how we'd ship
the metric: "this release emits N hallucinated sentences per 1000 produced."

Reuses the production stack end-to-end:
  - app.RAG._retrieve_ranked           (same retrieval as /query)
  - app.RAG.CONTEXT_CHUNKS             (same top-k slice)
  - app.RAG.MEDIRAG_SYSTEM_PROMPT      (same system prompt)
  - app.RAG.MAIN_QUERY_TEMPERATURE     (same temp)
  - Groq primary, Cohere fallback      (same chain)
  - app.guardrails.verify_entailment   (same NLI model)
  - app.guardrails.classify_claim      (same claim classifier)
  - app.guardrails._split_sentences    (same sentence splitter)

CLI
---
  # Quick smoke test (5 cases) before burning quota:
  python eval/score_hallucination.py --gold eval/gold/coverage.jsonl --limit 5

  # Full run against coverage.jsonl:
  python eval/score_hallucination.py --gold eval/gold/coverage.jsonl

  # Any gold file under eval/gold/ is accepted:
  python eval/score_hallucination.py --gold eval/gold/navigation.jsonl

  # Score an already-generated run (no retrieval, no LLM calls) — the
  # input jsonl must have fields: case_id, question, answer, chunks
  # (list of strings). Use this to re-score after threshold changes
  # without re-burning Cohere quota.
  python eval/score_hallucination.py --no-retrieval \
      --precomputed eval/baselines/hallucination_2026-04-20.precomputed.jsonl

Output
------
  eval/baselines/hallucination_<YYYY-MM-DD>.json
  (or --out <path> to override)

Exit codes
----------
  0 : run completed, OR Cohere quota exhausted (partial progress saved —
      this is a *budget* condition, not a code failure, per the plan)
  2 : CLI / config error (missing gold file, bad arg)
  3 : unrecoverable code failure (NLI model refuses to load, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
GOLD_DIR = EVAL_DIR / "gold"
BASELINES_DIR = EVAL_DIR / "baselines"

# Mirror of app.guardrails._NLI_SOFTEN_BELOW. Kept as a local constant so
# this scorer is pinned to a specific failure threshold for the recorded
# baseline — if guardrails.py tightens its threshold later, the old
# baselines stay interpretable.
NLI_FAIL_THRESHOLD = 0.5

# Strings in an exception message that mean "quota / rate-limit" rather
# than "the code is broken". Substring match is intentionally broad:
# Cohere, Groq, and the HTTP layer all phrase this differently.
_QUOTA_HINTS = (
    "quota",
    "rate limit",
    "rate_limit",
    "429",
    "too many requests",
    "tokens per day",
    "tpd",
    "insufficient_quota",
    "trial key",
)


def _is_quota_error(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(h in msg for h in _QUOTA_HINTS)


def load_gold(path: Path) -> list[dict[str, Any]]:
    """Load a gold jsonl. All 9 files under eval/gold/ share the
    (id, query, ...) shape — we only need those two fields here."""
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


def load_precomputed(path: Path) -> list[dict[str, Any]]:
    """Load --no-retrieval input. Expected schema per line:
        {case_id, question, answer, chunks: [str, ...]}
    Extra fields are ignored."""
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[error] {path}:{line_no} invalid JSON — {exc}", file=sys.stderr)
            continue
        if "question" not in rec or "answer" not in rec or "chunks" not in rec:
            print(
                f"[error] {path}:{line_no} missing required field "
                "(question/answer/chunks)",
                file=sys.stderr,
            )
            continue
        rows.append(rec)
    return rows


# ── Generation helpers (Groq primary → Cohere fallback) ──────────────────
#
# Kept local rather than imported because app.RAG's generate paths are
# fused with /query request plumbing (rate-limit check, session state,
# scope-guard, logging). Replicating the 20-line call pattern here gives
# us a clean cell to call from a script without mocking a FastAPI request.


def _generate_answer(question: str, context_text: str, *, RAG_mod) -> tuple[str, str]:
    """Return (answer, provider). Provider is one of 'groq', 'cohere_fallback',
    'cohere'. Raises on unrecoverable generation failure — caller decides
    whether to treat it as quota-exhausted or code-broken."""
    messages: list[dict] = [
        {"role": "system", "content": RAG_mod.MEDIRAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Sources:\n{context_text}\n\nQuestion: {question}",
        },
    ]
    temp = RAG_mod.MAIN_QUERY_TEMPERATURE

    if RAG_mod.groq_client is not None:
        try:
            resp = RAG_mod.groq_client.chat.completions.create(
                model=RAG_mod.GROQ_MODEL,
                messages=messages,
                max_tokens=250,
                temperature=temp,
            )
            return resp.choices[0].message.content or "", "groq"
        except Exception as exc:
            if _is_quota_error(exc):
                # Re-raise so the outer loop can save partial progress.
                raise
            print(f"[groq] generate failed, falling back to Cohere: {exc}")

    response = RAG_mod.co.chat(
        model="command-r-08-2024",
        messages=messages,
        max_tokens=250,
        temperature=temp,
    )
    try:
        answer = response.message.content[0].text
    except Exception as exc:
        raise RuntimeError(f"Failed to parse Cohere response: {exc}") from exc
    provider = "cohere_fallback" if RAG_mod.groq_client is not None else "cohere"
    return answer, provider


# ── Hallucination scoring ────────────────────────────────────────────────


def _score_answer(
    answer: str,
    chunk_texts: list[str],
    *,
    classify_claim_fn,
    verify_entailment_fn,
    split_sentences_fn,
) -> dict[str, Any]:
    """Walk the answer sentence-by-sentence, run claim classifier + NLI,
    return per-sentence + aggregate stats. Mirrors the logic in
    apply_guardrails but records max P(entail) and per-sentence outcomes
    instead of filtering the answer."""
    sentences = split_sentences_fn(answer)
    n_sentences = len(sentences)
    n_claim = 0
    n_failed = 0
    failed_sentences: list[dict[str, Any]] = []

    for s in sentences:
        feats = classify_claim_fn(s)
        if not feats.requires_nli:
            continue
        n_claim += 1
        best_p = 0.0
        best_idx = -1
        for i, ct in enumerate(chunk_texts):
            try:
                p = float(verify_entailment_fn(s, ct))
            except Exception as exc:
                # Fail-open matching guardrails.py: NLI failure doesn't
                # get counted as a hallucination (we can't prove it either
                # way). Record the error for the operator.
                print(f"[nli] verify_entailment failed on 1 chunk: {exc}")
                continue
            if p > best_p:
                best_p = p
                best_idx = i
        if best_p < NLI_FAIL_THRESHOLD:
            n_failed += 1
            failed_sentences.append({
                "text": s,
                "max_entailment": round(best_p, 3),
                "assigned_chunk_idx": best_idx,
            })
    return {
        "n_sentences": n_sentences,
        "n_claim_sentences": n_claim,
        "n_failed": n_failed,
        "failed_sentences": failed_sentences,
    }


# ── Main loop ────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MediRAG hallucination-rate scorer (Phase 2a, Metric 1)"
    )
    parser.add_argument(
        "--gold",
        default=str(GOLD_DIR / "coverage.jsonl"),
        help="Path to gold jsonl under eval/gold/ (default: coverage.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N cases (for quota-safe smoke tests).",
    )
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Skip retrieval+generation; read pre-computed answers+chunks.",
    )
    parser.add_argument(
        "--precomputed",
        default=None,
        help="jsonl with {case_id, question, answer, chunks} "
             "(required when --no-retrieval).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: eval/baselines/hallucination_<YYYY-MM-DD>.json)",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=7.0,
        help="Min seconds between generate calls — Cohere trial caps at "
             "10/min; 7s keeps us under. Ignored with --no-retrieval.",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    if not args.no_retrieval:
        if not gold_path.exists():
            print(f"[error] gold file not found: {gold_path}", file=sys.stderr)
            return 2
    else:
        if not args.precomputed:
            print(
                "[error] --no-retrieval requires --precomputed <path>",
                file=sys.stderr,
            )
            return 2
        if not Path(args.precomputed).exists():
            print(
                f"[error] precomputed file not found: {args.precomputed}",
                file=sys.stderr,
            )
            return 2

    out_path = Path(args.out) if args.out else (
        BASELINES_DIR / f"hallucination_{date.today().isoformat()}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Lazy imports — these pull in the 400 MB NLI model + DB clients.
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from app.guardrails import (  # type: ignore
            classify_claim,
            verify_entailment,
            _split_sentences,
        )
    except Exception as exc:
        print(f"[fatal] failed to import app.guardrails: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3

    RAG_mod = None
    if not args.no_retrieval:
        try:
            from app import RAG as RAG_mod  # type: ignore
        except Exception as exc:
            print(f"[fatal] failed to import app.RAG: {exc}", file=sys.stderr)
            traceback.print_exc()
            return 3

    # Build the work list.
    if args.no_retrieval:
        precomputed_rows = load_precomputed(Path(args.precomputed))
        work_items: list[dict[str, Any]] = [
            {
                "case_id": rec.get("case_id") or rec.get("id") or f"pre-{i}",
                "question": rec["question"],
                "precomputed_answer": rec["answer"],
                "precomputed_chunks": list(rec["chunks"] or []),
            }
            for i, rec in enumerate(precomputed_rows)
        ]
    else:
        gold_rows = load_gold(gold_path)
        work_items = [
            {
                "case_id": row.get("id") or f"case-{i}",
                "question": row.get("query") or row.get("question") or "",
            }
            for i, row in enumerate(gold_rows)
        ]

    if args.limit is not None:
        work_items = work_items[: args.limit]

    print(
        f"[hallucination] gold={gold_path.name} n={len(work_items)} "
        f"mode={'no-retrieval' if args.no_retrieval else 'live'}",
        file=sys.stderr,
    )

    per_example: list[dict[str, Any]] = []
    n_sent_total = 0
    n_claim_total = 0
    n_failed_total = 0

    quota_exhausted = False
    last_gen_t = 0.0
    started = time.time()

    for i, item in enumerate(work_items, start=1):
        case_id = item["case_id"]
        question = item["question"]
        try:
            if args.no_retrieval:
                answer = item["precomputed_answer"]
                chunk_texts = item["precomputed_chunks"]
                provider = "precomputed"
            else:
                # min-interval throttle (same pattern as score_ragas_lite).
                gap = time.time() - last_gen_t
                if gap < args.min_interval:
                    time.sleep(args.min_interval - gap)
                last_gen_t = time.time()

                rows = RAG_mod._retrieve_ranked(question)
                top_rows = rows[: RAG_mod.CONTEXT_CHUNKS]
                chunk_texts = [
                    r.get("content", "") for r in top_rows if r.get("content")
                ]
                context_blocks = []
                for j, r in enumerate(top_rows, start=1):
                    heading = r.get("section_heading") or ""
                    title = r.get("doc_title") or r.get("doc_source") or "source"
                    context_blocks.append(
                        f"[src:{j}] {title} — {heading}\n{r.get('content', '')}".strip()
                    )
                context_text = "\n\n".join(context_blocks)
                answer, provider = _generate_answer(
                    question, context_text, RAG_mod=RAG_mod
                )
        except Exception as exc:
            if _is_quota_error(exc):
                print(
                    f"[quota] exhausted at case {i}/{len(work_items)} "
                    f"({case_id}): {exc}",
                    file=sys.stderr,
                )
                quota_exhausted = True
                break
            print(
                f"[warn] {case_id}: generation/retrieval failed: {exc}",
                file=sys.stderr,
            )
            per_example.append({
                "case_id": case_id,
                "question": question,
                "answer": "",
                "n_sentences": 0,
                "n_failed": 0,
                "failed_sentences": [],
                "error": str(exc)[:300],
            })
            continue

        stats = _score_answer(
            answer,
            chunk_texts,
            classify_claim_fn=classify_claim,
            verify_entailment_fn=verify_entailment,
            split_sentences_fn=_split_sentences,
        )

        per_example.append({
            "case_id": case_id,
            "question": question,
            "answer": answer,
            "provider": provider,
            "n_sentences": stats["n_sentences"],
            "n_claim_sentences": stats["n_claim_sentences"],
            "n_failed": stats["n_failed"],
            "failed_sentences": stats["failed_sentences"],
        })
        n_sent_total += stats["n_sentences"]
        n_claim_total += stats["n_claim_sentences"]
        n_failed_total += stats["n_failed"]

        elapsed = time.time() - started
        print(
            f"[{i}/{len(work_items)}] {case_id} "
            f"sent={stats['n_sentences']} claims={stats['n_claim_sentences']} "
            f"failed={stats['n_failed']} ({elapsed:.1f}s)",
            flush=True,
        )

    rate_per_1000 = (
        (n_failed_total / n_sent_total) * 1000.0 if n_sent_total else 0.0
    )

    payload = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "gold_file": str(gold_path),
        "mode": "no-retrieval" if args.no_retrieval else "live",
        "nli_fail_threshold": NLI_FAIL_THRESHOLD,
        "n_cases": len(per_example),
        "n_cases_planned": len(work_items),
        "n_sentences_total": n_sent_total,
        "n_sentences_classified_as_claim": n_claim_total,
        "n_failed": n_failed_total,
        "rate_per_1000": round(rate_per_1000, 3),
        "quota_exhausted": quota_exhausted,
        "per_example": per_example,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        f"\n[hallucination] wrote {out_path} — "
        f"rate_per_1000={rate_per_1000:.2f} "
        f"(failed={n_failed_total} / total_sentences={n_sent_total}, "
        f"claim_sentences={n_claim_total})",
        file=sys.stderr,
    )

    if quota_exhausted:
        # Per the plan: budget exhaustion is not a code failure. Save
        # partial progress, print a resumable marker, exit 0.
        resumable_at = len(per_example)
        print(
            f"[hallucination] resumable at N={resumable_at} "
            "(re-run with --limit offset via your own slicing, or rerun "
            "after quota reset). Exit 0 — budget condition, not bug.",
            file=sys.stderr,
        )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
