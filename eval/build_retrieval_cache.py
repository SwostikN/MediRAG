"""Retrieve once per question per retrieval config, and cache it (Phase 2.2a).

WHY THIS EXISTS
---------------
The Phase 3 ablation runs 7 arms x 200 questions x up to 3 seeds. Retrieval is the
expensive, rate-limited, non-deterministic-in-wall-clock part of that: MedCPT encoding,
a Supabase hybrid query, and a Cohere rerank call whose trial tier is capped at roughly
10 requests/minute. Re-retrieving per arm would mean thousands of rerank calls, and the
Cohere limit -- not the LLM -- would become the bottleneck.

It would also make the comparison WRONG. If arm A4 and arm A6 each retrieve separately,
any difference between them mixes guardrail effects with retrieval jitter. The arms must
see byte-identical context so the only thing varying is the layer under test. Caching is
what makes the ablation a controlled experiment rather than a correlated one.

CORRECTION TO THE PLAN
----------------------
The plan says "retrieve once per question and reuse across all arms". That is not quite
right: the arms do not all share one retrieval.

    A0                 no retrieval at all (context-free upper bound)
    A1                 dense-only, no rerank          <- its own config
    A2..A6             hybrid + rerank                <- shared config

So it is one retrieval per (question, retrieval_config) -- two configs, not one, and not
seven. A1 exists precisely to show what reranking buys, so giving it A2's context would
erase the thing it is there to measure.

WHAT IS CACHED
--------------
Full ranked rows for both configs, with scores, so downstream arms can slice to
CONTEXT_CHUNKS and apply the abstention gate themselves without touching the network.
The abstention gate (A3+) is a threshold on rerank_score, so it is a pure function of the
cached rows -- no re-retrieval needed to add or remove it.

TWO INPUT SHAPES
----------------
`nepal_navigation` items carry an intake_summary rather than a question (see
eval/build_question_set.py). Retrieval runs against that summary, which is what
app/RAG.py:_retrieve_ranked already does for Stage 2, so semantics match production.

RESUMABLE
---------
Appends one JSON line per (question, config) and skips anything already present. Safe to
interrupt and re-run; a rate-limit stall costs only the item in flight.

USAGE
    .venv\\Scripts\\python.exe eval/build_retrieval_cache.py
    .venv\\Scripts\\python.exe eval/build_retrieval_cache.py --limit 5     # smoke test
    .venv\\Scripts\\python.exe eval/build_retrieval_cache.py --configs dense
    .venv\\Scripts\\python.exe eval/build_retrieval_cache.py --per-item-delay 6.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTIONS = ROOT / "eval" / "question_set_v1.jsonl"
CACHE = ROOT / "eval" / "results" / "retrieval_cache_v1.jsonl"

CONFIGS = ("dense", "hybrid_rerank")

# Fields kept per row. Deliberately includes the raw text: the cache must be able to
# rebuild an LLM context with no further network access, otherwise a later Cohere outage
# or corpus change would silently alter what the arms see.
ROW_FIELDS = ("chunk_id", "doc_id", "content", "doc_title", "doc_source",
              "source_url", "authority_tier", "publication_date",
              "similarity", "rerank_score", "final_score", "ord")


def slim(row: dict) -> dict:
    return {k: row.get(k) for k in ROW_FIELDS if k in row}


def load_questions(limit: int | None) -> list[dict]:
    if not QUESTIONS.exists():
        sys.exit(f"missing {QUESTIONS}\nRun: python eval/build_question_set.py")
    rows = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8") if l.strip()]
    return rows[:limit] if limit else rows


def load_done() -> set[tuple[str, str]]:
    if not CACHE.exists():
        return set()
    done = set()
    for line in CACHE.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a torn final line from an interrupted run
        if not r.get("error"):
            done.add((r["question_id"], r["config"]))
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help=f"comma-separated subset of {CONFIGS}")
    ap.add_argument("--per-item-delay", type=float, default=6.5,
                    help="seconds between retrievals; Cohere trial rerank is ~10 rpm "
                         "so 6.5s is the safe default. Lower it on a paid tier.")
    ap.add_argument("--force", action="store_true", help="ignore the cache and redo all")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    bad = [c for c in configs if c not in CONFIGS]
    if bad:
        sys.exit(f"unknown config(s): {bad}; valid: {CONFIGS}")

    questions = load_questions(args.limit)
    done = set() if args.force else load_done()

    todo = [(q, c) for q in questions for c in configs
            if (q["id"], c) not in done]
    print(f"questions {len(questions)} x configs {len(configs)} = "
          f"{len(questions)*len(configs)} pairs; {len(done)} cached, {len(todo)} to do")
    if not todo:
        print("nothing to do")
        return 0
    eta = len(todo) * args.per_item_delay / 60
    print(f"pacing {args.per_item_delay}s/item -> ~{eta:.0f} min minimum\n")

    # Imported here, after argument parsing, so --help does not pay the ~50s model load.
    from app import RAG
    from app.supabase_client import match_chunks
    from eval.run_manifest import build_manifest

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if args.force and CACHE.exists():
        CACHE.unlink()

    n_ok = n_err = 0
    started = time.time()
    with CACHE.open("a", encoding="utf-8", newline="\n") as fh:
        for i, (q, config) in enumerate(todo, 1):
            text = q.get("input") or ""
            rec = {
                "question_id": q["id"], "config": config,
                "stratum": q["stratum"], "input_kind": q["input_kind"],
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
            t0 = time.time()
            try:
                if config == "dense":
                    # A1: dense-only, no filter, no rerank. This is the "naive RAG"
                    # baseline, so it deliberately skips build_filter as well.
                    vec = RAG.get_query_encoder().encode_one(text)
                    rows = match_chunks(RAG.to_pgvector_literal(vec),
                                        match_count=RAG.RETRIEVE_K)
                else:
                    rows = RAG._retrieve_ranked(text)
                rec["n_rows"] = len(rows)
                rec["rows"] = [slim(r) for r in rows]
                rec["top_rerank_score"] = max(
                    (r.get("rerank_score") or 0) for r in rows) if rows else None
                n_ok += 1
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
                rec["traceback"] = traceback.format_exc()[-800:]
                n_err += 1
                print(f"  [{i}/{len(todo)}] {q['id']} {config} ERROR {rec['error']}")
            rec["seconds"] = round(time.time() - t0, 2)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()  # checkpoint every item, so a kill costs at most one retrieval

            if i % 10 == 0 or i == len(todo):
                rate = (time.time() - started) / i
                print(f"  [{i}/{len(todo)}] ok={n_ok} err={n_err} "
                      f"{rate:.1f}s/item eta {(len(todo)-i)*rate/60:.0f} min", flush=True)
            if i < len(todo) and args.per_item_delay:
                time.sleep(args.per_item_delay)

    manifest = build_manifest(notes="retrieval cache v1",
                             configs=configs, questions=len(questions),
                             ok=n_ok, errors=n_err)
    (CACHE.with_suffix(".manifest.json")).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\ncached {n_ok} ok, {n_err} errors -> {CACHE.relative_to(ROOT)}")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
