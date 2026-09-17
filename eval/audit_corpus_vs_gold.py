"""Audit how much of the gold `expected_sources` is gradable against the corpus.

WHY THIS EXISTS
---------------
Retrieval Recall@k can only be scored for a gold row when at least one corpus document
plausibly corresponds to the source the row expects. In April 2026 an ad-hoc audit found
only 21.4% of expected sources were matchable against a 131-document corpus, and retrieval
recall was consequently dropped as a headline metric (docs/DOCUMED.md §11.22).

That audit left behind `eval/reports/corpus_vs_gold_audit.json` but no script, so the
number could not be reproduced when the corpus grew. This script reconstructs the method
and makes it repeatable.

METHOD (reconstructed from the surviving JSON and docs/DOCUMED.md:6643)
    "Substring + token-overlap match of every expected_sources string across the six gold
     files against the full documents catalog."

For each gold source string, score it against every corpus document by token overlap:

    score = |tokens(gold) & tokens(title + " " + source)| / |tokens(gold)|

keeping the best-scoring document. A source is "gradable" at score >= 0.6. Verified
against two rows of the original output, which this reproduces exactly:
    "EDCD Nepal"  -> "Dengue Control Program (Nepal)" / EDCD  = 2/2 = 1.0
    "AAO patient" -> "NPHL Patient Frequently Asked Questions" = 1/2 = 0.5

TWO DENOMINATORS, because the original reported both and they are not the same number:
  * headline  — distinct source strings pooled across all gold files (518 originally)
  * per-stage — every reference occurrence in that file, duplicates included

USAGE
    .venv\\Scripts\\python.exe eval/audit_corpus_vs_gold.py            # offline, uses snapshot
    .venv\\Scripts\\python.exe eval/audit_corpus_vs_gold.py --live     # query Supabase instead
    .venv\\Scripts\\python.exe eval/audit_corpus_vs_gold.py --out FILE.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "eval" / "corpus_snapshot" / "documents.jsonl.gz"
GOLD_DIR = ROOT / "eval" / "gold"

# The six gold files the original audit covered. redflag.jsonl is excluded on purpose:
# the red-flag engine is deterministic YAML and never performs retrieval.
GOLD_STAGES = ["navigation_stage2", "navigation", "visit_prep",
               "results", "condition", "intake"]

GRADABLE_AT = 0.6

# Reference point from the April 2026 audit (corpus_size 131), for the delta table.
ORIGINAL = {
    "corpus_size": 131, "unique_expected": 518, "matchable_count": 111,
    "per_stage": {"navigation_stage2": 0.333, "navigation": 0.190, "visit_prep": 0.167,
                  "results": 0.319, "condition": 0.247, "intake": 0.123},
}


def tokens(s: str) -> set[str]:
    """Alphanumeric tokens longer than two characters, lowercased."""
    cleaned = "".join(c if c.isalnum() else " " for c in str(s).lower())
    return {w for w in cleaned.split() if len(w) > 2}


def load_docs_offline() -> list[dict]:
    if not SNAPSHOT.exists():
        sys.exit(f"no snapshot at {SNAPSHOT}\nRun: python eval/export_corpus.py")
    return [json.loads(line) for line in gzip.open(SNAPSHOT, "rt", encoding="utf-8")]


def load_docs_live() -> list[dict]:
    import os
    import requests
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(ROOT / ".env"))
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        sys.exit("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing from .env")
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    out: list[dict] = []
    offset = 0
    while True:
        r = requests.get(f"{url}/rest/v1/documents",
                         params={"select": "title,source", "limit": "1000",
                                 "offset": str(offset), "order": "doc_id.asc"},
                         headers=headers, timeout=120)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        offset += 1000
        if len(batch) < 1000:
            break
    return out


def best_match(gold: str, catalog: list[tuple[set[str], dict]]) -> tuple[float, dict | None]:
    gt = tokens(gold)
    if not gt:
        return 0.0, None
    best_score, best_doc = 0.0, None
    for dt, doc in catalog:
        score = len(gt & dt) / len(gt)
        if score > best_score:
            best_score, best_doc = score, doc
            if score == 1.0:
                break
    return best_score, best_doc


def like_for_like(catalog) -> dict | None:
    """Score the ORIGINAL April-2026 gold strings against the current corpus.

    The gold files were rewritten on 2026-04-21 (eval/scripts/apply_gold_rewrite_
    2026_04_21.py): 365 of 604 source entries were dropped as invalid labels and 30 were
    substituted with corpus-aligned titles. Gradability measured on the rewritten gold
    therefore cannot be compared with the April figure — the denominator lost its hardest
    entries.

    The old report preserved all 518 original strings, so scoring those against today's
    corpus isolates what corpus growth alone actually bought.
    """
    old_path = ROOT / "eval" / "reports" / "corpus_vs_gold_audit.json"
    if not old_path.exists():
        return None
    old = json.loads(old_path.read_text(encoding="utf-8"))
    strings = [r["gold"] for r in old.get("matchable", [])] + \
              [r["gold"] for r in old.get("unmatchable", [])]
    if not strings:
        return None
    n = sum(1 for g in strings if best_match(g, catalog)[0] >= GRADABLE_AT)
    return {
        "april_gold_strings": len(strings),
        "matchable_april_corpus": old["matchable_count"],
        "pct_april_corpus": round(old["matchable_count"] / len(strings), 4),
        "matchable_now": n,
        "pct_now": round(n / len(strings), 4),
        "delta_pp": round(100 * (n / len(strings) - old["matchable_count"] / len(strings)), 2),
        "interpretation": (
            "Corpus grew 131 -> 1009 documents (7.7x), almost entirely MedlinePlus "
            "patient-education pages, which do not correspond to the specific sources "
            "the gold rows cite (WHO IMAI/mhGAP/IMCI, named NHS topics, MoHP SOPs). "
            "Gradability is therefore essentially unchanged."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true",
                    help="query Supabase instead of the local snapshot")
    ap.add_argument("--out", type=Path, help="write the full report as JSON")
    args = ap.parse_args()

    docs = load_docs_live() if args.live else load_docs_offline()
    catalog = [(tokens(f"{d.get('title') or ''} {d.get('source') or ''}"), d) for d in docs]
    print(f"corpus: {len(docs):,} documents "
          f"({'live Supabase' if args.live else 'local snapshot'})\n")

    per_stage: dict[str, dict] = {}
    global_refs: Counter = Counter()          # distinct string -> total occurrences
    stage_of: dict[str, set[str]] = {}

    for stage in GOLD_STAGES:
        path = GOLD_DIR / f"{stage}.jsonl"
        if not path.exists():
            continue
        occurrences: Counter = Counter()
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            for src in (json.loads(line).get("expected_sources") or []):
                occurrences[src] += 1
                global_refs[src] += 1
        stage_of[stage] = set(occurrences)
        per_stage[stage] = {"occurrences": occurrences}

    # Score every distinct gold string once.
    scored: dict[str, tuple[float, dict | None]] = {}
    for i, gold in enumerate(global_refs, 1):
        scored[gold] = best_match(gold, catalog)
        print(f"  scoring {i}/{len(global_refs)}", end="\r", flush=True)
    print(" " * 40, end="\r")

    matchable, unmatchable = [], []
    for gold, refs in global_refs.most_common():
        score, doc = scored[gold]
        row = {"gold": gold, "refs": refs, "score": round(score, 3),
               "best_title": (doc or {}).get("title"),
               "best_source": (doc or {}).get("source")}
        (matchable if score >= GRADABLE_AT else unmatchable).append(row)

    # Per-stage, counted over occurrences (duplicates included) as the original did.
    stage_rows = []
    for stage, data in per_stage.items():
        occ = data["occurrences"]
        total = sum(occ.values())
        gradable = sum(n for src, n in occ.items() if scored[src][0] >= GRADABLE_AT)
        pct = gradable / total if total else 0.0
        data.clear()
        data.update({"gradable": gradable, "total": total, "pct": round(pct, 3)})
        stage_rows.append((stage, gradable, total, pct, ORIGINAL["per_stage"].get(stage)))

    head_n, head_d = len(matchable), len(global_refs)
    head_pct = head_n / head_d if head_d else 0.0
    orig_pct = ORIGINAL["matchable_count"] / ORIGINAL["unique_expected"]

    print(f"{'stage':<20}{'gradable':>10}{'total':>8}{'now':>9}")
    for stage, g, t, pct, _ in sorted(stage_rows, key=lambda r: -r[3]):
        print(f"{stage:<20}{g:>10}{t:>8}{pct:>8.1%}")
    print(f"{'CURRENT GOLD':<20}{head_n:>10}{head_d:>8}{head_pct:>8.1%}")

    like = like_for_like(catalog)
    print(f"\ncorpus: {ORIGINAL['corpus_size']} -> {len(docs):,} documents")
    print("\n" + "=" * 74)
    print("DO NOT compare the figure above with April 2026's 21.4%. Different gold set.")
    print("=" * 74)
    if like:
        print(f"  April gold set (518 strings) vs April corpus (131 docs) : "
              f"{ORIGINAL['matchable_count']}/518 = {orig_pct:.1%}")
        print(f"  April gold set (518 strings) vs THIS corpus ({len(docs):,} docs): "
              f"{like['matchable_now']}/518 = {like['pct_now']:.1%}")
        print(f"  -> like-for-like effect of corpus growth: "
              f"{(like['pct_now'] - orig_pct) * 100:+.1f} percentage points")
    print("\n  The current-gold figure is higher mainly because the 2026-04-21 gold")
    print("  rewrite DROPPED 365 of 604 source entries (60%) as invalid labels and")
    print("  SUBSTITUTED 30 more (5%) with corpus-aligned titles. Removing the")
    print("  ungradable entries raises the percentage by definition. Treat retrieval")
    print("  recall as a caveated secondary metric, not a headline result.")

    report = {
        "WARNING": (
            "matchable_pct is computed on the post-2026-04-21 gold. That rewrite dropped "
            "60% of source entries and substituted 5% with corpus-aligned titles, so this "
            "figure is NOT comparable to the April 2026 21.4%. See like_for_like below, "
            "which scores the original 518 April strings against the current corpus and "
            "isolates the actual effect of corpus growth."
        ),
        "like_for_like": like,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                  capture_output=True, text=True).stdout.strip() or "unknown",
        "source": "live" if args.live else "snapshot",
        "corpus_size": len(docs),
        "gradable_threshold": GRADABLE_AT,
        "unique_expected": head_d,
        "matchable_count": head_n,
        "matchable_pct": round(head_pct, 4),
        "comparison_april_2026": ORIGINAL,
        "per_stage": {s: d for s, d in per_stage.items()},
        "matchable": matchable,
        "unmatchable": unmatchable,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
