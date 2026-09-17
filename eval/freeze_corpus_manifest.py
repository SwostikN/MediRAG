"""Freeze the corpus record for the paper, and close the provenance gap.

WHY THIS EXISTS
---------------
`ingest/manifest/*.jsonl` is supposed to define which documents make up the corpus.
Reconciling the manifests against the live corpus showed they do not:

    manifest URLs present in corpus   185 / 210   (88%)
    manifest URLs never ingested       25
    corpus documents from NO manifest 824 / 1009  (82%)  -- all MedlinePlus

The 824 MedlinePlus documents were bulk-ingested from the cached topic index
(`ingest/cache/mplus_topics.xml`) rather than from a URL list, so the corpus could not be
described, audited, or rebuilt from the manifests alone. For a paper whose central claim
rests on a fixed corpus, "82% of it is undocumented" is not an acceptable position.

WHAT THIS WRITES
    ingest/manifest/medlineplus_bulk_v1.jsonl
        A real manifest for the 824 bulk-ingested documents, derived from the corpus
        itself, in the same schema as the hand-written manifests. After this, every
        document in the corpus is accounted for by some manifest.

    eval/corpus_manifest.json
        The frozen, paper-facing record: counts, composition breakdowns, the
        manifest reconciliation, and the Nepal-scope figures that Table 1 needs.

NOTE ON WHAT THIS IS AND IS NOT
    The derived manifest documents what WAS ingested. It is a record, not a recipe --
    re-running it would re-fetch those URLs as they exist today, which is not the same as
    the frozen corpus. The authoritative copy of the corpus remains
    eval/corpus_snapshot/, which carries the exact text and embeddings.

USAGE
    .venv\\Scripts\\python.exe eval/freeze_corpus_manifest.py
    .venv\\Scripts\\python.exe eval/freeze_corpus_manifest.py --check   # no writes
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "eval" / "corpus_snapshot" / "documents.jsonl.gz"
MANIFEST_DIR = ROOT / "ingest" / "manifest"
DERIVED = MANIFEST_DIR / "medlineplus_bulk_v1.jsonl"
OUT = ROOT / "eval" / "corpus_manifest.json"

# Manifest row schema, matching the hand-written files (e.g. seed_v1.jsonl).
MANIFEST_FIELDS = ["source", "source_url", "title", "authority_tier",
                   "doc_type", "domains", "population", "country_scope", "language"]


def norm_url(u: str | None) -> str:
    u = (u or "").strip().lower().rstrip("/")
    for prefix in ("https://", "http://"):
        if u.startswith(prefix):
            u = u[len(prefix):]
    return u[4:] if u.startswith("www.") else u


def load_docs() -> list[dict]:
    if not SNAPSHOT.exists():
        raise SystemExit(f"no snapshot at {SNAPSHOT}\nRun: python eval/export_corpus.py")
    return [json.loads(line) for line in gzip.open(SNAPSHOT, "rt", encoding="utf-8")]


def load_manifests() -> dict[str, str]:
    """normalised URL -> manifest filename (excluding the derived one)."""
    out: dict[str, str] = {}
    for path in sorted(MANIFEST_DIR.glob("*.jsonl")):
        if path.name == DERIVED.name:
            continue
        for line in path.open(encoding="utf-8"):
            if line.strip():
                out[norm_url(json.loads(line).get("source_url"))] = path.name
    return out


def breakdown(rows: list[dict], field: str) -> dict[str, int]:
    c: Counter = Counter()
    for r in rows:
        v = r.get(field)
        c[json.dumps(v) if isinstance(v, list) else str(v)] += 1
    return dict(c.most_common())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    docs = load_docs()
    manifests = load_manifests()
    doc_urls = {norm_url(d.get("source_url")) for d in docs}

    ingested = sorted(u for u in manifests if u in doc_urls)
    missing = sorted(u for u in manifests if u not in doc_urls)
    unaccounted = [d for d in docs if norm_url(d.get("source_url")) not in manifests]

    print(f"corpus documents                  : {len(docs):,}")
    print(f"manifest URLs (deduped)           : {len(manifests)}")
    print(f"  present in corpus               : {len(ingested)} ({len(ingested)/len(manifests):.0%})")
    print(f"  never ingested                  : {len(missing)}")
    print(f"corpus docs from no HAND-WRITTEN manifest : {len(unaccounted)} "
          f"({len(unaccounted)/len(docs):.0%})")
    by_source = Counter(d.get("source") for d in unaccounted)
    for s, n in by_source.most_common():
        print(f"    {str(s):<24} {n:>4}")
    # The derived manifest is excluded from the reconciliation above on purpose, so this
    # figure always reports hand-written coverage. State the resulting total explicitly,
    # or the output reads as though the gap were never closed.
    covered = len(ingested) + (len(unaccounted) if DERIVED.exists() else 0)
    print(f"\ncoverage INCLUDING {DERIVED.name}: {covered}/{len(docs)} "
          f"({covered/len(docs):.0%}) "
          f"{'-- every document accounted for' if covered >= len(docs) else '-- GAP REMAINS'}")

    missing_by_file = Counter(manifests[u] for u in missing)
    if missing_by_file:
        print("\nnever-ingested manifest URLs, by file:")
        for f, n in missing_by_file.most_common():
            print(f"    {f:<34} {n:>3}")

    # ── derive a manifest for the unaccounted documents ────────────────────
    derived_rows = [
        {f: (d.get("source_url") if f == "source_url" else d.get(f)) for f in MANIFEST_FIELDS}
        for d in sorted(unaccounted, key=lambda d: norm_url(d.get("source_url")))
    ]

    if not args.check and derived_rows:
        with DERIVED.open("w", encoding="utf-8", newline="\n") as fh:
            for row in derived_rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        readme = DERIVED.with_suffix("").with_suffix("") .parent / "medlineplus_bulk_v1.README.md"
        readme.write_text(
            "# MedlinePlus bulk v1 — DERIVED, not hand-written\n\n"
            f"{len(derived_rows)} rows, generated by `eval/freeze_corpus_manifest.py` on "
            f"{datetime.now(timezone.utc).date()}.\n\n"
            "## Why this file exists\n\n"
            "These documents were bulk-ingested from the cached MedlinePlus topic index\n"
            "(`ingest/cache/mplus_topics.xml`) rather than from a URL manifest, so 82% of\n"
            "the corpus was undocumented. This file reconstructs that provenance from the\n"
            "corpus itself so every document is accounted for by some manifest.\n\n"
            "## What it is and is not\n\n"
            "It is a **record of what was ingested**, not a recipe. Re-running it would\n"
            "re-fetch these URLs as they exist today, which is not the same as the frozen\n"
            "corpus. The authoritative copy of the corpus is `eval/corpus_snapshot/`,\n"
            "which carries the exact text and embeddings the paper's numbers were\n"
            "computed on.\n\n"
            "Do not hand-edit. Regenerate with:\n\n"
            "    .venv\\Scripts\\python.exe eval/freeze_corpus_manifest.py\n",
            encoding="utf-8")
        print(f"\nwrote {DERIVED.relative_to(ROOT)} ({len(derived_rows)} rows) + README")

    # ── the frozen paper-facing record ─────────────────────────────────────
    nepal = [d for d in docs
             if any("np" == str(c).lower() or "nepal" in str(c).lower()
                    for c in (d.get("country_scope") or []))]
    record = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "documents": len(docs),
            "chunks": 3816,  # from eval/corpus_snapshot/snapshot.json
            "nepal_scoped_documents": len(nepal),
            "nepal_scoped_pct": round(len(nepal) / len(docs), 4),
        },
        "composition": {f: breakdown(docs, f) for f in
                        ("source", "doc_type", "authority_tier", "country_scope", "language")},
        "manifest_reconciliation": {
            "manifest_urls_deduped": len(manifests),
            "ingested": len(ingested),
            "never_ingested": len(missing),
            "never_ingested_by_file": dict(missing_by_file),
            "never_ingested_urls": missing,
            "docs_without_manifest_before_derivation": len(unaccounted),
            "docs_without_manifest_after_derivation": 0 if derived_rows else len(unaccounted),
            "derived_manifest": str(DERIVED.relative_to(ROOT)).replace("\\", "/"),
        },
        "paper_notes": [
            "Table 1 must state that the corpus is 82% MedlinePlus and only "
            f"{len(nepal)} documents ({len(nepal)/len(docs):.1%}) are Nepal-scoped. The "
            "Nepal contribution is the ROUTING layer (nepal_care_tiers.yaml, "
            "redflag_rules.yaml), not the knowledge base. Do not claim a "
            "'Nepal-grounded corpus'.",
            "Authoritative corpus = eval/corpus_snapshot/ (exact text + MedCPT vector(768) "
            "embeddings, SHA-256 recorded). Manifests describe provenance, not content.",
            f"{len(missing)} manifest URLs were never ingested; they are listed here so the "
            "gap is documented rather than silently carried.",
        ],
    }
    if not args.check:
        OUT.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}")

    print(f"\nNepal-scoped documents: {len(nepal)}/{len(docs)} = {len(nepal)/len(docs):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
