"""Export the live Supabase corpus to local disk — the paper's reproducibility anchor.

WHY THIS EXISTS
---------------
The Supabase project hosting this corpus became unreachable once (Sept 2026) and was
later restored. Every evaluation number in the paper depends on the exact corpus, so the
corpus must not live in only one place. This script makes a complete, verifiable, local
copy that can be restored into any fresh Postgres/Supabase project.

WHAT IT WRITES  (into eval/corpus_snapshot/, all gzipped JSONL)
    documents.jsonl.gz   every row of public.documents
    chunks.jsonl.gz      every row of public.chunks EXCEPT `tsv`
    snapshot.json        counts, per-field breakdowns, SHA-256 of each file, timestamp

NOTE ON `tsv`: it is `GENERATED ALWAYS AS (to_tsvector('english', content)) STORED`
(supabase/002_rag_schema.sql). It cannot be inserted and Postgres regenerates it from
`content` on restore, so exporting it would be wrong as well as wasteful.

NOTE ON `embedding`: vector(768) — these are MedCPT vectors (ingest/medcpt.py), produced
by a local model. No paid embedding API is involved in restoring this corpus.

USAGE
    .venv\\Scripts\\python.exe eval/export_corpus.py              # export
    .venv\\Scripts\\python.exe eval/export_corpus.py --verify     # re-check an export
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "eval" / "corpus_snapshot"
PAGE = 500  # chunks carry 768-dim embeddings; keep pages modest so responses stay sane

# `tsv` is a generated column — never export it (see module docstring).
CHUNK_COLUMNS = [
    "chunk_id", "doc_id", "ord", "content",
    "section_heading", "token_count", "embedding",
]


def _headers() -> dict[str, str]:
    load_dotenv(dotenv_path=str(ROOT / ".env"))
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        sys.exit("SUPABASE_SERVICE_ROLE_KEY missing from .env")
    return {"apikey": key, "Authorization": f"Bearer {key}"}


def _base_url() -> str:
    url = os.getenv("SUPABASE_URL")
    if not url:
        sys.exit("SUPABASE_URL missing from .env")
    return url.rstrip("/")


def fetch_all(table: str, columns: str, headers: dict, order: str) -> list[dict]:
    """Page through a table deterministically (stable ORDER BY, so reruns match)."""
    url = _base_url()
    rows: list[dict] = []
    offset = 0
    while True:
        r = requests.get(
            f"{url}/rest/v1/{table}",
            params={"select": columns, "order": order,
                    "limit": str(PAGE), "offset": str(offset)},
            headers=headers, timeout=120,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        offset += PAGE
        print(f"    {table}: {len(rows)} rows", end="\r", flush=True)
        if len(batch) < PAGE:
            break
    print(f"    {table}: {len(rows)} rows        ")
    return rows


def write_jsonl_gz(path: Path, rows: list[dict]) -> str:
    """Write gzipped JSONL with sorted keys, and return the SHA-256 of the raw bytes.

    mtime=0 keeps the gzip header byte-identical across runs, so the checksum reflects
    content only — a rerun with unchanged data produces an unchanged hash.
    """
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as fh:
        for row in rows:
            fh.write((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    return sha256(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def breakdown(rows: list[dict], field: str) -> dict[str, int]:
    c: Counter = Counter()
    for r in rows:
        v = r.get(field)
        c[json.dumps(v) if isinstance(v, list) else str(v)] += 1
    return dict(c.most_common())


def do_export() -> int:
    headers = _headers()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Exporting corpus from Supabase")
    docs = fetch_all("documents", "*", headers, order="doc_id.asc")
    chunks = fetch_all("chunks", ",".join(CHUNK_COLUMNS), headers,
                       order="doc_id.asc,ord.asc")

    if not docs or not chunks:
        sys.exit("refusing to write an empty snapshot")

    # Integrity checks before anything is declared good.
    doc_ids = {d["doc_id"] for d in docs}
    orphans = sorted({c["doc_id"] for c in chunks} - doc_ids)
    missing_emb = sum(1 for c in chunks if not c.get("embedding"))
    dims = Counter()
    for c in chunks[:200]:
        emb = c.get("embedding")
        if isinstance(emb, str):
            dims[emb.count(",") + 1] += 1
        elif isinstance(emb, list):
            dims[len(emb)] += 1

    docs_path = OUT_DIR / "documents.jsonl.gz"
    chunks_path = OUT_DIR / "chunks.jsonl.gz"
    print("  writing…")
    docs_sha = write_jsonl_gz(docs_path, docs)
    chunks_sha = write_jsonl_gz(chunks_path, chunks)

    snapshot = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "supabase_host": _base_url().split("//", 1)[-1],
        "counts": {"documents": len(docs), "chunks": len(chunks)},
        "integrity": {
            "orphan_chunk_doc_ids": orphans,
            "chunks_missing_embedding": missing_emb,
            "embedding_dims_sampled": dict(dims),
            "chunks_per_doc_mean": round(len(chunks) / len(docs), 2),
        },
        "files": {
            "documents.jsonl.gz": {
                "sha256": docs_sha, "bytes": docs_path.stat().st_size, "rows": len(docs),
            },
            "chunks.jsonl.gz": {
                "sha256": chunks_sha, "bytes": chunks_path.stat().st_size, "rows": len(chunks),
                "excluded_columns": ["tsv (GENERATED ALWAYS — regenerated on restore)"],
            },
        },
        "documents_breakdown": {
            f: breakdown(docs, f)
            for f in ("source", "doc_type", "authority_tier", "country_scope", "language")
        },
        "restore_notes": [
            "Apply supabase/002_rag_schema.sql first, then insert documents, then chunks.",
            "Do not insert `tsv`; Postgres regenerates it from `content`.",
            "Embeddings are MedCPT vector(768) — no paid embedding API needed to restore.",
            "Rebuild indexes after bulk insert (supabase/004_fix_vector_index.sql).",
        ],
    }
    (OUT_DIR / "snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    mb = (docs_path.stat().st_size + chunks_path.stat().st_size) / 1e6
    print(f"\n  documents : {len(docs):,}")
    print(f"  chunks    : {len(chunks):,}")
    print(f"  orphans   : {len(orphans)}   missing embeddings: {missing_emb}")
    print(f"  emb dims  : {dict(dims)}")
    print(f"  total     : {mb:.1f} MB  ->  {OUT_DIR}")
    if orphans or missing_emb:
        print("  WARNING: integrity problems recorded in snapshot.json")
        return 1
    return 0


def do_verify() -> int:
    snap_path = OUT_DIR / "snapshot.json"
    if not snap_path.exists():
        sys.exit(f"no snapshot found at {snap_path}")
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    ok = True
    for name, meta in snap["files"].items():
        path = OUT_DIR / name
        if not path.exists():
            print(f"  MISSING  {name}")
            ok = False
            continue
        actual = sha256(path)
        good = actual == meta["sha256"]
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'}  {name}  {actual[:16]}…")
        n = sum(1 for _ in gzip.open(path, "rt", encoding="utf-8"))
        if n != meta["rows"]:
            print(f"        row count {n} != recorded {meta['rows']}")
            ok = False
    print("\nsnapshot verified" if ok else "\nSNAPSHOT CORRUPT")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="re-check checksums and row counts of an existing export")
    args = ap.parse_args()
    raise SystemExit(do_verify() if args.verify else do_export())
