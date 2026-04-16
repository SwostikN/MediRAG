"""MediRAG ingestion driver (Week 3).

Reads a JSONL manifest of corpus sources, fetches each URL, parses (HTML or
PDF), chunks to ~1500 chars with 200 overlap, embeds with MedCPT Article
Encoder, and upserts `documents` + `chunks` rows into Supabase via the
PostgREST API.

Usage:
    python -m ingest.run                               # ingest default manifest
    python -m ingest.run --manifest <path>             # custom manifest
    python -m ingest.run --limit 5                     # first 5 entries (dev)
    python -m ingest.run --dry-run                     # validate + fetch, no writes
    python -m ingest.run --skip-embed                  # skip MedCPT (debug)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from app.supabase_client import (
        find_document_by_url,
        insert_chunk,
        insert_document,
    )
except ImportError:
    import os
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.supabase_client import (  # type: ignore  # noqa: E402
        find_document_by_url,
        insert_chunk,
        insert_document,
    )

from ingest.fetch import fetch
from ingest.parse import parse_html, parse_pdf


INGEST_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = INGEST_DIR / "manifest" / "seed_v1.jsonl"

CHUNK_SIZE_CHARS = 1500
CHUNK_OVERLAP_CHARS = 200
MIN_DOC_CHARS = 400

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE_CHARS,
    chunk_overlap=CHUNK_OVERLAP_CHARS,
)


@dataclass
class ManifestEntry:
    raw: dict[str, Any]

    @property
    def source(self) -> str:
        return self.raw["source"]

    @property
    def source_url(self) -> str:
        return self.raw["source_url"]

    @property
    def title_hint(self) -> Optional[str]:
        return self.raw.get("title")

    def as_document_fields(self, parsed_meta: dict) -> dict:
        return {
            "title": self.title_hint or parsed_meta.get("title") or self.source_url,
            "source": self.source,
            "source_url": self.source_url,
            "authority_tier": int(self.raw.get("authority_tier", 3)),
            "doc_type": self.raw.get("doc_type", "patient-ed"),
            "publication_date": self.raw.get("publication_date") or parsed_meta.get("publication_date"),
            "last_revised_date": self.raw.get("last_revised_date") or parsed_meta.get("last_revised_date"),
            "language": self.raw.get("language", "en"),
            "domains": self.raw.get("domains"),
            "population": self.raw.get("population"),
            "country_scope": self.raw.get("country_scope"),
        }


def load_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[error] {path}:{line_no} bad JSON — {exc}", file=sys.stderr)
            continue
        if not obj.get("source") or not obj.get("source_url"):
            print(f"[error] {path}:{line_no} missing source or source_url", file=sys.stderr)
            continue
        entries.append(ManifestEntry(raw=obj))
    return entries


def chunk_text(text: str) -> list[str]:
    return [c.strip() for c in _splitter.split_text(text) if c.strip()]


@dataclass
class IngestStats:
    attempted: int = 0
    succeeded: int = 0
    skipped_existing: int = 0
    fetch_errors: int = 0
    parse_errors: int = 0
    short_docs: int = 0
    embed_errors: int = 0
    db_errors: int = 0
    total_chunks: int = 0
    started_at: float = 0.0

    def pretty(self) -> str:
        dur = time.time() - self.started_at
        return (
            f"attempted={self.attempted} succeeded={self.succeeded} "
            f"skipped_existing={self.skipped_existing} "
            f"fetch_err={self.fetch_errors} parse_err={self.parse_errors} "
            f"short_doc={self.short_docs} embed_err={self.embed_errors} "
            f"db_err={self.db_errors} chunks={self.total_chunks} "
            f"duration={dur:.1f}s"
        )


def ingest(
    manifest_path: Path,
    *,
    limit: Optional[int] = None,
    dry_run: bool = False,
    skip_embed: bool = False,
    skip_existing: bool = True,
) -> IngestStats:
    entries = load_manifest(manifest_path)
    if limit is not None:
        entries = entries[:limit]

    article_encoder = None
    if not skip_embed and not dry_run:
        from ingest.medcpt import ArticleEncoder, to_pgvector_literal  # lazy import
        article_encoder = ArticleEncoder()

    stats = IngestStats(started_at=time.time())

    for entry in entries:
        stats.attempted += 1
        url = entry.source_url
        print(f"[{stats.attempted}/{len(entries)}] {entry.source} :: {url}")

        if skip_existing and not dry_run:
            existing = find_document_by_url(url)
            if existing:
                print(f"  skip: already ingested (doc_id={existing})")
                stats.skipped_existing += 1
                continue

        fetched = fetch(url)
        if fetched.error:
            print(f"  fetch error: {fetched.error}")
            stats.fetch_errors += 1
            continue

        try:
            if fetched.is_pdf:
                text, parsed_meta = parse_pdf(fetched.body)
            else:
                text, parsed_meta = parse_html(fetched.body, url)
        except Exception as exc:
            print(f"  parse error: {exc}")
            stats.parse_errors += 1
            continue

        if len(text) < MIN_DOC_CHARS:
            print(f"  short doc ({len(text)} chars) — skipping")
            stats.short_docs += 1
            continue

        chunks = chunk_text(text)
        print(f"  extracted {len(text)} chars → {len(chunks)} chunks")

        if dry_run:
            stats.succeeded += 1
            stats.total_chunks += len(chunks)
            continue

        doc_fields = entry.as_document_fields(parsed_meta)
        doc_res = insert_document(**doc_fields)
        if isinstance(doc_res, dict) and doc_res.get("error"):
            print(f"  db error (document): {doc_res['error']}")
            stats.db_errors += 1
            continue
        try:
            doc_id = doc_res[0].get("doc_id")
        except Exception:
            print("  db error (document): could not parse response")
            stats.db_errors += 1
            continue

        embeddings: list[Optional[str]] = [None] * len(chunks)
        if article_encoder is not None:
            try:
                pairs = [(doc_fields["title"], c) for c in chunks]
                vecs = article_encoder.encode(pairs)
                embeddings = [to_pgvector_literal(v) for v in vecs]
            except Exception as exc:
                print(f"  embed error: {exc}")
                stats.embed_errors += 1

        chunk_errors = 0
        for ord_i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            res = insert_chunk(
                doc_id,
                ord_i,
                chunk,
                token_count=len(chunk.split()),
                embedding=emb,
            )
            if isinstance(res, dict) and res.get("error"):
                chunk_errors += 1
        if chunk_errors:
            print(f"  db warning: {chunk_errors}/{len(chunks)} chunks failed")
            stats.db_errors += chunk_errors
        stats.succeeded += 1
        stats.total_chunks += len(chunks) - chunk_errors

    return stats


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MediRAG corpus ingestion")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    stats = ingest(
        args.manifest,
        limit=args.limit,
        dry_run=args.dry_run,
        skip_embed=args.skip_embed,
        skip_existing=not args.no_skip_existing,
    )
    print("=" * 60)
    print(stats.pretty())
    return 0 if stats.fetch_errors + stats.db_errors + stats.embed_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
