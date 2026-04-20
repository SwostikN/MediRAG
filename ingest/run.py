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

import re

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

# Phase 4a structural chunker: splits markdown/HTML content on heading
# boundaries before falling back to the character splitter.
#
# Each output chunk is a dict: {"text": str, "section_heading": Optional[str]}.
# Plain-text sources with no detected headings yield chunks with
# section_heading=None — the character splitter does the work and the
# metadata shape stays uniform so downstream `insert_chunk(..., section_heading=...)`
# is a direct passthrough.
#
# Heading detection is deliberately conservative: only lines that start with
# '#' / '##' / '###' (markdown) or that are wrapped in <h1>..<h3> tags (HTML
# fragments parse.py may leave in). We do NOT try to detect all-caps pseudo-
# headings in plain text — false positives there would shred paragraphs.

# Markdown headings: '# H1', '## H2', '### H3' at line start.
_MD_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)
# HTML headings (in case parse.py passes through raw <h1>..<h3> tags for
# some source that trafilatura couldn't fully flatten).
_HTML_HEADING_RE = re.compile(
    r"<h([1-3])\b[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL
)


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def _split_into_sections(text: str) -> list[tuple[Optional[str], str]]:
    """Return a list of (heading, body) tuples.

    If no headings are detected the whole text is returned as a single
    (None, text) tuple so callers can treat it uniformly.
    """
    if not text or not text.strip():
        return []

    # Find heading positions from both detectors, merged and sorted.
    spans: list[tuple[int, int, str]] = []  # (start, end, heading_text)
    for m in _MD_HEADING_RE.finditer(text):
        spans.append((m.start(), m.end(), m.group(2).strip()))
    for m in _HTML_HEADING_RE.finditer(text):
        spans.append((m.start(), m.end(), _strip_html(m.group(2))))

    if not spans:
        return [(None, text.strip())]

    spans.sort(key=lambda s: s[0])

    sections: list[tuple[Optional[str], str]] = []
    # Any preamble before the first heading becomes a headingless section.
    first_start = spans[0][0]
    if first_start > 0:
        preamble = text[:first_start].strip()
        if preamble:
            sections.append((None, preamble))

    for i, (_start, end, heading) in enumerate(spans):
        body_start = end
        body_end = spans[i + 1][0] if i + 1 < len(spans) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append((heading or None, body))
    return sections


def chunk_text(text: str) -> list[dict]:
    """Structural-first chunker.

    Returns a list of {"text": str, "section_heading": Optional[str]} dicts.
    Flow:
      1. Split on markdown/HTML headings into logical sections.
      2. Any section > CHUNK_SIZE_CHARS is further broken by the existing
         RecursiveCharacterTextSplitter; sub-chunks inherit the section heading.
      3. Plain-text (no headings) falls through to the character splitter,
         yielding chunks with section_heading=None.
    """
    out: list[dict] = []
    for heading, body in _split_into_sections(text):
        body = body.strip()
        if not body:
            continue
        if len(body) <= CHUNK_SIZE_CHARS:
            out.append({"text": body, "section_heading": heading})
            continue
        for piece in _splitter.split_text(body):
            piece = piece.strip()
            if piece:
                out.append({"text": piece, "section_heading": heading})
    return out


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

        # chunks is a list of {"text": str, "section_heading": Optional[str]}
        # under the Phase 4a structural chunker. Heading-aware retrieval is
        # backward-compatible: chunks.section_heading is nullable on the DB
        # schema so older pre-Phase-4a rows keep working.
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
                # MedCPT Article encoder takes (title, body). When a
                # section heading is present we fold it into the title
                # segment ("{doc_title} — {section}") so the encoder can
                # use the sectional anchor; falls back to doc title alone
                # when heading is None.
                pairs = [
                    (
                        f"{doc_fields['title']} — {c['section_heading']}"
                        if c.get("section_heading")
                        else doc_fields["title"],
                        c["text"],
                    )
                    for c in chunks
                ]
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
                chunk["text"],
                token_count=len(chunk["text"].split()),
                embedding=emb,
                section_heading=chunk.get("section_heading"),
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
