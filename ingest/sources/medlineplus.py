"""MedlinePlus Health Topics → MediRAG corpus ingest.

Source: https://medlineplus.gov/xml.html — daily bulk XML, public domain,
no API key, no rate limit on the download endpoint. One file (~29 MB)
contains all ~2000 English patient-education topics with full-summary
prose, MeSH terms, group taxonomy, and also-called synonyms.

Why MedlinePlus (not PubMed / PMC): register. MedlinePlus is consumer
patient-ed — the same register as our existing NHS corpus. PubMed is
research-paper register, which would bias the generator toward a
clinician tone we don't want for a health navigator.

Scope: we filter by group to match the corpus scope defined in
`memory/project_corpus_scope.md`. We include primary-care-wide body
systems, common conditions, diagnostic tests, drug therapy, and
population health. We exclude surgical procedures, transplantation
specifics, disaster response, genetics/birth-defect specialist depth,
and complementary/alternative therapies (register / evidence-gating).

Usage:
    # One-time download of the latest topic XML (cached locally)
    python -m ingest.sources.medlineplus download

    # Smoke-test the filter + chunking, no DB writes
    python -m ingest.sources.medlineplus --limit 5 --dry-run

    # Full ingest (skips topics already in documents table by URL)
    python -m ingest.sources.medlineplus
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set
from xml.etree import ElementTree as ET

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Reuse ingest-driver constants so chunk size matches existing corpus
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ingest.run import (  # noqa: E402
    CHUNK_SIZE_CHARS,
    CHUNK_OVERLAP_CHARS,
    MIN_DOC_CHARS,
)
from app.supabase_client import (  # noqa: E402
    find_document_by_url,
    insert_chunk,
    insert_document,
)


INGEST_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = INGEST_DIR / "cache"
CACHE_XML = CACHE_DIR / "mplus_topics.xml"
INDEX_URL = "https://medlineplus.gov/xml.html"
XML_URL_PATTERN = re.compile(r"mplus_topics_\d{4}-\d{2}-\d{2}\.xml")
BASE_XML_URL = "https://medlineplus.gov/xml/"

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE_CHARS,
    chunk_overlap=CHUNK_OVERLAP_CHARS,
)


# ─── scope filter ───────────────────────────────────────────────────────

# MedlinePlus group name → MediRAG domain tags. Multiple groups can map
# to the same domain (e.g. "Female Reproductive System" + "Pregnancy and
# Reproduction" both → "reproductive-health"). Domain tags match the
# existing vocabulary used in ingest/manifest/*.jsonl.
GROUP_TO_DOMAINS: dict[str, List[str]] = {
    "Blood, Heart and Circulation": ["cardiovascular"],
    "Bones, Joints and Muscles": ["musculoskeletal"],
    "Brain and Nerves": ["neurology"],
    "Children and Teenagers": ["paediatric"],
    "Diabetes Mellitus": ["endocrine", "diabetes"],
    "Diagnostic Tests": ["lab-explainer", "diagnostic-tests"],
    "Digestive System": ["gi"],
    "Drug Therapy": ["pharmacology", "medications"],
    "Ear, Nose and Throat": ["ent"],
    "Endocrine System": ["endocrine"],
    "Eyes and Vision": ["ophthalmology"],
    "Female Reproductive System": ["reproductive-health", "women"],
    "Fitness and Exercise": ["wellness"],
    "Food and Nutrition": ["nutrition", "wellness"],
    "Health System": ["health-system"],
    "Immune System": ["immunology"],
    "Infections": ["infectious-disease"],
    "Injuries and Wounds": ["trauma", "safety"],
    "Kidneys and Urinary System": ["renal", "genitourinary"],
    "Lungs and Breathing": ["respiratory"],
    "Male Reproductive System": ["reproductive-health", "men"],
    "Men": ["men"],
    "Mental Health and Behavior": ["mental-health"],
    "Metabolic Problems": ["endocrine", "metabolic"],
    "Mouth and Teeth": ["dental", "oral-health"],
    "Older Adults": ["elderly"],
    "Personal Health Issues": ["wellness"],
    "Poisoning, Toxicology, Environmental Health": ["toxicology", "safety"],
    "Pregnancy and Reproduction": ["pregnancy", "reproductive-health"],
    "Safety Issues": ["safety"],
    "Sexual Health Issues": ["sexual-health"],
    "Skin, Hair and Nails": ["dermatology"],
    "Social/Family Issues": ["mental-health", "social"],
    "Substance Use and Disorders": ["mental-health", "substance-use"],
    "Symptoms": ["symptoms"],
    "Wellness and Lifestyle": ["wellness"],
    "Women": ["women"],
}

# Groups we deliberately skip. Reasons inline.
EXCLUDED_GROUPS: Set[str] = {
    "Cancers",                         # deferred — oncology expansion comes later once simple scope is solid
    "Surgery and Rehabilitation",      # specialist — out of primary-care scope
    "Transplantation and Donation",    # specialist
    "Disasters",                        # not our remit
    "Genetics/Birth Defects",           # specialist depth beyond navigator
    "Complementary and Alternative Therapies",  # evidence-gating issue
    "Population Groups",                # demographic metadata, not content
}

# Population tag heuristics — some groups imply a population filter
# that's useful for downstream retrieval filtering.
GROUP_TO_POPULATION: dict[str, str] = {
    "Children and Teenagers": "paediatric",
    "Older Adults": "elderly",
    "Pregnancy and Reproduction": "pregnant",
    "Women": "women",
    "Men": "men",
}


# ─── XML parsing ────────────────────────────────────────────────────────

@dataclass
class Topic:
    mplus_id: str
    title: str
    url: str
    groups: List[str]
    also_called: List[str]
    summary_html: str
    date_created: Optional[str]

    @property
    def summary_text(self) -> str:
        """Strip HTML tags/entities from the embedded full-summary HTML."""
        s = html.unescape(self.summary_html or "")
        s = re.sub(r"<[^>]+>", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def domains(self) -> List[str]:
        tags: List[str] = []
        seen: Set[str] = set()
        for g in self.groups:
            for d in GROUP_TO_DOMAINS.get(g, []):
                if d not in seen:
                    seen.add(d)
                    tags.append(d)
        return tags

    def population(self) -> List[str]:
        pops: List[str] = []
        for g in self.groups:
            p = GROUP_TO_POPULATION.get(g)
            if p and p not in pops:
                pops.append(p)
        return pops or ["general"]

    def is_in_scope(self) -> bool:
        # Strict filter — drop if ANY group is excluded. This is the
        # "simple first, expand later" stance (2026-04-20): when we
        # decide to add a specialty area we reverse the exclusion; for
        # now we'd rather miss a relevant "Colorectal Cancer" topic
        # than accidentally pull "Chemotherapy" into the corpus.
        if any(g in EXCLUDED_GROUPS for g in self.groups):
            return False
        return any(g in GROUP_TO_DOMAINS for g in self.groups)


def parse_topics(xml_path: Path) -> Iterable[Topic]:
    root = ET.parse(xml_path).getroot()
    for el in root.findall("health-topic"):
        if el.get("language") != "English":
            continue
        groups = [g.text for g in el.findall("group") if g.text]
        summary = el.findtext("full-summary") or ""
        also = [a.text for a in el.findall("also-called") if a.text]
        yield Topic(
            mplus_id=el.get("id", ""),
            title=el.get("title", "").strip(),
            url=el.get("url", "").strip(),
            groups=[g for g in groups if g],
            also_called=[a for a in also if a],
            summary_html=summary,
            date_created=el.get("date-created"),
        )


# ─── download ───────────────────────────────────────────────────────────

def _discover_latest_xml_url() -> str:
    """Scrape the index page to find the newest mplus_topics_*.xml filename."""
    with urllib.request.urlopen(INDEX_URL, timeout=30) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    matches = XML_URL_PATTERN.findall(body)
    if not matches:
        raise RuntimeError("could not find a topic XML filename on " + INDEX_URL)
    matches.sort(reverse=True)
    return BASE_XML_URL + matches[0]


def download(force: bool = False) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_XML.exists() and not force:
        print(f"[medlineplus] cache present: {CACHE_XML} "
              f"({CACHE_XML.stat().st_size // 1024} KiB). Use --force to re-download.")
        return CACHE_XML
    url = _discover_latest_xml_url()
    print(f"[medlineplus] downloading {url} …")
    with urllib.request.urlopen(url, timeout=120) as resp:
        CACHE_XML.write_bytes(resp.read())
    print(f"[medlineplus] saved {CACHE_XML} ({CACHE_XML.stat().st_size // 1024} KiB)")
    return CACHE_XML


# ─── ingest ─────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    return [c.strip() for c in _splitter.split_text(text) if c.strip()]


def ingest_topics(
    topics: Iterable[Topic],
    *,
    limit: Optional[int],
    dry_run: bool,
    skip_embed: bool,
    skip_existing: bool,
) -> dict:
    stats = dict(
        attempted=0, succeeded=0, skipped_existing=0,
        out_of_scope=0, short_docs=0, embed_errors=0, db_errors=0,
        total_chunks=0, started_at=time.time(),
    )

    article_encoder = None
    to_pgvector_literal = None
    if not skip_embed and not dry_run:
        from ingest.medcpt import ArticleEncoder, to_pgvector_literal  # lazy
        article_encoder = ArticleEncoder()

    in_scope: List[Topic] = []
    for t in topics:
        if not t.is_in_scope():
            stats["out_of_scope"] += 1
            continue
        in_scope.append(t)
    print(f"[medlineplus] {len(in_scope)} in-scope topics after group filter "
          f"({stats['out_of_scope']} dropped)")

    if limit is not None:
        in_scope = in_scope[:limit]
        print(f"[medlineplus] limiting to first {limit} for this run")

    for i, t in enumerate(in_scope, start=1):
        stats["attempted"] += 1
        print(f"[{i}/{len(in_scope)}] {t.title} — {t.url}")

        if skip_existing and not dry_run and find_document_by_url(t.url):
            print("  skip: already ingested")
            stats["skipped_existing"] += 1
            continue

        text = t.summary_text
        if t.also_called:
            # Prepend also-called synonyms so retrieval catches alt names
            # ("DM" → Diabetes). Short prefix; doesn't dominate the chunk.
            text = "Also called: " + ", ".join(t.also_called) + ".\n\n" + text

        if len(text) < MIN_DOC_CHARS:
            print(f"  short doc ({len(text)} chars) — skipping")
            stats["short_docs"] += 1
            continue

        chunks = chunk_text(text)
        print(f"  {len(text)} chars → {len(chunks)} chunks "
              f"[{', '.join(t.groups[:3])}{'…' if len(t.groups) > 3 else ''}]")

        if dry_run:
            stats["succeeded"] += 1
            stats["total_chunks"] += len(chunks)
            continue

        doc_res = insert_document(
            title=t.title,
            source="MedlinePlus",
            source_url=t.url,
            authority_tier=1,  # NIH — top-tier
            doc_type="patient-ed",
            publication_date=_mdy_to_iso(t.date_created),
            language="en",
            domains=t.domains(),
            population=t.population(),
            country_scope=["global"],
        )
        if isinstance(doc_res, dict) and doc_res.get("error"):
            print(f"  db error (document): {doc_res['error']}")
            stats["db_errors"] += 1
            continue
        try:
            doc_id = doc_res[0].get("doc_id")
        except Exception:
            print("  db error (document): could not parse response")
            stats["db_errors"] += 1
            continue

        embeddings: list[Optional[str]] = [None] * len(chunks)
        if article_encoder is not None:
            try:
                pairs = [(t.title, c) for c in chunks]
                vecs = article_encoder.encode(pairs)
                embeddings = [to_pgvector_literal(v) for v in vecs]
            except Exception as exc:
                print(f"  embed error: {exc}")
                stats["embed_errors"] += 1

        chunk_errors = 0
        for ord_i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            res = insert_chunk(
                doc_id, ord_i, chunk,
                token_count=len(chunk.split()), embedding=emb,
            )
            if isinstance(res, dict) and res.get("error"):
                chunk_errors += 1
        if chunk_errors:
            print(f"  db error: {chunk_errors} chunks failed")
            stats["db_errors"] += 1
        stats["succeeded"] += 1
        stats["total_chunks"] += len(chunks)

    stats["duration_s"] = round(time.time() - stats["started_at"], 1)
    return stats


def _mdy_to_iso(date_str: Optional[str]) -> Optional[str]:
    """Convert MedlinePlus 'MM/DD/YYYY' → 'YYYY-MM-DD' for Postgres DATE."""
    if not date_str:
        return None
    parts = date_str.strip().split("/")
    if len(parts) != 3:
        return None
    mm, dd, yyyy = parts
    try:
        return f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
    except ValueError:
        return None


# ─── CLI ────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest MedlinePlus Health Topics into MediRAG corpus."
    )
    sub = parser.add_subparsers(dest="cmd")

    dl = sub.add_parser("download", help="Download latest topic XML to cache.")
    dl.add_argument("--force", action="store_true")

    # Default command: ingest
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-embed", action="store_true",
                        help="Skip MedCPT embeddings (debug only).")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Re-ingest topics whose URL is already in documents.")
    args = parser.parse_args(argv)

    if args.cmd == "download":
        download(force=args.force)
        return 0

    if not CACHE_XML.exists():
        print("[medlineplus] cache missing — run "
              "`python -m ingest.sources.medlineplus download` first.",
              file=sys.stderr)
        return 2

    topics = parse_topics(CACHE_XML)
    stats = ingest_topics(
        topics,
        limit=args.limit,
        dry_run=args.dry_run,
        skip_embed=args.skip_embed,
        skip_existing=not args.no_skip_existing,
    )
    print()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0 if stats.get("db_errors", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
