"""PubMed / PMC Open Access → MediRAG corpus ingest (Phase 5, §3.1 #3).

Source: NCBI E-utilities (esearch + efetch) against the `pmc` database.
Authority tier: 2 when publication type is Systematic Review / RCT /
    Practice Guideline, 3 otherwise. Per HALLUCINATION_ZERO_PLAN §3.2
    item 2 the tier is assigned from the source-emitted `pub_types`, not
    from chunk content.
Licensing posture: STRICT. PMC contains a mix of CC-BY, CC-BY-NC,
    CC0 / public-domain, and "author manuscript" submissions that are NOT
    redistributable. We ingest only rows whose OA license is CC-BY or
    public-domain (including CC0). Anything else — NC, ND, "open access"
    without an explicit license tag, or a bare `oa = "Y"` flag — is logged
    and skipped. We never want a downstream chunk snippet to be a
    copyright violation; missing a paper is fine, re-hosting paywalled
    text is not.
Rate-limit posture: NCBI allows 3 req/sec without an api_key and 10 req/sec
    with one. This module sleeps 0.35s between HTTP calls (headroom under
    3/s) and does NOT read NCBI_API_KEY from env — if/when the user
    registers one, add it as a `&api_key=` query param and drop the sleep
    to 0.11s.
Seed-list integration plan:
    Phase 5 Week A ingests against the primary-care seed list
    (ingest/manifest/primary_care_v1.jsonl). The caller loops the seed
    conditions, calls `fetch_pmc_oa_for_condition`, filters, and hands
    rows to ingest.run.run() via the same (documents, chunks) row shape
    used by medlineplus.py. This module does not write to the DB itself —
    it is a fetcher, not a driver.

Usage:
    # Print usage note (this module does NOT auto-fetch):
    python -m ingest.sources.pubmed_oa

    # When the user wires it up, the intended call signature is:
    #     fetch_pmc_oa_for_condition("pneumonia",
    #         pub_types=["Systematic Review", "Practice Guideline"],
    #         date_from="2019-01-01", limit=25)
"""
from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree as ET

import requests

log = logging.getLogger("ingest.pubmed_oa")

# ─── endpoints / constants ──────────────────────────────────────────────

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"
ESUMMARY_URL = f"{EUTILS_BASE}/esummary.fcgi"

# Conservative sleep under NCBI's 3 req/sec anonymous cap.
RATE_LIMIT_SLEEP = 0.35

REQUEST_TIMEOUT = 30

# OA licenses we are willing to redistribute chunk text from.
# CC-BY (any version) and CC0 / public domain. Anything else is a skip.
_ALLOWED_LICENSE_RE = re.compile(
    r"\b(CC[\s-]?BY(?!\s*-?\s*(NC|ND))|CC0|PUBLIC[\s-]?DOMAIN|PD)\b",
    re.IGNORECASE,
)

# Pub-type → authority tier contribution. Presence of any of these
# upgrades the row from tier 3 to tier 2.
TIER_2_PUB_TYPES = {
    "systematic review",
    "meta-analysis",
    "randomized controlled trial",
    "practice guideline",
    "guideline",
}


# ─── tier helpers ───────────────────────────────────────────────────────

def _assign_tier(pub_types: List[str]) -> int:
    lowered = {p.lower() for p in pub_types or []}
    return 2 if (lowered & TIER_2_PUB_TYPES) else 3


def _license_allowed(license_str: Optional[str]) -> bool:
    if not license_str:
        return False
    # Explicit disallow for NC / ND even if the string also says CC-BY-NC.
    if re.search(r"\bCC[\s-]?BY[\s-]?(NC|ND)", license_str, re.IGNORECASE):
        return False
    return bool(_ALLOWED_LICENSE_RE.search(license_str))


# ─── XML → plain text ───────────────────────────────────────────────────

def _xml_to_plain_text(root: ET.Element) -> str:
    """JATS/PMC XML → whitespace-normalised plain text.

    We strip all tags, keep text-node content only. This is deliberately
    naive — PMC articles have tables, figures, and math that we don't
    attempt to render. The downstream chunker (ingest.run.chunk_text) is
    section-aware via a separate heading pass, but at this layer we just
    flatten.
    """
    if root is None:
        return ""
    parts: List[str] = []
    for elem in root.iter():
        if elem.text and elem.text.strip():
            parts.append(elem.text.strip())
        if elem.tail and elem.tail.strip():
            parts.append(elem.tail.strip())
    text = " ".join(parts)
    return re.sub(r"\s+", " ", text).strip()


def _findtext(elem: Optional[ET.Element], path: str, default: str = "") -> str:
    if elem is None:
        return default
    found = elem.findtext(path)
    return (found or default).strip()


# ─── fetch primitives ───────────────────────────────────────────────────

def _http_get(url: str, params: dict) -> Optional[bytes]:
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as exc:
        log.warning("pubmed_oa http error on %s: %s", url, exc)
        return None
    finally:
        time.sleep(RATE_LIMIT_SLEEP)


def _esearch_pmc(query: str, limit: int) -> List[str]:
    """Run esearch on the `pmc` DB, return PMC IDs (no 'PMC' prefix)."""
    params = {
        "db": "pmc",
        "term": query,
        "retmax": str(limit),
        "retmode": "xml",
    }
    body = _http_get(ESEARCH_URL, params)
    if not body:
        return []
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        log.warning("pubmed_oa esearch XML parse error: %s", exc)
        return []
    return [el.text for el in root.findall(".//IdList/Id") if el.text]


def _build_query(condition: str, pub_types: Optional[List[str]],
                 date_from: Optional[str]) -> str:
    """Assemble an E-utilities query string.

    condition AND (pt1[PT] OR pt2[PT]) AND ("date_from"[PDAT] : "3000"[PDAT])
    AND "open access"[Filter]
    """
    parts = [f'("{condition}"[Title/Abstract] OR "{condition}"[MeSH Terms])']
    if pub_types:
        pt_clause = " OR ".join(f'"{pt}"[Publication Type]' for pt in pub_types)
        parts.append(f"({pt_clause})")
    if date_from:
        parts.append(f'("{date_from}"[PDAT] : "3000"[PDAT])')
    parts.append('"open access"[Filter]')
    return " AND ".join(parts)


# ─── row parsing ────────────────────────────────────────────────────────

def _parse_article(article: ET.Element) -> Optional[dict]:
    """Parse a single <article> JATS element into the MediRAG row shape."""
    # IDs
    pmcid = ""
    pmid = ""
    doi = ""
    for aid in article.findall(".//article-id"):
        kind = aid.get("pub-id-type", "")
        if kind == "pmc" and aid.text:
            pmcid = aid.text.strip()
        elif kind == "pmid" and aid.text:
            pmid = aid.text.strip()
        elif kind == "doi" and aid.text:
            doi = aid.text.strip()

    # Title
    title_el = article.find(".//title-group/article-title")
    title = _xml_to_plain_text(title_el) if title_el is not None else ""

    # Abstract
    abs_el = article.find(".//abstract")
    abstract = _xml_to_plain_text(abs_el) if abs_el is not None else ""

    # Full text body
    body_el = article.find(".//body")
    full_text = _xml_to_plain_text(body_el) if body_el is not None else ""

    # Journal
    journal = _findtext(article, ".//journal-title")

    # Pub date (prefer epub, then ppub, then collection)
    publication_date = ""
    for pub_type in ("epub", "ppub", "collection"):
        date_el = article.find(f".//pub-date[@pub-type='{pub_type}']")
        if date_el is None:
            date_el = article.find(f".//pub-date[@date-type='{pub_type}']")
        if date_el is not None:
            y = _findtext(date_el, "year")
            m = _findtext(date_el, "month") or "01"
            d = _findtext(date_el, "day") or "01"
            if y:
                try:
                    publication_date = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                    break
                except ValueError:
                    continue

    # Authors — first 3
    authors_out: List[str] = []
    for contrib in article.findall(".//contrib[@contrib-type='author']"):
        surname = _findtext(contrib, ".//surname")
        given = _findtext(contrib, ".//given-names")
        name = " ".join(p for p in (given, surname) if p).strip()
        if name:
            authors_out.append(name)
        if len(authors_out) >= 3:
            break
    authors = ", ".join(authors_out)

    # Publication types
    pub_types: List[str] = []
    for subj in article.findall(".//subj-group[@subj-group-type='heading']/subject"):
        if subj.text:
            pub_types.append(subj.text.strip())
    # Also pick up <article-categories>/<subj-group>/<subject>
    for subj in article.findall(".//article-categories//subject"):
        t = (subj.text or "").strip()
        if t and t not in pub_types:
            pub_types.append(t)

    # OA license — look at <permissions><license> / @license-type
    oa_license = ""
    lic_el = article.find(".//permissions/license")
    if lic_el is not None:
        # Prefer an explicit license-type attribute; else flatten child text.
        oa_license = (lic_el.get("license-type")
                      or lic_el.get("xlink:href")
                      or _xml_to_plain_text(lic_el))
    if not oa_license:
        # Some PMC records put a <copyright-statement> free-text clue.
        oa_license = _findtext(article, ".//copyright-statement")

    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/" if pmcid else ""

    return {
        "pmcid": pmcid,
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "full_text": full_text,
        "publication_date": publication_date or None,
        "authors": authors,
        "journal": journal,
        "pub_types": pub_types,
        "oa_license": oa_license,
        "doi": doi,
        "url": url,
    }


def _efetch_pmc_articles(pmcids: List[str]) -> List[dict]:
    """Bulk-fetch full JATS XML for a list of PMC IDs."""
    if not pmcids:
        return []
    params = {
        "db": "pmc",
        "id": ",".join(pmcids),
        "retmode": "xml",
    }
    body = _http_get(EFETCH_URL, params)
    if not body:
        return []
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        log.warning("pubmed_oa efetch XML parse error: %s", exc)
        return []
    out: List[dict] = []
    for article in root.findall(".//article"):
        try:
            row = _parse_article(article)
        except Exception as exc:  # noqa: BLE001 — per-article robustness
            log.warning("pubmed_oa parse error on one article: %s", exc)
            continue
        if row is not None:
            out.append(row)
    return out


# ─── public entrypoint ──────────────────────────────────────────────────

def fetch_pmc_oa_for_condition(
    condition: str,
    pub_types: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    limit: int = 50,
) -> List[dict]:
    """Fetch PMC OA articles for a condition.

    Returns a list of row dicts normalised to the MediRAG ingest shape.
    Rows that fail the CC-BY / public-domain license gate are dropped
    with a log line. On any network error, returns [] rather than raising
    (so a batch run over many conditions does not abort on one flaky
    condition).
    """
    if not condition:
        log.warning("pubmed_oa: empty condition")
        return []

    query = _build_query(condition, pub_types, date_from)
    ids = _esearch_pmc(query, limit)
    log.info("pubmed_oa esearch '%s' → %d ids", condition, len(ids))
    if not ids:
        return []

    rows = _efetch_pmc_articles(ids)
    allowed: List[dict] = []
    for row in rows:
        if not _license_allowed(row.get("oa_license")):
            log.info(
                "pubmed_oa license skip PMC%s (license=%r) — not CC-BY/PD",
                row.get("pmcid"), row.get("oa_license"),
            )
            continue
        row["authority_tier"] = _assign_tier(row.get("pub_types", []))
        allowed.append(row)

    log.info("pubmed_oa '%s' → %d allowed after license gate (of %d)",
             condition, len(allowed), len(rows))
    return allowed


# ─── row → ingest.run document shape ────────────────────────────────────

def to_document_fields(row: dict, *, default_domains: Optional[List[str]] = None,
                       country_scope: Optional[List[str]] = None) -> dict:
    """Map a fetch row → insert_document() kwargs.

    Mirrors ingest/run.py::ManifestEntry.as_document_fields so the caller
    can feed these straight to app.supabase_client.insert_document.
    """
    # doc_type: 'guideline' if tier 2 practice guideline, else 'research'.
    lowered = {p.lower() for p in row.get("pub_types", [])}
    if "practice guideline" in lowered or "guideline" in lowered:
        doc_type = "clinical-guideline"
    elif lowered & {"systematic review", "meta-analysis"}:
        doc_type = "systematic-review"
    elif "randomized controlled trial" in lowered:
        doc_type = "rct"
    else:
        doc_type = "research"

    return {
        "title": row.get("title") or row.get("url") or "",
        "source": "PubMed/PMC-OA",
        "source_url": row.get("url"),
        "authority_tier": row.get("authority_tier", 3),
        "doc_type": doc_type,
        "publication_date": row.get("publication_date"),
        "last_revised_date": None,
        "language": "en",
        "domains": default_domains,
        "population": None,
        "country_scope": country_scope or ["global"],
    }


# ─── CLI ────────────────────────────────────────────────────────────────

def _usage_note() -> str:
    return (
        "ingest.sources.pubmed_oa — fetcher module (PMC Open Access).\n"
        "\n"
        "This module does NOT auto-fetch from the CLI. It exports\n"
        "fetch_pmc_oa_for_condition(condition, pub_types, date_from, limit).\n"
        "\n"
        "Before first real ingest:\n"
        "  1. Confirm you accept NCBI's usage policy\n"
        "     (https://www.ncbi.nlm.nih.gov/books/NBK25497/).\n"
        "  2. (Optional) Register for an NCBI api_key and wire it in\n"
        "     via env var NCBI_API_KEY — this module currently does not\n"
        "     read it; add &api_key= to _http_get() params when you do.\n"
        "  3. Pick a seed condition list (ingest/manifest/primary_care_v1.jsonl\n"
        "     is the canonical source; do NOT hard-code a new list here).\n"
        "  4. The license gate drops rows whose OA license is not CC-BY /\n"
        "     CC0 / public-domain. This is intentional. Do not loosen it\n"
        "     without a legal review — `feedback_clinical_safety.md` says\n"
        "     we don't soften safety-floor rules for coverage.\n"
        "\n"
        "Typical programmatic use (from a driver the user writes later):\n"
        "    from ingest.sources.pubmed_oa import (\n"
        "        fetch_pmc_oa_for_condition, to_document_fields,\n"
        "    )\n"
        "    rows = fetch_pmc_oa_for_condition('pneumonia',\n"
        "        pub_types=['Systematic Review', 'Practice Guideline'],\n"
        "        date_from='2019-01-01', limit=25)\n"
    )


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover
    sys.stdout.write(_usage_note())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
