"""Europe PMC → MediRAG corpus ingest (Phase 5, §3.1 #4).

Source: Europe PMC REST Search API
    https://europepmc.org/RestfulWebService
    GET https://www.ebi.ac.uk/europepmc/webservices/rest/search
Authority tier: 2 for systematic reviews / RCTs / practice guidelines
    (as published in Europe PMC's `publicationType` facet), 3 otherwise.
    Same convention as pubmed_oa.py so downstream retrieval weighting is
    consistent across both feeds.
Licensing posture: Europe PMC aggregates from many upstream OA sources
    and exposes a `license` field on each row. We keep rows whose license
    is CC-BY (any version) or CC0 / public-domain. Everything else is
    dropped with a log line. No scraping of author-manuscript PDFs.
Rate-limit posture: Europe PMC has no published per-second cap but asks
    for "reasonable" use. We sleep 0.25s between requests to stay well
    under 10/s. If the user registers for higher-volume access, that
    window can be tightened.
Seed-list integration plan:
    Run AFTER pubmed_oa.py for the same seed condition. The de-dup by
    PMID is the whole point — Europe PMC is a superset of PMC for most
    rows, so we use it to catch the papers PMC OA doesn't have open, and
    we skip anything whose PMID already shipped via pubmed_oa.py. The
    caller must pass the set of already-emitted PMIDs into
    `fetch_europepmc_for_condition(..., seen_pmids=...)`.

Usage:
    # Print usage note (this module does NOT auto-fetch):
    python -m ingest.sources.europepmc
"""
from __future__ import annotations

import logging
import re
import sys
import time
from typing import Iterable, List, Optional, Set

import requests

log = logging.getLogger("ingest.europepmc")

ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Europe PMC doesn't enforce a strict RPS on this endpoint but their
# docs ask for reasonable use. 0.25s sleep → ~4 req/sec ceiling.
RATE_LIMIT_SLEEP = 0.25

REQUEST_TIMEOUT = 30
PAGE_SIZE = 25  # Europe PMC default; can be bumped to 1000 but 25 is plenty for our seed list.

# Same license policy as pubmed_oa.py.
_ALLOWED_LICENSE_RE = re.compile(
    r"\b(CC[\s-]?BY(?!\s*-?\s*(NC|ND))|CC0|PUBLIC[\s-]?DOMAIN|PD)\b",
    re.IGNORECASE,
)

# Europe PMC labels tier-2 content under `publicationType` with these.
TIER_2_PUB_TYPES = {
    "systematic review",
    "meta-analysis",
    "randomized controlled trial",
    "practice guideline",
    "guideline",
}


def _license_allowed(license_str: Optional[str]) -> bool:
    if not license_str:
        return False
    if re.search(r"\bCC[\s-]?BY[\s-]?(NC|ND)", license_str, re.IGNORECASE):
        return False
    return bool(_ALLOWED_LICENSE_RE.search(license_str))


def _assign_tier(pub_types: List[str]) -> int:
    lowered = {p.lower() for p in (pub_types or [])}
    return 2 if (lowered & TIER_2_PUB_TYPES) else 3


def _build_query(condition: str) -> str:
    """Europe PMC Lucene-ish query string.

    We scope to open-access full text (`OPEN_ACCESS:y`) and to PubMed-
    indexed sources (`SRC:MED`) to avoid the preprint / agricola noise.
    """
    return (
        f'"{condition}" AND OPEN_ACCESS:y AND SRC:MED'
    )


def _parse_row(obj: dict) -> Optional[dict]:
    """Europe PMC JSON result row → MediRAG row shape.

    Field names come from the Europe PMC search response doc schema.
    """
    pmid = (obj.get("pmid") or "").strip()
    pmcid = (obj.get("pmcid") or "").strip().replace("PMC", "")
    doi = (obj.get("doi") or "").strip()
    title = (obj.get("title") or "").strip().rstrip(".")
    abstract = (obj.get("abstractText") or "").strip()
    journal = (obj.get("journalTitle") or "").strip()
    authors = (obj.get("authorString") or "").strip()
    # Trim to first 3 authors to match pubmed_oa.py convention.
    if authors:
        parts = [p.strip() for p in authors.split(",") if p.strip()]
        authors = ", ".join(parts[:3])

    pub_date = (obj.get("firstPublicationDate") or obj.get("pubYear") or "").strip()
    # Normalise pubYear-only → YYYY-01-01 for Postgres DATE compat.
    if pub_date and re.fullmatch(r"\d{4}", pub_date):
        pub_date = f"{pub_date}-01-01"
    # Europe PMC firstPublicationDate is already ISO (YYYY-MM-DD).

    # publicationType in Europe PMC is a dict or list depending on
    # endpoint version; normalise to List[str].
    pub_types_raw = obj.get("pubTypeList") or {}
    pub_types: List[str] = []
    if isinstance(pub_types_raw, dict):
        pt = pub_types_raw.get("pubType")
        if isinstance(pt, str):
            pub_types = [pt]
        elif isinstance(pt, list):
            pub_types = [p for p in pt if isinstance(p, str)]
    elif isinstance(pub_types_raw, list):
        pub_types = [p for p in pub_types_raw if isinstance(p, str)]

    oa_license = (obj.get("license") or "").strip()

    # URL — prefer Europe PMC article page; fall back to PMC if we have pmcid.
    if pmcid:
        url = f"https://europepmc.org/article/PMC/{pmcid}"
    elif pmid:
        url = f"https://europepmc.org/article/MED/{pmid}"
    else:
        url = ""

    return {
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "full_text": "",  # Europe PMC full-text requires a second call; defer.
        "publication_date": pub_date or None,
        "authors": authors,
        "journal": journal,
        "pub_types": pub_types,
        "oa_license": oa_license,
        "url": url,
    }


def _http_get_json(params: dict) -> Optional[dict]:
    try:
        resp = requests.get(ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("europepmc http/json error: %s", exc)
        return None
    finally:
        time.sleep(RATE_LIMIT_SLEEP)


def fetch_europepmc_for_condition(
    condition: str,
    limit: int = 50,
    *,
    seen_pmids: Optional[Iterable[str]] = None,
) -> List[dict]:
    """Fetch Europe PMC OA results for a condition, deduped against PubMed.

    Parameters
    ----------
    condition : str
        Seed-list condition name (e.g. "pneumonia").
    limit : int
        Max rows to emit after all filters. Europe PMC pagination handled
        transparently.
    seen_pmids : iterable of str, optional
        PMIDs already emitted by pubmed_oa.py for this run. Any row whose
        PMID matches one of these is skipped with a log line. This is
        the primary reason this module exists — without the dedup we'd
        double-ingest every paper Europe PMC mirrors from PMC.

    Returns
    -------
    list[dict]
        Rows in the MediRAG row shape. [] on network failure.
    """
    if not condition:
        log.warning("europepmc: empty condition")
        return []

    skip = {p for p in (seen_pmids or []) if p}

    out: List[dict] = []
    cursor_mark = "*"  # Europe PMC cursor pagination.
    seen_in_run: Set[str] = set()

    while len(out) < limit:
        page_size = min(PAGE_SIZE, limit - len(out))
        params = {
            "query": _build_query(condition),
            "format": "json",
            "resultType": "core",  # core gives abstract + license
            "pageSize": str(page_size),
            "cursorMark": cursor_mark,
        }
        payload = _http_get_json(params)
        if payload is None:
            break

        result_list = (payload.get("resultList") or {}).get("result") or []
        if not result_list:
            break

        for obj in result_list:
            row = _parse_row(obj)
            if row is None:
                continue
            pmid = row.get("pmid", "")
            if pmid and pmid in skip:
                log.info("europepmc dedup: skip PMID %s (already emitted by pubmed_oa)", pmid)
                continue
            # Also dedupe within this run to protect against cursor retries.
            dedup_key = pmid or row.get("pmcid") or row.get("doi") or row.get("url")
            if dedup_key and dedup_key in seen_in_run:
                continue
            if dedup_key:
                seen_in_run.add(dedup_key)

            if not _license_allowed(row.get("oa_license")):
                log.info(
                    "europepmc license skip %s (license=%r)",
                    row.get("pmid") or row.get("pmcid"),
                    row.get("oa_license"),
                )
                continue

            row["authority_tier"] = _assign_tier(row.get("pub_types", []))
            out.append(row)
            if len(out) >= limit:
                break

        next_cursor = payload.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor_mark:
            break
        cursor_mark = next_cursor

    log.info("europepmc '%s' → %d rows after dedup + license gate",
             condition, len(out))
    return out


def to_document_fields(row: dict, *, default_domains: Optional[List[str]] = None,
                       country_scope: Optional[List[str]] = None) -> dict:
    """Map a fetch row → insert_document() kwargs. See pubmed_oa.to_document_fields."""
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
        "source": "EuropePMC",
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


def _usage_note() -> str:
    return (
        "ingest.sources.europepmc — fetcher module (Europe PMC REST).\n"
        "\n"
        "This module does NOT auto-fetch from the CLI. It exports\n"
        "fetch_europepmc_for_condition(condition, limit, seen_pmids=...).\n"
        "\n"
        "Before first real ingest:\n"
        "  1. Read https://europepmc.org/RestfulWebService#termsofuse.\n"
        "  2. This module MUST be run AFTER pubmed_oa.py for the same\n"
        "     condition — pass the set of already-emitted PMIDs as\n"
        "     `seen_pmids` to avoid double-ingesting PMC-mirrored papers.\n"
        "  3. License gate drops anything not CC-BY / CC0 / PD — same\n"
        "     posture as pubmed_oa. Do not loosen without legal review.\n"
        "\n"
        "Typical programmatic use:\n"
        "    from ingest.sources.pubmed_oa import fetch_pmc_oa_for_condition\n"
        "    from ingest.sources.europepmc import fetch_europepmc_for_condition\n"
        "    pmc_rows = fetch_pmc_oa_for_condition('pneumonia', limit=25)\n"
        "    seen = {r['pmid'] for r in pmc_rows if r.get('pmid')}\n"
        "    epmc_rows = fetch_europepmc_for_condition('pneumonia',\n"
        "        limit=25, seen_pmids=seen)\n"
    )


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover
    sys.stdout.write(_usage_note())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
