"""ClinicalTrials.gov v2 → MediRAG corpus ingest (Phase 5, §3.1 #5).

Source: ClinicalTrials.gov REST v2 API
    GET https://clinicaltrials.gov/api/v2/studies
Authority tier: 3 (fixed). Trial records are primary outputs, useful for
    "is there evidence for X intervention?" navigation questions but NOT
    ground truth for clinical guidance — a single RCT is a data point,
    not a guideline. We intentionally rank them below systematic reviews
    and practice guidelines in retrieval weighting. Per
    HALLUCINATION_ZERO_PLAN §3.1 the authority tier here is set by source
    (not by content), and the table pins CT.gov at tier 3.
Licensing posture: ClinicalTrials.gov content is US government work
    (public domain). No license gate is required, but we DO strip and
    trim records to the fields we actually use — the full API response
    includes large Protocol/Results sections we don't need to re-host.
    (https://clinicaltrials.gov/about-site/terms-conditions)
Rate-limit posture: ClinicalTrials.gov v2 does not advertise a per-second
    cap and the portal explicitly supports bulk access, but we sleep
    0.25s between requests anyway to be polite (≤4 req/sec) and to match
    the behaviour of the other fetchers in this directory.
Seed-list integration plan:
    Run LAST in Phase 5 Week C, scoped to condition-intervention pairs
    that showed up in the prior week's eval as weak-coverage. This
    module is the narrowest of the three — it's per-intervention, not
    per-condition, because a trial record's value in MediRAG is
    answering "did X intervention show benefit for Y?" not "what is Y?".

Usage:
    # Print usage note (this module does NOT auto-fetch):
    python -m ingest.sources.clinicaltrials_gov
"""
from __future__ import annotations

import logging
import sys
import time
from typing import Any, List, Optional

import requests

log = logging.getLogger("ingest.clinicaltrials_gov")

ENDPOINT = "https://clinicaltrials.gov/api/v2/studies"

RATE_LIMIT_SLEEP = 0.25
REQUEST_TIMEOUT = 30
PAGE_SIZE = 50  # CT.gov v2 default; max 1000. 50 is ample per condition.

# Fields we want back. Restricting the `fields` param makes the response
# ~10x smaller and keeps us honest about what we're storing.
REQUESTED_FIELDS = [
    "NCTId",
    "BriefTitle",
    "OfficialTitle",
    "OverallStatus",
    "HasResults",
    "PrimaryOutcomeMeasure",
    "PrimaryOutcomeDescription",
    "CompletionDate",
    "InterventionName",
    "InterventionType",
    "Condition",
    "StudyType",
    "StartDate",
    "ResultsFirstPostDate",
]


def _safe_get(obj: Any, *keys: str) -> Any:
    """Walk a nested dict/list; return None on any missing step."""
    cur = obj
    for k in keys:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(k)
            continue
        return None
    return cur


def _extract_primary_outcome(study: dict) -> Optional[str]:
    """Flatten PrimaryOutcomeMeasure + Description into one string."""
    po_list = _safe_get(
        study, "protocolSection", "outcomesModule", "primaryOutcomes"
    ) or []
    parts: List[str] = []
    for po in po_list:
        if not isinstance(po, dict):
            continue
        measure = (po.get("measure") or "").strip()
        desc = (po.get("description") or "").strip()
        if measure and desc:
            parts.append(f"{measure}: {desc}")
        elif measure:
            parts.append(measure)
    return "; ".join(parts) or None


def _extract_results_summary(study: dict) -> Optional[str]:
    """Pull the human-readable results summary, when the study has results.

    CT.gov v2 exposes this via resultsSection → several sub-modules. We
    want the baselineCharacteristicsModule / outcomeMeasuresModule
    free-text where available. Since the schema is deep, we stringify a
    small set of high-signal fields rather than recursively serialise.
    """
    rs = _safe_get(study, "resultsSection")
    if not rs:
        return None

    chunks: List[str] = []

    # Adverse events module often has a shared overview.
    ae_desc = _safe_get(rs, "adverseEventsModule", "description")
    if isinstance(ae_desc, str) and ae_desc.strip():
        chunks.append("Adverse events: " + ae_desc.strip())

    # Outcome measures — take `description` per outcome.
    om_list = _safe_get(rs, "outcomeMeasuresModule", "outcomeMeasures") or []
    for om in om_list:
        if not isinstance(om, dict):
            continue
        title = (om.get("title") or "").strip()
        desc = (om.get("description") or "").strip()
        if title and desc:
            chunks.append(f"{title}: {desc}")
        elif desc:
            chunks.append(desc)

    return " | ".join(chunks) or None


def _extract_intervention_names(study: dict) -> List[str]:
    ints = _safe_get(
        study, "protocolSection", "armsInterventionsModule", "interventions"
    ) or []
    names: List[str] = []
    for it in ints:
        if isinstance(it, dict):
            name = (it.get("name") or "").strip()
            if name:
                names.append(name)
    return names


def _build_query_params(
    condition: str,
    intervention: Optional[str],
    has_results: bool,
    page_size: int,
    page_token: Optional[str],
) -> dict:
    params: dict[str, Any] = {
        "format": "json",
        "pageSize": str(page_size),
        "fields": ",".join(REQUESTED_FIELDS),
        "query.cond": condition,
    }
    if intervention:
        params["query.intr"] = intervention
    if has_results:
        # v2 API filter — studies that have posted results.
        params["filter.overallStatus"] = "COMPLETED"
        params["aggFilters"] = "results:with"
    if page_token:
        params["pageToken"] = page_token
    return params


def _parse_study(study: dict) -> Optional[dict]:
    """CT.gov v2 study JSON → MediRAG row shape."""
    ident = _safe_get(study, "protocolSection", "identificationModule") or {}
    status_mod = _safe_get(study, "protocolSection", "statusModule") or {}

    nct_id = (ident.get("nctId") or "").strip()
    if not nct_id:
        return None

    title = (ident.get("officialTitle")
             or ident.get("briefTitle")
             or "").strip()
    status = (status_mod.get("overallStatus") or "").strip()

    completion = _safe_get(status_mod, "completionDateStruct", "date") or ""
    # CT.gov returns dates as "YYYY" or "YYYY-MM" or "YYYY-MM-DD"; pad to ISO.
    if completion and len(completion) == 4:
        completion = f"{completion}-01-01"
    elif completion and len(completion) == 7:
        completion = f"{completion}-01"

    primary_outcome = _extract_primary_outcome(study)
    results_summary = _extract_results_summary(study)
    interventions = _extract_intervention_names(study)

    url = f"https://clinicaltrials.gov/study/{nct_id}"

    return {
        "nct_id": nct_id,
        "title": title,
        "status": status,
        "results_summary": results_summary,
        "primary_outcome": primary_outcome,
        "completion_date": completion or None,
        "interventions": interventions,
        "url": url,
        "authority_tier": 3,  # fixed, per §3.1
    }


def _http_get_json(params: dict) -> Optional[dict]:
    try:
        resp = requests.get(ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("clinicaltrials_gov http/json error: %s", exc)
        return None
    finally:
        time.sleep(RATE_LIMIT_SLEEP)


def fetch_ct_results_for_intervention(
    condition: str,
    intervention: Optional[str] = None,
    has_results: bool = True,
    limit: int = 50,
) -> List[dict]:
    """Fetch CT.gov studies for a condition / optional intervention pair.

    Parameters
    ----------
    condition : str
        Required. Matches CT.gov `query.cond`.
    intervention : str, optional
        If provided, further restricts by `query.intr`. This is the
        primary use case for MediRAG (condition-intervention pairs
        flagged as weak-coverage in eval).
    has_results : bool
        If True (default), only studies with posted results are
        returned. A trial without results is of limited navigational
        value — we'd just be telling the user "yes someone is trying
        this" which isn't evidence.
    limit : int
        Max rows to emit.

    Returns
    -------
    list[dict]
        Rows with nct_id, title, status, results_summary, primary_outcome,
        completion_date, interventions, url, authority_tier. [] on
        network failure.
    """
    if not condition:
        log.warning("clinicaltrials_gov: empty condition")
        return []

    out: List[dict] = []
    page_token: Optional[str] = None

    while len(out) < limit:
        params = _build_query_params(
            condition, intervention, has_results,
            page_size=min(PAGE_SIZE, limit - len(out)),
            page_token=page_token,
        )
        payload = _http_get_json(params)
        if payload is None:
            break

        studies = payload.get("studies") or []
        if not studies:
            break

        for s in studies:
            row = _parse_study(s)
            if row is None:
                continue
            out.append(row)
            if len(out) >= limit:
                break

        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    log.info("clinicaltrials_gov '%s' / intr=%r → %d rows",
             condition, intervention, len(out))
    return out


def to_document_fields(row: dict, *, default_domains: Optional[List[str]] = None,
                       country_scope: Optional[List[str]] = None) -> dict:
    """Map a fetch row → insert_document() kwargs.

    `doc_type` is always 'clinical-trial' for CT.gov rows. Language is
    English (CT.gov's default; multilingual records are rare).
    """
    return {
        "title": row.get("title") or row.get("nct_id") or "",
        "source": "ClinicalTrials.gov",
        "source_url": row.get("url"),
        "authority_tier": row.get("authority_tier", 3),
        "doc_type": "clinical-trial",
        "publication_date": row.get("completion_date"),
        "last_revised_date": None,
        "language": "en",
        "domains": default_domains,
        "population": None,
        "country_scope": country_scope or ["global"],
    }


def _usage_note() -> str:
    return (
        "ingest.sources.clinicaltrials_gov — fetcher module (CT.gov v2).\n"
        "\n"
        "This module does NOT auto-fetch from the CLI. It exports\n"
        "fetch_ct_results_for_intervention(condition, intervention,\n"
        "    has_results=True, limit=50).\n"
        "\n"
        "Before first real ingest:\n"
        "  1. Read https://clinicaltrials.gov/about-site/terms-conditions.\n"
        "     Content is US government public-domain; no license gate.\n"
        "  2. Decide on condition-intervention PAIRS, not just conditions.\n"
        "     Per HALLUCINATION_ZERO_PLAN §3.3 Week C, this fetcher is\n"
        "     meant to target eval-weak pairs, not blanket ingest.\n"
        "  3. Authority tier is fixed at 3. Do not bump — a trial record\n"
        "     is not a guideline and retrieval weighting depends on the\n"
        "     tier being honest.\n"
        "\n"
        "Typical programmatic use:\n"
        "    from ingest.sources.clinicaltrials_gov import (\n"
        "        fetch_ct_results_for_intervention,\n"
        "    )\n"
        "    rows = fetch_ct_results_for_intervention(\n"
        "        condition='pneumonia',\n"
        "        intervention='amoxicillin',\n"
        "        has_results=True, limit=20)\n"
    )


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover
    sys.stdout.write(_usage_note())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
