import os
from typing import Optional, Any, Dict, List

# requests is optional at import time; if missing, we fallback and warn
try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    requests = None  # type: ignore
    _HAS_REQUESTS = False

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
HAS_REAL_SERVICE_KEY = bool(
    SUPABASE_SERVICE_KEY and not SUPABASE_SERVICE_KEY.startswith("sb_publishable_")
)

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("[supabase_client] WARNING: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set in env")
elif not HAS_REAL_SERVICE_KEY:
    print(
        "[supabase_client] WARNING: SUPABASE_SERVICE_ROLE_KEY is using a publishable key. "
        "Server-side Supabase inserts are disabled until you set the real service role key."
    )

HEADERS = {
    "apikey": SUPABASE_SERVICE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY or ''}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}


def _post_table(table: str, payload: Any) -> Dict[str, Any]:
    if not _HAS_REQUESTS:
        return {"error": "requests library not installed"}

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return {"error": "supabase configuration missing"}

    if not HAS_REAL_SERVICE_KEY:
        return {"error": "supabase service role key is missing or invalid"}

    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{table}"
    resp = requests.post(url, json=payload, headers=HEADERS)
    try:
        resp.raise_for_status()
    except Exception:
        print(f"[supabase_client] POST {table} failed: {resp.status_code} {resp.text}")
        return {"error": resp.text}
    try:
        return resp.json()
    except Exception:
        return {"data": None}


def insert_document(
    title: str,
    source: str,
    *,
    source_url: Optional[str] = None,
    authority_tier: int = 5,
    doc_type: str = "patient-ed",
    publication_date: Optional[str] = None,
    last_revised_date: Optional[str] = None,
    language: str = "en",
    domains: Optional[List[str]] = None,
    population: Optional[List[str]] = None,
    country_scope: Optional[List[str]] = None,
) -> Optional[dict]:
    payload = {
        "title": title,
        "source": source,
        "source_url": source_url,
        "authority_tier": authority_tier,
        "doc_type": doc_type,
        "publication_date": publication_date,
        "last_revised_date": last_revised_date,
        "language": language,
        "domains": domains,
        "population": population,
        "country_scope": country_scope,
    }
    return _post_table("documents", [payload])


def insert_chunk(
    doc_id: Any,
    ord: int,
    content: str,
    *,
    section_heading: Optional[str] = None,
    token_count: Optional[int] = None,
) -> Optional[dict]:
    payload = {
        "doc_id": doc_id,
        "ord": ord,
        "content": content,
        "section_heading": section_heading,
        "token_count": token_count,
    }
    return _post_table("chunks", [payload])


def insert_user_report(
    user_id: Any,
    filename: str,
    extracted_values: Optional[List[dict]] = None,
) -> Optional[dict]:
    payload = {
        "user_id": user_id,
        "filename": filename,
        "extracted_values": extracted_values,
    }
    return _post_table("user_reports", [payload])


def insert_query(query_text: str, user_id: Optional[str] = None) -> Optional[dict]:
    payload = {"query_text": query_text, "user_id": user_id, "timestamp": None}
    return _post_table("query", [payload])


def insert_response(query_id: Any, answer: str, confidence_score: Optional[float] = None, freshness_score: Optional[float] = None) -> Optional[dict]:
    payload = {"query_id": query_id, "answer": answer, "confidence_score": confidence_score, "freshness_score": freshness_score}
    return _post_table("response", [payload])
