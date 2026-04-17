import os
from typing import Optional, Any, Dict, List
from urllib.parse import quote

from dotenv import load_dotenv

# requests is optional at import time; if missing, we fallback and warn
try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    requests = None  # type: ignore
    _HAS_REQUESTS = False

load_dotenv()

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


def _config_ok() -> Optional[Dict[str, Any]]:
    if not _HAS_REQUESTS:
        return {"error": "requests library not installed"}
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return {"error": "supabase configuration missing"}
    if not HAS_REAL_SERVICE_KEY:
        return {"error": "supabase service role key is missing or invalid"}
    return None


def _post_table(table: str, payload: Any) -> Dict[str, Any]:
    err = _config_ok()
    if err is not None:
        return err
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


def _rpc(name: str, payload: Any) -> Any:
    err = _config_ok()
    if err is not None:
        return err
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/rpc/{name}"
    resp = requests.post(url, json=payload, headers=HEADERS)
    try:
        resp.raise_for_status()
    except Exception:
        print(f"[supabase_client] RPC {name} failed: {resp.status_code} {resp.text}")
        return {"error": resp.text}
    try:
        return resp.json()
    except Exception:
        return {"data": None}


def _get(table: str, params: Dict[str, str]) -> Any:
    err = _config_ok()
    if err is not None:
        return err
    query = "&".join(f"{k}={quote(v, safe='.,*()')}" for k, v in params.items())
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{table}?{query}"
    resp = requests.get(url, headers=HEADERS)
    try:
        resp.raise_for_status()
    except Exception:
        print(f"[supabase_client] GET {table} failed: {resp.status_code} {resp.text}")
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
    embedding: Optional[str] = None,
) -> Optional[dict]:
    payload: Dict[str, Any] = {
        "doc_id": doc_id,
        "ord": ord,
        "content": content,
        "section_heading": section_heading,
        "token_count": token_count,
    }
    if embedding is not None:
        payload["embedding"] = embedding
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


def find_document_by_url(source_url: str) -> Optional[str]:
    """Return doc_id for a document with this source_url, or None if not found."""
    res = _get("documents", {"select": "doc_id", "source_url": f"eq.{source_url}", "limit": "1"})
    if isinstance(res, dict) and res.get("error"):
        return None
    if isinstance(res, list) and res:
        return res[0].get("doc_id")
    return None


def match_chunks(query_embedding: str, match_count: int = 50) -> List[dict]:
    """Call the match_chunks RPC. Returns a list of rows (or empty list on error)."""
    res = _rpc("match_chunks", {"query_embedding": query_embedding, "match_count": match_count})
    if isinstance(res, dict) and res.get("error"):
        print(f"[supabase_client] match_chunks failed: {res['error']}")
        return []
    if isinstance(res, list):
        return res
    return []


def match_chunks_hybrid(
    query_embedding: str,
    query_text: str,
    match_count: int = 30,
    candidate_count: int = 50,
    rrf_k: int = 60,
) -> List[dict]:
    """Hybrid retrieval (dense + BM25 via RRF). See supabase/005_match_chunks_hybrid.sql."""
    res = _rpc(
        "match_chunks_hybrid",
        {
            "query_embedding": query_embedding,
            "query_text": query_text,
            "match_count": match_count,
            "candidate_count": candidate_count,
            "rrf_k": rrf_k,
        },
    )
    if isinstance(res, dict) and res.get("error"):
        print(f"[supabase_client] match_chunks_hybrid failed: {res['error']}")
        return []
    if isinstance(res, list):
        return res
    return []


def match_chunks_hybrid_filtered(
    query_embedding: str,
    query_text: str,
    *,
    match_count: int = 30,
    candidate_count: int = 50,
    rrf_k: int = 60,
    filter_domains: Optional[List[str]] = None,
    filter_country_scope: Optional[List[str]] = None,
    filter_min_authority_tier: Optional[int] = None,
    filter_max_age_years: Optional[int] = None,
) -> List[dict]:
    """Filtered hybrid retrieval. See supabase/006_match_chunks_hybrid_filtered.sql."""
    res = _rpc(
        "match_chunks_hybrid_filtered",
        {
            "query_embedding": query_embedding,
            "query_text": query_text,
            "match_count": match_count,
            "candidate_count": candidate_count,
            "rrf_k": rrf_k,
            "filter_domains": filter_domains,
            "filter_country_scope": filter_country_scope,
            "filter_min_authority_tier": filter_min_authority_tier,
            "filter_max_age_years": filter_max_age_years,
        },
    )
    if isinstance(res, dict) and res.get("error"):
        print(f"[supabase_client] match_chunks_hybrid_filtered failed: {res['error']}")
        return []
    if isinstance(res, list):
        return res
    return []


def insert_query(query_text: str, user_id: Optional[str] = None) -> Optional[dict]:
    payload = {"query_text": query_text, "user_id": user_id, "timestamp": None}
    return _post_table("query", [payload])


def insert_response(query_id: Any, answer: str, confidence_score: Optional[float] = None, freshness_score: Optional[float] = None) -> Optional[dict]:
    payload = {"query_id": query_id, "answer": answer, "confidence_score": confidence_score, "freshness_score": freshness_score}
    return _post_table("response", [payload])
