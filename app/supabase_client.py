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
    resp = requests.get(url, headers=HEADERS, timeout=30)
    try:
        resp.raise_for_status()
    except Exception:
        print(f"[supabase_client] GET {table} failed: {resp.status_code} {resp.text}")
        return {"error": resp.text}
    try:
        return resp.json()
    except Exception:
        return {"data": None}


def _patch(table: str, params: Dict[str, str], payload: Dict[str, Any]) -> Any:
    err = _config_ok()
    if err is not None:
        return err
    query = "&".join(f"{k}={quote(v, safe='.,*()')}" for k, v in params.items())
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{table}?{query}"
    resp = requests.patch(url, json=payload, headers=HEADERS, timeout=30)
    try:
        resp.raise_for_status()
    except Exception:
        print(f"[supabase_client] PATCH {table} failed: {resp.status_code} {resp.text}")
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


def get_chat_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Fetch the stage-state fields for one chat_sessions row. Returns None
    on any failure (missing row, config error, network error)."""
    res = _get(
        "chat_sessions",
        {
            "id": f"eq.{session_id}",
            "select": "id,user_id,current_stage,intent_bucket,intake_summary",
            "limit": "1",
        },
    )
    if isinstance(res, dict) and res.get("error"):
        return None
    if isinstance(res, list) and res:
        return res[0]
    return None


def update_chat_session(session_id: str, **fields: Any) -> bool:
    """Patch selected fields on one chat_sessions row. Returns True on
    success, False on any failure."""
    if not fields:
        return True
    res = _patch("chat_sessions", {"id": f"eq.{session_id}"}, fields)
    if isinstance(res, dict) and res.get("error"):
        return False
    return True


def insert_session_document(
    session_id: str,
    user_id: str,
    *,
    filename: str,
    doc_type: str,
    content_hash: Optional[str] = None,
    page_count: Optional[int] = None,
    byte_size: Optional[int] = None,
) -> Optional[dict]:
    """Insert one row into session_documents (per-session uploaded file).

    doc_type must be one of: 'lab_report', 'research_paper', 'other'.
    The unique (session_id, content_hash) index dedups re-uploads of
    the same PDF in one session — caller should detect the conflict
    via the returned error and look up the existing row instead.
    """
    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "filename": filename,
        "doc_type": doc_type,
        "content_hash": content_hash,
        "page_count": page_count,
        "byte_size": byte_size,
    }
    return _post_table("session_documents", [payload])


def get_session_document(session_doc_id: str) -> Optional[Dict[str, Any]]:
    """Fetch one session_documents row by id. Includes extracted_text
    (used by /upload/resolve to re-run a handler on a previously
    uploaded 'other'-bucket file without a re-upload)."""
    res = _get(
        "session_documents",
        {
            "id": f"eq.{session_doc_id}",
            "select": "id,session_id,user_id,filename,doc_type,page_count,byte_size,extracted_text",
            "limit": "1",
        },
    )
    if isinstance(res, dict) and res.get("error"):
        return None
    if isinstance(res, list) and res:
        return res[0]
    return None


def update_session_document(session_doc_id: str, **fields: Any) -> bool:
    """Patch selected fields on one session_documents row. Used by
    /upload/resolve to flip doc_type and clear extracted_text after
    the user's chosen handler has finished."""
    if not fields:
        return True
    res = _patch("session_documents", {"id": f"eq.{session_doc_id}"}, fields)
    if isinstance(res, dict) and res.get("error"):
        return False
    return True


def find_session_document_by_hash(
    session_id: str, content_hash: str
) -> Optional[Dict[str, Any]]:
    """Return the existing session_documents row for this session+hash if
    it already exists (re-upload of the same PDF). None when absent."""
    res = _get(
        "session_documents",
        {
            "session_id": f"eq.{session_id}",
            "content_hash": f"eq.{content_hash}",
            "select": "id,filename,doc_type,page_count,byte_size,uploaded_at",
            "limit": "1",
        },
    )
    if isinstance(res, dict) and res.get("error"):
        return None
    if isinstance(res, list) and res:
        return res[0]
    return None


def insert_session_chunk(
    session_doc_id: str,
    ord: int,
    content: str,
    *,
    section_heading: Optional[str] = None,
    token_count: Optional[int] = None,
    embedding: Optional[str] = None,
) -> Optional[dict]:
    """Insert one chunk into session_chunks (private to the owning session)."""
    payload: Dict[str, Any] = {
        "session_doc_id": session_doc_id,
        "ord": ord,
        "content": content,
        "section_heading": section_heading,
        "token_count": token_count,
    }
    if embedding is not None:
        payload["embedding"] = embedding
    return _post_table("session_chunks", [payload])


def match_session_chunks(
    p_session_id: str,
    query_embedding: str,
    query_text: str,
    *,
    match_count: int = 10,
    candidate_count: int = 30,
    rrf_k: int = 60,
) -> List[dict]:
    """RRF hybrid retrieval restricted to ONE session. See migration 009."""
    res = _rpc(
        "match_session_chunks",
        {
            "p_session_id": p_session_id,
            "query_embedding": query_embedding,
            "query_text": query_text,
            "match_count": match_count,
            "candidate_count": candidate_count,
            "rrf_k": rrf_k,
        },
    )
    if isinstance(res, dict) and res.get("error"):
        print(f"[supabase_client] match_session_chunks failed: {res['error']}")
        return []
    if isinstance(res, list):
        return res
    return []


def insert_user_lab_markers(
    user_id: str,
    session_doc_id: str,
    markers: List[Dict[str, Any]],
) -> Optional[dict]:
    """Bulk insert parsed lab markers for one uploaded report.

    `markers` items shape (matching app.stages.results.LabMarker):
        {
          "marker_name":     "TSH",
          "value":           8.4,
          "unit":            "mIU/L",
          "reference_range": "0.4 - 4.0" | None,
          "status":          "low" | "normal" | "high" | "unknown",
          "taken_at":        ISO timestamp | None  # falls back to now()
        }
    """
    if not markers:
        return {"data": None}
    payload = []
    for m in markers:
        row: Dict[str, Any] = {
            "user_id": user_id,
            "session_doc_id": session_doc_id,
            "marker_name": m["marker_name"],
            "value": m["value"],
            "unit": m["unit"],
            "reference_range": m.get("reference_range"),
            "status": m.get("status") or "unknown",
        }
        if m.get("taken_at"):
            row["taken_at"] = m["taken_at"]
        payload.append(row)
    return _post_table("user_lab_markers", payload)


def update_session_attached_documents(
    session_id: str, attached: List[Dict[str, Any]]
) -> bool:
    """Overwrite chat_sessions.attached_documents with the given list.

    The frontend uses this to show the doc-chip strip in the chat header.
    Caller is responsible for fetching the current list, appending the
    new entry, and passing the merged result here.
    """
    return update_chat_session(session_id, attached_documents=attached)


def get_session_attached_documents(session_id: str) -> List[Dict[str, Any]]:
    """Read the attached_documents JSONB column for one session.
    Returns [] on any failure or empty state."""
    res = _get(
        "chat_sessions",
        {
            "id": f"eq.{session_id}",
            "select": "attached_documents",
            "limit": "1",
        },
    )
    if isinstance(res, dict) and res.get("error"):
        return []
    if isinstance(res, list) and res:
        att = res[0].get("attached_documents") or []
        if isinstance(att, list):
            return att
    return []


def insert_query_log(
    *,
    user_id: Optional[str],
    session_id: Optional[str],
    stage: Optional[str],
    query_text: Optional[str],
    response_text: Optional[str],
    citations: Optional[List[Any]] = None,
    retrieved_chunk_ids: Optional[List[str]] = None,
    prompt_hash: Optional[str] = None,
    refusal_triggered: bool = False,
    refusal_reason: Optional[str] = None,
    red_flag_fired: bool = False,
    red_flag_rule_id: Optional[str] = None,
) -> Optional[dict]:
    """Insert one row into public.query_log (P2.11). Fire-and-forget —
    the caller is expected to swallow exceptions so a logging outage
    never breaks the user response."""
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "stage": stage,
        "query_text": query_text,
        "response_text": response_text,
        "citations": citations,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "prompt_hash": prompt_hash,
        "refusal_triggered": refusal_triggered,
        "refusal_reason": refusal_reason,
        "red_flag_fired": red_flag_fired,
        "red_flag_rule_id": red_flag_rule_id,
    }
    return _post_table("query_log", [payload])


def insert_query(query_text: str, user_id: Optional[str] = None) -> Optional[dict]:
    payload = {"query_text": query_text, "user_id": user_id, "timestamp": None}
    return _post_table("query", [payload])


def insert_response(query_id: Any, answer: str, confidence_score: Optional[float] = None, freshness_score: Optional[float] = None) -> Optional[dict]:
    payload = {"query_id": query_id, "answer": answer, "confidence_score": confidence_score, "freshness_score": freshness_score}
    return _post_table("response", [payload])
