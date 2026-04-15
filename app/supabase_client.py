import os
from typing import Optional, Any, Dict

# requests is optional at import time; if missing, we fallback and warn
try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    requests = None  # type: ignore
    _HAS_REQUESTS = False

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("[supabase_client] WARNING: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set in env")

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


def insert_document(title: str, user_id: Optional[str] = None) -> Optional[dict]:
    payload = {"title": title, "user_id": user_id, "upload_date": None}
    return _post_table("document", [payload])


def insert_chunk(doc_id: Any, content: str) -> Optional[dict]:
    payload = {"doc_id": doc_id, "content": content}
    return _post_table("chunk", [payload])


def insert_query(query_text: str, user_id: Optional[str] = None) -> Optional[dict]:
    payload = {"query_text": query_text, "user_id": user_id, "timestamp": None}
    return _post_table("query", [payload])


def insert_response(query_id: Any, answer: str, confidence_score: Optional[float] = None, freshness_score: Optional[float] = None) -> Optional[dict]:
    payload = {"query_id": query_id, "answer": answer, "confidence_score": confidence_score, "freshness_score": freshness_score}
    return _post_table("response", [payload])
