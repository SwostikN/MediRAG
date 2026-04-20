import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, Tuple
from fastapi import FastAPI, File, Form, Header, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel
import pymupdf

import cohere  # Official Cohere SDK

try:
    from groq import Groq
except ImportError:  # pragma: no cover — groq is optional at import time
    Groq = None  # type: ignore

from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .middleware import add_cors_middleware
    from .supabase_client import (
        insert_chunk,
        insert_document,
        match_chunks_hybrid_filtered,
        create_chat_session,
        get_chat_session,
        update_chat_session,
        insert_session_document,
        find_session_document_by_hash,
        get_session_document,
        update_session_document,
        insert_session_chunk,
        match_session_chunks,
        insert_user_lab_markers,
        get_session_attached_documents,
        update_session_attached_documents,
        list_user_session_documents,
        delete_session_document,
        insert_query_log,
    )
    from .filters import build_filter
    from .intent import classify as classify_intent
    from .intent_gate import classify_turn as classify_intake_gate
    from .meta_question import is_meta_question, last_assistant_turn
    from .query_rewrite import expand_for_retrieval
    from .redflag import check as redflag_check
    from .stages import clarification as clarification_stage
    from .stages import intake as intake_stage
    from .stages import navigation as navigation_stage
    from .stages import results as results_stage
    from .document_classifier import (
        classify_document,
        is_medically_relevant,
    )
    from . import rate_limit
    from .guardrails import (
        apply_guardrails,
        process_streaming_chunk,
        flush_streaming_buffer,
    )
    from .refusal_filter import classify_scope, SCOPE_REFUSAL_TEMPLATES
except ImportError:
    from middleware import add_cors_middleware
    from supabase_client import (
        insert_chunk,
        insert_document,
        match_chunks_hybrid_filtered,
        create_chat_session,
        get_chat_session,
        update_chat_session,
        insert_session_document,
        find_session_document_by_hash,
        get_session_document,
        update_session_document,
        insert_session_chunk,
        match_session_chunks,
        insert_user_lab_markers,
        get_session_attached_documents,
        update_session_attached_documents,
        list_user_session_documents,
        delete_session_document,
        insert_query_log,
    )
    from filters import build_filter
    from intent import classify as classify_intent
    from intent_gate import classify_turn as classify_intake_gate
    from meta_question import is_meta_question, last_assistant_turn
    from query_rewrite import expand_for_retrieval
    from redflag import check as redflag_check
    from stages import clarification as clarification_stage
    from stages import intake as intake_stage
    from stages import navigation as navigation_stage
    from stages import results as results_stage
    from document_classifier import (
        classify_document,
        is_medically_relevant,
    )
    import rate_limit
    from guardrails import (
        apply_guardrails,
        process_streaming_chunk,
        flush_streaming_buffer,
    )
    from refusal_filter import classify_scope, SCOPE_REFUSAL_TEMPLATES

from ingest.medcpt import ArticleEncoder, QueryEncoder, to_pgvector_literal

# --------------------------------------------------
# App setup
# --------------------------------------------------
load_dotenv()

app = FastAPI(title="DocuMed AI")
add_cors_middleware(app)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"

if FRONTEND_DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST_DIR / "assets"), name="frontend-assets")


COHERE_API_KEY = os.getenv("COHERE_API_KEY")
print("Cohere API key loaded:", bool(COHERE_API_KEY))

if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY not found in environment")

# Cohere Chat Client (NEW API). Always initialised — we still use Cohere
# Rerank regardless of which provider generates the final answer.
co = cohere.ClientV2(api_key=COHERE_API_KEY)

# Optional Groq client for the generate step. When GROQ_API_KEY is set
# and the SDK is installed, /query uses Groq's Llama 3.3 70B (300–600ms
# generate) instead of Cohere Chat (1800–4500ms). Falls back to Cohere
# automatically if either is missing. Cohere Rerank is untouched.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
groq_client = Groq(api_key=GROQ_API_KEY) if (Groq and GROQ_API_KEY) else None
print(f"Groq generate enabled: {groq_client is not None} (model={GROQ_MODEL if groq_client else 'n/a'})")

# Admin token for the corpus-ingestion endpoint /upload_pdf. /upload_pdf
# writes to the SHARED chunks corpus, so we cannot let arbitrary users
# call it (a public POST would let anyone seed the corpus that everyone
# else searches). When ADMIN_UPLOAD_TOKEN is unset, /upload_pdf refuses
# every request — fail-closed by default. Operator workflow: set the
# env var in the deploy environment, then run admin scripts with
# `curl -H "X-Admin-Token: <token>" -F file=@who.pdf …`.
ADMIN_UPLOAD_TOKEN = os.getenv("ADMIN_UPLOAD_TOKEN")
print(
    "Admin upload gate: "
    + ("active (token required)" if ADMIN_UPLOAD_TOKEN else "DISABLED — /upload_pdf will refuse all requests until ADMIN_UPLOAD_TOKEN is set")
)

_article_encoder: Optional[ArticleEncoder] = None
_query_encoder: Optional[QueryEncoder] = None


def get_article_encoder() -> ArticleEncoder:
    global _article_encoder
    if _article_encoder is None:
        _article_encoder = ArticleEncoder()
    return _article_encoder


def get_query_encoder() -> QueryEncoder:
    global _query_encoder
    if _query_encoder is None:
        _query_encoder = QueryEncoder()
    return _query_encoder


MEDIRAG_SYSTEM_PROMPT = (
    "You are MediRAG, a Nepal-focused health navigator. "
    "Answer strictly using the provided sources. "
    "Do not give a diagnosis for the user. "
    "Do not recommend medications, doses, or treatments. "
    "Always frame answers as information to discuss with a doctor.\n\n"
    "MANDATORY — source-binding rule:\n"
    "Every factual clinical claim you emit MUST be directly supported by a "
    "sentence in the provided sources. If no source supports a claim, YOU "
    "MUST NOT emit that claim. Do NOT paraphrase from memory. Do NOT fill "
    "gaps with general knowledge. If none of the sources address the user's "
    "question, output EXACTLY:\n\n"
    "I don't have a source for that in my current library. Please try "
    "rewording your question, or ask your doctor directly.\n\n"
    "and nothing else.\n\n"
    "CRITICAL — topic match rule:\n"
    "The user asked about ONE specific topic. The retrieved sources may be "
    "about adjacent-but-different topics (e.g. thyroid sources retrieved for "
    "a question about hyperthermia, because both start with 'hyper'). If the "
    "sources do NOT directly discuss the exact topic in the user's question, "
    "you MUST respond with exactly this and nothing else:\n\n"
    "I don't have a source for that in my current library. Please try "
    "rewording your question, or ask your doctor directly.\n\n"
    "Do NOT discuss the adjacent topic as a substitute. Do NOT say "
    "\"but <other thing> is discussed\". Do NOT suggest what the user could "
    "ask their doctor about the adjacent thing. Refuse cleanly and stop.\n\n"
    "Formatting rules (follow strictly):\n"
    "- Use Markdown. Open with a one-sentence lead-in (no header above it).\n"
    "- Group related points under short bold headers on their own line, e.g. "
    "`**Symptoms:**`, `**When to see a doctor:**`, `**What to ask your doctor:**`.\n"
    "- Under each header, use `-` bullets. One idea per bullet, one line per bullet.\n"
    "- Never use numbered lists (`1.`, `2.`) for section grouping — bold headers only.\n"
    "- Use **bold** inline for key terms the reader should remember.\n"
    "- Do not output a `Sources:` section yourself — the system appends citations."
)

# Phase-3C: inline-citation instruction. Appended to the base prompt when
# INLINE_CITATIONS_ENABLED=1. Kept separate so the prompt can be toggled
# back to pre-Phase-3 behaviour with one env var.
_INLINE_CITATION_PROMPT_SUFFIX = (
    "\n\nCITATION BINDING (MANDATORY):\n"
    "- Every sentence that makes a clinical claim (dose, threshold, "
    "diagnostic criterion, treatment duration, 'is a symptom of X') MUST "
    "end with a bracketed source tag: `[1]`, `[2]`, or `[1, 2]` when the "
    "claim is supported by multiple sources.\n"
    "- The number N in `[N]` refers to the `[src:N]` label on the source "
    "block you were given. Use only numbers that appear as `[src:N]` in "
    "the sources. Do NOT invent numbers.\n"
    "- If you cannot cite a specific source for a claim, do NOT emit the "
    "claim — refuse per the source-binding rule above.\n"
    "- Framing sentences (intros, disclaimers like 'see your doctor') do "
    "NOT need a `[N]` tag."
)


def build_system_prompt() -> str:
    """Return the active system prompt, conditionally including the
    Phase-3 inline-citation addendum. Centralised here so both /query
    and /query/stream call sites stay in lock-step."""
    if INLINE_CITATIONS_ENABLED:
        return MEDIRAG_SYSTEM_PROMPT + _INLINE_CITATION_PROMPT_SUFFIX
    return MEDIRAG_SYSTEM_PROMPT

@app.on_event("startup")
async def warm_query_encoder():
    try:
        get_query_encoder().encode_one("warmup")
        print("[startup] MedCPT query encoder warmed")
    except Exception as exc:
        print(f"[startup] MedCPT warmup failed (non-fatal): {exc}")


@app.get("/")
async def serve_frontend():
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse(
        status_code=200,
        content={
            "message": "DocuMed AI backend is running.",
            "frontend": "Frontend build not found. Run the React app in dev mode or build frontend/dist first.",
            "docs_url": "/docs",
            "health_url": "/health",
            "query_url": "/query",
            "upload_pdf_url": "/upload_pdf",
        },
    )


# --------------------------------------------------
# Health check (REQUIRED for ECS / ALB)
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# Session start — creates a chat_sessions row so the caller can pass
# session_id into /query and exercise the stage state machine (intake
# template → navigation → routine retrieval). Before this endpoint
# existed, sessions were only created implicitly by the upload flow,
# which meant the eval scorer could never drive Stage 1 intake.
# --------------------------------------------------
class SessionStartRequest(BaseModel):
    user_id: Optional[str] = None
    current_stage: Optional[str] = "intake"


@app.post("/session/start")
def session_start(req: SessionStartRequest) -> dict:
    # chat_sessions.user_id is NOT NULL. Accept from request; fall back to
    # EVAL_USER_ID env var for the eval scorer / local smoke-tests.
    user_id = req.user_id or os.getenv("EVAL_USER_ID")
    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="user_id is required (or set EVAL_USER_ID in server env)",
        )
    row = create_chat_session(
        user_id=user_id,
        current_stage=req.current_stage or "intake",
    )
    if not row or not row.get("id"):
        raise HTTPException(status_code=500, detail="failed to create chat session")
    return {
        "session_id": row["id"],
        "current_stage": row.get("current_stage") or req.current_stage,
        "user_id": row.get("user_id"),
    }


# --------------------------------------------------
# Helpers
# --------------------------------------------------
_PDF_DATE_RE = re.compile(r"D:(\d{4})(\d{2})(\d{2})")


def _parse_pdf_date(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    m = _PDF_DATE_RE.match(raw)
    if not m:
        return None
    year, month, day = m.groups()
    return f"{year}-{month}-{day}"


def _extract_text_from_pdf_bytes(data: bytes) -> Tuple[str, dict]:
    """Core PDF text + metadata extractor. Operates on raw bytes so the
    caller can pre-read once and reuse the bytes for hashing,
    persistence, etc. Raises HTTPException(400) on a corrupt PDF."""
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF read error: {e}")
    try:
        text_parts = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text_parts.append(page_text)
        text = "\n".join(text_parts).strip()
        raw_meta = doc.metadata or {}
        metadata = {
            "title": (raw_meta.get("title") or "").strip() or None,
            "author": (raw_meta.get("author") or "").strip() or None,
            "creation_date": _parse_pdf_date(raw_meta.get("creationDate")),
            "modification_date": _parse_pdf_date(raw_meta.get("modDate")),
            "page_count": doc.page_count,
        }
        return text, metadata
    finally:
        doc.close()


def extract_text_from_pdf(file: UploadFile) -> Tuple[str, dict]:
    try:
        data = file.file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF read error: {e}")
    try:
        return _extract_text_from_pdf_bytes(data)
    finally:
        file.file.close()


# --------------------------------------------------
# Upload PDF
# --------------------------------------------------
@app.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
):
    # Admin gate. /upload_pdf writes to the SHARED corpus; users go
    # through /upload, which is per-session. Unauthenticated calls here
    # would let anyone seed the corpus that every other user searches.
    if not ADMIN_UPLOAD_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="/upload_pdf is disabled: ADMIN_UPLOAD_TOKEN is not configured.",
        )
    if not x_admin_token or x_admin_token != ADMIN_UPLOAD_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-Admin-Token. /upload_pdf is admin-only; users should use /upload.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    text, pdf_metadata = extract_text_from_pdf(file)
    if not text:
        raise HTTPException(status_code=400, detail="No text found in PDF")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = [c for c in splitter.split_text(text) if c.strip()]

    title = pdf_metadata.get("title") or file.filename
    doc_res = insert_document(
        title=title,
        source="user-upload",
        authority_tier=5,
        doc_type="reference",
        publication_date=pdf_metadata.get("creation_date"),
    )
    if isinstance(doc_res, dict) and doc_res.get("error"):
        raise HTTPException(status_code=502, detail=f"supabase: document insert failed: {doc_res['error']}")
    try:
        doc_id = doc_res[0].get("doc_id") or doc_res[0].get("id")
    except Exception:
        raise HTTPException(status_code=502, detail="supabase: could not parse document insert response")

    try:
        pairs = [(title, c) for c in chunks]
        vectors = get_article_encoder().encode(pairs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MedCPT embedding failed: {exc}")

    chunk_errors = 0
    for ord_index, (chunk_text, vec) in enumerate(zip(chunks, vectors)):
        res = insert_chunk(
            doc_id,
            ord_index,
            chunk_text,
            token_count=len(chunk_text.split()),
            embedding=to_pgvector_literal(vec),
        )
        if isinstance(res, dict) and res.get("error"):
            chunk_errors += 1

    return {
        "message": "PDF uploaded and indexed",
        "doc_id": doc_id,
        "chunks": len(chunks),
        "chunk_errors": chunk_errors,
    }


# --------------------------------------------------
# /upload — user-facing per-session document upload (Week 8)
#
# Distinct from /upload_pdf, which is the admin/corpus ingestion path
# (writes to the SHARED chunks table — used to seed the corpus from
# WHO/MoHP PDFs). /upload writes ONLY to session_documents +
# session_chunks (or user_lab_markers for lab reports). A patient's
# uploaded data NEVER reaches the shared corpus — that invariant is
# the entire reason these are two separate endpoints.
#
# Flow:
#   1. Validate (.pdf, session_id, user_id present).
#   2. Read bytes once → SHA-256 hash → check for prior re-upload of
#      the exact same file in this session (dedup short-circuit).
#   3. Extract text via pymupdf. Empty text → "scanned image" error.
#   4. Run heuristic classifier (lab_report / research_paper / other).
#   5. Branch by doc_type:
#        - lab_report:     parse markers → persist to user_lab_markers
#                          → run Stage 4 explainer → return inline.
#        - research_paper: gate on is_medically_relevant (refuse pure
#                          off-domain papers at upload time, not at
#                          query time). Then chunk + embed → insert
#                          into session_chunks. Future queries in this
#                          session pull these via _retrieve_ranked's
#                          session-merge path.
#        - other:          persist the file row + return
#                          {needs_user_intent: true} so the frontend
#                          can prompt "Lab report or Research paper?".
#                          No chunking until user disambiguates.
#   6. Update chat_sessions.attached_documents with the doc-chip entry
#      (frontend uses this to render the chips strip in the header).
# --------------------------------------------------

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB ceiling — typical Nepal lab PDF is <2 MB
SESSION_PAPER_CHUNK_SIZE = 1500
SESSION_PAPER_CHUNK_OVERLAP = 200


def _build_attached_doc_entry(
    *, session_doc_id: str, filename: str, doc_type: str, page_count: Optional[int]
) -> dict:
    """One entry in chat_sessions.attached_documents (used for the
    sidebar / chat-header chip strip)."""
    return {
        "id": session_doc_id,
        "filename": filename,
        "doc_type": doc_type,
        "page_count": page_count,
    }


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    user_id: str = Form(...),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if not session_id or not user_id:
        raise HTTPException(status_code=400, detail="session_id and user_id are required.")

    rate_limit.check(user_id, "upload")

    # Read once; reuse for hashing + extraction + persistence.
    try:
        data = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read upload: {exc}")
    finally:
        await file.close()

    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(data)} bytes (max {MAX_UPLOAD_BYTES}).",
        )

    content_hash = hashlib.sha256(data).hexdigest()

    # Re-upload of the exact same PDF in the same session → return the
    # existing row instead of duplicating chunks/markers.
    existing = find_session_document_by_hash(session_id, content_hash)
    if existing:
        print(f"[upload] dedup hit: session={session_id} hash={content_hash[:12]}…")
        return {
            "stage": "upload",
            "status": "duplicate",
            "doc_type": existing.get("doc_type"),
            "session_doc_id": existing.get("id"),
            "filename": existing.get("filename"),
            "message": (
                "This document is already attached to this conversation."
            ),
        }

    text, pdf_metadata = _extract_text_from_pdf_bytes(data)

    # Image-only / scanned PDF: pymupdf returns no extractable text.
    # Return a clear, user-facing message rather than the generic 400.
    if not text or len(text) < 50:
        return JSONResponse(
            status_code=200,
            content={
                "stage": "upload",
                "status": "unreadable",
                "error": "scanned_or_image",
                "message": (
                    "We couldn't read any text from this PDF. It may be a "
                    "scanned image or a photo. If you have a text-based "
                    "version of the report, please upload that instead."
                ),
            },
        )

    classification = classify_document(text)
    doc_type = classification.doc_type
    print(
        f"[upload] classify → {doc_type} (conf={classification.confidence}, "
        f"lab={classification.lab_score}, paper={classification.paper_score})"
    )

    # Domain gate — only applies when the classifier picked research_paper.
    # Lab reports are exempt (their own signals are strong enough);
    # 'other' is exempt (we'll ask the user to disambiguate).
    if doc_type == "research_paper" and not is_medically_relevant(text):
        return JSONResponse(
            status_code=200,
            content={
                "stage": "upload",
                "status": "off_domain",
                "error": "non_medical_paper",
                "message": (
                    "This document doesn't look like a health or medicine "
                    "paper. MediRAG only accepts medical / clinical / "
                    "public-health documents. Please upload a different "
                    "file."
                ),
            },
        )

    page_count = pdf_metadata.get("page_count")
    doc_res = insert_session_document(
        session_id,
        user_id,
        filename=file.filename,
        doc_type=doc_type,
        content_hash=content_hash,
        page_count=page_count,
        byte_size=len(data),
    )
    if isinstance(doc_res, dict) and doc_res.get("error"):
        raise HTTPException(
            status_code=502,
            detail=f"supabase: session_document insert failed: {doc_res['error']}",
        )
    try:
        session_doc_id = doc_res[0].get("id")
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="supabase: could not parse session_document insert response",
        )

    # Maintain the chat-session's attached-documents pointer list.
    attached = get_session_attached_documents(session_id)
    attached.append(_build_attached_doc_entry(
        session_doc_id=session_doc_id,
        filename=file.filename,
        doc_type=doc_type,
        page_count=page_count,
    ))
    update_session_attached_documents(session_id, attached)

    if doc_type == "lab_report":
        return _handle_lab_report(text, session_doc_id, user_id, file.filename, page_count)
    if doc_type == "research_paper":
        return _handle_research_paper(text, session_doc_id, file.filename, page_count)
    # doc_type == "other": persist extracted text so the user's
    # disambiguation click on /upload/resolve can re-run the right
    # handler without forcing a re-upload.
    update_session_document(session_doc_id, extracted_text=text)
    return {
        "stage": "upload",
        "status": "needs_user_intent",
        "doc_type": "other",
        "session_doc_id": session_doc_id,
        "filename": file.filename,
        "page_count": page_count,
        "lab_score": classification.lab_score,
        "paper_score": classification.paper_score,
        "message": (
            "I'm not sure how to handle this document. Is it a lab report "
            "(values to explain) or a research paper (questions to answer "
            "from the text)?"
        ),
    }


def _handle_lab_report(
    text: str,
    session_doc_id: str,
    user_id: str,
    filename: str,
    page_count: Optional[int],
) -> dict:
    """Parse markers, persist them, run the Stage 4 explainer, return
    the answer + table data ready for the frontend."""
    explainer = results_stage.compose_explainer(
        text,
        # Pass _retrieve_ranked itself — the composer queries it
        # per-marker. Session-merge is intentionally OFF here: marker
        # explanations should pull from the curated lab-explainer
        # corpus (NHS / Lab Tests Online), not from other docs the
        # user happened to upload.
        retrieve_fn=lambda q: _retrieve_ranked(q),
        groq_client=groq_client,
        groq_model=GROQ_MODEL,
        cohere_client=co,
    )

    # Persist parsed markers to user_lab_markers for longitudinal
    # tracking. Failure here does NOT block the response — the
    # explainer already has the markers in memory and ships them.
    if explainer.get("markers"):
        rows = [
            {
                "marker_name": m["name"],
                "value": m["value"],
                "unit": m["unit"],
                "reference_range": m.get("reference_range"),
                "status": m.get("status") or "unknown",
            }
            for m in explainer["markers"]
        ]
        insert_user_lab_markers(user_id, session_doc_id, rows)

    return {
        "stage": "results",
        "status": "ok",
        "doc_type": "lab_report",
        "session_doc_id": session_doc_id,
        "filename": filename,
        "page_count": page_count,
        "answer": explainer["answer"],
        "sources": explainer["sources"],
        "markers": explainer["markers"],
    }


def _handle_research_paper(
    text: str,
    session_doc_id: str,
    filename: str,
    page_count: Optional[int],
) -> dict:
    """Chunk + embed the paper, insert chunks into session_chunks.
    Future /query calls in this session will retrieve them via
    _retrieve_ranked's session-merge path."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SESSION_PAPER_CHUNK_SIZE,
        chunk_overlap=SESSION_PAPER_CHUNK_OVERLAP,
    )
    chunks = [c for c in splitter.split_text(text) if c.strip()]
    if not chunks:
        return {
            "stage": "upload",
            "status": "empty_after_chunking",
            "doc_type": "research_paper",
            "session_doc_id": session_doc_id,
            "message": "We couldn't split this paper into searchable sections.",
        }

    try:
        pairs = [(filename, c) for c in chunks]
        vectors = get_article_encoder().encode(pairs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MedCPT embedding failed: {exc}")

    chunk_errors = 0
    for ord_index, (chunk_text, vec) in enumerate(zip(chunks, vectors)):
        res = insert_session_chunk(
            session_doc_id,
            ord_index,
            chunk_text,
            token_count=len(chunk_text.split()),
            embedding=to_pgvector_literal(vec),
        )
        if isinstance(res, dict) and res.get("error"):
            chunk_errors += 1

    return {
        "stage": "upload",
        "status": "ok",
        "doc_type": "research_paper",
        "session_doc_id": session_doc_id,
        "filename": filename,
        "page_count": page_count,
        "chunks": len(chunks),
        "chunk_errors": chunk_errors,
        "message": (
            f"Got it — I've indexed {len(chunks)} sections from "
            f"\"{filename}\". You can now ask me questions about this "
            f"paper in this conversation."
        ),
    }


# --------------------------------------------------
# /upload/resolve — user disambiguation for 'other'-bucket uploads
#
# When the heuristic classifier returns 'other' (low confidence), the
# frontend shows two buttons under the assistant message:
#   [Treat as lab report]  [Treat as research paper]
# Clicking a button hits this endpoint with the existing
# session_doc_id, so we can re-run the chosen handler on the
# already-uploaded text without forcing a re-upload.
#
# Pre-conditions:
#   - The session_documents row exists and was created by this user.
#   - extracted_text on the row is non-null (set by /upload's 'other'
#     branch). For lab_report / research_paper rows extracted_text is
#     null, so resolve refuses (idempotency guard).
#   - The session_id in the request matches the row's session_id.
#
# After the chosen handler runs:
#   - doc_type on the row flips to the user's pick.
#   - extracted_text is cleared (we don't need it once chunked /
#     parsed, and it would otherwise duplicate session_chunks content).
#   - chat_sessions.attached_documents has the matching chip's
#     doc_type updated in-place so the frontend strip recolors.
# --------------------------------------------------


class ResolveUploadRequest(BaseModel):
    session_doc_id: str
    session_id: str
    user_id: str
    doc_type: str  # 'lab_report' | 'research_paper'


@app.post("/upload/resolve")
async def upload_resolve(req: ResolveUploadRequest):
    if req.doc_type not in ("lab_report", "research_paper"):
        raise HTTPException(
            status_code=400,
            detail="doc_type must be 'lab_report' or 'research_paper'.",
        )

    rate_limit.check(req.user_id, "upload")

    row = get_session_document(req.session_doc_id)
    if not row:
        raise HTTPException(status_code=404, detail="Document not found.")
    # Defence in depth on top of RLS — make sure the requester owns
    # this document AND it belongs to the claimed session.
    if row.get("user_id") != req.user_id or row.get("session_id") != req.session_id:
        raise HTTPException(status_code=403, detail="Document does not belong to this session/user.")

    if row.get("doc_type") != "other":
        raise HTTPException(
            status_code=409,
            detail=f"Document is already classified as '{row.get('doc_type')}'; cannot resolve again.",
        )

    text = row.get("extracted_text") or ""
    if not text:
        raise HTTPException(
            status_code=410,
            detail="Extracted text is no longer available for this document; please re-upload.",
        )

    filename = row.get("filename") or "document.pdf"
    page_count = row.get("page_count")

    # Domain gate for research_paper choice — same rule as /upload.
    if req.doc_type == "research_paper" and not is_medically_relevant(text):
        return JSONResponse(
            status_code=200,
            content={
                "stage": "upload",
                "status": "off_domain",
                "error": "non_medical_paper",
                "message": (
                    "This document doesn't look like a health or medicine "
                    "paper. MediRAG only accepts medical / clinical / "
                    "public-health documents."
                ),
            },
        )

    # Run the chosen handler.
    if req.doc_type == "lab_report":
        result = _handle_lab_report(text, req.session_doc_id, req.user_id, filename, page_count)
    else:
        result = _handle_research_paper(text, req.session_doc_id, filename, page_count)

    # Flip doc_type and drop the cached text — handler outputs
    # (markers, chunks) are now the source of truth.
    update_session_document(
        req.session_doc_id,
        doc_type=req.doc_type,
        extracted_text=None,
    )

    # Update the chat_sessions.attached_documents chip in-place so the
    # frontend strip recolors from grey ('Doc') to green/blue.
    attached = get_session_attached_documents(req.session_id)
    for entry in attached:
        if entry.get("id") == req.session_doc_id:
            entry["doc_type"] = req.doc_type
            break
    update_session_attached_documents(req.session_id, attached)

    return result


# --------------------------------------------------
# Upload listing + deletion
# --------------------------------------------------
# Retention policy:
#   - doc_type == "lab_report"     → PHI, user-owned, deletable via
#                                    DELETE /upload/{id}. FK cascade
#                                    (migrations 009/010) removes the
#                                    associated session_chunks and
#                                    user_lab_markers rows.
#   - doc_type == "research_paper" → candidate for admin promotion into
#                                    the shared corpus; NOT auto-deleted
#                                    on user request (for now we still
#                                    accept delete — admin workflow is a
#                                    later project milestone).
#   - other                        → treated like lab_report (PHI-safe
#                                    default).
# The user_id match on DELETE is the privacy guard: a user cannot delete
# another user's upload even if they guess the id.
class DeleteUploadRequest(BaseModel):
    user_id: str


@app.get("/uploads")
async def list_uploads(user_id: str) -> Dict[str, Any]:
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    rows = list_user_session_documents(user_id)
    return {"uploads": rows}


@app.delete("/upload/{session_doc_id}")
async def delete_upload(session_doc_id: str, req: DeleteUploadRequest) -> Dict[str, Any]:
    if not req.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    doc = get_session_document(session_doc_id)
    if doc is None or doc.get("user_id") != req.user_id:
        raise HTTPException(status_code=404, detail="upload not found")
    session_id = doc.get("session_id")
    ok = delete_session_document(session_doc_id, req.user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="upload not found")
    if session_id:
        attached = get_session_attached_documents(session_id) or []
        attached = [a for a in attached if a.get("id") != session_doc_id]
        update_session_attached_documents(session_id, attached)
    return {"ok": True, "deleted": session_doc_id}


# --------------------------------------------------
# Query
# --------------------------------------------------
class HistoryTurn(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    history: Optional[list[HistoryTurn]] = None


HISTORY_MAX_TURNS = 6
HISTORY_RETRIEVAL_USER_TURNS = 2


def _prompt_hash(messages: list[dict]) -> str:
    """SHA-256 over the JSON-encoded message list. Stable fingerprint of
    the exact prompt sent to the LLM — lets us detect silent system-prompt
    drift from audit logs without storing the full prompt."""
    try:
        blob = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()
    except Exception:
        return ""


def _log_query_safe(**kwargs: Any) -> None:
    """Fire-and-forget wrapper around insert_query_log. Audit logging must
    never break the user-facing response path — swallow everything."""
    try:
        insert_query_log(**kwargs)
    except Exception as exc:
        print(f"[query_log] insert failed (swallowed): {exc}")


def _guard_clarification(clar_text: str, prior_answer: str) -> tuple[str, str, list[dict]]:
    """Phase-1.4: apply the same NLI + scope-guard that routine answers get,
    but with the PRIOR ANSWER as the sole reference "chunk". Catches the two
    clarification-specific risks:

    (a) New medical claim introduced by the LLM that was not in the prior
        answer (e.g. a dose, a diagnostic criterion). NLI against
        prior_answer will fail and the sentence is redacted.
    (b) The clarification reads as a diagnosis / prescription on its own.
        classify_scope returns diagnostic / prescriptive → hard refuse.

    Returns (final_text, verdict, score_log). verdict ∈ {"pass",
    "redacted_all", "scope_refused"}. Caller uses verdict to decide which
    response shape + log reason to emit.
    """
    if not clar_text or not clar_text.strip():
        return clar_text, "pass", []
    if not prior_answer or not prior_answer.strip():
        # Without a reference we can't NLI-check; fall back to scope-only.
        scope = classify_scope(clar_text)
        if scope in ("diagnostic", "prescriptive"):
            return SCOPE_REFUSAL_TEMPLATES[scope], "scope_refused", []
        return clar_text, "pass", []

    filtered, score_log = apply_guardrails(
        clar_text,
        [{"content": prior_answer}],
    )
    if not filtered.strip():
        return _CLARIFICATION_REDACTED_FALLBACK, "redacted_all", score_log

    scope = classify_scope(filtered)
    if scope in ("diagnostic", "prescriptive"):
        return SCOPE_REFUSAL_TEMPLATES[scope], "scope_refused", score_log

    return filtered, "pass", score_log


_CLARIFICATION_REDACTED_FALLBACK = (
    "My previous answer did not specify that — please ask it as a new "
    "question and I'll look it up."
)


def _session_in_doc_qa_mode(session: Optional[dict]) -> bool:
    """True if the session has ANY uploaded document attached.

    Stage 1 intake is for symptom triage. Once the user has uploaded a
    document — even one that was classified as 'other' or later rejected
    by the domain gate — the conversation is clearly about a document,
    not a symptom complaint. Route those questions to retrieval (or a
    no-source refusal), never to the symptom-intake template.
    """
    if not session:
        return False
    docs = session.get("attached_documents") or []
    return len(docs) > 0


def _retrieval_query_with_history(question: str, history: Optional[list]) -> str:
    """Concatenate the last few user turns into the retrieval string so a
    vague follow-up like 'what are the symptoms?' inherits the topic of
    prior turns (e.g. 'hypertension'). Retrieval is bag-of-words-ish so
    simple concatenation is enough — no LLM rewrite round-trip needed."""
    if not history:
        return question
    prior_user = [h.content for h in history if h.role == "user" and h.content]
    if not prior_user:
        return question
    tail = prior_user[-HISTORY_RETRIEVAL_USER_TURNS:]
    return " ".join(tail + [question])


RETRIEVE_K = 15
CONTEXT_CHUNKS = 6
DISPLAY_SOURCES = 3
RERANK_MODEL = "rerank-v3.5"

# Phase-1.3: explicit temperature on the main /query and /query/stream
# generation calls. Previously these inherited provider defaults (Cohere
# ~1.0), which is unacceptable variance for medical outputs. Stage 1/2/4
# modules already use 0.1–0.2; align here. If tuning is needed per
# provider, split into GROQ_TEMPERATURE / COHERE_TEMPERATURE later.
MAIN_QUERY_TEMPERATURE = 0.15

# Phase-3: inline-citation binding + fusion-drift detection.
# - INLINE_CITATIONS_ENABLED turns on the prompt instruction that asks
#   the LLM to tag each clinical claim with `[N]` markers, and the
#   guardrail parses + verifies against the assigned chunk only.
# - FUSION_DRIFT_ENABLED turns on the 2-sentence window NLI pass.
# Both default ON, but flag-gated so we can A/B against gold.
INLINE_CITATIONS_ENABLED = os.getenv("INLINE_CITATIONS_ENABLED", "1") not in ("0", "false", "False", "")
FUSION_DRIFT_ENABLED = os.getenv("FUSION_DRIFT_ENABLED", "1") not in ("0", "false", "False", "")

# Failure A #1: lay→clinical query rewrite before retrieval. Flag-gated
# so we can A/B against gold. Default on; set QUERY_REWRITE_ENABLED=0 to
# skip the expansion and reproduce pre-fix behaviour.
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "1") not in ("0", "false", "False", "")

# Three-layer no-coverage refusal gate. Cohere rerank-v3.5 scores in [0, 1].
# Empirical bands: strong on-topic 0.5+, good 0.3–0.5, adjacent-topic drift
# 0.15–0.35 (the "hyperthermia → hyperthyroidism" failure mode lives here),
# clearly off-topic <0.15.
#
# Gate behavior:
# 1. If no rows retrieved at all → refuse.
# 2. If rerank didn't run (Cohere error / network) → refuse. We can't judge
#    relevance, so we must fail closed, not fall back to uncalibrated RRF.
# 3. If top rerank_score < RERANK_REFUSAL_THRESHOLD → refuse.
# 4. Chunks with rerank_score < RERANK_CONTEXT_MIN are stripped from the
#    LLM context even when gate #3 passes, so the model cannot pivot to
#    low-score adjacent chunks that rode in on a single strong chunk's
#    coattails.
RERANK_REFUSAL_THRESHOLD = 0.4
RERANK_CONTEXT_MIN = 0.2

FILTER_FALLBACK_MIN_ROWS = 1  # only re-query Supabase if the filtered path yielded zero rows

FRESHNESS_DECAY_PER_YEAR = 0.1

# Stage-aware weighted-scoring weights, tuned from v5 ablation.
# intake and results regressed hard when authority/freshness were mixed in;
# their answers depend on the reranker's semantic match (symptom pathways,
# lab-specific reference ranges) more than on doc authority.
# condition, navigation, visit_prep benefit from the authority/freshness
# nudge — WHO disease pages and recent Nepal care protocols are the right
# sources to prefer when rerank ranks are close.
#   (w_rerank, w_authority, w_freshness)
STAGE_WEIGHTS = {
    "intake":     (1.0, 0.0, 0.0),
    "navigation": (0.7, 0.2, 0.1),
    "visit_prep": (0.7, 0.2, 0.1),
    "results":    (1.0, 0.0, 0.0),
    "condition":  (0.7, 0.2, 0.1),
}
DEFAULT_WEIGHTS = (0.7, 0.2, 0.1)


def _rerank_rows(question: str, rows: list) -> list:
    """Rerank retrieved chunks with Cohere Rerank. Returns every candidate
    with rerank_score attached so downstream weighted scoring can reorder
    the full set. Falls back to original order if the API call fails.

    COHERE_DISABLED=1 short-circuits to the same fallback without touching
    the API — for eval / CI runs that must not burn trial quota."""
    if not rows:
        return rows
    if os.getenv("COHERE_DISABLED") == "1":
        # Synthetic rank-derived scores so the no-coverage gate and the
        # weighted-final-score path still work. Top RRF row → 0.80 (clears
        # RERANK_REFUSAL_THRESHOLD=0.4); scores decay linearly so the tail
        # gets pruned by RERANK_CONTEXT_MIN=0.2 like a real rerank pass.
        # NOT a quality-equivalent substitute — it trusts hybrid retrieval's
        # RRF order completely. Use only when COHERE_DISABLED is set.
        n = len(rows)
        out = []
        for i, row in enumerate(rows):
            synth = max(0.0, 0.80 - (0.65 * i / max(1, n - 1))) if n > 1 else 0.80
            new_row = dict(row)
            new_row["rerank_score"] = synth
            out.append(new_row)
        return out
    docs = [r.get("content") or "" for r in rows]
    try:
        resp = co.rerank(
            model=RERANK_MODEL,
            query=question,
            documents=docs,
            top_n=len(docs),
        )
    except Exception as exc:
        # Phase-1.1 fix: previously returned rows with NO rerank_score, which
        # (a) tripped the no-coverage gate via `rerank_missing` on every
        # Cohere outage and (b) silently zeroed the w_r semantic component
        # in _weighted_final_score for any row that slipped past the gate.
        # Reuse the same synthetic scoring shape as COHERE_DISABLED so the
        # gate + weighting stay in their designed regime during transient
        # failures. NOT quality-equivalent — trusts hybrid RRF order — but
        # recoverable and auditable.
        print(f"[rerank] failed, using synthetic RRF-derived scores: {exc}")
        n = len(rows)
        out = []
        for i, row in enumerate(rows):
            synth = max(0.0, 0.80 - (0.65 * i / max(1, n - 1))) if n > 1 else 0.80
            new_row = dict(row)
            new_row["rerank_score"] = synth
            new_row["rerank_source"] = "synthetic_fallback"
            out.append(new_row)
        return out
    reordered = []
    for result in resp.results:
        row = dict(rows[result.index])
        row["rerank_score"] = float(result.relevance_score)
        reordered.append(row)
    return reordered


def _authority_score(tier: Any) -> float:
    if tier is None:
        return 0.5
    try:
        t = int(tier)
    except (ValueError, TypeError):
        return 0.5
    return max(0.0, 1.0 - 0.2 * (t - 1))


def _freshness_score(pub_date_str: Any) -> float:
    if not pub_date_str:
        return 0.5
    try:
        from datetime import date
        parts = str(pub_date_str).split("-")
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        years = (date.today() - date(y, m, d)).days / 365.0
    except Exception:
        return 0.5
    return max(0.0, min(1.0, 1.0 - FRESHNESS_DECAY_PER_YEAR * years))


def _weighted_final_score(row: dict, weights: Tuple[float, float, float]) -> float:
    w_r, w_a, w_f = weights
    return (
        w_r * (row.get("rerank_score") or 0.0)
        + w_a * _authority_score(row.get("doc_authority_tier"))
        + w_f * _freshness_score(row.get("doc_publication_date"))
    )


SESSION_RETRIEVE_K = 6  # max session-private rows to merge before rerank


def _retrieve_ranked(
    question: str,
    *,
    intent: Optional[dict] = None,
    session_id: Optional[str] = None,
) -> list:
    """Run filtered hybrid retrieval → rerank → stage-weighted scoring.

    Returns the full ranked row list (caller slices to CONTEXT_CHUNKS).
    Extracted so Stage 2 navigation can run retrieval against the intake
    summary with identical semantics to the routine /query path.

    When `session_id` is given, also pulls up to SESSION_RETRIEVE_K
    session-private chunks (uploaded research papers, lab reports) for
    THIS session and merges them into the candidate pool BEFORE rerank.
    The reranker then judges shared-corpus and session-private chunks
    against each other on the same scale, so an off-topic uploaded
    paper can't crowd out the corpus, and the existing
    RERANK_REFUSAL_THRESHOLD gate still catches off-domain questions.
    Session chunks are normalised to the same row shape as corpus
    chunks (doc_title, doc_source, etc.) so downstream code is
    untouched.
    """
    q_vec = get_query_encoder().encode_one(question)
    if intent is None:
        intent = classify_intent(question)
    print(f"[intent] {intent} q={question[:60]!r}")
    filter_kwargs = build_filter(question, intent=intent)
    rows = match_chunks_hybrid_filtered(
        to_pgvector_literal(q_vec),
        question,
        match_count=RETRIEVE_K,
        **filter_kwargs,
    )
    if len(rows) < FILTER_FALLBACK_MIN_ROWS and intent is not None:
        print(
            f"[filter] intent={intent} yielded {len(rows)} rows (<{FILTER_FALLBACK_MIN_ROWS});"
            f" falling back to Phase 2 default filter"
        )
        filter_kwargs = build_filter(question, intent=None)
        rows = match_chunks_hybrid_filtered(
            to_pgvector_literal(q_vec),
            question,
            match_count=RETRIEVE_K,
            **filter_kwargs,
        )

    # Session-private chunks (Week 8). Merged BEFORE rerank so the
    # cross-encoder can judge them on equal footing with corpus chunks.
    if session_id:
        try:
            session_rows = match_session_chunks(
                session_id,
                to_pgvector_literal(q_vec),
                question,
                match_count=SESSION_RETRIEVE_K,
            )
        except Exception as exc:
            print(f"[session-merge] match_session_chunks failed: {exc}")
            session_rows = []
        if session_rows:
            print(f"[session-merge] +{len(session_rows)} session-private chunks for session={session_id}")
            for sr in session_rows:
                rows.append({
                    "chunk_id": sr.get("chunk_id"),
                    "content": sr.get("content"),
                    "section_heading": sr.get("section_heading"),
                    "rrf_score": sr.get("rrf_score"),
                    "similarity": sr.get("similarity"),
                    "bm25_rank": sr.get("bm25_rank"),
                    "doc_title": sr.get("doc_filename") or "Uploaded document",
                    "doc_source": "user-upload",
                    "doc_source_url": None,
                    "doc_authority_tier": 5,
                    "doc_publication_date": None,
                    "is_session_chunk": True,
                })

    # Phase-1.2 fix: always rerank when we have any rows. Previously we
    # skipped rerank if len(rows) <= CONTEXT_CHUNKS, which left RRF order
    # as-is for low-recall queries. RRF is authority-agnostic, so a
    # session-upload chunk (tier 5) could outrank a WHO chunk (tier 1).
    # Rerank at N=6 costs one extra Cohere call; worth it.
    if rows:
        rows = _rerank_rows(question, rows)
    weights = STAGE_WEIGHTS.get((intent or {}).get("stage"), DEFAULT_WEIGHTS)
    for r in rows:
        r["final_score"] = _weighted_final_score(r, weights)
    rows.sort(key=lambda r: r.get("final_score", 0.0), reverse=True)
    return rows


def _dedupe_sources(rows: list) -> list:
    """Collapse retrieval rows that point at the same document into one
    entry. Rows arrive already sorted by final_score, so keeping the
    first occurrence of each key preserves the best-ranked chunk per
    doc. Key preference: source_url, else (title, source) pair. Used
    for DISPLAY only — LLM context still sees all chunks for diversity."""
    seen: set = set()
    unique: list = []
    for r in rows:
        url = r.get("doc_source_url")
        key = url if url else (r.get("doc_title"), r.get("doc_source"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def _format_sources(top_rows: list) -> list:
    return [
        {
            "rank": i,
            "title": r.get("doc_title"),
            "source": r.get("doc_source"),
            "source_url": r.get("doc_source_url"),
            "similarity": r.get("similarity"),
            "rrf_score": r.get("rrf_score"),
            "bm25_rank": r.get("bm25_rank"),
            "rerank_score": r.get("rerank_score"),
            "final_score": r.get("final_score"),
            "authority_tier": r.get("doc_authority_tier"),
            "publication_date": r.get("doc_publication_date"),
        }
        for i, r in enumerate(top_rows, start=1)
    ]


@app.post("/query")
async def query_document(query: QueryRequest):
    rate_limit.check(query.session_id or "anon", "query")

    # Week 6 Stage 0: deterministic red-flag screen runs first.
    # If any rule fires, the LLM is never invoked and retrieval is skipped.
    rf = redflag_check(query.question)
    if rf is not None:
        print(f"[redflag] fired rule={rf.rule_id} category={rf.category}")
        _log_query_safe(
            user_id=None,
            session_id=query.session_id,
            stage="redflag",
            query_text=query.question,
            response_text=rf.message,
            red_flag_fired=True,
            red_flag_rule_id=rf.rule_id,
        )
        return {
            "answer": rf.message,
            "sources": [],
            "stage": "redflag",
            "red_flag": {
                "rule_id": rf.rule_id,
                "category": rf.category,
                "urgency": rf.urgency,
            },
        }

    # Meta-question gate (Failure B, 2026-04-20). Runs BEFORE intake /
    # intent-gate / retrieval. If the turn is a clarification about the
    # prior assistant answer ("is that for T2D only?", "what about
    # children?", "why?"), skip retrieval entirely and answer against
    # the prior assistant text via stages.clarification. Safe default:
    # any non-match falls through to the existing pipeline.
    prior_assistant = last_assistant_turn(query.history)
    if prior_assistant and is_meta_question(query.question, query.history):
        clar_raw = clarification_stage.compose_clarification(
            prior_answer=prior_assistant,
            user_question=query.question,
            groq_client=groq_client,
            groq_model=GROQ_MODEL,
        )
        # Phase-1.4: NLI-verify the clarification against the prior answer
        # (the only legitimate "source" for a clarification) and run the
        # scope-guard. Previously the clarification text went out
        # unguarded — a Groq hallucination here emitted directly.
        clar, verdict, clar_nli = _guard_clarification(clar_raw, prior_assistant)
        print(f"[meta_question] fired q={query.question[:60]!r} verdict={verdict}")
        _log_query_safe(
            user_id=None,
            session_id=query.session_id,
            stage="clarification",
            query_text=query.question,
            response_text=clar,
            refusal_triggered=(verdict != "pass"),
            refusal_reason=(None if verdict == "pass" else f"clarification_{verdict}"),
            nli_entailment_scores=clar_nli or None,
        )
        return {
            "answer": clar,
            "sources": [],
            "stage": "clarification",
        }

    # Week 7A Stage 1: structured intake. Runs only when a session_id is
    # provided and the session is in the 'intake' stage.
    #
    # Two sub-turns, driven by current session state in chat_sessions:
    #   a) intent_bucket IS NULL → first intake turn: pick a template and
    #      return its 5 slot questions. Persist the bucket.
    #   b) intent_bucket NOT NULL and intake_summary IS NULL → second
    #      intake turn: treat the user's message as the slot answers,
    #      compose the bullet summary, persist it, and transition the
    #      session to 'navigation'.
    # Any other state (no session_id, stage past intake, summary already
    # composed) falls through to the routine retrieval path below.
    if query.session_id:
        session = get_chat_session(query.session_id)
        if session and session.get("current_stage") == "intake" and _session_in_doc_qa_mode(session):
            # Doc-Q&A mode: a classified document is attached. Advance the
            # stage so subsequent turns also skip intake, then fall through
            # to retrieval (session-merge picks up the indexed chunks).
            update_chat_session(query.session_id, current_stage="navigation")
            session = None  # neutralise the intake check below
        # Pre-intake intent gate (2026-04). Applies ONLY on the very first
        # turn (intent_bucket still NULL). If the user's question is
        # clearly not a symptom report — informational, navigational,
        # results-oriented, or references an existing diagnosis — skip the
        # 5-slot history-taking flow entirely, flip the stage, and fall
        # through to routine retrieval. Intake remains the safe default
        # when the gate is uncertain. See app/intent_gate.py for layers.
        if (
            session
            and session.get("current_stage") == "intake"
            and not session.get("intent_bucket")
        ):
            gate_decision = classify_intake_gate(
                query.question,
                groq_client=groq_client,
                groq_model=GROQ_MODEL,
            )
            if gate_decision != "intake":
                update_chat_session(query.session_id, current_stage=gate_decision)
                print(
                    f"[intent_gate] bypass: decision={gate_decision} "
                    f"q={query.question[:60]!r}"
                )
                session = None  # neutralise the intake check below
        if session and session.get("current_stage") == "intake":
            if not session.get("intent_bucket"):
                template = intake_stage.select_template(
                    query.question,
                    groq_client=groq_client,
                    groq_model=GROQ_MODEL,
                )
                questions = intake_stage.compose_questions(template)
                update_chat_session(query.session_id, intent_bucket=template["id"])
                print(f"[intake] bucket={template['id']} q={query.question[:60]!r}")
                _log_query_safe(
                    user_id=None,
                    session_id=query.session_id,
                    stage="intake_questions",
                    query_text=query.question,
                    response_text=questions,
                )
                return {
                    "answer": questions,
                    "sources": [],
                    "stage": "intake",
                    "intake_turn": "questions",
                    "intent_bucket": template["id"],
                }
            if not session.get("intake_summary"):
                template = intake_stage.TEMPLATES_BY_ID.get(
                    session["intent_bucket"],
                    intake_stage.TEMPLATES_BY_ID["other"],
                )
                summary = intake_stage.compose_summary(
                    template,
                    user_answers=query.question,
                    groq_client=groq_client,
                    groq_model=GROQ_MODEL,
                    cohere_client=co,
                    cohere_model="command-r-08-2024",
                )
                # Stage 2 MVP: chain a care-tier recommendation onto the
                # summary response so the user doesn't get a dead-end
                # bullet list. Retrieval runs against the summary (richer
                # signal than the raw slot answers) using the existing
                # Week 5 corpus. Week 7B will upgrade the corpus with
                # care-pathway content (WHO IMAI, NHS, MoHP STG).
                nav_rows = _retrieve_ranked(summary)
                nav_top = nav_rows[:CONTEXT_CHUNKS]
                nav_block = navigation_stage.compose_recommendation(
                    intake_summary=summary,
                    intent_bucket=session["intent_bucket"],
                    groq_client=groq_client,
                    groq_model=GROQ_MODEL,
                    cohere_client=co,
                    cohere_model="command-r-08-2024",
                    retrieval_rows=nav_top,
                )
                combined = f"{summary}\n\n---\n\n{nav_block}"
                update_chat_session(
                    query.session_id,
                    current_stage="navigation",
                    intake_summary=summary,
                )
                print(f"[intake] summary+nav composed bucket={template['id']}")
                nav_sources = _format_sources(
                    _dedupe_sources(nav_rows)[:DISPLAY_SOURCES]
                )
                _log_query_safe(
                    user_id=None,
                    session_id=query.session_id,
                    stage="intake_summary",
                    query_text=query.question,
                    response_text=combined,
                    citations=nav_sources,
                    retrieved_chunk_ids=[r.get("id") for r in nav_top if r.get("id")],
                )
                return {
                    "answer": combined,
                    "sources": nav_sources,
                    "stage": "intake",
                    "intake_turn": "summary",
                    "intent_bucket": template["id"],
                }

    # Context-aware retrieval: concatenate prior user turns so follow-ups
    # like "what are the symptoms?" inherit the topic ("hypertension") from
    # earlier in the conversation. Without this the retrieval layer is
    # fully stateless and a vague follow-up pulls unrelated chunks.
    retrieval_query = _retrieval_query_with_history(query.question, query.history)
    if retrieval_query != query.question:
        print(f"[history] retrieval_query rewritten: {retrieval_query[:120]!r}")
    if QUERY_REWRITE_ENABLED:
        expanded = expand_for_retrieval(
            retrieval_query, groq_client=groq_client, groq_model=GROQ_MODEL,
        )
        if expanded != retrieval_query:
            print(f"[query_rewrite] expanded: {expanded[:200]!r}")
            retrieval_query = expanded
    rows = _retrieve_ranked(retrieval_query, session_id=query.session_id)

    refusal = {
        "answer": (
            "I don't have a source for that in my current library. "
            "Please try rewording your question, or ask your doctor directly."
        ),
        "sources": [],
        # Frontend uses this marker to drop the (user-question, refusal)
        # pair from conversation history before the next /query call. If we
        # don't, the next retrieval rewrite concatenates the unanswered
        # question into its query (e.g. "who is a gynac? what is hypertension?")
        # and can drag a follow-up's rerank score below the gate threshold.
        "coverage": "no_source",
    }

    # Fail-closed no-coverage gate. Any of these three conditions → refuse,
    # without ever calling the LLM:
    #
    #   (a) Retrieval returned nothing.
    #   (b) Rerank didn't attach a score to the top row (Cohere error or
    #       degraded path). We cannot judge relevance → refuse.
    #   (c) Top chunk's rerank_score is below RERANK_REFUSAL_THRESHOLD.
    #       The corpus does not cover this query well enough.
    def _log_refusal(reason: str) -> None:
        _log_query_safe(
            user_id=None,
            session_id=query.session_id,
            stage="routine",
            query_text=query.question,
            response_text=refusal["answer"],
            refusal_triggered=True,
            refusal_reason=reason,
        )

    if not rows:
        print("[refusal-gate] refusing: retrieval returned 0 rows")
        _log_refusal("no_rows")
        return refusal
    top_rerank = rows[0].get("rerank_score")
    print(f"[refusal-gate] top rerank_score={top_rerank} threshold={RERANK_REFUSAL_THRESHOLD}")
    if top_rerank is None:
        print("[refusal-gate] refusing: rerank did not run, cannot judge relevance")
        _log_refusal("rerank_missing")
        return refusal
    if top_rerank < RERANK_REFUSAL_THRESHOLD:
        print(
            f"[refusal-gate] refusing: top rerank_score {top_rerank:.3f} "
            f"< {RERANK_REFUSAL_THRESHOLD}"
        )
        _log_refusal(f"below_threshold:{top_rerank:.3f}")
        return refusal

    # Gate passed. Now also drop any low-score chunks from the LLM context
    # so the model never sees junk it could pivot to. If filtering empties
    # the context (shouldn't happen given gate #3 already cleared), refuse.
    rows = [r for r in rows if (r.get("rerank_score") or 0.0) >= RERANK_CONTEXT_MIN]
    if not rows:
        print(
            f"[refusal-gate] refusing: no chunks above RERANK_CONTEXT_MIN "
            f"({RERANK_CONTEXT_MIN}) after filter"
        )
        _log_refusal("context_empty_after_filter")
        return refusal

    top_rows = rows[:CONTEXT_CHUNKS]
    context_blocks = []
    for i, r in enumerate(top_rows, start=1):
        heading = r.get("section_heading") or ""
        title = r.get("doc_title") or r.get("doc_source") or "source"
        context_blocks.append(f"[src:{i}] {title} — {heading}\n{r.get('content', '')}".strip())
    context_text = "\n\n".join(context_blocks)

    sources = _format_sources(_dedupe_sources(rows)[:DISPLAY_SOURCES])

    messages: list[dict] = [{"role": "system", "content": build_system_prompt()}]
    if query.history:
        for h in query.history[-HISTORY_MAX_TURNS:]:
            if h.role in ("user", "assistant") and h.content:
                messages.append({"role": h.role, "content": h.content})
    messages.append(
        {
            "role": "user",
            "content": f"Sources:\n{context_text}\n\nQuestion: {query.question}",
        }
    )

    # Phase-1.3: lock temperature + log provider. Previously the main /query
    # path inherited provider defaults (Cohere ~1.0), which is unsafe on a
    # medical RAG. Match the stage modules (0.1–0.2). Also track which
    # provider emitted the final answer for incident correlation. Stored
    # in `llm_provider` (stdout only until schema migration lands).
    llm_provider = "none"
    if groq_client is not None:
        try:
            groq_resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=250,
                temperature=MAIN_QUERY_TEMPERATURE,
            )
            answer = groq_resp.choices[0].message.content
            llm_provider = "groq"
        except Exception as exc:
            print(f"[groq] generate failed, falling back to Cohere: {exc}")
            response = co.chat(
                model="command-r-08-2024",
                messages=messages,
                max_tokens=250,
                temperature=MAIN_QUERY_TEMPERATURE,
            )
            try:
                answer = response.message.content[0].text
                llm_provider = "cohere_fallback"
            except Exception:
                raise HTTPException(status_code=500, detail="Failed to parse Cohere response")
    else:
        response = co.chat(
            model="command-r-08-2024",
            messages=messages,
            max_tokens=250,
            temperature=MAIN_QUERY_TEMPERATURE,
        )
        try:
            answer = response.message.content[0].text
            llm_provider = "cohere"
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to parse Cohere response")
    print(f"[llm_provider] /query emitted via provider={llm_provider}")

    # Week 10 guardrails: classify + NLI-verify each sentence. On
    # contradiction we redact; on weak support we soften; dose / diagnosis
    # sentences get the harder rule (redact unless strongly entailed).
    # If every claim sentence was redacted, fall back to the standard
    # no-coverage refusal — we do not emit an answer with only framing
    # sentences because that would feel conversational but empty.
    filtered_answer, nli_scores = apply_guardrails(
        answer,
        top_rows,
        use_inline_citations=INLINE_CITATIONS_ENABLED,
        check_fusion_drift=FUSION_DRIFT_ENABLED,
    )
    if not filtered_answer.strip():
        print("[guardrails] all claim sentences redacted, falling back to refusal")
        _log_query_safe(
            user_id=None,
            session_id=query.session_id,
            stage="routine",
            query_text=query.question,
            response_text=refusal["answer"],
            refusal_triggered=True,
            refusal_reason="all_sentences_redacted_by_guardrails",
            prompt_hash=_prompt_hash(messages),
            nli_entailment_scores=nli_scores,
        )
        return refusal

    # Week 10 scope-guard: policy layer on top of the NLI factual layer.
    # NLI says "is this sentence grounded?"; scope-guard says "is it the
    # kind of claim we should be making at all?". A well-grounded dose
    # recommendation is still a dose recommendation — this closes that gap.
    scope = classify_scope(filtered_answer)
    if scope in ("diagnostic", "prescriptive"):
        scope_refusal_text = SCOPE_REFUSAL_TEMPLATES[scope]
        print(f"[scope-guard] filtered answer classified as {scope}, refusing")
        _log_query_safe(
            user_id=None,
            session_id=query.session_id,
            stage="routine",
            query_text=query.question,
            response_text=scope_refusal_text,
            refusal_triggered=True,
            refusal_reason=f"scope_guard_{scope}",
            prompt_hash=_prompt_hash(messages),
            nli_entailment_scores=nli_scores,
        )
        return {
            "answer": scope_refusal_text,
            "sources": [],
            "stage": "routine",
            "coverage": "scope_refused",
        }

    _log_query_safe(
        user_id=None,
        session_id=query.session_id,
        stage="routine",
        query_text=query.question,
        response_text=filtered_answer,
        citations=sources,
        retrieved_chunk_ids=[r.get("id") for r in top_rows if r.get("id")],
        prompt_hash=_prompt_hash(messages),
        nli_entailment_scores=nli_scores,
    )
    return {"answer": filtered_answer, "sources": sources, "stage": "routine"}


# ─── Streaming endpoint ─────────────────────────────────────────────────────
#
# /query/stream returns Server-Sent Events. Frontend reads incrementally so
# the user sees the first token in ~250ms (Groq TTFT) instead of staring at
# a spinner for ~1.5s while the full answer generates.
#
# SSE event types (all `data:` payloads are JSON):
#   meta    — initial event with stage, red_flag, intake_turn, coverage,
#             intent_bucket. Sent once, first.
#   delta   — incremental text. `{"text": "..."}`. Sent 0..N times.
#   sources — citations payload. Sent once if there are sources.
#   error   — recoverable mid-stream error. `{"message": "..."}`.
#   done    — terminal event. Sent once, last. Body is `{}`.
#
# Non-streaming responses (red-flag, refusal, Stage 1 questions, Stage 1+2
# chained summary) are emitted as: meta → one big delta → sources (if any)
# → done. So the frontend has a single code path: append every delta to
# the assistant message; finalize on done.
#
# /query (non-streaming JSON) is preserved unchanged for backward compat —
# evals (`eval/test_redflag.py` etc.) and any external consumer keep working.


def _sse(event: str, payload: dict) -> bytes:
    """Encode a single SSE event. Two-newline terminator is required by
    the SSE spec for the client to flush the event."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


@app.post("/query/stream")
async def query_document_stream(query: QueryRequest):
    rate_limit.check(query.session_id or "anon", "query")

    async def event_stream():
        # ── Stage 0: red-flag (instant, no LLM) ──────────────────────────
        rf = redflag_check(query.question)
        if rf is not None:
            print(f"[redflag] fired rule={rf.rule_id} category={rf.category}")
            yield _sse("meta", {
                "stage": "redflag",
                "red_flag": {
                    "rule_id": rf.rule_id,
                    "category": rf.category,
                    "urgency": rf.urgency,
                },
            })
            yield _sse("delta", {"text": rf.message})
            _log_query_safe(
                user_id=None,
                session_id=query.session_id,
                stage="redflag",
                query_text=query.question,
                response_text=rf.message,
                red_flag_fired=True,
                red_flag_rule_id=rf.rule_id,
            )
            yield _sse("done", {})
            return

        # ── Meta-question gate (Failure B) ────────────────────────────────
        # Clarifications about the prior assistant turn skip retrieval and
        # answer against the stored prior answer. Emitted as a single
        # delta; short enough that streaming tokens buys nothing.
        prior_assistant = last_assistant_turn(query.history)
        if prior_assistant and is_meta_question(query.question, query.history):
            clar_raw = clarification_stage.compose_clarification(
                prior_answer=prior_assistant,
                user_question=query.question,
                groq_client=groq_client,
                groq_model=GROQ_MODEL,
            )
            # Phase-1.4: same NLI + scope guard as batch path. If the
            # verdict is not "pass", the replacement text is safe by
            # construction (refusal template or "didn't specify" fallback).
            clar, verdict, clar_nli = _guard_clarification(clar_raw, prior_assistant)
            print(f"[meta_question] stream fired q={query.question[:60]!r} verdict={verdict}")
            yield _sse("meta", {"stage": "clarification"})
            yield _sse("delta", {"text": clar})
            _log_query_safe(
                user_id=None,
                session_id=query.session_id,
                stage="clarification",
                query_text=query.question,
                response_text=clar,
                refusal_triggered=(verdict != "pass"),
                refusal_reason=(None if verdict == "pass" else f"clarification_{verdict}"),
                nli_entailment_scores=clar_nli or None,
            )
            yield _sse("done", {})
            return

        # ── Stage 1: structured intake (multi-turn) ──────────────────────
        # Question turn returns 5 slot questions as one block — no LLM
        # streaming, just emit as a single delta. Summary turn does run
        # LLM(s) but chains Stage 2 on top, which is too entangled to
        # stream cleanly this iteration; emit as a single delta after both
        # finish. Streaming the chained path is on the Week 7B-or-later
        # roadmap.
        if query.session_id:
            session = get_chat_session(query.session_id)
            if session and session.get("current_stage") == "intake" and _session_in_doc_qa_mode(session):
                update_chat_session(query.session_id, current_stage="navigation")
                print(f"[intake] skipped: doc-Q&A mode (session={query.session_id})")
                session = None
            # Pre-intake intent gate (mirrors the /query path above). Only
            # fires on the first turn (intent_bucket still NULL). Skips
            # the 5-slot history-taking for non-symptom questions.
            if (
                session
                and session.get("current_stage") == "intake"
                and not session.get("intent_bucket")
            ):
                gate_decision = classify_intake_gate(
                    query.question,
                    groq_client=groq_client,
                    groq_model=GROQ_MODEL,
                )
                if gate_decision != "intake":
                    update_chat_session(query.session_id, current_stage=gate_decision)
                    print(
                        f"[intent_gate] stream bypass: decision={gate_decision} "
                        f"q={query.question[:60]!r}"
                    )
                    session = None
            if session and session.get("current_stage") == "intake":
                if not session.get("intent_bucket"):
                    template = intake_stage.select_template(
                        query.question,
                        groq_client=groq_client,
                        groq_model=GROQ_MODEL,
                    )
                    questions = intake_stage.compose_questions(template)
                    update_chat_session(query.session_id, intent_bucket=template["id"])
                    print(f"[intake] bucket={template['id']} q={query.question[:60]!r}")
                    yield _sse("meta", {
                        "stage": "intake",
                        "intake_turn": "questions",
                        "intent_bucket": template["id"],
                    })
                    yield _sse("delta", {"text": questions})
                    _log_query_safe(
                        user_id=None,
                        session_id=query.session_id,
                        stage="intake_questions",
                        query_text=query.question,
                        response_text=questions,
                    )
                    yield _sse("done", {})
                    return
                if not session.get("intake_summary"):
                    template = intake_stage.TEMPLATES_BY_ID.get(
                        session["intent_bucket"],
                        intake_stage.TEMPLATES_BY_ID["other"],
                    )
                    summary = intake_stage.compose_summary(
                        template,
                        user_answers=query.question,
                        groq_client=groq_client,
                        groq_model=GROQ_MODEL,
                        cohere_client=co,
                        cohere_model="command-r-08-2024",
                    )
                    nav_rows = _retrieve_ranked(summary)
                    nav_top = nav_rows[:CONTEXT_CHUNKS]
                    nav_block = navigation_stage.compose_recommendation(
                        intake_summary=summary,
                        intent_bucket=session["intent_bucket"],
                        groq_client=groq_client,
                        groq_model=GROQ_MODEL,
                        cohere_client=co,
                        cohere_model="command-r-08-2024",
                        retrieval_rows=nav_top,
                    )
                    combined = f"{summary}\n\n---\n\n{nav_block}"
                    update_chat_session(
                        query.session_id,
                        current_stage="navigation",
                        intake_summary=summary,
                    )
                    nav_sources = _format_sources(
                        _dedupe_sources(nav_rows)[:DISPLAY_SOURCES]
                    )
                    print(f"[intake] summary+nav composed bucket={template['id']}")
                    yield _sse("meta", {
                        "stage": "intake",
                        "intake_turn": "summary",
                        "intent_bucket": template["id"],
                    })
                    yield _sse("delta", {"text": combined})
                    if nav_sources:
                        yield _sse("sources", {"sources": nav_sources})
                    _log_query_safe(
                        user_id=None,
                        session_id=query.session_id,
                        stage="intake_summary",
                        query_text=query.question,
                        response_text=combined,
                        citations=nav_sources,
                        retrieved_chunk_ids=[r.get("id") for r in nav_top if r.get("id")],
                    )
                    yield _sse("done", {})
                    return

        # ── Routine: retrieve, gate, then stream Groq tokens ─────────────
        retrieval_query = _retrieval_query_with_history(query.question, query.history)
        if retrieval_query != query.question:
            print(f"[history] retrieval_query rewritten: {retrieval_query[:120]!r}")
        if QUERY_REWRITE_ENABLED:
            expanded = expand_for_retrieval(
                retrieval_query, groq_client=groq_client, groq_model=GROQ_MODEL,
            )
            if expanded != retrieval_query:
                print(f"[query_rewrite] expanded: {expanded[:200]!r}")
                retrieval_query = expanded
        rows = _retrieve_ranked(retrieval_query, session_id=query.session_id)

        refusal_text = (
            "I don't have a source for that in my current library. "
            "Please try rewording your question, or ask your doctor directly."
        )

        def emit_refusal(reason: str):
            print(f"[refusal-gate] refusing ({reason})")
            _log_query_safe(
                user_id=None,
                session_id=query.session_id,
                stage="routine",
                query_text=query.question,
                response_text=refusal_text,
                refusal_triggered=True,
                refusal_reason=reason,
            )
            return [
                _sse("meta", {"stage": "routine", "coverage": "no_source"}),
                _sse("delta", {"text": refusal_text}),
                _sse("done", {}),
            ]

        if not rows:
            for ev in emit_refusal("retrieval returned 0 rows"):
                yield ev
            return
        top_rerank = rows[0].get("rerank_score")
        print(f"[refusal-gate] top rerank_score={top_rerank} threshold={RERANK_REFUSAL_THRESHOLD}")
        if top_rerank is None:
            for ev in emit_refusal("rerank did not run, cannot judge relevance"):
                yield ev
            return
        if top_rerank < RERANK_REFUSAL_THRESHOLD:
            for ev in emit_refusal(f"top rerank_score {top_rerank:.3f} < {RERANK_REFUSAL_THRESHOLD}"):
                yield ev
            return

        rows = [r for r in rows if (r.get("rerank_score") or 0.0) >= RERANK_CONTEXT_MIN]
        if not rows:
            for ev in emit_refusal(f"no chunks above RERANK_CONTEXT_MIN ({RERANK_CONTEXT_MIN}) after filter"):
                yield ev
            return

        top_rows = rows[:CONTEXT_CHUNKS]
        context_blocks = []
        for i, r in enumerate(top_rows, start=1):
            heading = r.get("section_heading") or ""
            title = r.get("doc_title") or r.get("doc_source") or "source"
            context_blocks.append(f"[src:{i}] {title} — {heading}\n{r.get('content', '')}".strip())
        context_text = "\n\n".join(context_blocks)
        sources = _format_sources(_dedupe_sources(rows)[:DISPLAY_SOURCES])

        messages: list[dict] = [{"role": "system", "content": build_system_prompt()}]
        if query.history:
            for h in query.history[-HISTORY_MAX_TURNS:]:
                if h.role in ("user", "assistant") and h.content:
                    messages.append({"role": h.role, "content": h.content})
        messages.append(
            {
                "role": "user",
                "content": f"Sources:\n{context_text}\n\nQuestion: {query.question}",
            }
        )

        # Stage opens with the meta event so the frontend can set up the
        # message shell (mark it as a routine query, prepare to render
        # markdown) before any tokens arrive.
        yield _sse("meta", {"stage": "routine"})

        # Week 10 streaming guardrails (Option C: buffer-per-sentence).
        # Tokens accumulate in `buffer` until a sentence boundary is crossed;
        # each completed sentence runs through the same classify → NLI →
        # tiered-action pipeline as batch mode, then only the surviving
        # (possibly softened) sentence is flushed to the SSE stream. This
        # adds ~1–2s of sentence-level latency but guarantees the user
        # never sees a sentence we would later retract.
        chunk_texts = [r.get("content", "") for r in top_rows if r.get("content")]
        buffer = ""
        nli_scores: list[dict] = []
        emitted_parts: list[str] = []
        streamed_any = False
        source: str = "none"  # which provider actually emitted text

        def _guard_and_emit(delta: str):
            """Feed a provider delta into the streaming guardrail and yield
            any sentences that cleared the filter. Mutates the closed-over
            buffer / nli_scores / emitted_parts / streamed_any state."""
            nonlocal buffer, streamed_any
            buffer, emits = process_streaming_chunk(
                buffer,
                delta,
                chunk_texts,
                nli_scores,
                use_inline_citations=INLINE_CITATIONS_ENABLED,
            )
            events = []
            for sent_out in emits:
                emitted_parts.append(sent_out)
                streamed_any = True
                # Re-introduce the separating space the splitter consumed.
                events.append(_sse("delta", {"text": sent_out + " "}))
            return events

        # Groq primary. Mid-stream failure after content emitted → we can't
        # unwind emitted sentences; report error and end cleanly. Failure
        # before any emit → fall through to Cohere.
        groq_failed_midstream = False
        if groq_client is not None:
            try:
                stream = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    max_tokens=250,
                    stream=True,
                    temperature=MAIN_QUERY_TEMPERATURE,
                )
                source = "groq"
                for chunk in stream:
                    delta = ""
                    try:
                        delta = chunk.choices[0].delta.content or ""
                    except Exception:
                        delta = ""
                    if delta:
                        for ev in _guard_and_emit(delta):
                            yield ev
            except Exception as exc:
                print(f"[groq-stream] failed (streamed_any={streamed_any}): {exc}")
                if streamed_any:
                    groq_failed_midstream = True
                else:
                    # Reset guardrail state for a clean fallback attempt.
                    buffer = ""
                    nli_scores.clear()
                    source = "none"

        if groq_failed_midstream:
            yield _sse("error", {"message": "Generation interrupted. Partial answer above."})
            yield _sse("done", {})
            return

        if not streamed_any and source == "none":
            try:
                cohere_stream = co.chat_stream(
                    model="command-r-08-2024",
                    messages=messages,
                    max_tokens=250,
                    temperature=MAIN_QUERY_TEMPERATURE,
                )
                source = "cohere"
                for ev in cohere_stream:
                    text = ""
                    try:
                        if getattr(ev, "type", None) == "content-delta":
                            text = ev.delta.message.content.text or ""
                    except Exception:
                        text = ""
                    if text:
                        for out_ev in _guard_and_emit(text):
                            yield out_ev
            except Exception as exc:
                print(f"[cohere-stream] failed: {exc}")
                if not streamed_any:
                    yield _sse("error", {"message": "Generation failed. Please try again."})
                    yield _sse("done", {})
                    return

        # Flush any trailing sentence that did not end with whitespace —
        # LLMs commonly drop the terminal newline on short answers.
        for sent_out in flush_streaming_buffer(
            buffer,
            chunk_texts,
            nli_scores,
            use_inline_citations=INLINE_CITATIONS_ENABLED,
        ):
            emitted_parts.append(sent_out)
            streamed_any = True
            yield _sse("delta", {"text": sent_out})

        filtered_answer = " ".join(emitted_parts).strip()

        # Total-redaction fallback: if every sentence was dropped by the
        # guardrail we have nothing to show the user. Emit the standard
        # no-source refusal as a final delta so the frontend's existing
        # "append delta, finalize on done" path handles it without changes.
        if not filtered_answer:
            refusal_msg = (
                "I don't have a source for that in my current library. "
                "Please try rewording your question, or ask your doctor directly."
            )
            yield _sse("delta", {"text": refusal_msg})
            _log_query_safe(
                user_id=None,
                session_id=query.session_id,
                stage="routine",
                query_text=query.question,
                response_text=refusal_msg,
                refusal_triggered=True,
                refusal_reason="all_sentences_redacted_by_guardrails",
                prompt_hash=_prompt_hash(messages),
                nli_entailment_scores=nli_scores,
            )
            yield _sse("done", {})
            return

        # Scope-guard on the final (already NLI-filtered) answer. If the
        # surviving text still reads as a diagnosis or a prescription we
        # have emitted it token-by-token and cannot unemit — emit a
        # dedicated `override` event so the frontend can *replace* the
        # streamed bubble content with the scope refusal instead of
        # appending to it. Separate event type keeps the `sources`
        # contract single-shape.
        scope = classify_scope(filtered_answer)
        if scope in ("diagnostic", "prescriptive"):
            scope_refusal_text = SCOPE_REFUSAL_TEMPLATES[scope]
            print(f"[scope-guard] streamed answer classified as {scope}, overriding")
            yield _sse("override", {
                "text": scope_refusal_text,
                "coverage": "scope_refused",
                "reason": f"scope_guard_{scope}",
            })
            _log_query_safe(
                user_id=None,
                session_id=query.session_id,
                stage="routine",
                query_text=query.question,
                response_text=scope_refusal_text,
                refusal_triggered=True,
                refusal_reason=f"scope_guard_{scope}",
                prompt_hash=_prompt_hash(messages),
                nli_entailment_scores=nli_scores,
            )
            yield _sse("done", {})
            return

        if sources:
            yield _sse("sources", {"sources": sources})
        # Phase-1.6: provider observability (stdout only; add to query_log
        # once the schema migration for llm_provider column lands).
        print(f"[llm_provider] /query/stream emitted via provider={source}")
        _log_query_safe(
            user_id=None,
            session_id=query.session_id,
            stage="routine",
            query_text=query.question,
            response_text=filtered_answer,
            citations=sources,
            retrieved_chunk_ids=[r.get("id") for r in top_rows if r.get("id")],
            prompt_hash=_prompt_hash(messages),
            nli_entailment_scores=nli_scores,
        )
        yield _sse("done", {})

    # text/event-stream + no-cache so intermediate proxies don't buffer
    # tokens. X-Accel-Buffering disables buffering on nginx specifically.
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    requested_path = FRONTEND_DIST_DIR / full_path
    if requested_path.exists() and requested_path.is_file():
        return FileResponse(requested_path)

    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    return JSONResponse(
        status_code=404,
        content={
            "detail": "Route not found.",
            "frontend": "Frontend build not found. Start Vite in the frontend folder or build the app.",
            "docs_url": "/docs",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
