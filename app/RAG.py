import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel
import pymupdf

import cohere  # Official Cohere SDK

from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .middleware import add_cors_middleware
    from .supabase_client import (
        insert_chunk,
        insert_document,
        match_chunks,
    )
except ImportError:
    from middleware import add_cors_middleware
    from supabase_client import (
        insert_chunk,
        insert_document,
        match_chunks,
    )

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

# Cohere Chat Client (NEW API)
co = cohere.ClientV2(api_key=COHERE_API_KEY)

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
    "Always frame answers as information to discuss with a doctor. "
    "If a claim is not supported by the sources, say you don't have a source for it."
)

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


def extract_text_from_pdf(file: UploadFile) -> Tuple[str, dict]:
    try:
        data = file.file.read()
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
        file.file.close()


# --------------------------------------------------
# Upload PDF
# --------------------------------------------------
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
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
# Query
# --------------------------------------------------
class QueryRequest(BaseModel):
    question: str


TOP_K = 10
CONTEXT_CHUNKS = 6


@app.post("/query")
async def query_document(query: QueryRequest):
    q_vec = get_query_encoder().encode_one(query.question)
    rows = match_chunks(to_pgvector_literal(q_vec), match_count=TOP_K)

    if not rows:
        return {
            "answer": (
                "I don't have enough reliable information to answer this safely yet. "
                "The corpus may not be populated. Please ingest sources first or ask your doctor."
            ),
            "sources": [],
        }

    top_rows = rows[:CONTEXT_CHUNKS]
    context_blocks = []
    for i, r in enumerate(top_rows, start=1):
        heading = r.get("section_heading") or ""
        title = r.get("doc_title") or r.get("doc_source") or "source"
        context_blocks.append(f"[src:{i}] {title} — {heading}\n{r.get('content', '')}".strip())
    context_text = "\n\n".join(context_blocks)

    sources = [
        {
            "rank": i,
            "title": r.get("doc_title"),
            "source": r.get("doc_source"),
            "source_url": r.get("doc_source_url"),
            "similarity": r.get("similarity"),
            "authority_tier": r.get("doc_authority_tier"),
            "publication_date": r.get("doc_publication_date"),
        }
        for i, r in enumerate(top_rows, start=1)
    ]

    messages = [
        {"role": "system", "content": MEDIRAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Sources:\n{context_text}\n\nQuestion: {query.question}",
        },
    ]

    response = co.chat(model="command-a-03-2025", messages=messages)
    try:
        answer = response.message.content[0].text
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to parse Cohere response")

    return {"answer": answer, "sources": sources}


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
