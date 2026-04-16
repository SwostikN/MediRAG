import os
import re
from pathlib import Path
from typing import Optional, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel
import pymupdf

import cohere  # Official Cohere SDK

from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    from .middleware import add_cors_middleware
    from .supabase_client import (
        insert_chunk,
        insert_document,
        insert_query,
        insert_response,
    )
except ImportError:
    from middleware import add_cors_middleware
    from supabase_client import (
        insert_chunk,
        insert_document,
        insert_query,
        insert_response,
    )

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

# In-memory vector store (Fargate-safe)
retriever = None

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
    global retriever

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    text, pdf_metadata = extract_text_from_pdf(file)
    if not text:
        raise HTTPException(status_code=400, detail="No text found in PDF")

    documents = [Document(page_content=text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=COHERE_API_KEY
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Persist document and chunks to Supabase if configured.
    # NOTE: this still treats user uploads as corpus documents. Stage 4
    # (Week 9) will route patient lab PDFs into `user_reports` instead.
    try:
        title = pdf_metadata.get("title") or file.filename
        doc_res = insert_document(
            title=title,
            source="user-upload",
            authority_tier=5,
            doc_type="reference",
            publication_date=pdf_metadata.get("creation_date"),
        )
        if isinstance(doc_res, dict) and doc_res.get("error"):
            print("supabase: document insert error", doc_res.get("error"))
        else:
            try:
                inserted = doc_res[0]
                doc_id = inserted.get("doc_id") or inserted.get("id")
            except Exception:
                doc_id = None

            if doc_id is not None:
                for ord_index, chunk in enumerate(chunks):
                    try:
                        insert_chunk(
                            doc_id,
                            ord_index,
                            chunk.page_content,
                            token_count=len(chunk.page_content.split()),
                        )
                    except Exception as e:
                        print("supabase: insert_chunk failed", e)
    except Exception as e:
        print("supabase: failed to persist document/chunks", e)

    return {"message": "PDF uploaded and indexed successfully"}


# --------------------------------------------------
# Query
# --------------------------------------------------
class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query_document(query: QueryRequest, request: Request):
    if retriever is None:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet")

    # ✅ Correct modern LangChain retriever usage
    docs = retriever.invoke(query.question)

    if not docs:
        return {"answer": "The information is not mentioned in the provided context."}

    context_text = "\n\n".join(doc.page_content for doc in docs)
    # sources = []
    # for index, doc in enumerate(docs, start=1):
    #     excerpt = doc.page_content[:280].strip()
    #     if len(doc.page_content) > 280:
    #         excerpt += "..."
    #     sources.append(
    #         {
    #             "title": f"Retrieved chunk {index}",
    #             "excerpt": excerpt,
    #             "confidence": max(0.6, 0.95 - (index - 1) * 0.1),
    #         }
    #     )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant for healthcare data that answers strictly using the given context. "
                "If the answer is not present, say so clearly."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {query.question}",
        },
    ]

    response = co.chat(
        model="command-a-03-2025",
        messages=messages
    )

    try:
        answer = response.message.content[0].text
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to parse Cohere response"
        )

    # Persist query + response to Supabase (server-side)
    try:
        user_id = None
        # try to read authenticated user id from request headers if client sends it
        if "x-user-id" in request.headers:
            user_id = request.headers.get("x-user-id")

        qres = insert_query(query.question, user_id)
        inserted_query_id = None
        if isinstance(qres, dict) and qres.get("error"):
            print("supabase: query insert error", qres.get("error"))
        else:
            try:
                inserted = qres[0]
                inserted_query_id = inserted.get("query_id") or inserted.get("id")
            except Exception:
                inserted_query_id = None

        # insert response row
        try:
            insert_response(inserted_query_id, answer, None, None)
        except Exception as e:
            print("supabase: insert_response failed", e)
    except Exception as e:
        print("supabase: failed to persist query/response", e)

    return {"answer": answer}


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
