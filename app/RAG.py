import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel
from PyPDF2 import PdfReader

import cohere  # Official Cohere SDK

from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    from .middleware import add_cors_middleware
except ImportError:
    from middleware import add_cors_middleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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
def extract_text_from_pdf(file: UploadFile) -> str:
    try:
        reader = PdfReader(file.file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF read error: {e}")
    finally:
        file.file.close()


# --------------------------------------------------
# Upload PDF
# --------------------------------------------------
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global retriever

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    text = extract_text_from_pdf(file)
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

    return {"message": "PDF uploaded and indexed successfully"}


# --------------------------------------------------
# Query
# --------------------------------------------------
class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query_document(query: QueryRequest):
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
