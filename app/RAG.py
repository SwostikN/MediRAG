import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException
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
        get_chat_session,
        update_chat_session,
    )
    from .filters import build_filter
    from .intent import classify as classify_intent
    from .redflag import check as redflag_check
    from .stages import intake as intake_stage
    from .stages import navigation as navigation_stage
except ImportError:
    from middleware import add_cors_middleware
    from supabase_client import (
        insert_chunk,
        insert_document,
        match_chunks_hybrid_filtered,
        get_chat_session,
        update_chat_session,
    )
    from filters import build_filter
    from intent import classify as classify_intent
    from redflag import check as redflag_check
    from stages import intake as intake_stage
    from stages import navigation as navigation_stage

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
    "If a claim is not supported by the sources, say you don't have a source for it.\n\n"
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
class HistoryTurn(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    history: Optional[list[HistoryTurn]] = None


HISTORY_MAX_TURNS = 6
HISTORY_RETRIEVAL_USER_TURNS = 2


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
    the full set. Falls back to original order if the API call fails."""
    if not rows:
        return rows
    docs = [r.get("content") or "" for r in rows]
    try:
        resp = co.rerank(
            model=RERANK_MODEL,
            query=question,
            documents=docs,
            top_n=len(docs),
        )
    except Exception as exc:
        print(f"[rerank] failed, falling back to RRF order: {exc}")
        return rows
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


def _retrieve_ranked(question: str, *, intent: Optional[dict] = None) -> list:
    """Run filtered hybrid retrieval → rerank → stage-weighted scoring.

    Returns the full ranked row list (caller slices to CONTEXT_CHUNKS).
    Extracted so Stage 2 navigation can run retrieval against the intake
    summary with identical semantics to the routine /query path.
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
    if len(rows) > CONTEXT_CHUNKS:
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
    # Week 6 Stage 0: deterministic red-flag screen runs first.
    # If any rule fires, the LLM is never invoked and retrieval is skipped.
    rf = redflag_check(query.question)
    if rf is not None:
        print(f"[redflag] fired rule={rf.rule_id} category={rf.category}")
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
                return {
                    "answer": combined,
                    "sources": _format_sources(
                        _dedupe_sources(nav_rows)[:DISPLAY_SOURCES]
                    ),
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
    rows = _retrieve_ranked(retrieval_query)

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
    if not rows:
        print("[refusal-gate] refusing: retrieval returned 0 rows")
        return refusal
    top_rerank = rows[0].get("rerank_score")
    print(f"[refusal-gate] top rerank_score={top_rerank} threshold={RERANK_REFUSAL_THRESHOLD}")
    if top_rerank is None:
        print("[refusal-gate] refusing: rerank did not run, cannot judge relevance")
        return refusal
    if top_rerank < RERANK_REFUSAL_THRESHOLD:
        print(
            f"[refusal-gate] refusing: top rerank_score {top_rerank:.3f} "
            f"< {RERANK_REFUSAL_THRESHOLD}"
        )
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
        return refusal

    top_rows = rows[:CONTEXT_CHUNKS]
    context_blocks = []
    for i, r in enumerate(top_rows, start=1):
        heading = r.get("section_heading") or ""
        title = r.get("doc_title") or r.get("doc_source") or "source"
        context_blocks.append(f"[src:{i}] {title} — {heading}\n{r.get('content', '')}".strip())
    context_text = "\n\n".join(context_blocks)

    sources = _format_sources(_dedupe_sources(rows)[:DISPLAY_SOURCES])

    messages: list[dict] = [{"role": "system", "content": MEDIRAG_SYSTEM_PROMPT}]
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

    if groq_client is not None:
        try:
            groq_resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=250,
            )
            answer = groq_resp.choices[0].message.content
        except Exception as exc:
            print(f"[groq] generate failed, falling back to Cohere: {exc}")
            response = co.chat(
                model="command-r-08-2024",
                messages=messages,
                max_tokens=250,
            )
            try:
                answer = response.message.content[0].text
            except Exception:
                raise HTTPException(status_code=500, detail="Failed to parse Cohere response")
    else:
        response = co.chat(
            model="command-r-08-2024",
            messages=messages,
            max_tokens=250,
        )
        try:
            answer = response.message.content[0].text
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to parse Cohere response")

    return {"answer": answer, "sources": sources, "stage": "routine"}


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
                    yield _sse("done", {})
                    return

        # ── Routine: retrieve, gate, then stream Groq tokens ─────────────
        retrieval_query = _retrieval_query_with_history(query.question, query.history)
        if retrieval_query != query.question:
            print(f"[history] retrieval_query rewritten: {retrieval_query[:120]!r}")
        rows = _retrieve_ranked(retrieval_query)

        refusal_text = (
            "I don't have a source for that in my current library. "
            "Please try rewording your question, or ask your doctor directly."
        )

        def emit_refusal(reason: str):
            print(f"[refusal-gate] refusing ({reason})")
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

        messages: list[dict] = [{"role": "system", "content": MEDIRAG_SYSTEM_PROMPT}]
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

        streamed_any = False
        # Groq primary streaming path. If it fails BEFORE any token is
        # streamed, fall back to Cohere streaming. If it fails MID-stream,
        # we can't unwind the partial answer — emit an error event and end
        # cleanly so the frontend marks the message as incomplete.
        if groq_client is not None:
            try:
                stream = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    max_tokens=250,
                    stream=True,
                )
                for chunk in stream:
                    delta = ""
                    try:
                        delta = chunk.choices[0].delta.content or ""
                    except Exception:
                        delta = ""
                    if delta:
                        streamed_any = True
                        yield _sse("delta", {"text": delta})
            except Exception as exc:
                print(f"[groq-stream] failed (streamed_any={streamed_any}): {exc}")
                if streamed_any:
                    yield _sse("error", {"message": "Generation interrupted. Partial answer above."})
                    yield _sse("done", {})
                    return
                # else: fall through to Cohere fallback below

        if not streamed_any:
            try:
                cohere_stream = co.chat_stream(
                    model="command-r-08-2024",
                    messages=messages,
                    max_tokens=250,
                )
                for ev in cohere_stream:
                    text = ""
                    try:
                        # cohere v2 stream events: type=='content-delta'
                        if getattr(ev, "type", None) == "content-delta":
                            text = ev.delta.message.content.text or ""
                    except Exception:
                        text = ""
                    if text:
                        streamed_any = True
                        yield _sse("delta", {"text": text})
            except Exception as exc:
                print(f"[cohere-stream] failed: {exc}")
                if not streamed_any:
                    yield _sse("error", {"message": "Generation failed. Please try again."})
                    yield _sse("done", {})
                    return

        if sources:
            yield _sse("sources", {"sources": sources})
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
