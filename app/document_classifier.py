"""Document classifier — routes user uploads to the right downstream
flow based on detected content type.

Three buckets:
  - lab_report      → Stage 4 (results explainer; values stored in
                      user_reports + session_documents, never
                      ingested to shared corpus)
  - research_paper  → session-scoped Q&A (chunks stored in
                      session_chunks, retrieved alongside corpus
                      only for THIS session)
  - other           → ask the user what to do with it

Classification is heuristic (regex over the first ~2 pages of text).
No LLM call — the classifier sits on the upload critical path and
adding ~500ms-1s of LLM latency per upload is unacceptable.

Failure asymmetry. The cost of mis-routing is asymmetric:
  - lab_report misclassified as research_paper: patient lab values
    end up in the session-Q&A flow. They still never reach the
    shared corpus (both buckets use session_chunks), so privacy is
    preserved — but the user gets a useless Q&A flow instead of a
    table.
  - research_paper misclassified as lab_report: the marker parser
    extracts garbage; user gets confusing output.
  - either misclassified as 'other': user is asked, gives explicit
    intent, no harm done.

Default policy when uncertain: 'other'. The user prompt is one extra
tap; the wrong-flow output is much worse UX.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

DocType = Literal["lab_report", "research_paper", "other"]


@dataclass(frozen=True)
class ClassifyResult:
    doc_type: DocType
    confidence: Literal["high", "medium", "low"]
    lab_score: int
    paper_score: int
    matched_signals: dict[str, list[str]]


# ---------------------------------------------------------------------------
# Lab report signals — regex patterns scored by presence
# ---------------------------------------------------------------------------

# Marker-name + value + unit + reference-range pattern. The most
# specific signal: if we see this 2+ times, it's almost certainly a
# lab report.
#   e.g. "TSH 8.4 mIU/L (0.4 - 4.0)"  /  "Hb: 11.8 g/dL  12.0 - 15.5"
_LAB_VALUE_LINE = re.compile(
    r"\b\d+(?:\.\d+)?\s*"
    r"(?:mg/dL|mg/dl|mmol/L|mmol/l|mIU/L|mIU/l|µIU/mL|uIU/mL|"
    r"g/dL|g/dl|U/L|u/l|%|ng/mL|ng/ml|pg/mL|pg/ml|cells/[uµ]L|"
    r"million/[uµ]L|10\^?[0-9]+/[uµ]L|/cmm|fl|fL|pg)"
    r"\s*\(?\s*(?:[<>]?\s*\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?|"
    r"[<>]\s*\d+(?:\.\d+)?)\s*\)?",
    re.IGNORECASE,
)

# Common Nepali lab brand names + generic pathology lab keywords
_LAB_BRAND = re.compile(
    r"\b(?:samyak|prime\s+(?:medical|diagnostic|hospital)|star\s+hospital|"
    r"national\s+medical\s+college|nmc\s+pathology|grande\s+international|"
    r"norvic|metro\s+pathology|medsci|medi\s*care|"
    r"pathology\s+(?:lab|department|report)|clinical\s+laboratory|"
    r"laboratory\s+report|lab\s+report)\b",
    re.IGNORECASE,
)

# Common test-panel headings
_LAB_PANEL_HEADINGS = re.compile(
    r"\b(?:complete\s+blood\s+count|CBC|"
    r"lipid\s+profile|liver\s+function\s+test|LFT|"
    r"renal\s+function\s+test|RFT|kidney\s+function|"
    r"thyroid\s+(?:function|profile)|TFT|"
    r"fasting\s+(?:blood\s+)?(?:glucose|sugar)|FBS|HbA1c|"
    r"urine\s+(?:routine|analysis|examination)|"
    r"electrolyte\s+panel|serum\s+electrolytes)\b",
    re.IGNORECASE,
)

# Patient/sample metadata fields — present on every printed lab report
_LAB_PATIENT_FIELDS = re.compile(
    r"\b(?:patient\s+(?:name|id|age)|age\s*/\s*sex|"
    r"ref(?:erred)?\s*(?:by|by\s+dr)|sample\s+(?:collected|received)|"
    r"specimen|collection\s+date|report\s+date|"
    r"bio\.?\s*ref\.?\s*interval|reference\s+(?:range|interval|value)|"
    r"normal\s+range)\b",
    re.IGNORECASE,
)

# Specific common marker abbreviations
_LAB_MARKERS = re.compile(
    r"\b(?:TSH|FT[34]|T[34]|HbA1c|LDL|HDL|VLDL|"
    r"ALT|AST|ALP|GGT|"
    r"Hb|Hgb|Hct|RBC|WBC|MCV|MCH|MCHC|"
    r"creatinine|urea|BUN|sodium|potassium|chloride|"
    r"vitamin\s+[BD](?:12)?|ferritin|iron|TIBC)\b",
)


# ---------------------------------------------------------------------------
# Research paper signals
# ---------------------------------------------------------------------------

_PAPER_SECTIONS = re.compile(
    r"(?:^|\n)\s*(?:abstract|introduction|"
    r"materials\s+and\s+methods|methods|methodology|"
    r"results|discussion|conclusion|conclusions|"
    r"references|bibliography|acknowledg(?:e?ments?))\s*(?:\n|$|:)",
    re.IGNORECASE | re.MULTILINE,
)

_DOI = re.compile(r"\b(?:doi:?\s*)?10\.\d{4,9}/[^\s]+\b", re.IGNORECASE)

_ARXIV = re.compile(r"\barXiv:\s*\d{4}\.\d{4,5}\b", re.IGNORECASE)

_CITATION_BRACKETS = re.compile(r"\[\d+(?:\s*[-,]\s*\d+)*\]")

_CITATION_PARENS = re.compile(r"\([A-Z][a-z]+(?:\s+et\s+al\.?)?,\s*\d{4}\)")

_PAPER_KEYWORDS = re.compile(
    r"\b(?:keywords?\s*:|funding\s*:|conflict\s+of\s+interest|"
    r"corresponding\s+author|orcid|received\s+\d{1,2}\s+\w+\s+\d{4}|"
    r"accepted\s+\d{1,2}\s+\w+\s+\d{4}|published\s+online|"
    r"affiliations?\s*:|department\s+of\s+\w+,\s+\w+\s+university)\b",
    re.IGNORECASE,
)


_SAMPLE_CHARS = 8000  # ~first 2 pages of a typical PDF


# ---------------------------------------------------------------------------
# Medical-relevance probe (used by the upload gate, not the doc-type vote)
# ---------------------------------------------------------------------------
# A research paper that is clearly off-domain (pure ML theory, civil
# engineering, pure mathematics) gets rejected at upload time so the
# user isn't stuck uploading something MediRAG will refuse every
# question about.
#
# Lab reports are exempt from this probe — they have their own strong
# signals (the marker abbreviations + value+unit+range pattern) and we
# don't want to risk filtering a Nepali lab format we haven't seen.
#
# Pattern: any single-word hit counts. The threshold is intentionally
# low — false-positive (accepting a marginally-medical paper) is fine
# (the rerank gate will refuse irrelevant questions later). The thing
# we are catching is "user uploaded a paper about distributed
# consensus algorithms by mistake".

_MEDICAL_TERMS = re.compile(
    r"\b(?:patient|patients|clinical|clinic|disease|diseases|diagnosis|"
    r"diagnostic|treatment|therapy|therapies|symptom|symptoms|"
    r"medication|medications|drug|drugs|dose|dosage|"
    r"hospital|physician|doctor|nurse|nursing|"
    r"medicine|medical|health|healthcare|public\s+health|"
    r"epidemiolog\w*|prevalence|incidence|mortality|morbidity|"
    r"cohort|trial|randomi[sz]ed|placebo|"
    r"pathology|patholog\w*|oncolog\w*|cardiolog\w*|neurolog\w*|"
    r"endocrin\w*|nephrolog\w*|gastroenterolog\w*|paediatric|pediatric|"
    r"surgery|surgical|anatomy|physiology|pharmacolog\w*|"
    r"infection|infectious|antibiotic|antiviral|vaccine|vaccination|"
    r"hypertension|diabetes|cancer|tuberculosis|pneumonia|asthma|"
    r"covid|sars|hiv|hepatitis|malaria|dengue|"
    r"WHO\b|MoHP\b|NICE\b|NHS\b|CDC\b|PubMed)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Non-medical counter-signals — catches meta-documents about medical RAG /
# health software that trivially mention "patient", "diagnosis", "NICE", etc.
# but are actually CS / engineering design docs.
#
# The original is_medically_relevant probe was a simple keyword-density
# check. A software design document describing a medical RAG system
# (e.g. "RAG_Filtering_Documentation.pdf") hits the medical-term list
# easily — it discusses the domain — but is NOT a medical paper and
# MediRAG should refuse to ingest it.
#
# Policy: "we only cater for medical papers" → bias toward REJECTING
# borderline. False-positive rejection of an edge medical paper is
# preferable to admitting a non-medical paper.

# Programming / code-block signals. Any one of these is a smoke-signal
# that the document contains source code — not a research paper.
_CODE_TERMS = re.compile(
    r"(?:"
    r"\bdef\s+\w+\s*\(|"                  # python function def
    r"\bclass\s+\w+\s*[:\(]|"             # python class def
    r"\bimport\s+\w+|"                    # python import
    r"\bfrom\s+\w+\s+import\b|"           # python from-import
    r"\breturn\s+\w+|"                    # generic return statement
    r"\bfunction\s+\w+\s*\(|"             # JS/TS function
    r"\bconst\s+\w+\s*=|\blet\s+\w+\s*=|" # JS/TS decl
    r"\basync\s+def\b|\bawait\s+\w+|"     # python async
    r"={2,3}|!={1,2}|"                    # equality operators
    r"\{[\s\S]{0,40}\":|"                 # JSON-ish object
    r"-->|<--|"                           # arrow comments
    r"```[a-zA-Z]*|"                      # markdown code fence
    r"\[\s*[\"'][^\"']+[\"']\s*,"         # list-literal of strings
    r")",
    re.MULTILINE,
)

# Engineering / CS / ML vocabulary — the giveaway that the doc is
# ABOUT a system, not a clinical study.
_CS_TERMS = re.compile(
    r"\b(?:"
    r"RAG|retrieval[- ]augmented|vector\s*store|vectorstore|"
    r"embedding|embeddings|embed|re[- ]?ranker|reranker|rerank|"
    r"FAISS|pgvector|chromadb|qdrant|pinecone|weaviate|"
    r"LLM|LLMs|large\s+language\s+model|"
    r"GPT|ChatGPT|Claude|Llama|Mistral|Gemini|Cohere|OpenAI|Anthropic|"
    r"transformer|attention\s+mechanism|fine[- ]?tun(?:e|ing|ed)|"
    r"tokeniz(?:er|ation|e)|prompt\s+engineer\w*|"
    r"cosine\s+similarity|semantic\s+search|"
    r"chunk(?:ing|ed|s)?|chunk\s+size|sliding\s+window|"
    r"API|endpoint|backend|frontend|middleware|microservice|"
    r"pipeline|data\s+pipeline|ETL|"
    r"docker|kubernetes|postgres|sqlite|redis|fastapi|flask|django|"
    r"numpy|pandas|pytorch|tensorflow|scikit[- ]learn|sklearn|"
    r"hyperparameter|loss\s+function|gradient\s+descent|"
    r"precision|recall|f1[- ]score|benchmark|"
    r"repository|repo|commit|pull\s+request|github|gitlab"
    r")\b",
    re.IGNORECASE,
)


def _count_code_signals(text: str) -> int:
    """Distinct code / CS signal hits in `text`. Used as a counter-weight
    to the medical-term probe so meta-documents about medical software
    get rejected at the upload gate."""
    hits = _CODE_TERMS.findall(text) + _CS_TERMS.findall(text)
    # Normalize case so "RAG" and "rag" don't both count.
    return len({h.lower().strip() for h in hits if h})


def _approx_token_count(text: str) -> int:
    """Cheap whitespace token count — used for density normalisation."""
    return max(1, len(text.split()))


def is_medically_relevant(
    text: str,
    *,
    min_hits: int = 5,
    min_density_per_1k: float = 4.0,
    max_cs_ratio: float = 0.6,
) -> bool:
    """Cheap medical-domain probe over the first ~2 pages.

    A document passes only if ALL of:
      (1) At least `min_hits` distinct medical terms appear (original rule).
      (2) Medical-term density per 1k tokens meets `min_density_per_1k` —
          a long CS doc that mentions "patient" / "trial" in passing
          will fail this even if the raw count is high.
      (3) The non-medical counter-signal (programming + CS/ML vocab) is
          not dominant. Specifically, if #cs_signals > max_cs_ratio *
          #medical_signals, reject — this catches "medical RAG design
          document" false positives.

    Policy bias: "we only cater for medical papers". Rejecting a
    borderline medical paper is cheaper than admitting a CS doc whose
    chunks poison the session corpus.

    Keywords-only, no ML dependency. Sits on the upload critical path.
    """
    if not text:
        return False
    sample = text[:_SAMPLE_CHARS]

    med_hits_all = _MEDICAL_TERMS.findall(sample)
    distinct_med = {h.lower() for h in med_hits_all}
    if len(distinct_med) < min_hits:
        return False

    # Density guard — real medical papers carry ~10-40 medical terms per
    # 1000 tokens in their abstract+intro. A CS design document that
    # discusses medical topics tends to sit well below that.
    tokens = _approx_token_count(sample)
    density_per_1k = (len(med_hits_all) / tokens) * 1000.0
    if density_per_1k < min_density_per_1k:
        return False

    # Counter-signal guard — reject if CS/code vocabulary rivals the
    # medical vocabulary. Using distinct-medical (not raw hits) so a
    # doc that repeats "patient" 40 times can't swamp the ratio.
    cs_count = _count_code_signals(sample)
    if cs_count > max_cs_ratio * len(distinct_med):
        return False

    return True


# ---------------------------------------------------------------------------
# Scoring + decision
# ---------------------------------------------------------------------------


def _score_lab_report(text: str) -> tuple[int, dict[str, list[str]]]:
    matches: dict[str, list[str]] = {}
    score = 0

    value_lines = _LAB_VALUE_LINE.findall(text)
    if value_lines:
        # Each value+unit+range line is strong evidence; cap at 5 pts to
        # avoid one very long lab report dominating the signal.
        n = min(len(value_lines), 5)
        score += n * 2
        matches["value_lines"] = value_lines[:5]

    if (m := _LAB_BRAND.findall(text)):
        score += 3
        matches["brand"] = m[:3]
    if (m := _LAB_PANEL_HEADINGS.findall(text)):
        score += 2
        matches["panels"] = m[:3]
    if (m := _LAB_PATIENT_FIELDS.findall(text)):
        # Multiple patient fields → strong signal; cap at 3.
        score += min(len(m), 3)
        matches["patient_fields"] = m[:3]
    if (m := _LAB_MARKERS.findall(text)):
        # Marker abbreviations alone are weak (research papers cite them
        # too) — count once.
        if len(m) >= 2:
            score += 2
            matches["markers"] = list(set(m))[:5]

    return score, matches


def _score_research_paper(text: str) -> tuple[int, dict[str, list[str]]]:
    matches: dict[str, list[str]] = {}
    score = 0

    sections = _PAPER_SECTIONS.findall(text)
    if sections:
        # Need 2+ canonical sections to count strongly (a single
        # "Conclusion" header could appear in a clinical guideline).
        unique = {s.strip().lower() for s in sections}
        if len(unique) >= 3:
            score += 4
        elif len(unique) >= 2:
            score += 2
        matches["sections"] = sorted(unique)[:5]

    if (m := _DOI.findall(text)):
        score += 3
        matches["doi"] = m[:2]
    if (m := _ARXIV.findall(text)):
        score += 3
        matches["arxiv"] = m[:2]
    if len(_CITATION_BRACKETS.findall(text)) >= 3:
        score += 2
        matches["citations_bracket"] = _CITATION_BRACKETS.findall(text)[:3]
    if len(_CITATION_PARENS.findall(text)) >= 2:
        score += 2
        matches["citations_paren"] = _CITATION_PARENS.findall(text)[:3]
    if (m := _PAPER_KEYWORDS.findall(text)):
        score += min(len(m), 2)
        matches["paper_keywords"] = m[:3]

    return score, matches


# Decision thresholds, set conservatively. A single strong signal
# (e.g. one DOI, one lab brand) on its own should NOT win — we want
# multiple converging signals before committing to a doc_type.
_LAB_HIGH = 6
_LAB_MEDIUM = 3
_PAPER_HIGH = 6
_PAPER_MEDIUM = 4


def classify_document(text: str) -> ClassifyResult:
    """Classify a document by scoring its first ~2 pages of extracted text.

    Returns a ClassifyResult. When confidence is 'low' (returned as
    doc_type='other'), the caller should prompt the user instead of
    silently routing to a downstream flow.
    """
    sample = (text or "")[:_SAMPLE_CHARS]
    lab_score, lab_matches = _score_lab_report(sample)
    paper_score, paper_matches = _score_research_paper(sample)
    matched = {"lab": lab_matches, "paper": paper_matches}

    if lab_score >= _LAB_HIGH and lab_score > paper_score:
        return ClassifyResult("lab_report", "high", lab_score, paper_score, matched)
    if paper_score >= _PAPER_HIGH and paper_score > lab_score:
        return ClassifyResult("research_paper", "high", lab_score, paper_score, matched)
    if lab_score >= _LAB_MEDIUM and lab_score > paper_score + 1:
        return ClassifyResult("lab_report", "medium", lab_score, paper_score, matched)
    if paper_score >= _PAPER_MEDIUM and paper_score > lab_score + 1:
        return ClassifyResult("research_paper", "medium", lab_score, paper_score, matched)
    return ClassifyResult("other", "low", lab_score, paper_score, matched)
