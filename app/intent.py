"""Zero-shot intent classifier (docs/IMPROVEMENTS.md §4.4, Week 5 Phase 3).

Maps a user question to (stage, domain) via one Cohere chat call. Output
feeds build_filter(). Fails closed: returns None on any error, which
causes build_filter() to fall back to the Phase 2 default filter.

Both the stage vocabulary and the domain list are bounded — the classifier
output is validated against them. Unknown/malformed responses → None.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import cohere

STAGES = ["intake", "navigation", "visit_prep", "results", "condition"]

DOMAINS = [
    "cardiovascular", "endocrine", "respiratory", "gi", "infectious",
    "mental-health", "renal", "neurology", "reproductive", "maternal",
    "dermatology", "rheumatology", "ent", "hematology", "nutrition",
    "general",
]

MODEL = "command-r-08-2024"

_PROMPT = """Classify the user query into a (stage, domain) pair.

Stages:
- intake: patient describes symptoms; unsure how serious
- navigation: asks where/when to seek care (which level of hospital, urgency)
- visit_prep: preparing questions for an upcoming doctor visit
- results: asking what lab values or test reports mean
- condition: asking about a diagnosed disease

Domains (pick the closest one): {domains}

Respond with a single JSON object only, e.g. {{"stage": "condition", "domain": "endocrine"}}.

Query: {question}"""

_client: Optional[cohere.ClientV2] = None


def _get_client() -> cohere.ClientV2:
    global _client
    if _client is None:
        key = os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY not set")
        _client = cohere.ClientV2(api_key=key)
    return _client


def _keyword_classify(question: str) -> Optional[Dict[str, str]]:
    """Deterministic keyword fallback for when Cohere is unavailable or
    COHERE_DISABLED=1 (eval / CI runs that must not burn trial quota).

    Rule order matters — first match wins. Accuracy is ~75-85% vs the
    Cohere classifier on our gold set; good enough for eval-mode routing,
    not a permanent replacement. Domain defaults to 'general' because
    misrouting a domain is cheap (retrieval is still filtered by stage)."""
    import re as _re
    q = (question or "").lower()
    # whitespace-padded copy for word-boundary-style contains() checks
    qp = " " + _re.sub(r"[^\w\s]", " ", q) + " "

    # Stage 4 (results): lab values or test names
    if any(k in q for k in (
        "lab ", "labs ", "test result", "my report", "tsh", "hba1c", "ldl",
        "hdl", "hemoglobin", "haemoglobin", "creatinine", "alt ", "ast ",
        "mg/dl", "mmol", "ng/ml", "cbc", "bilirubin", "ferritin",
        "vitamin d", "vitamin b12", "b12 level", "egfr", "platelet",
    )):
        stage = "results"
    # Stage 5 (condition): diagnosed disease education (check BEFORE
    # navigation — "I was diagnosed" is condition, not triage)
    elif any(k in q for k in (
        "diagnosed", "my doctor said i have", "i have type", "i was told i have",
        "explain my ", "hashimoto", "hypertension",
        "diabetes", "asthma", "gastritis", "copd", "ckd ", "pcos", "migraine",
        "epilepsy", "parkinson", "eczema", "psoriasis",
    )):
        stage = "condition"
    # Stage 3 (visit_prep): appointment framing
    elif any(k in q for k in (
        "prepare for", "appointment", "visit the doctor", "see the specialist",
        "what to ask", "questions to ask", "ask my", "ask the doctor",
        "referral", "before my visit",
    )):
        stage = "visit_prep"
    # Stage 2 (navigation): triage / where to go (word-boundary match for
    # short tokens like "ed" to avoid "diagnosed"/"prescribed" false hits)
    elif any(k in q for k in (
        "should i go", "which hospital", "which level", "emergency room",
        "urgent care", "a&e", "call an ambulance", "is this an emergency",
        "can i wait",
    )) or any(k in qp for k in (" 102 ", " 103 ", " ed ", " a e ")):
        stage = "navigation"
    else:
        stage = "intake"

    # Domain routing — coarse keyword map
    domain = "general"
    if any(k in q for k in ("heart", "chest pain", "blood pressure", "bp ", "cholesterol", "mi ", "stroke")):
        domain = "cardiovascular"
    elif any(k in q for k in ("thyroid", "diabetes", "insulin", "hashimoto", "adrenal", "cortisol")):
        domain = "endocrine"
    elif any(k in q for k in ("cough", "breath", "asthma", "copd", "pneumonia", "wheez")):
        domain = "respiratory"
    elif any(k in q for k in ("stomach", "abdom", "gastr", "ulcer", "liver", "alt", "ast", "jaundice")):
        domain = "gi"
    elif any(k in q for k in ("fever", "tb", "dengue", "covid", "typhoid", "leptospir", "malaria")):
        domain = "infectious"
    elif any(k in q for k in ("depress", "anxiety", "mental", "suicid", "panic", "bipolar")):
        domain = "mental-health"
    elif any(k in q for k in ("kidney", "renal", "urine", "uti ", "creatinine", "egfr")):
        domain = "renal"
    elif any(k in q for k in ("head", "migraine", "seizure", "epilepsy", "parkinson", "stroke")):
        domain = "neurology"
    elif any(k in q for k in ("pregnan", "period", "menstru", "fibroid", "pcos", "breast")):
        domain = "reproductive"

    return {"stage": stage, "domain": domain}


def classify(question: str) -> Optional[Dict[str, str]]:
    """Return {'stage': ..., 'domain': ...} or None on any failure.

    Unknown stage → None (filter falls back to defaults).
    Unknown domain → coerced to 'general'.
    """
    if os.getenv("COHERE_DISABLED") == "1":
        return _keyword_classify(question)
    try:
        resp = _get_client().chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": _PROMPT.format(
                        domains=", ".join(DOMAINS),
                        question=question,
                    ),
                }
            ],
            max_tokens=50,
            response_format={"type": "json_object"},
        )
        text = resp.message.content[0].text
        parsed = json.loads(text)
    except Exception as exc:
        print(f"[intent] classifier failed: {exc}")
        return None

    stage = parsed.get("stage")
    domain = parsed.get("domain")
    if stage not in STAGES:
        return None
    if domain not in DOMAINS:
        domain = "general"
    return {"stage": stage, "domain": domain}
