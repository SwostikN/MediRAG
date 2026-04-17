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


def classify(question: str) -> Optional[Dict[str, str]]:
    """Return {'stage': ..., 'domain': ...} or None on any failure.

    Unknown stage → None (filter falls back to defaults).
    Unknown domain → coerced to 'general'.
    """
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
