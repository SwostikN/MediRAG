"""Lay → clinical retrieval-query expansion (Failure A #1).

Closes the Failure A shape from docs/HALLUCINATION_ZERO_PLAN.md §4 #1:
the user asks in colloquial language ("weather remedies", "my pee is
brown", "I have a runny nose and sore throat") and the raw MedCPT +
BM25 retrieval misses because the corpus uses clinical vocabulary
("upper respiratory tract infection", "haematuria", "viral rhinitis").

The gate is deliberately narrow — it ONLY adds clinical synonyms to
the retrieval string. It does NOT rewrite the user-facing question and
it does NOT change retrieval fan-out. The user still sees their
original phrasing; retrieval sees "original + synonyms."

Fails open: any Groq failure → return the original question unchanged.
The worst case is we degrade to today's behaviour (over-refusal), never
corrupt retrieval.
"""
from __future__ import annotations

from typing import Any, Optional


_SYSTEM_PROMPT = """You expand lay health questions into medical retrieval \
queries. Given a user's question in colloquial English, Hindi, or Nepali, \
output a SHORT list of clinical SYNONYMS for the same topic — NOT \
diagnostic guesses.

First classify the question, then expand accordingly.

1. SYMPTOM / EXPERIENCE questions ("my X hurts", "I feel", "my pee is \
brown", "runny nose", "chest crushing"):
   - Output ONLY symptom and anatomy synonyms. \
FORBIDDEN: specific disease names, diagnoses, or causes.
   - Allowed: describing the sensation ("dry cough, persistent cough") \
and anatomical terms ("haematuria" for blood-in-urine).
   - NOT allowed: "cough variant asthma", "myocardial infarction", \
"bronchitis", "angina" — those are diagnoses.

2. CONDITION / INFO questions ("what is X", "how does X work", \
"symptoms of X"):
   - Output the main term + 1-3 obvious synonyms.
   - E.g. "type 2 diabetes" → "type 2 diabetes mellitus, T2DM, \
non-insulin-dependent diabetes".

3. LAB / MEDICATION / NAVIGATION / HEALTH-SYSTEM questions:
   - Output the domain terms involved (lab name, drug class, tier \
names). No disease guesses.

4. EMERGENCY-shaped symptom questions ("chest crushing", "can't \
breathe", "one-sided weakness", "severe bleeding", "unresponsive"): \
output SKIP. The emergency path handles these without rewrite.

5. NON-HEALTH questions: output SKIP.

Output format:
- Terms only, comma-separated. No sentences, no questions, no \
explanations. Max 20 tokens. British + American spellings when \
different.

Examples:
Q: my pee is brown
A: dark urine, haematuria, blood in urine, tea-coloured urine

Q: runny nose and sore throat with cold weather
A: rhinorrhoea, nasal discharge, sore throat, pharyngeal pain, \
upper respiratory symptoms

Q: persistent dry cough for 3 weeks
A: persistent cough, chronic cough, dry cough, non-productive cough

Q: what is type 2 diabetes
A: type 2 diabetes mellitus, T2DM, non-insulin-dependent diabetes

Q: my chest is crushing and my left arm tingles
A: SKIP

Q: how do i cook rice
A: SKIP
"""


_MAX_TOKENS = 80
_TEMPERATURE = 0.0


def expand_for_retrieval(
    question: str,
    *,
    groq_client: Any,
    groq_model: str,
) -> str:
    """Return `question + " " + synonyms` if Groq gives useful expansion,
    else the original `question`. Never raises."""
    q = (question or "").strip()
    if not q:
        return question
    if groq_client is None or not groq_model:
        return question
    # Skip obvious non-questions — 1-2 word inputs are too short for the
    # rewrite to help and risk adding noise.
    if len(q.split()) < 2:
        return question

    try:
        resp = groq_client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Q: {q}\nA:"},
            ],
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
        )
    except Exception as exc:
        print(f"[query_rewrite] groq failed: {exc}")
        return question

    try:
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return question

    if not raw:
        return question
    # First line only — model occasionally leaks a second line.
    raw = raw.splitlines()[0].strip()
    if raw.upper().startswith("SKIP"):
        return question
    # Strip a leading "A:" if the model echoed the prompt shape.
    if raw.lower().startswith("a:"):
        raw = raw[2:].strip()
    if not raw:
        return question

    # Concatenate original + expansion. Retrieval is bag-of-words-ish
    # (BM25 + MedCPT dense); simple concatenation beats replacing the
    # original string, which could drop user-specific cues.
    return f"{q} {raw}"
