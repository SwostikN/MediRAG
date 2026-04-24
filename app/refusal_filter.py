"""Runtime refusal filter — post-generation guard for diagnostic /
prescriptive language.

Centralises the forbidden-phrase regex previously duplicated in
eval/score_stage2.py. Used by Stage 4 (results explainer) and any
other stage that needs to enforce the no-diagnosis / no-dosing
contract at runtime, not just in eval.

Two patterns of failure this catches:
  1. Diagnostic claims — "you have X", "diagnosis is Y", "sounds like Z"
  2. Dosing recommendations — "take 50 mg", "start taking N", "you
     should take ..."

Design choice: pattern-match, not classifier. A false-positive
(filtering a safe response) is preferable to a false-negative
(letting a diagnostic claim through) — the failure asymmetry is the
same as Stage 0/2 safety logic: over-refuse, never under-refuse.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Diagnostic / interpretive phrases — same set used by eval/score_stage2.py
# plus stronger "diagnosis" variants for Stage 4 risk surface.
_DIAGNOSTIC_PATTERNS = [
    r"\bsounds?\s+like\b",
    r"\bmight\s+be\b",
    r"\bprobably\b",
    r"\byou\s+have\b",
    r"\byou\s+are\s+diagnosed\b",
    r"\bit\s+could\s+be\b",
    r"\bmost\s+likely\b",
    r"\bdiagnosis\s+is\b",
    r"\byour\s+diagnosis\b",
    r"\bI\s+diagnose\b",
    r"\byou\s+(?:are|have\s+got)\s+(?:diabetic|hypothyroid|anemic|hypertensive)\b",
]

# Dosing / prescription phrases. Tightened to "take N mg/mcg/g" rather
# than bare numbers + units (which appear in lab tables: "TSH 8.4 mIU/L").
# "should take" caught separately to handle "you should take this medication".
_DOSING_PATTERNS = [
    r"\btake\s+\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|mL|IU|iu|units?)\b",
    r"\bstart(?:\s+taking)?\s+\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|mL|IU|iu)\b",
    r"\byou\s+should\s+take\b",
    r"\bI\s+recommend\s+(?:taking|starting)\b",
    r"\bdose\s+(?:should\s+be|is)\s+\d+\b",
]

_FORBIDDEN_RE = re.compile(
    "|".join(_DIAGNOSTIC_PATTERNS + _DOSING_PATTERNS),
    re.IGNORECASE,
)

# Default replacement when a generated response is filtered out.
# Stage-specific callers can override via the `fallback` argument.
SAFE_REFUSAL_TEMPLATE = (
    "I should not give a diagnosis or recommend a specific dose. "
    "What I *can* do: explain general patterns, suggest tests a doctor "
    "may consider, and help you prepare for your visit. Please share "
    "the details of your case with a qualified clinician."
)


# Emergency-number localisation. NHS-sourced chunks literally say "call
# 999" and US-sourced chunks say "call 911". The LLM quotes verbatim,
# which is clinically wrong for a Nepal-focused navigator — 999 / 911
# don't route to an ambulance here. Rewrite the answer body (NOT the
# source titles in the citations panel) so the user sees the correct
# Nepal number. Source links stay honest.
_NEPAL_EMERGENCY_REWRITES: list[tuple[str, str]] = [
    # Most specific first so "call 999" is rewritten before "999" alone.
    (r"\bcall\s*999\b", "call 102"),
    (r"\bcall\s*911\b", "call 102"),
    (r"\bdial\s*999\b", "dial 102"),
    (r"\bdial\s*911\b", "dial 102"),
    (r"\b999\s*straight\s*away\b", "102 straight away"),
    (r"\b911\s*straight\s*away\b", "102 straight away"),
    # "999 vs see GP" appears in NHS page titles — only rewrite body
    # prose where the number stands alone and paired with a verb, so
    # we don't rewrite "the 999-point scale" or source-titles the LLM
    # quoted. Use a narrow post-verb pattern.
    (r"\b(?:phone|ring|contact)\s*999\b", "call 102"),
    (r"\b(?:phone|ring|contact)\s*911\b", "call 102"),
    # A&E / emergency-room Britishisms / Americanisms → neutral phrasing
    # that makes sense in Nepal.
    (r"\bA&E\s+department\b", "nearest hospital emergency department"),
    (r"\baccident\s+and\s+emergency\s+department\b", "nearest hospital emergency department"),
    (r"\bgo\s+to\s+A&E\b", "go to the nearest hospital emergency department"),
    (r"\bgo\s+to\s+a&e\b", "go to the nearest hospital emergency department"),
]

_NEPAL_EMERGENCY_COMPILED = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in _NEPAL_EMERGENCY_REWRITES
]


def rewrite_emergency_numbers(text: str) -> str:
    """Post-generation rewrite so the user sees Nepal emergency numbers
    (102 for ambulance) regardless of whether the retrieved source was
    NHS (999) or US (911). Safety floor is unchanged — the LLM's claim
    was already NLI-grounded against the source that literally says
    'call 999'; we're localising the call-to-action, not inventing one.
    """
    if not text:
        return text
    out = text
    for pat, repl in _NEPAL_EMERGENCY_COMPILED:
        out = pat.sub(repl, out)
    return out


def has_forbidden_phrase(text: str) -> bool:
    """True if `text` contains any diagnostic or dosing phrase."""
    return bool(_FORBIDDEN_RE.search(text or ""))


def find_forbidden_phrases(text: str) -> List[str]:
    """Return the list of matched substrings (for logging / eval)."""
    if not text:
        return []
    return [m.group(0) for m in _FORBIDDEN_RE.finditer(text)]


def filter_response(
    text: str,
    *,
    fallback: Optional[str] = None,
) -> Tuple[str, bool]:
    """Run the refusal gate.

    Returns `(text_to_show_user, was_filtered)`. When a forbidden
    phrase is found, returns `(fallback or SAFE_REFUSAL_TEMPLATE,
    True)` — the original generation is discarded. Caller should log
    the original + matched phrases for safety review.
    """
    if has_forbidden_phrase(text):
        return (fallback or SAFE_REFUSAL_TEMPLATE, True)
    return (text, False)


# ──────────────────────────────────────────────────────────────────────────
# Week 10 — scope-guard classifier (policy layer)
# ──────────────────────────────────────────────────────────────────────────
#
# The regex filter above catches specific forbidden *phrases* the LLM may
# emit. This scope classifier works one level up: it looks at the whole
# answer and buckets it by the kind of clinical work it's trying to do.
# That lets us refuse POLICY-incompatible responses (e.g. "take X mg") even
# when they sneak past the phrase regex because of novel wording.
#
# Defense-in-depth with the NLI guardrail: NLI checks whether claims are
# grounded (factual layer), scope-guard checks whether we should be making
# this kind of claim at all (policy layer). A grounded dose recommendation
# is still a dose recommendation — that's the gap this closes.
#
# Keyword-cluster scoring matches the redflag engine pattern (app/redflag.py)
# rather than introducing a new ML dependency. Precision over recall again:
# a false-positive refusal annoys a user, a false-negative is a safety event.


_DIAGNOSTIC_SCOPE_CUES = (
    # "you have been diagnosed" is specifically diagnostic; the earlier
    # bare "you have" catch-all (2026-04-20: removed) was over-firing on
    # patient-ed lab explanations like "your HbA1c of 6.8% means you have
    # a slightly elevated blood sugar." Adversarial diagnosis requests
    # (coverage.jsonl) still fire on "you are diabetic/hypothyroid",
    # "your diagnosis is", "most likely", "sounds like" below.
    "you have been diagnosed",
    "you've been diagnosed",
    "you are diagnosed",
    "your diagnosis",
    "sounds like",
    "most likely",
    "it could be",
    "it might be",
    "it's probably",
    "diagnosis is",
    "i diagnose",
    "you are diabetic",
    "you are hypertensive",
    "you are anemic",
    "you are hypothyroid",
    "this is asthma",
    "this is diabetes",
)

# "You're experiencing X" / "your symptoms suggest X" / "showing signs of X":
# these phrasings ARE diagnostic when X is a named condition, but the cues
# also fire on benign patient-ed framing like "if you're experiencing chest
# pain, go to the nearest emergency" or "your symptoms suggest seeing a
# doctor soon". We only treat these as diagnostic when a named condition
# co-occurs in the same answer. This keeps the original v0 leak closed
# (LLM emitting "you're experiencing symptoms of depression") without
# retracting safe navigation prose (the 2026-04-21 regression).
_CONTEXT_DIAGNOSTIC_CUES = (
    "you're experiencing",
    "you are experiencing",
    "you're showing signs",
    "you are showing signs",
    "your symptoms suggest",
    "your symptoms indicate",
    "your symptoms are consistent with",
    "symptoms of",  # e.g. "symptoms of depression" when paired with a condition
)

_CONDITION_NAMES = (
    "depression", "anxiety", "panic disorder", "bipolar",
    "diabetes", "diabetic", "hypertension", "hypertensive",
    "asthma", "copd", "bronchitis", "pneumonia",
    "angina", "heart attack", "mi ", "myocardial",
    "stroke", "tia",
    "hypothyroid", "hyperthyroid",
    "anaemia", "anemia", "anemic",
    "migraine",
    "covid", "influenza",
    "anaphylaxis",
    "sepsis", "meningitis",
    "tuberculosis",
    "cancer", "lymphoma", "leukaemia", "leukemia",
)

_PRESCRIPTIVE_SCOPE_CUES = (
    "take ",
    "you should take",
    "start taking",
    "i recommend taking",
    "begin with",
    "dose should be",
    "dose is ",
    "increase the dose",
    "reduce the dose",
    "prescribe",
    "i'd prescribe",
    "try taking",
    "consume ",
    "use ",  # paired with dose-unit check below
)

# Emergency override — if the response is routing to ER / ambulance / red-flag,
# scope-guard MUST let it through even if phrasing would otherwise trip the
# diagnostic cluster. Emergencies trump scope rules.
_EMERGENCY_OVERRIDE_CUES = (
    # Nepal emergency numbers (primary for this product)
    "call 102",
    "call 100",
    "dial 102",
    "dial 100",
    # NHS- and US-sourced chunks come through with these; the LLM
    # paraphrases rather than re-numbering, so we must recognise them.
    "call 999",
    "call 911",
    "dial 999",
    "dial 911",
    "nearest emergency",
    "emergency department",
    "emergency room",
    "accident and emergency",
    "go to the er",
    "go to the emergency",
    "go to a&e",
    "go to a & e",
    "call an ambulance",
    "get an ambulance",
    "red flag",
    "this is a medical emergency",
    "seek emergency care",
    "seek immediate",
    "immediate medical help",
    "urgent medical attention",
    "seek urgent medical",
)

# Dose-unit regex — paired with prescriptive cues to avoid false-positive on
# "take your time" / "use the inhaler at night" (no numeric dose attached).
_DOSE_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|l|iu|units?|tablets?|capsules?|drops?|puffs?)\b",
    re.IGNORECASE,
)


SCOPE_REFUSAL_TEMPLATES = {
    "diagnostic": (
        "I shouldn't tell you what condition you have — that's a clinician's call. "
        "What I can do: explain what the symptoms you described could relate to in "
        "general terms, list tests a doctor may consider, and help you prepare "
        "questions for your visit."
    ),
    "prescriptive": (
        "I can't recommend a specific medicine or dose — prescribing is a clinician's "
        "job and depends on your history, allergies, and other meds. I can explain what "
        "a class of medicines does in general, or help you frame what to ask a doctor."
    ),
}


# Informational-vs-personal question classifier. "What happens if…",
# "What are the symptoms of…", "How does X work?" are general-knowledge
# questions — the LLM's answer is describing a phenomenon, not
# diagnosing the user. Applying the diagnostic-scope refusal to those
# answers is a user-hostile false positive.
#
# Personal-advice patterns override informational prefixes, so
# "What medicine should I take for flu?" is still classified as
# personal (and prescriptive scope-guard / safety layers still apply).
_INFORMATIONAL_PREFIXES_RE = re.compile(
    r"^\s*(?:what|how|why|when|where|which|who|"
    r"does|do|is|are|can|could|would|will|"
    r"tell me about|explain|describe)\b",
    re.IGNORECASE,
)

_PERSONAL_ADVICE_RE = re.compile(
    r"\b(?:"
    r"do\s+i\s+have|am\s+i\s+(?:having|getting)|"
    r"should\s+i\s+(?:take|have|eat|drink|stop|start|use|try|see|visit|go)|"
    r"can\s+i\s+(?:take|have|eat|drink|stop|start|use|try)|"
    r"what\s+(?:medicine|medication|drug|dose|pill)\s+should\s+i|"
    r"which\s+(?:medicine|medication|drug|pill)\s+should\s+i|"
    r"what\s+should\s+i\s+(?:take|do|eat|drink)|"
    r"how\s+much\s+(?:should\s+i|of\s+\w+\s+should\s+i|can\s+i\s+take)|"
    r"is\s+it\s+safe\s+(?:for\s+me|if\s+i)"
    r")\b",
    re.IGNORECASE,
)


_ATTACHED_DOC_QUERY_RE = re.compile(
    r"\b("
    r"my\s+(?:report|reports|result|results|lab|labs|blood\s+work|pdf|document|test|tests)"
    r"|these\s+(?:report|reports|result|results|lab|labs|pdf|documents|tests)"
    r"|the\s+(?:report|reports|result|results|lab|labs|pdf|document|uploaded\s+\w+)"
    r"|uploaded\s+(?:report|reports|result|results|lab|labs|pdf|document)"
    r"|based\s+on\s+(?:my|these|the|this)"
    r"|from\s+(?:my|these|the|this)\s+(?:report|reports|result|results|lab|labs|pdf)"
    r")\b",
    re.IGNORECASE,
)


def is_attached_doc_query(question: str) -> bool:
    """True when the user's question explicitly references their
    uploaded lab reports / research PDFs. Used to gate a narrower
    rerank-threshold relax (0.4 → 0.3) ONLY when the user is asking
    about content they themselves attached — not when they're asking
    a general health question that happens to have poor retrieval.

    NLI entailment still runs unchanged. Only the coverage gate
    (rerank_score) is relaxed, and only for this query class.
    """
    if not question or not question.strip():
        return False
    return bool(_ATTACHED_DOC_QUERY_RE.search(question))


def is_informational_question(question: str) -> bool:
    """True when the question asks for general information rather than
    personal medical advice. Informational questions should not trigger
    diagnostic-scope retraction. NLI entailment still runs, so the
    answer must still be source-grounded — safety is not softened."""
    if not question or not question.strip():
        return False
    lower = question.lower().strip()
    if _PERSONAL_ADVICE_RE.search(lower):
        return False
    return bool(_INFORMATIONAL_PREFIXES_RE.search(lower))


def classify_scope(text: str) -> str:
    """Bucket an answer into: 'safe', 'diagnostic', 'prescriptive', or
    'emergency_override'.

    Order of checks matters:
      1. Emergency override FIRST — emergency routing takes precedence
         even if the text also contains a diagnostic cue (e.g. "this is
         a medical emergency — call an ambulance").
      2. Diagnostic cluster — "you have X" / "your diagnosis is Y".
      3. Prescriptive cluster — only fires when a prescriptive cue
         co-occurs with a dose-unit pattern. "Take this medication" on
         its own is fine; "take 500 mg of X" is not.
      4. Default — 'safe'.
    """
    if not text:
        return "safe"
    lower = text.lower()

    if any(cue in lower for cue in _EMERGENCY_OVERRIDE_CUES):
        return "emergency_override"

    if any(cue in lower for cue in _DIAGNOSTIC_SCOPE_CUES):
        return "diagnostic"

    # Context-sensitive cluster — only diagnostic when paired with a
    # named condition. Stops "if you're experiencing chest pain, go to
    # the ER" from tripping diagnostic retraction on navigation prose.
    if any(cue in lower for cue in _CONTEXT_DIAGNOSTIC_CUES):
        if any(cond in lower for cond in _CONDITION_NAMES):
            return "diagnostic"

    has_dose_unit = bool(_DOSE_UNIT_RE.search(text))
    if has_dose_unit and any(cue in lower for cue in _PRESCRIPTIVE_SCOPE_CUES):
        return "prescriptive"

    return "safe"
