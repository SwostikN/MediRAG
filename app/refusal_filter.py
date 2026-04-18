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
    "you have",
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
    # "You're experiencing symptoms of X" / "you are experiencing X" —
    # caught after the baseline_v0 calibration run showed an LLM leak
    # where this phrasing passed the classifier because it doesn't
    # use "you have". Also cover "showing signs of" variants.
    "you're experiencing",
    "you are experiencing",
    "you're showing signs",
    "you are showing signs",
    "your symptoms suggest",
    "your symptoms indicate",
    "your symptoms are consistent with",
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
    "call 102",
    "call 100",
    "nearest emergency",
    "go to the er",
    "go to the emergency",
    "call an ambulance",
    "red flag",
    "this is a medical emergency",
    "seek emergency care",
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

    has_dose_unit = bool(_DOSE_UNIT_RE.search(text))
    if has_dose_unit and any(cue in lower for cue in _PRESCRIPTIVE_SCOPE_CUES):
        return "prescriptive"

    return "safe"
