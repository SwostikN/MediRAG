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
