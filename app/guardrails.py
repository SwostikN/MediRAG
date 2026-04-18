"""Week 10 hallucination guardrails.

This module hosts the layers that sit AFTER generation and BEFORE the
response is sent back to the user. First piece: a claim classifier.

Why a claim classifier first? The NLI verifier is expensive (model load +
per-sentence inference) and too harsh if applied blanket: honest paraphrases
and navigation prose score 0.5–0.7 on entailment and would be wrongly
redacted. The classifier decides which sentences are *clinical claims*
worth NLI-checking. Navigation, disclaimers, and generic framing skip NLI.

The classifier is deliberately simple — regex + keyword cues, no ML.
Precision over recall: when we're unsure, treat the sentence as a claim
(False positives here just mean an NLI check runs on benign prose;
False negatives would let a clinical claim escape the verifier).
"""

from __future__ import annotations

from dataclasses import dataclass
import re


# Dose units with an attached number. \b number \b (mg|mcg|g|ml|IU|units)
# The \b before the number catches "500mg" (no space) and "500 mg".
_DOSE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|l|iu|units?|tablets?|capsules?|drops?|puffs?)\b",
    re.IGNORECASE,
)

# Numeric clinical thresholds. Covers BP (150/90 mmHg, or bare 140/90),
# percentages (HbA1c 6.5%), mmol/L, mg/dL, bpm, kg/m², temperatures.
# Note: trailing `\b` doesn't work for `%` / `°c` / `°f` because those
# characters are non-word. Use a unit-specific boundary instead.
_THRESHOLD_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*"
    r"(?:/\s*\d+(?:\.\d+)?\s*)?"  # optional "/90" for BP
    r"(?:mmhg\b|mmol/?l\b|mg/?dl\b|bpm\b|kg/m2\b|kg/m²|%|°c|°f)",
    re.IGNORECASE,
)

# Bare BP pattern ("150/90") — common in patient-facing prose without units.
_BP_BARE_RE = re.compile(r"\b\d{2,3}\s*/\s*\d{2,3}\b")

# Diagnostic verbs. "You have hypertension", "diagnosed with diabetes",
# "this is asthma". Kept narrow — generic "have" on its own is too loose.
_DIAGNOSIS_RE = re.compile(
    r"\b(?:you\s+(?:have|are\s+having)|diagnosed\s+with|this\s+is\s+(?:a|an)?\s*\w+)\b",
    re.IGNORECASE,
)

# Duration of treatment/symptoms. "for 5 days", "every 6 hours", "twice
# daily". Not every duration is a claim — "you've had this for 3 days"
# is the USER's history, not a claim BY the assistant. Caller handles that.
_DURATION_RE = re.compile(
    r"\b(?:for|every|over|within)\s+\d+\s*(?:days?|weeks?|months?|hours?|minutes?)\b"
    r"|\b(?:once|twice|three\s+times|four\s+times)\s+(?:a\s+)?(?:day|daily|week|weekly)\b",
    re.IGNORECASE,
)

# Disclaimers / policy prose. These ALWAYS pass NLI because they are
# template strings the assistant emits independently of retrieval.
_DISCLAIMER_CUES = (
    "i can't diagnose",
    "i cannot diagnose",
    "i'm not a doctor",
    "i am not a doctor",
    "please see a",
    "please consult",
    "see a clinician",
    "see a doctor",
    "go to the",  # "go to the nearest emergency"
    "call 102",
    "call 100",
    "this is not medical advice",
    "based on what you",
    "here's what",
    "here is what",
)


@dataclass
class ClaimFeatures:
    has_dose: bool
    has_threshold: bool
    has_diagnosis_verb: bool
    has_duration: bool
    is_disclaimer: bool

    @property
    def requires_nli(self) -> bool:
        """True if this sentence makes a clinical claim worth verifying.
        Disclaimers short-circuit to False — they are template output, not
        corpus-grounded claims."""
        if self.is_disclaimer:
            return False
        return (
            self.has_dose
            or self.has_threshold
            or self.has_diagnosis_verb
            or self.has_duration
        )


def classify_claim(sentence: str) -> ClaimFeatures:
    """Classify a single sentence for NLI-worthy clinical claims.

    Splitting a response into sentences is the CALLER's job — this function
    treats `sentence` as atomic. Callers can use e.g. `re.split(r'(?<=[.!?])\\s+', text)`
    for a cheap split, or a proper tokeniser when one is on hand.
    """
    if not sentence or not sentence.strip():
        return ClaimFeatures(False, False, False, False, False)

    lower = sentence.lower()
    is_disclaimer = any(cue in lower for cue in _DISCLAIMER_CUES)

    has_dose = bool(_DOSE_RE.search(sentence))
    has_threshold = bool(_THRESHOLD_RE.search(sentence)) or bool(
        _BP_BARE_RE.search(sentence)
    )
    has_diagnosis_verb = bool(_DIAGNOSIS_RE.search(sentence))
    has_duration = bool(_DURATION_RE.search(sentence))

    return ClaimFeatures(
        has_dose=has_dose,
        has_threshold=has_threshold,
        has_diagnosis_verb=has_diagnosis_verb,
        has_duration=has_duration,
        is_disclaimer=is_disclaimer,
    )
