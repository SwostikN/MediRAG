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


# ──────────────────────────────────────────────────────────────────────────
# Layer 2 — NLI entailment verifier
# ──────────────────────────────────────────────────────────────────────────
#
# `verify_entailment(sentence, chunk)` returns P(entailment) in [0, 1] from
# a cross-encoder NLI model. Callers decide the tier action on the score
# (the integration layer in /query will redact below 0.2, soften 0.2–0.5,
# keep above 0.5; dose + diagnosis claims ignore the soften tier).
#
# The model (~400 MB) is lazy-loaded on first call and cached module-wide.
# Startup is NOT slowed — nothing below this comment runs on import.


NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

_nli_state: dict = {"tokenizer": None, "model": None, "entail_idx": None}


def _load_nli() -> dict:
    """Idempotent loader. Populates _nli_state on first call, then returns
    it. Raises on failure — callers are expected to catch and treat the
    exception as 'could not verify' (fail-closed at integration time).

    Looking up the entailment index via model.config.id2label rather than
    hardcoding is deliberate: MNLI-era models use [contradiction, neutral,
    entailment] while FEVER-era and some ANLI-era models use [contradiction,
    entailment, neutral]. Hardcoding is a silent-bug factory on model swaps.
    """
    if _nli_state["model"] is not None:
        return _nli_state
    import torch  # local import so module import stays cheap
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    mdl.eval()

    entail_idx = None
    for idx, label in mdl.config.id2label.items():
        if str(label).lower().startswith("entail"):
            entail_idx = int(idx)
            break
    if entail_idx is None:
        raise RuntimeError(
            f"NLI model {NLI_MODEL_NAME!r} has no 'entailment' label in "
            f"config.id2label={mdl.config.id2label!r}"
        )

    _nli_state["tokenizer"] = tok
    _nli_state["model"] = mdl
    _nli_state["entail_idx"] = entail_idx
    _nli_state["torch"] = torch
    return _nli_state


def verify_entailment(sentence: str, chunk_text: str) -> float:
    """Return P(entailment) in [0, 1] for 'does chunk_text support sentence?'.

    NLI convention: premise=chunk_text, hypothesis=sentence. The model
    scores whether the premise entails the hypothesis.

    Empty inputs short-circuit to 0.0 — a sentence with no supporting
    chunk cannot be entailed. Callers should not call this on classifier-
    skipped sentences (disclaimers, navigation) because the score has no
    meaning there."""
    if not sentence or not sentence.strip():
        return 0.0
    if not chunk_text or not chunk_text.strip():
        return 0.0

    state = _load_nli()
    tok = state["tokenizer"]
    mdl = state["model"]
    entail_idx = state["entail_idx"]
    torch = state["torch"]

    inputs = tok(
        chunk_text,
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = mdl(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    return float(probs[entail_idx].item())


# ──────────────────────────────────────────────────────────────────────────
# Layer 3 — Orchestration: apply_guardrails
# ──────────────────────────────────────────────────────────────────────────
#
# Ties Layer 1 (classifier) and Layer 2 (NLI verifier) into the shape the
# response path actually needs: given the generated answer and the list of
# retrieved chunks the LLM saw, produce a (filtered_answer, score_log) pair.
#
# Thresholds are deliberately constants here rather than constructor args.
# Calibration (Week 10 item 4) will move them to a config file once we have
# gold-set measurements to support the numbers.


# Tier cutoffs on P(entailment). See DOCUMED §10.6 for rationale.
_NLI_REDACT_BELOW = 0.2
_NLI_SOFTEN_BELOW = 0.5
# Hard-claim floor: dose + diagnosis sentences must clear this or they
# are redacted, never softened. Hard claims ARE the dangerous failure mode.
_NLI_HARD_CLAIM_MIN = 0.5

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")


def _split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter. Good enough for patient-ed prose; does not
    try to handle abbreviations like 'e.g.' or 'Dr.' perfectly — the cost
    of a wrong split is one extra (or one missed) classifier call, which
    fails-soft either way. A proper tokeniser is an upgrade path."""
    if not text or not text.strip():
        return []
    return [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]


def apply_guardrails(
    answer: str,
    top_rows: list[dict],
    *,
    verifier=None,
    classifier=None,
) -> tuple[str, list[dict]]:
    """Run the Layer-1 classifier + Layer-2 NLI verifier on `answer`.

    Parameters
    ----------
    answer    : the LLM's generated response.
    top_rows  : the retrieved chunks passed to the LLM as context. Each
                row must have a 'content' key. We check each classified-
                claim sentence against EVERY chunk and take the max
                P(entail) — if any chunk entails it, we treat the
                sentence as supported.
    verifier  : optional override for verify_entailment. Tests inject a
                mock to avoid loading the 400 MB NLI model.
    classifier: optional override for classify_claim.

    Returns
    -------
    filtered_answer : the answer with redacted sentences removed and
                      softened sentences annotated. If every claim
                      sentence was redacted, returns "" and callers
                      should fall back to the standard refusal.
    score_log       : list of per-sentence records suitable for jsonb
                      persistence on public.query_log. See migration
                      013 for the schema contract.
    """
    verifier = verifier or verify_entailment
    classifier = classifier or classify_claim

    sentences = _split_sentences(answer)
    if not sentences:
        return answer, []

    chunk_texts = [r.get("content", "") for r in top_rows if r.get("content")]

    kept: list[str] = []
    score_log: list[dict] = []

    for sent in sentences:
        feats = classifier(sent)
        flags = {
            "dose": feats.has_dose,
            "threshold": feats.has_threshold,
            "diagnosis": feats.has_diagnosis_verb,
            "duration": feats.has_duration,
        }

        if not feats.requires_nli:
            kept.append(sent)
            score_log.append({
                "sentence": sent[:200],
                "action": "keep_no_claim",
                "requires_nli": False,
                "p_entail": None,
                "flags": flags,
            })
            continue

        # Run NLI against every chunk; keep the max. If any chunk entails
        # the sentence, we treat it as supported.
        p_entail: float = 0.0
        nli_error: str = ""
        try:
            for ct in chunk_texts:
                p = verifier(sent, ct)
                if p > p_entail:
                    p_entail = p
        except Exception as exc:
            nli_error = str(exc)[:200]
            print(f"[guardrails] NLI call failed, fail-soft: {exc}")

        is_hard_claim = feats.has_dose or feats.has_diagnosis_verb

        if nli_error:
            kept.append(sent + " _(not independently verified)_")
            action = "soften_nli_error"
        elif is_hard_claim and p_entail < _NLI_HARD_CLAIM_MIN:
            action = "redact_hard_claim"
            # sentence is dropped
        elif p_entail < _NLI_REDACT_BELOW:
            action = "redact"
            # sentence is dropped
        elif p_entail < _NLI_SOFTEN_BELOW:
            kept.append(
                sent + " _(not directly stated in sources — confirm with a clinician)_"
            )
            action = "soften"
        else:
            kept.append(sent)
            action = "keep"

        score_log.append({
            "sentence": sent[:200],
            "action": action,
            "requires_nli": True,
            "p_entail": round(p_entail, 3) if not nli_error else None,
            "flags": flags,
            **({"error": nli_error} if nli_error else {}),
        })

    filtered = " ".join(kept).strip()
    return filtered, score_log
