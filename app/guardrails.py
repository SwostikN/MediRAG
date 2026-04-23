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
# "this is asthma". The older `this\s+is\s+(?:a|an)?\s*\w+` shape was
# catastrophically broad — it matched "this is a common symptom",
# "this is a warning sign", "this is a serious condition" — every
# such sentence was promoted to a hard claim with a 0.5 entailment
# floor and then redacted, leaving the total-redaction refusal shown
# to users for anxiety / stress / trauma / allergy questions. Restrict
# "this is <condition>" to an explicit disease list; generic patient-ed
# framing is NOT a diagnosis.
_DIAGNOSIS_CONDITIONS = (
    "asthma|diabetes|hypertension|depression|anxiety|bronchitis|pneumonia|"
    "covid|influenza|flu|migraine|stroke|heart\\s+attack|angina|"
    "anaphylaxis|sepsis|meningitis|tuberculosis|tb|cancer"
)
_DIAGNOSIS_RE = re.compile(
    rf"\b(?:you\s+(?:have|are\s+having)|diagnosed\s+with|"
    rf"this\s+is\s+(?:a|an)?\s*(?:\w+\s+){{0,3}}(?:{_DIAGNOSIS_CONDITIONS}))\b",
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


# ──────────────────────────────────────────────────────────────────────────
# Phase 3 — Inline citation parsing + per-claim source assignment
# ──────────────────────────────────────────────────────────────────────────
#
# When the LLM tags each clinical claim with `[N]` or `[src:N]` markers,
# we can NLI-verify the sentence against JUST the chunks it claimed to cite,
# not the max across all chunks. This closes the "right claim, wrong
# citation" binding error that max-across-all cannot detect.
#
# Markers the parser recognises (inclusive):
#   [1]   [2, 3]   [src:1]   [src:1,2]   [1,2]
#
# Absence of a marker on a claim sentence is NOT treated as a failure here;
# the fallback is the original max-across-all NLI (same behaviour as before).
# A stricter mode that REQUIRES markers on claim sentences is a separate
# opt-in, controlled by the `require_markers` flag in apply_guardrails.

_CITATION_MARKER_RE = re.compile(r"\[(?:src\s*:\s*)?((?:\d+\s*,?\s*)+)\]", re.IGNORECASE)


def parse_inline_citations(sentence: str) -> tuple[str, list[int]]:
    """Extract citation markers and return (clean_sentence, [0-indexed chunk idxs]).

    LLM emits 1-based markers (`[1]`, `[2, 3]`). We convert to 0-based
    internally for list indexing. Duplicate indices are deduped. The
    clean sentence has the marker tokens removed so NLI is not distracted
    by bracketed digits that are not part of the claim.
    """
    if not sentence:
        return sentence, []
    idxs: list[int] = []
    for m in _CITATION_MARKER_RE.finditer(sentence):
        raw = m.group(1)
        for part in re.split(r",\s*", raw):
            part = part.strip()
            if not part:
                continue
            try:
                idxs.append(int(part) - 1)
            except ValueError:
                continue
    # Dedupe preserving order, drop negatives.
    seen: set[int] = set()
    deduped: list[int] = []
    for i in idxs:
        if i >= 0 and i not in seen:
            seen.add(i)
            deduped.append(i)
    clean = _CITATION_MARKER_RE.sub("", sentence).strip()
    # Collapse any double spaces left by the substitution.
    clean = re.sub(r"\s{2,}", " ", clean)
    return clean, deduped


def _split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter. Good enough for patient-ed prose; does not
    try to handle abbreviations like 'e.g.' or 'Dr.' perfectly — the cost
    of a wrong split is one extra (or one missed) classifier call, which
    fails-soft either way. A proper tokeniser is an upgrade path."""
    if not text or not text.strip():
        return []
    return [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]


def _process_sentence(
    sent: str,
    chunk_texts: list[str],
    *,
    verifier,
    classifier,
    use_inline_citations: bool = False,
) -> tuple[str | None, dict]:
    """Apply classify → NLI → tiered-action to one sentence.

    Returns `(emit_text, score_entry)`. `emit_text` is the string to append
    to the filtered answer, or `None` if the sentence is redacted. Shared
    between batch (apply_guardrails) and streaming (process_streaming_chunk)
    so both paths apply exactly the same policy.

    When `use_inline_citations=True` and the sentence carries `[N]` markers,
    the NLI pass is restricted to the assigned chunks (1-based markers,
    0-indexed internally). Missing / out-of-range markers fall back to the
    max-across-all behaviour so the model cannot silently bypass
    verification by citing a non-existent chunk.
    """
    # Phase-3A: strip inline citation markers before classification + NLI.
    assigned_idxs: list[int] = []
    sent_raw = sent
    if use_inline_citations:
        sent, assigned_idxs = parse_inline_citations(sent_raw)
        if not sent:
            # Sentence was nothing but a citation marker — treat as empty.
            return None, {
                "sentence": sent_raw[:200],
                "action": "redact_empty_after_marker_strip",
                "requires_nli": False,
                "p_entail": None,
                "flags": {},
                "assigned_idxs": assigned_idxs,
            }

    feats = classifier(sent)
    flags = {
        "dose": feats.has_dose,
        "threshold": feats.has_threshold,
        "diagnosis": feats.has_diagnosis_verb,
        "duration": feats.has_duration,
    }

    if not feats.requires_nli:
        entry = {
            "sentence": sent[:200],
            "action": "keep_no_claim",
            "requires_nli": False,
            "p_entail": None,
            "flags": flags,
        }
        if use_inline_citations:
            entry["assigned_idxs"] = assigned_idxs
        return sent, entry

    # Pick which chunks this sentence will be NLI-verified against. When
    # inline markers are valid, restrict to those; otherwise fall back to
    # all chunks (preserves legacy behaviour when markers are absent or
    # malformed).
    nli_sources_mode = "all"
    nli_targets = chunk_texts
    if use_inline_citations and assigned_idxs:
        picked = [chunk_texts[i] for i in assigned_idxs if 0 <= i < len(chunk_texts)]
        if picked:
            nli_targets = picked
            nli_sources_mode = "assigned"
        else:
            nli_sources_mode = "assigned_invalid_fallback_all"

    p_entail: float = 0.0
    nli_error: str = ""
    try:
        for ct in nli_targets:
            p = verifier(sent, ct)
            if p > p_entail:
                p_entail = p
    except Exception as exc:
        nli_error = str(exc)[:200]
        print(f"[guardrails] NLI call failed, fail-soft: {exc}")

    is_hard_claim = feats.has_dose or feats.has_diagnosis_verb

    if nli_error:
        emit = sent + " _(not independently verified)_"
        action = "soften_nli_error"
    elif is_hard_claim and p_entail < _NLI_HARD_CLAIM_MIN:
        emit = None
        action = "redact_hard_claim"
    elif p_entail < _NLI_REDACT_BELOW:
        emit = None
        action = "redact"
    elif p_entail < _NLI_SOFTEN_BELOW:
        emit = sent + " _(not directly stated in sources — confirm with a clinician)_"
        action = "soften"
    else:
        emit = sent
        action = "keep"

    entry = {
        "sentence": sent[:200],
        "action": action,
        "requires_nli": True,
        "p_entail": round(p_entail, 3) if not nli_error else None,
        "flags": flags,
    }
    if nli_error:
        entry["error"] = nli_error
    if use_inline_citations:
        entry["assigned_idxs"] = assigned_idxs
        entry["nli_sources_mode"] = nli_sources_mode
    return emit, entry


_FUSION_DISCLAIMER = " _(combines claims from multiple sources — confirm with a clinician)_"


def _fusion_drift_check(
    sent_a: str,
    sent_b: str,
    chunk_texts: list[str],
    *,
    verifier,
) -> tuple[bool, float]:
    """Phase-3B: two-sentence fusion-drift detector.

    NLI each individual sentence passed, but the CONCATENATION is a
    single compound claim that may not be supported by any single chunk
    (the "ACE inhibitors cause cough in 5% of hypertensive patients"
    shape). Re-NLI the concatenated pair; returns (fused_fails, p_fused).

    fused_fails = True when the concatenation's max-entailment falls
    below the individual-sentence soften threshold. Callers decide
    whether to soften, redact, or just log.
    """
    fused = f"{sent_a.rstrip('. ')}. {sent_b.lstrip()}"
    p_fused: float = 0.0
    try:
        for ct in chunk_texts:
            p = verifier(fused, ct)
            if p > p_fused:
                p_fused = p
    except Exception as exc:
        print(f"[guardrails] fusion NLI failed, fail-soft: {exc}")
        return False, 0.0
    return p_fused < _NLI_SOFTEN_BELOW, p_fused


def apply_guardrails(
    answer: str,
    top_rows: list[dict],
    *,
    verifier=None,
    classifier=None,
    use_inline_citations: bool = False,
    check_fusion_drift: bool = False,
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
    use_inline_citations : when True, parse `[N]` markers on each sentence
                and NLI against the assigned chunks only (Phase-3A
                binding). Legacy max-across-all when False or no markers.
    check_fusion_drift   : when True, also NLI consecutive claim-sentence
                pairs as a compound claim. Pairs where each sentence passes
                individually but the fusion fails get a soft-fusion
                disclaimer appended to the second sentence and a
                `fusion_drift` entry added to the score_log. (Phase-3B.)

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
    # Track the last kept sentence for fusion-drift pairing. We only pair
    # two sentences that BOTH classified as claims AND both passed
    # individual NLI (i.e. emit is not None and action in keep/soften).
    prev_sent_for_fusion: str | None = None
    prev_entry_for_fusion: dict | None = None

    for sent in sentences:
        emit, entry = _process_sentence(
            sent,
            chunk_texts,
            verifier=verifier,
            classifier=classifier,
            use_inline_citations=use_inline_citations,
        )
        # Fusion-drift check fires when BOTH the previous kept sentence
        # and the current one are individually-verified claims. We
        # append the fusion disclaimer to the current sentence (softens
        # forward, never re-edits what was already appended).
        if (
            check_fusion_drift
            and emit is not None
            and entry.get("requires_nli")
            and entry.get("action") in ("keep", "soften")
            and prev_sent_for_fusion is not None
            and prev_entry_for_fusion is not None
            and prev_entry_for_fusion.get("action") in ("keep", "soften")
        ):
            fails, p_fused = _fusion_drift_check(
                prev_sent_for_fusion, emit, chunk_texts, verifier=verifier
            )
            if fails:
                emit = emit + _FUSION_DISCLAIMER
                entry["fusion_drift"] = {
                    "flagged": True,
                    "p_fused": round(p_fused, 3),
                    "paired_with_prev_sentence": True,
                }
            else:
                entry["fusion_drift"] = {
                    "flagged": False,
                    "p_fused": round(p_fused, 3),
                }

        if emit is not None:
            kept.append(emit)
            if entry.get("requires_nli") and entry.get("action") in ("keep", "soften"):
                prev_sent_for_fusion = emit
                prev_entry_for_fusion = entry
            else:
                # Non-claim sentence breaks the fusion chain; do not pair
                # a non-claim "framing" sentence with the next claim.
                prev_sent_for_fusion = None
                prev_entry_for_fusion = None
        else:
            prev_sent_for_fusion = None
            prev_entry_for_fusion = None
        score_log.append(entry)

    filtered = " ".join(kept).strip()
    return filtered, score_log


# ──────────────────────────────────────────────────────────────────────────
# Phase-3B: post-stream fusion-drift check for /query/stream
# ──────────────────────────────────────────────────────────────────────────
#
# `process_streaming_chunk` can only see one sentence at a time — it has no
# way to pair consecutive claim sentences and NLI their concatenation (the
# "ACE inhibitors cause cough in 5% of patients" shape where each half
# entails its own chunk but the fusion is unsupported).
#
# To close that gap without rewriting the streaming pipeline, we run a
# post-stream pass over the list of already-emitted sentences. We cannot
# retroactively edit text that's already on the user's screen, so the
# caller uses the returned records to emit a trailing fusion-drift
# disclaimer delta + persists the records to query_log for offline audit.
def post_stream_fusion_check(
    emitted_sentences: list[str],
    chunk_texts: list[str],
    *,
    verifier=None,
    classifier=None,
) -> list[dict]:
    """Run `_fusion_drift_check` on consecutive claim-sentence pairs from
    a completed stream. Returns a list of fusion-drift records, one per
    flagged pair, each with keys:
        - `sent_a_index`, `sent_b_index`: positions in `emitted_sentences`
        - `p_fused`: max-entailment of the concatenated pair
        - `flagged`: always True in returned records (non-flagged pairs omitted)

    Empty list when no pairs flag. Safe to call with empty input.
    """
    if not emitted_sentences or not chunk_texts:
        return []

    verifier = verifier or verify_entailment
    classifier = classifier or classify_claim

    drifts: list[dict] = []
    prev_claim_sent: str | None = None
    prev_claim_idx: int | None = None

    for idx, sent in enumerate(emitted_sentences):
        if not sent or not sent.strip():
            continue
        # classify_claim returns a ClaimFeatures dataclass. Pair only
        # sentences that carry NLI-worthy clinical claims; otherwise we
        # flag benign framing-to-framing pairs and spray the spurious
        # "combines claims from multiple sources" disclaimer on answers
        # that never cross-source-fuse anything.
        feats = classifier(sent)
        is_claim = getattr(feats, "requires_nli", False)
        if is_claim and prev_claim_sent is not None and prev_claim_idx is not None:
            fails, p_fused = _fusion_drift_check(
                prev_claim_sent, sent, chunk_texts, verifier=verifier
            )
            if fails:
                drifts.append({
                    "flagged": True,
                    "sent_a_index": prev_claim_idx,
                    "sent_b_index": idx,
                    "p_fused": p_fused,
                })
        if is_claim:
            prev_claim_sent = sent
            prev_claim_idx = idx
        else:
            # Non-claim sentence breaks the chain — do not pair a framing
            # sentence with the next claim (mirrors apply_guardrails).
            prev_claim_sent = None
            prev_claim_idx = None

    return drifts


# ──────────────────────────────────────────────────────────────────────────
# Streaming guardrails — Option C (buffer-per-sentence)
# ──────────────────────────────────────────────────────────────────────────
#
# `/query/stream` receives tokens one at a time from Groq/Cohere. We can't
# guardrail a partial sentence because the NLI verifier needs a complete
# clinical claim. Option C buffers tokens until a sentence boundary, runs
# the exact same _process_sentence pipeline as batch mode, then emits the
# (possibly redacted or softened) sentence to the SSE stream.
#
# Why not stream-then-replace (Option B)? The user would briefly see an
# unsafe sentence on screen before the redaction edit landed. For a health
# tool that is a nonstarter. Option C adds ~1–2s of sentence-level latency
# but never shows the user text we later retract.


# Sentence boundary used for streaming. Splits on `.!?` followed by whitespace,
# tolerating either uppercase / digit / `[` start (same shape as _SENTENCE_SPLIT_RE)
# but also happy to emit on pure whitespace since tokens arrive mid-word.
_STREAM_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def process_streaming_chunk(
    buffer: str,
    token: str,
    chunk_texts: list[str],
    score_log: list[dict],
    *,
    verifier=None,
    classifier=None,
    use_inline_citations: bool = False,
) -> tuple[str, list[str]]:
    """Incrementally feed `token` into `buffer`; return any complete
    sentences that passed guardrails and are safe to flush to the client.

    The caller owns `buffer` and `score_log` across calls — we mutate
    `score_log` in place so the final record at end-of-stream matches
    the batch-mode shape exactly (same JSON, same schema, same logging).

    Returns
    -------
    (new_buffer, emits)
        new_buffer : remaining text that is still a partial sentence
        emits      : list of sentence strings safe to yield to the client
                     (may be empty if no sentence boundary crossed)
    """
    verifier = verifier or verify_entailment
    classifier = classifier or classify_claim

    buffer = buffer + token
    emits: list[str] = []

    while True:
        m = _STREAM_BOUNDARY_RE.search(buffer)
        if not m:
            break
        sent = buffer[: m.start() + 1].strip()  # include the punctuation
        buffer = buffer[m.end():]
        if not sent:
            continue
        emit, entry = _process_sentence(
            sent,
            chunk_texts,
            verifier=verifier,
            classifier=classifier,
            use_inline_citations=use_inline_citations,
        )
        score_log.append(entry)
        if emit is not None:
            emits.append(emit)

    return buffer, emits


def flush_streaming_buffer(
    buffer: str,
    chunk_texts: list[str],
    score_log: list[dict],
    *,
    verifier=None,
    classifier=None,
    use_inline_citations: bool = False,
) -> list[str]:
    """End-of-stream flush. If `buffer` still holds a partial sentence
    (the model's final sentence often lacks a trailing space), run it
    through the guardrail pipeline one last time.

    Mutates `score_log` in place. Returns the list of surviving sentence
    strings (0 or 1 elements in practice).

    Note: fusion-drift checking is batch-only. The streaming path emits
    one sentence at a time with no cross-sentence buffer, so we cannot
    pair consecutive claims without reintroducing the latency the
    streaming architecture was designed to avoid. Batch callers get
    full Phase-3 coverage; stream callers get Phase-3A (binding) only.
    """
    verifier = verifier or verify_entailment
    classifier = classifier or classify_claim

    sent = buffer.strip()
    if not sent:
        return []
    emit, entry = _process_sentence(
        sent,
        chunk_texts,
        verifier=verifier,
        classifier=classifier,
        use_inline_citations=use_inline_citations,
    )
    score_log.append(entry)
    return [emit] if emit is not None else []
