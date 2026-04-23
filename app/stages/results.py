"""Stage 4 — lab-results explainer.

Runs after a user uploads a document classified as `lab_report` (see
app/document_classifier.py). Two halves:

  1. extract_lab_markers(text) — regex parser over the Nepal lab
     formats we've seen in practice (Samyak, Prime, NMC Pathology,
     Norvic, Grande, Star) plus generic "marker: value unit (range)"
     patterns common to most printed reports. No LLM call. Output is a
     list of `LabMarker` dataclasses with value, unit, reference range
     (when detectable), and a high/low/normal status flag.

  2. compose_explainer(markers, retrieve_fn, ...) — for each detected
     marker, pull supporting context from the corpus (NHS / Lab Tests
     Online / WHO / MoHP STG once Week 8 corpus is in), and assemble a
     Markdown block with:
       - what the marker measures (one sentence, sourced)
       - the patient's value vs. the report's reference range
       - what high/low values commonly indicate, GENERALLY (no
         diagnosis for THIS patient)
       - what to discuss with a doctor at the visit
       - a fixed escalation footer with red-flag cues.

Hard rules (mirror the eval/score_stage4 contract):

  - Never output a diagnosis ("you have hypothyroidism", "this is
    diabetes"). The composer enforces this with a final pass through
    app.refusal_filter — any forbidden phrase replaces the entire
    block with the safe-refusal template.
  - Never recommend a dose, drug, or treatment. Same enforcement.
  - Reference ranges are LAB-SPECIFIC. We only ever quote the range
    printed on the patient's report — never substitute a "standard"
    range. If the report omits a range for a marker, we say so and
    skip the comparison rather than guess.
  - When LLM generation fails for a marker, that marker block is
    replaced with a conservative "couldn't safely interpret —
    discuss with your doctor" stub. The other markers still render.
  - The escalation footer is appended deterministically, not asked
    of the LLM, so it is always present even on degraded paths.

Failure asymmetry: same as Stage 0/2. False-positive on the refusal
filter (filtering a safe block) is annoying. False-negative (a
diagnostic phrase reaching the user) is the failure mode that makes
this entire feature unsafe to ship. Bias every threshold toward
filtering.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

try:
    from ..refusal_filter import filter_response, find_forbidden_phrases
except ImportError:  # pragma: no cover — when imported from script context
    from refusal_filter import filter_response, find_forbidden_phrases  # type: ignore


# ---------------------------------------------------------------------------
# Marker dictionary
# ---------------------------------------------------------------------------
# Canonical name → list of regex-friendly aliases as they appear on Nepali
# lab reports. Aliases are matched case-insensitively. Order matters only
# for display; the parser picks whichever alias appears first in the text.
#
# Scope: the ~15 markers we will ship lab-explainer corpus content for in
# Week 8. Anything outside this list is detected only by the generic
# "value + unit + range" regex and surfaced as `name="unknown_<unit>"`,
# which the composer skips with a "marker not in our reference library"
# note rather than guessing.

_MARKER_ALIASES: dict[str, list[str]] = {
    "TSH":          ["TSH", "thyroid stimulating hormone", "thyrotropin"],
    "FT4":          ["FT4", "free T4", "free thyroxine"],
    "FT3":          ["FT3", "free T3", "free triiodothyronine"],
    "HbA1c":        ["HbA1c", "glycated haemoglobin", "glycated hemoglobin", "A1c"],
    "FBS":          ["FBS", "fasting blood sugar", "fasting blood glucose", "fasting glucose"],
    "Hb":           ["Hb", "Hgb", "haemoglobin", "hemoglobin"],
    "LDL":          ["LDL", "LDL cholesterol", "LDL-C"],
    "HDL":          ["HDL", "HDL cholesterol", "HDL-C"],
    "Triglycerides": ["triglycerides", "TG"],
    "Total cholesterol": ["total cholesterol", "cholesterol total", "TC"],
    "ALT":          ["ALT", "SGPT", "alanine aminotransferase"],
    "AST":          ["AST", "SGOT", "aspartate aminotransferase"],
    "Creatinine":   ["creatinine", "serum creatinine"],
    "Vitamin D":    ["vitamin D", "25-OH vitamin D", "25(OH)D", "vit D"],
    "Vitamin B12":  ["vitamin B12", "B12", "cobalamin"],
    "Ferritin":     ["ferritin", "serum ferritin"],
    # Renal function panel — Nepali labs routinely bundle these in the
    # RFT section. Parser was missing all of these before, so a real
    # NARANATH-style RFT report returned zero markers and was wrongly
    # classified as unreadable.
    "Urea":         ["urea", "blood urea", "BUN", "blood urea nitrogen"],
    "Sodium":       ["sodium", "Na+", "serum sodium"],
    "Potassium":    ["potassium", "K+", "serum potassium"],
    "Chloride":     ["chloride", "Cl-", "serum chloride"],
    "eGFR":         ["eGFR", "estimated GFR", "estimated glomerular filtration rate"],
    "Uric acid":    ["uric acid", "serum uric acid"],
    # Common supplementary panels.
    "Calcium":      ["calcium", "serum calcium", "Ca2+", "total calcium"],
    "Phosphate":    ["phosphate", "phosphorus", "serum phosphate", "serum phosphorus"],
    "Albumin":      ["albumin", "serum albumin"],
    "Total protein": ["total protein", "serum protein"],
    "Bilirubin":    ["bilirubin", "total bilirubin", "direct bilirubin", "indirect bilirubin"],
    "ALP":          ["ALP", "alkaline phosphatase"],
    "GGT":          ["GGT", "gamma GT", "gamma-glutamyl transferase"],
    "CRP":          ["CRP", "C-reactive protein"],
    "WBC":          ["WBC", "white blood cell", "total leucocyte count", "TLC"],
    "RBC":          ["RBC", "red blood cell", "erythrocyte count"],
    "Platelet":     ["platelet", "platelet count", "PLT"],
}

# Build one combined alias regex with named groups indexed by canonical
# name, so a single scan of the text identifies every marker line.
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in _MARKER_ALIASES.items():
    for a in aliases:
        _ALIAS_TO_CANONICAL[a.lower()] = canonical


# ---------------------------------------------------------------------------
# Value + unit + range patterns
# ---------------------------------------------------------------------------
# Reused from the document classifier's lab-value pattern but split into
# value, unit, and range groups so we can capture them. Tolerates the
# common Nepali-lab formatting variations:
#   "TSH      8.4   mIU/L     0.4 - 4.0"
#   "TSH: 8.4 mIU/L (Normal: 0.4 - 4.0)"
#   "TSH      8.4    mIU/L      <4.5"
#   "Hb       11.8  g/dL       12.0-15.5"
_UNITS = (
    r"mg/dL|mg/dl|mmol/L|mmol/l|mIU/L|mIU/l|µIU/mL|uIU/mL|"
    r"g/dL|g/dl|U/L|u/l|IU/L|iu/l|%|ng/mL|ng/ml|ng/dL|ng/dl|"
    r"pg/mL|pg/ml|cells/[uµ]L|"
    r"million/[uµ]L|10\^?[0-9]+/[uµ]L|/cmm|fl|fL|pg|"
    # Electrolyte units used across Nepali and international labs.
    # "mEq/l", "mEq/L", "meq/l" are the primary sodium/potassium units.
    r"mEq/L|mEq/l|meq/l|meq/L|mEq|meq|"
    # eGFR is often reported with "mL/min/1.73m2" or "ml/min/1.73m²".
    r"mL/min/1\.73m2|mL/min/1\.73m²|ml/min/1\.73m2|ml/min/1\.73m²|"
    r"mL/min|ml/min"
)

# Range can be "0.4 - 4.0" / "0.4-4.0" / "<4.5" / ">40" / "40 to 150".
_RANGE_RE = re.compile(
    r"(?P<range>"
    r"(?:[<>]\s*\d+(?:\.\d+)?)"
    r"|(?:\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?)"
    r"|(?:\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?)"
    r")",
    re.IGNORECASE,
)


@dataclass
class LabMarker:
    name: str                             # canonical name e.g. "TSH"
    value: float                          # numeric value
    unit: str                             # e.g. "mIU/L"
    reference_range: Optional[str]        # raw range string from the report
    status: str                           # "low" | "normal" | "high" | "unknown"
    raw_match: str = ""                   # original line snippet for debugging
    aliases_used: str = ""                # alias actually matched in text


def _classify_status(value: float, range_str: Optional[str]) -> str:
    """Compare `value` against `range_str` literally; do not invent
    bounds. Returns 'unknown' if the range can't be parsed — the
    composer must say "no reference range on the report" rather than
    guess one."""
    if not range_str:
        return "unknown"
    s = range_str.strip()

    # "<4.5" or "< 4.5"
    m = re.fullmatch(r"<\s*(\d+(?:\.\d+)?)", s)
    if m:
        upper = float(m.group(1))
        return "high" if value > upper else "normal"
    # ">40"
    m = re.fullmatch(r">\s*(\d+(?:\.\d+)?)", s)
    if m:
        lower = float(m.group(1))
        return "low" if value < lower else "normal"
    # "0.4 - 4.0" / "0.4-4.0" / "40 to 150"
    m = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*(?:[-–]|to)\s*(\d+(?:\.\d+)?)",
        s,
        re.IGNORECASE,
    )
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if value < lo:
            return "low"
        if value > hi:
            return "high"
        return "normal"
    return "unknown"


# Per-line scan: walk text by lines, look for any alias near a value+unit.
_VALUE_UNIT_RE = re.compile(
    rf"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>{_UNITS})",
    re.IGNORECASE,
)


def extract_lab_markers(text: str) -> list[LabMarker]:
    """Parse a (lab report) text blob into a list of LabMarker objects.

    Strategy is line-by-line: for each non-blank line, find the first
    canonical alias and the first value+unit+range. If both are present
    and on the same line, emit a marker. This matches Nepali lab
    formatting where each marker is one line.

    Multiple aliases for the same canonical marker on different lines
    (e.g. "TSH" header followed by "Thyroid Stimulating Hormone" detail)
    are deduplicated by canonical name — first occurrence wins.

    The parser is deliberately conservative: it skips any line where it
    can't bind a value to a single canonical marker, rather than
    inferring. The cost of a missed marker (user sees fewer rows in the
    explainer) is much lower than the cost of misattributing a value to
    the wrong marker.
    """
    if not text:
        return []

    seen: dict[str, LabMarker] = {}
    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        lower = line_stripped.lower()

        matched_alias: Optional[str] = None
        matched_canonical: Optional[str] = None
        for alias_lower, canonical in _ALIAS_TO_CANONICAL.items():
            if re.search(rf"\b{re.escape(alias_lower)}\b", lower):
                matched_alias = alias_lower
                matched_canonical = canonical
                break
        if not matched_canonical or matched_canonical in seen:
            continue

        vu = _VALUE_UNIT_RE.search(line_stripped)
        if not vu:
            continue
        try:
            value = float(vu.group("value"))
        except ValueError:
            continue
        unit = vu.group("unit")

        # Look for a range AFTER the value+unit on the same line, so we
        # don't mistake a separate numeric on the line (e.g. patient ID,
        # date) for the reference range.
        range_str: Optional[str] = None
        tail = line_stripped[vu.end():]
        rm = _RANGE_RE.search(tail)
        if rm:
            range_str = rm.group("range")

        marker = LabMarker(
            name=matched_canonical,
            value=value,
            unit=unit,
            reference_range=range_str,
            status=_classify_status(value, range_str),
            raw_match=line_stripped[:200],
            aliases_used=matched_alias or "",
        )
        seen[matched_canonical] = marker

    return list(seen.values())


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

# Retrieved-row callable signature — lets RAG.py pass _retrieve_ranked
# in without us having to import it (which would be a circular import
# back to RAG.py at module load time).
RetrieveFn = Callable[[str], list[dict]]


_PER_MARKER_SYSTEM_PROMPT = """\
You are DocuMed AI's lab-results explainer for patients in Nepal. You are
NOT a doctor. Your job is to explain ONE lab marker at a time using ONLY
the supplied sources, in a way the user can take to their next clinical
visit.

ABSOLUTE RULES (violating ANY of these means the entire response is
discarded by a downstream safety filter):
1. Never give a diagnosis. Forbidden phrases include but are not
   limited to: "you have", "you are diabetic", "diagnosis is",
   "sounds like", "might be", "probably", "most likely", "it could
   be".
2. Never recommend a dose, drug, brand, or treatment. Forbidden
   phrases include "take 50 mg", "you should take", "start taking",
   "I recommend".
3. Never invent a reference range. If the report did not print one,
   say "your report did not include a reference range for this
   marker — your doctor can interpret it against their lab's range."
4. Never claim certainty about cause. Use "commonly associated with"
   or "can indicate", never "means" or "is caused by".

Output EXACTLY this Markdown structure for the single marker:

**{marker_name}: {value} {unit}** ({status_label})
- **What it measures:** <one sentence, sourced>
- **Your value vs. the report's range:** <compare to the reference
  range printed on THIS report, or say the report did not include
  one>
- **What this can indicate:** <2 short sentences, general
  population terms, NOT applied to the user — use phrases like
  "high values are commonly associated with…">
- **What to ask your doctor:** <1-2 concrete questions>

Keep the whole block under 110 words. No preamble, no closing
disclaimer (the system appends one). No Sources section (the system
appends sources separately).
"""


_STATUS_LABEL = {
    "low": "below the report's range",
    "high": "above the report's range",
    "normal": "within the report's range",
    "unknown": "no reference range on the report",
}


# Fixed escalation footer appended to every explainer response. Listed
# triggers are the ones a primary-care clinician would want a patient
# with abnormal markers to recognise — kept short, generic, and
# pattern-based (no diagnosis).
_ESCALATION_FOOTER = (
    "**Go to the nearest emergency department or call 102 right away "
    "if you develop:** chest pain or pressure lasting more than 15 "
    "minutes, severe shortness of breath at rest, sudden weakness on "
    "one side or slurred speech, fainting or persistent confusion, "
    "uncontrolled bleeding, severe dehydration (no urine for 12+ "
    "hours, sunken eyes), or a high fever (≥39°C) with shivering or "
    "drowsiness.\n\n"
    "*This is general information from public health sources, not a "
    "diagnosis. Take this report to a qualified clinician for review.*"
)

# Conservative per-marker stub used when LLM generation fails or the
# marker isn't in our reference library. Keeps the rest of the table
# rendering rather than failing the whole response.
def _safe_marker_stub(marker: LabMarker, *, reason: str) -> str:
    label = _STATUS_LABEL.get(marker.status, marker.status)
    return (
        f"**{marker.name}: {marker.value} {marker.unit}** ({label})\n"
        f"- I couldn't safely interpret this marker right now ({reason}). "
        f"Please discuss this value with a qualified clinician.\n"
    )


def _format_sources_block(rows: list[dict], limit: int = 3) -> str:
    if not rows:
        return ""
    lines: list[str] = []
    seen: set[str] = set()
    for r in rows:
        url = r.get("doc_source_url") or ""
        title = r.get("doc_title") or r.get("doc_source") or "source"
        if url and url in seen:
            continue
        if url:
            seen.add(url)
            lines.append(f"- [{title}]({url})")
        else:
            lines.append(f"- {title}")
        if len(lines) >= limit:
            break
    if not lines:
        return ""
    return "**Sources:**\n" + "\n".join(lines)


def _retrieval_query_for(marker: LabMarker) -> str:
    """Compose a retrieval query for one marker. We want chunks
    explaining the marker generally — so we ask about its meaning, not
    about the patient's specific value. Including the status word
    ("high"/"low") biases retrieval toward chunks that discuss
    abnormalities, which is usually what the user needs."""
    if marker.status in ("low", "high"):
        return f"{marker.name} {marker.status} value meaning interpretation"
    return f"{marker.name} blood test what it measures"


def _compose_one_marker(
    marker: LabMarker,
    retrieved_rows: list[dict],
    *,
    groq_client: Any,
    groq_model: str,
    cohere_client: Any,
    cohere_model: str,
    max_tokens: int,
) -> tuple[str, list[dict]]:
    """Compose the per-marker explainer block. Returns (block_text, rows_used)."""
    if not retrieved_rows:
        return _safe_marker_stub(marker, reason="no source coverage"), []

    context_blocks = []
    for i, r in enumerate(retrieved_rows[:4], start=1):
        title = r.get("doc_title") or r.get("doc_source") or "source"
        heading = r.get("section_heading") or ""
        content = (r.get("content") or "").strip()
        context_blocks.append(f"[src:{i}] {title} — {heading}\n{content}")
    context_text = "\n\n".join(context_blocks)

    status_label = _STATUS_LABEL.get(marker.status, marker.status)
    range_line = (
        f"Reference range printed on the report: {marker.reference_range}"
        if marker.reference_range
        else "The report did NOT include a reference range for this marker."
    )
    user_prompt = (
        f"Sources:\n{context_text}\n\n"
        f"Marker: {marker.name}\n"
        f"Patient's value: {marker.value} {marker.unit}\n"
        f"{range_line}\n"
        f"Status vs. report range: {status_label}\n\n"
        "Produce the per-marker explainer block now in the EXACT structure "
        "specified by the system prompt."
    )

    raw: Optional[str] = None
    if groq_client is not None:
        try:
            resp = groq_client.chat.completions.create(
                model=groq_model,
                messages=[
                    {"role": "system", "content": _PER_MARKER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content
        except Exception as exc:
            print(f"[results] Groq compose failed for {marker.name}, falling back to Cohere: {exc}")

    if raw is None and cohere_client is not None:
        try:
            resp = cohere_client.chat(
                model=cohere_model,
                messages=[
                    {"role": "system", "content": _PER_MARKER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
            )
            raw = resp.message.content[0].text
        except Exception as exc:
            print(f"[results] Cohere compose also failed for {marker.name}: {exc}")

    if not raw or not raw.strip():
        return _safe_marker_stub(marker, reason="generation unavailable"), retrieved_rows

    # Per-marker refusal gate. If the LLM smuggled a diagnosis or dose
    # in, replace this single block with the safe stub — keep the
    # other markers rendering. Logging the matched phrases makes it
    # possible to tune _PER_MARKER_SYSTEM_PROMPT against real misses.
    matches = find_forbidden_phrases(raw)
    if matches:
        print(
            f"[results] refusal-filter matched {marker.name}: "
            f"{matches[:3]}"
        )
        return _safe_marker_stub(marker, reason="failed safety check"), retrieved_rows

    return raw.strip(), retrieved_rows


_NO_MARKERS_BLOCK = (
    "I couldn't pick out any lab markers I'm confident about from this "
    "report. The text may be image-only (a scanned PDF), or the layout "
    "is one I haven't seen before. You can:\n"
    "- Re-upload a text-based PDF if you have one, or\n"
    "- Ask me about a specific marker by name and I'll explain what it "
    "generally measures, or\n"
    "- Take the report directly to a clinician for review.\n"
)


def compose_explainer(
    text: str,
    *,
    retrieve_fn: RetrieveFn,
    groq_client: Any = None,
    groq_model: str = "",
    cohere_client: Any = None,
    cohere_model: str = "command-r-08-2024",
    max_tokens_per_marker: int = 220,
    max_markers: int = 8,
) -> dict:
    """Top-level Stage 4 entry point. Given the raw text of an uploaded
    lab report, return a dict ready to ship as an /upload response:

        {
          "answer":   <full markdown body>,
          "sources":  <flat list of (title, url) for the sources panel>,
          "markers":  <list of dicts for the frontend lab-table>,
          "stage":    "results",
        }

    The caller (RAG.py /upload handler) is responsible for persisting the
    parsed markers to user_reports / session_documents — the composer
    deals only with text-in / text-out so it stays unit-testable.
    """
    markers = extract_lab_markers(text)[:max_markers]
    if not markers:
        return {
            "answer": _NO_MARKERS_BLOCK + "\n\n" + _ESCALATION_FOOTER,
            "sources": [],
            "markers": [],
            "stage": "results",
        }

    blocks: list[str] = []
    all_source_rows: list[dict] = []
    seen_source_keys: set = set()

    for marker in markers:
        retrieval_query = _retrieval_query_for(marker)
        try:
            rows = retrieve_fn(retrieval_query)
        except Exception as exc:
            print(f"[results] retrieve_fn failed for {marker.name}: {exc}")
            rows = []
        block, rows_used = _compose_one_marker(
            marker,
            rows,
            groq_client=groq_client,
            groq_model=groq_model,
            cohere_client=cohere_client,
            cohere_model=cohere_model,
            max_tokens=max_tokens_per_marker,
        )
        blocks.append(block)

        # Collect deduplicated sources across markers.
        for r in rows_used[:3]:
            key = r.get("doc_source_url") or (r.get("doc_title"), r.get("doc_source"))
            if key in seen_source_keys:
                continue
            seen_source_keys.add(key)
            all_source_rows.append(r)

    body = "\n\n".join(blocks)
    sources_block = _format_sources_block(all_source_rows)

    parts: list[str] = [body]
    if sources_block:
        parts.append(sources_block)
    parts.append(_ESCALATION_FOOTER)
    full_answer = "\n\n".join(parts)

    # Final whole-response refusal gate. The per-marker gate already
    # filters individual blocks, but this catches anything that leaked
    # through (e.g. forbidden phrase split across two block edits).
    safe_answer, was_filtered = filter_response(full_answer)
    if was_filtered:
        print("[results] WHOLE-RESPONSE refusal filter triggered — discarding generation")
        # Even on whole-response refusal, surface the parsed markers
        # (just the table data) — they came from the user's own
        # uploaded report, not from generation, so they're safe to
        # show.
        safe_answer = (
            safe_answer
            + "\n\nWhat I can still tell you: I parsed the following "
              "values from your report. Please share them with a "
              "qualified clinician for interpretation.\n\n"
            + _markers_as_table(markers)
            + "\n\n"
            + _ESCALATION_FOOTER
        )

    return {
        "answer": safe_answer,
        "sources": [
            {
                "title": r.get("doc_title") or r.get("doc_source") or "source",
                "source_url": r.get("doc_source_url"),
            }
            for r in all_source_rows[:6]
        ],
        "markers": [
            {
                "name": m.name,
                "value": m.value,
                "unit": m.unit,
                "reference_range": m.reference_range,
                "status": m.status,
            }
            for m in markers
        ],
        "stage": "results",
    }


def _markers_as_table(markers: list[LabMarker]) -> str:
    """Plain Markdown table — used as a last-resort fallback when the
    composed prose is filtered out. Pure data from the user's own
    report, no LLM-generated text."""
    if not markers:
        return ""
    lines = [
        "| Marker | Value | Unit | Reference range | Status |",
        "|---|---|---|---|---|",
    ]
    for m in markers:
        lines.append(
            f"| {m.name} | {m.value} | {m.unit} | "
            f"{m.reference_range or '—'} | {m.status} |"
        )
    return "\n".join(lines)
