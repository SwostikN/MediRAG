"""Stage 2 — care-tier navigation (Week 7A MVP).

Runs right after Stage 1 intake. Given the bullet summary Stage 1 just
produced, plus the user's intent bucket, plus supporting chunks retrieved
from the existing corpus, recommend WHERE to go in Nepal's health system
and HOW urgently.

Why it chains directly after intake:
    Stage 1 on its own is useful but passive — we ask 5 questions, hand
    back a bullet summary, and leave the user to figure out what to do
    with it. For a health navigator that's a hole. Stage 2 closes it with
    a concrete next action in the SAME response.

Design constraints (MVP scope, Week 7A):
    1. Use the existing Week 5 corpus for sources. Week 7B adds
       care-pathway content (WHO IMAI, NHS "when to see a GP", MoHP STG).
    2. Tier reasoning is LLM-inferred against the static tier ladder in
       app/nepal_care_tiers.yaml. That ladder is reference data, not
       retrieved context.
    3. Never output "self-care, no doctor needed" as the PRIMARY tier
       when symptoms have persisted beyond ~48h — the intake summary
       already implies a multi-day complaint. Default to District
       Hospital OPD if uncertain.
    4. Always list concrete ED-escalation triggers under "Go to 102
       right away if". This is where the Stage 0 red-flag engine's
       coverage becomes user-visible.
    5. Fails closed: any LLM failure returns a conservative fallback
       block pointing at District Hospital + call 102 for red flags.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

_TIERS_PATH = Path(__file__).resolve().parent.parent / "nepal_care_tiers.yaml"


def _load_tiers() -> dict:
    with open(_TIERS_PATH) as f:
        data = yaml.safe_load(f)
    if not data.get("tiers"):
        raise RuntimeError(f"no tiers found in {_TIERS_PATH}")
    return data


_TIERS_DATA: dict = _load_tiers()
TIERS: list[dict] = _TIERS_DATA["tiers"]
TIER_IDS: list[str] = [t["id"] for t in TIERS]

print(f"[navigation] loaded {len(TIERS)} Nepal care tiers: {TIER_IDS}")


_SYSTEM_PROMPT = """\
You are DocuMed AI's care-navigation stage for patients in Nepal. Your job
is NOT to diagnose. Your job is to tell the user which tier of Nepal's
health system is appropriate for the complaint they just described, and
how urgently to go.

Strict rules:
1. Recommend exactly one primary tier from the provided Nepal care-tier
   ladder. Use the tier's 'label' verbatim.
2. Never recommend specific medications, doses, or treatments.
3. Never give a differential diagnosis. Do not use phrases like "sounds
   like", "might be", "probably", "you have", "most likely".
4. Tier-selection guidance — match the complaint to the LOWEST tier that
   safely handles it. Do not default everything to District Hospital.
   - Self-care at home: mild self-limiting symptoms under ~48h, no red
     flags, no high-risk factors (pregnancy, infant, elderly, chronic
     disease). Pair with a clear worsening-trigger list.
   - Health Post or Urban Health Centre: common acute illness needing
     basic examination but not imaging or lab — uncomplicated URI,
     mild diarrhoea, simple wound care, stable symptoms present a few
     days.
   - PHCC: primary-care consultation needing basic lab (CBC, urine,
     malaria RDT, glucose). Good fit for subacute but stable complaints.
   - District Hospital OPD: persistent complaints (weeks) needing lab
     + X-ray/ultrasound, or when a Health Post/PHCC cannot resolve.
   - Private hospital or specialist OPD: reasonable parallel to District
     when the user needs faster specialist access and can afford it.
   - Zonal/Central/Tertiary: only by referral, or when a clear
     sub-specialty need is present (cardiology, neurology, oncology).
   When genuinely uncertain between two adjacent tiers, pick the lower
   one and raise urgency. Never default upward to ED unless an emergency
   override below applies.
5. "When" must reflect actual clinical urgency — not a blanket
   "same-day". Use: "Routine, in the next 1–2 weeks" for stable chronic
   complaints; "Within the week" for persistent but non-worsening;
   "Within 24 hours" or "Same-day walk-in" for acute but non-red-flag;
   "Go now — do not wait" only for red flags.
6. The "Go to 102 right away if" section MUST list concrete symptom
   triggers specific to this complaint — never generic platitudes.
7. Keep the whole block under 150 words.

URGENT SYMPTOM OVERRIDES — these specific presentations ALWAYS route
to "District Hospital — general medicine OPD" with "Within 24 hours"
or "Same-day walk-in" urgency, NEVER routine 1–2 weeks. Match on the
pattern; the symptom itself drives urgency regardless of tier ladder
defaults:
- New yellow skin, yellow eyes (jaundice), or dark tea-coloured urine
  in any adult → District Hospital within 24 hours (needs urgent LFTs,
  viral-hepatitis / obstruction workup — delay risks acute liver
  failure or missed ascending cholangitis).
- New-onset unintended weight loss >5% in 4–6 weeks with any of:
  night sweats, persistent cough >2 weeks, or palpable lump →
  District Hospital within the week.
- Persistent vomiting preventing any oral intake for >24 hours in
  an adult → District Hospital same-day walk-in (dehydration risk;
  lower tiers cannot rehydrate IV).
- Any new focal neurological symptom that resolved spontaneously
  (transient weakness, speech change, vision loss) — TIA → District
  Hospital within 24 hours for stroke workup.

EMERGENCY OVERRIDES — these patterns ALWAYS route to "Nearest
Emergency Department — call 102 for ambulance", overriding rule 4.
Match on the pattern; you are not diagnosing, you are recognising a
known dangerous presentation:
- Postpartum within 6 weeks AND any fever (≥38°C) or foul-smelling
  vaginal discharge → ED (puerperal sepsis is a leading maternal
  mortality cause in Nepal).
- Known severe lung disease (COPD, asthma) AND inability to speak full
  sentences OR reduced response to the patient's usual reliever inhaler
  → ED.
- New weakness on one side, slurred speech, facial droop, or sudden
  confusion in any adult → ED (FAST stroke pattern; time-critical).
- Snake bite with swelling spreading above the bite site OR any
  neurological symptom (drooping eyelids, weakness, difficulty
  swallowing) → ED.
- Chest pain at rest in adult ≥40 yr AND any of: lasting >15 minutes,
  sweating with the pain, radiating to arm/jaw/back, history of
  hypertension/diabetes/smoking → ED.
- Severe headache + neck stiffness + fever (any age) → ED (meningitis
  pattern).
- Infant under 2 months with any fever, OR infant 2–12 months with
  fever + lethargy/poor feeding → ED (IMCI danger signs).

Output EXACTLY this markdown structure and nothing else:

**Where to go:** <tier label from the ladder>
**When:** <urgency — e.g. "Routine appointment in the next 1–2 weeks", "Same-day walk-in", "Within 24 hours">
**Why this tier, not others:** <2 short sentences explaining the choice>
**Go to 102 right away if:** <comma-separated list of concrete escalation triggers for THIS complaint>
"""


def _render_tier_ladder() -> str:
    """Render the tier ladder as a compact prompt block."""
    lines = ["Nepal care-tier ladder (pick one by label):"]
    for t in TIERS:
        lines.append(
            f"- {t['label']} (id: {t['id']}). Typical urgency: "
            f"{t['typical_urgency']}. Handles: {t['handles'].strip()}"
        )
    return "\n".join(lines)


_FALLBACK_BLOCK = (
    "**Where to go:** District Hospital — general medicine OPD\n"
    "**When:** Routine appointment in the next 1–2 weeks\n"
    "**Why this tier, not others:** A District Hospital has the lab, "
    "imaging, and medical-officer review needed for a persistent "
    "complaint. Lower tiers can refer you upward; higher tiers are for "
    "referrals or clear emergencies.\n"
    "**Go to 102 right away if:** severe breathing difficulty, chest "
    "pain, fainting, confusion, uncontrolled bleeding, or a rapidly "
    "worsening condition."
)


def _compose_sources_block(retrieval_rows: list[dict], limit: int = 3) -> str:
    """Format up to `limit` retrieved rows as a Sources footer. Returns
    empty string if there are no usable rows."""
    if not retrieval_rows:
        return ""
    lines: list[str] = []
    seen_urls: set[str] = set()
    for r in retrieval_rows:
        url = r.get("doc_source_url") or ""
        title = r.get("doc_title") or r.get("doc_source") or "source"
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
            lines.append(f"- [{title}]({url})")
        else:
            lines.append(f"- {title}")
        if len(lines) >= limit:
            break
    if not lines:
        return ""
    return "**Sources:**\n" + "\n".join(lines)


def compose_recommendation(
    intake_summary: str,
    intent_bucket: str,
    *,
    groq_client: Any = None,
    groq_model: str = "",
    cohere_client: Any = None,
    cohere_model: str = "command-r-08-2024",
    retrieval_rows: Optional[list[dict]] = None,
    max_tokens: int = 300,
) -> str:
    """Generate the care-tier recommendation block.

    Groq primary, Cohere fallback. On total LLM failure, returns the
    conservative _FALLBACK_BLOCK (District Hospital + red-flag triggers)
    so the user always gets SOME navigation, never silence.

    Sources block is appended separately from the LLM output — the LLM
    is not trusted to cite accurately from retrieval, so we format
    citations deterministically from `retrieval_rows`.
    """
    retrieval_rows = retrieval_rows or []

    user_prompt = (
        f"{_render_tier_ladder()}\n\n"
        f"Intent bucket (for context only): {intent_bucket}\n\n"
        f"Patient's Stage 1 intake summary:\n{intake_summary.strip()}\n\n"
        "Produce the recommendation block now. Output ONLY the four "
        "bolded fields in the exact structure specified — no preamble, "
        "no disclaimers, no Sources section (the system will append "
        "sources separately)."
    )

    raw: Optional[str] = None

    if groq_client is not None:
        try:
            resp = groq_client.chat.completions.create(
                model=groq_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content
        except Exception as exc:
            print(f"[navigation] Groq recommendation failed, falling back to Cohere: {exc}")

    if raw is None and cohere_client is not None:
        try:
            resp = cohere_client.chat(
                model=cohere_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
            )
            raw = resp.message.content[0].text
        except Exception as exc:
            print(f"[navigation] Cohere recommendation also failed: {exc}")

    block = (raw or "").strip() or _FALLBACK_BLOCK

    sources_block = _compose_sources_block(retrieval_rows)
    if sources_block:
        return f"{block}\n\n{sources_block}"
    return block
