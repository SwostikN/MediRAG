"""Pre-intake intent gate.

Decides whether a user's first turn in a new chat session should enter
the structured Stage-1 intake flow (5-slot history-taking questions)
or bypass it and route straight to retrieval / navigation / results /
condition education.

The existing app.intent.classify is a retrieval-filter classifier —
its vocabulary and prompt bias everything symptom-shaped toward
"intake", which is fine for filtering but wrong for gating. The gate
asks a different binary question: **is this a symptom report, or
something else?**

Design: four layers, highest-precedence match wins. Deliberately
asymmetric — layer 2 (symptom detector) is greedy on purpose because
intake is the safe failure mode (asking extra clarifying questions
wastes one turn; misrouting a real symptom to educational retrieval
can miss a red flag).

    Layer 1  hard signals (disjoint, zero-ambiguity)
             - RESULTS    lab keyword + value/"mean"
             - NAVIGATION explicit care-tier phrases
             - DIAGNOSED  explicit diagnosis marker

    Layer 2  symptom-report detector (1st-person + verb, temporal,
             bare symptom tokens). Matches →  intake.

    Layer 3  information-query detector (WH-fronting, definition
             verbs, feature-of-condition phrases, bare condition
             lexicon). Only consulted if layer 2 did NOT match.
             Matches →  condition (retrieval-based education).

    Layer 4  LLM tie-breaker. Fires only when layers 2 AND 3 both
             matched (genuine ambiguity like "what is this rash").
             One Groq call, constrained two-token output. On Groq
             failure or absence → default to intake (safe).

    Fallback  nothing matched → intake (safe default).

No side effects. No network except layer 4. Import-safe.
"""
from __future__ import annotations

import re
from typing import Any, Literal, Optional

Decision = Literal["intake", "condition", "navigation", "results"]


# ---------------------------------------------------------------------------
# Layer 1 — hard signals. Each is a list of regex strings compiled once.
# ---------------------------------------------------------------------------

_RESULTS_KEYWORDS = (
    r"\btsh\b", r"\bt3\b", r"\bt4\b", r"\bhba1c\b", r"\bldl\b", r"\bhdl\b",
    r"\bhemoglobin\b", r"\bhaemoglobin\b", r"\bcreatinine\b", r"\begfr\b",
    r"\balt\b", r"\bast\b", r"\bcbc\b", r"\bbilirubin\b", r"\bferritin\b",
    r"\bvitamin\s+d\b", r"\bvitamin\s+b12\b", r"\bb12\s+level\b",
    r"\bplatelet", r"\bwbc\b", r"\brbc\b", r"\bmcv\b",
    r"\bfasting\s+sugar\b", r"\brandom\s+sugar\b",
)
# Lab keyword within 20 chars of a numeric value OR a mean/indicate verb.
# "my hba1c is 7.2" → match. "what does hba1c measure" → no match (no
# number, no interpretation verb — the user wants the definition, not
# an interpretation of their own value).
_RESULTS_VALUE_PAT = re.compile(
    r"(" + "|".join(_RESULTS_KEYWORDS) + r").{0,20}?"
    r"(\d+(\.\d+)?|mean|means|indicate|abnormal|mg/?dl|mmol|ng/?ml|µg|mcg|"
    r"iu/?l|u/?l|g/?dl)",
    re.IGNORECASE,
)
# 1st-person lab framing: "my report", "my labs", or explicit
# interpretation question ("what does my X mean").
_RESULTS_PHRASE_PAT = re.compile(
    r"\b(my\s+(report|labs?|test\s+result|tsh|hba1c|ldl|hdl|creatinine|"
    r"egfr|hemoglobin|haemoglobin|vitamin\s+d|vitamin\s+b12|b12|platelet|"
    r"cholesterol|sugar)\b|what\s+does\s+my\s+[a-z0-9\s]{1,30}\s+(mean|"
    r"means|indicate|show))",
    re.IGNORECASE,
)

_NAVIGATION_PAT = re.compile(
    r"\b(should\s+i\s+go\s+to|which\s+hospital|which\s+(level|tier)\s+of|"
    r"emergency\s+room|urgent\s+care|a\s*&\s*e|"
    r"call\s+(an?\s+)?ambulance|is\s+this\s+an\s+emergency|can\s+i\s+wait|"
    r"where\s+(can|should|do)\s+i\s+(go|get))\b"
    r"|\b(call\s+)?102\b|\bcall\s+103\b",
    re.IGNORECASE,
)

_DIAGNOSED_PAT = re.compile(
    r"\b(i\s+was\s+(just\s+)?diagnosed|i\s+(have\s+been|was)\s+told\s+i\s+have|"
    r"doctor\s+said\s+i\s+have|my\s+doctor\s+said\s+i\s+have|"
    r"i'?ve\s+been\s+diagnosed|recently\s+diagnosed|"
    r"living\s+with\s+(type\s*[12]\s+)?(diabetes|hypertension|asthma|copd|"
    r"epilepsy|parkinson|ckd|pcos))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Layer 2 — symptom-report detector. Intentionally greedy.
# ---------------------------------------------------------------------------

_SYMPTOM_VERB_PAT = re.compile(
    r"\b(i\s+have|i'?ve\s+(been|got)|i\s+(am|'m)\s+(having|feeling|"
    r"experiencing|suffering)|i\s+feel|i\s+felt|i\s+am\s+feeling|"
    r"been\s+(having|feeling|experiencing)|"
    r"m\s+(having|feeling|experiencing|suffering)|"
    r"am\s+suffering|i\s+keep\s+getting|"
    # Subject-aux inversion in WH-fronted questions: "why am I having",
    # "how long have I been feeling" — 1st-person symptom report in
    # question form, still intake-territory when stacked with WH.
    r"(am|'m)\s+i\s+(having|feeling|experiencing)|"
    r"have\s+i\s+been\s+(having|feeling|experiencing))\b",
    re.IGNORECASE,
)

# 1st-person possessive + body part / symptom noun = symptom report.
# Deliberately narrow to body/organ terms so "my father has X" (3rd
# person by proxy) is caught by DIAGNOSED, not symptom.
_MY_BODY_PAT = re.compile(
    r"\bmy\s+(head|chest|stomach|back|neck|throat|ear|eye|eyes|nose|"
    r"tooth|teeth|leg|legs|arm|arms|hand|hands|foot|feet|knee|knees|"
    r"skin|bp|blood\s+pressure|sugar|period|periods)\b"
    r".*\b(hurt|hurts|pain|pains|aching|burning|swollen|itchy|bleeding|"
    r"is\s+high|is\s+low|feels?)\b",
    re.IGNORECASE,
)

# Temporal clause implying an ongoing symptom (combined with symptom
# noun below). "for 3 days", "since yesterday", "this morning" etc.
_TEMPORAL_PAT = re.compile(
    r"\b(since\s+(yesterday|last\s+night|last\s+week|\d+\s+days?\s+ago)|"
    r"for\s+(the\s+)?(past\s+)?\d+\s+(hours?|days?|weeks?|months?)|"
    r"this\s+(morning|afternoon|evening)|last\s+night|past\s+few\s+days)\b",
    re.IGNORECASE,
)

# Common symptom nouns. Used with temporal OR as standalone-token
# detection for bare-symptom phrases like "fever cough runny nose".
_SYMPTOM_NOUNS = {
    "fever", "cough", "pain", "ache", "headache", "nausea", "vomiting",
    "diarrhea", "diarrhoea", "dizzy", "dizziness", "fatigue", "weakness",
    "breathless", "breathlessness", "palpitation", "palpitations",
    "swelling", "rash", "itching", "bleeding", "bruising", "numbness",
    "tingling", "shortness", "weak", "tired", "sweating", "chills",
    "constipation", "insomnia", "anxious", "anxiety", "depressed",
    "depression", "cramp", "cramps", "bloating", "reflux", "heartburn",
    "wheeze", "wheezing", "sneezing", "runny", "sore", "burn", "burns",
    "stiff", "stiffness",
}
# Extra tokens that often appear as symptoms but could be condition
# names; kept separate so "hypertension" alone doesn't trigger symptom.
_SYMPTOM_NOUNS_STRICT = _SYMPTOM_NOUNS

_TEMPORAL_SYMPTOM_PAT = re.compile(
    r"(" + "|".join(_SYMPTOM_NOUNS) + r")", re.IGNORECASE,
)

# "this [symptom]" — demonstrative reference usually = 1st-person body.
_DEMONSTRATIVE_BODY_PAT = re.compile(
    r"\bthis\s+(rash|pain|ache|swelling|lump|bump|spot|bruise|cough|"
    r"fever|headache|dizziness|itch|burning|numbness|tingling)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Layer 3 — information-query detector.
# ---------------------------------------------------------------------------

_WH_FRONT_PAT = re.compile(
    r"^\s*(what\s+(is|are|does|do|causes?|happens?)|how\s+(does|do|to|is|are|"
    r"can)|why\s+(is|does|do|am|are)|when\s+(should|does|do|is)|"
    r"who\s+(gets|has)|which\s+(is|are)|where\s+(does|is|can))\b",
    re.IGNORECASE,
)

_DEFINITION_VERB_PAT = re.compile(
    r"\b(define|explain|describe|tell\s+me\s+about|give\s+me\s+info\s+on|"
    r"what\s+do\s+you\s+know\s+about|overview\s+of|meaning\s+of)\b",
    re.IGNORECASE,
)

_FEATURE_OF_PAT = re.compile(
    r"\b(symptoms?|causes?|risk\s+factors?|treatments?|treatment\s+for|"
    r"side\s+effects?|complications?|stages?|types?|prognosis|"
    r"prevention|diet\s+for|signs?\s+of)\s+(of\s+)?",
    re.IGNORECASE,
)

# Bare condition names — same lexicon used in app/intent.py keyword
# fallback, plus common Nepali clinical vocabulary.
_CONDITION_LEXICON = {
    "hypertension", "diabetes", "asthma", "copd", "tb", "tuberculosis",
    "dengue", "malaria", "typhoid", "covid", "covid-19", "hepatitis",
    "jaundice", "pneumonia", "bronchitis", "migraine", "epilepsy",
    "parkinson", "parkinsons", "eczema", "psoriasis", "gastritis",
    "ulcer", "ckd", "pcos", "hashimoto", "hashimotos", "hyperthyroidism",
    "hypothyroidism", "anemia", "anaemia", "leptospirosis", "cholera",
    "stroke", "mi", "heart\\s+attack", "arthritis", "osteoporosis",
}


# ---------------------------------------------------------------------------
# Scoring helpers.
# ---------------------------------------------------------------------------

def _normalize(q: str) -> str:
    return (q or "").strip()


def _layer1(q: str) -> Optional[Decision]:
    """Hard signals. None if no match."""
    if _RESULTS_VALUE_PAT.search(q) or _RESULTS_PHRASE_PAT.search(q):
        return "results"
    if _NAVIGATION_PAT.search(q):
        return "navigation"
    if _DIAGNOSED_PAT.search(q):
        return "condition"
    return None


def _layer2_is_symptom(q: str) -> bool:
    """Symptom-report detector. Greedy by design."""
    if _SYMPTOM_VERB_PAT.search(q):
        return True
    if _MY_BODY_PAT.search(q):
        return True
    if _DEMONSTRATIVE_BODY_PAT.search(q):
        return True
    # Temporal clause + any symptom noun nearby.
    if _TEMPORAL_PAT.search(q) and _TEMPORAL_SYMPTOM_PAT.search(q):
        return True
    # Bare symptom-token list: short query, no WH-word, no verb, ≥2
    # symptom nouns. Catches "fever cough runny nose" style inputs.
    words = [w for w in re.split(r"[^a-zA-Z]+", q.lower()) if w]
    if len(words) <= 8 and not _WH_FRONT_PAT.search(q):
        hits = sum(1 for w in words if w in _SYMPTOM_NOUNS_STRICT)
        if hits >= 2:
            return True
    return False


def _layer3_is_info(q: str) -> bool:
    """Information-query detector."""
    if _WH_FRONT_PAT.search(q):
        return True
    if _DEFINITION_VERB_PAT.search(q):
        return True
    if _FEATURE_OF_PAT.search(q):
        return True
    # Bare condition name(s) only, ≤3 tokens.
    words = [w for w in re.split(r"[^a-zA-Z0-9-]+", q.lower()) if w]
    if 1 <= len(words) <= 3:
        joined = " ".join(words)
        for cond in _CONDITION_LEXICON:
            if re.fullmatch(cond, joined) or re.fullmatch(cond, words[0]):
                return True
    return False


# ---------------------------------------------------------------------------
# Layer 4 — LLM tie-breaker. Optional; off by default.
# ---------------------------------------------------------------------------

_LLM_PROMPT = (
    "You classify a user message into one of two intents.\n"
    "A = the user is reporting their OWN current symptoms or bodily "
    "experience and wants advice about what to do.\n"
    "B = the user is asking for factual information, a definition, an "
    "explanation, or education about a topic — not reporting their own "
    "current symptoms.\n\n"
    "Output exactly one character: A or B. No other text.\n\n"
    "Message: {q}"
)


def _llm_tiebreak(q: str, groq_client: Any, groq_model: str) -> Decision:
    """One-token Groq call. On any failure → intake (safe default)."""
    try:
        resp = groq_client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": _LLM_PROMPT.format(q=q)}],
            max_tokens=2,
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip().upper()
        if text.startswith("B"):
            return "condition"
        return "intake"
    except Exception as exc:
        print(f"[intent_gate] LLM tiebreak failed: {exc}")
        return "intake"


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------

def classify_turn(
    question: str,
    *,
    groq_client: Any = None,
    groq_model: str = "",
) -> Decision:
    """Decide how to route the first user turn in a fresh session.

    Returns one of: "intake", "condition", "navigation", "results".

    Passing a groq_client enables layer-4 LLM tie-breaking for the
    genuinely ambiguous queries where layers 2 AND 3 both match.
    Without it, ambiguous queries fall through to intake (safe).
    """
    q = _normalize(question)
    if not q:
        return "intake"

    hard = _layer1(q)
    if hard is not None:
        return hard

    is_symptom = _layer2_is_symptom(q)
    is_info = _layer3_is_info(q)

    if is_symptom and not is_info:
        return "intake"
    if is_info and not is_symptom:
        return "condition"
    if is_symptom and is_info:
        # Genuine ambiguity: "what is this rash", "why am i having headaches".
        if groq_client is not None and groq_model:
            return _llm_tiebreak(q, groq_client, groq_model)
        return "intake"

    # Neither fired. Could be a one-word greeting, a typo, a non-medical
    # question, or something novel. Intake is the safe default — the
    # scope_guard and redflag layers run further down the pipe and will
    # catch non-medical / emergency cases.
    return "intake"
