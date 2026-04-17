"""Stage 1 — structured intake (Week 7A).

Turn a vague complaint into a clinically-useful summary using history-taking
frameworks (SOCRATES / OPQRST / WHO IMAI / PHQ-2+GAD-2 prompts).

Flow per conversation:
    Turn N     : user describes complaint vaguely.
    Turn N+1   : system picks a template (keyword → LLM fallback) and asks
                 its 5 slot questions as ONE assistant message.
    Turn N+2   : user answers (ideally all 5 at once).
    Turn N+3   : system composes a structured bullet summary the user can
                 show a doctor, then hands off to Stage 2 (navigation).

Fails closed: template load failure raises at import time. LLM failures
fall through to a safe fallback message.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import yaml

_TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "intake_templates.yaml"

# Forbidden phrases from intake_templates.yaml global constraints.
# Any line containing one of these is dropped from the final summary.
_FORBIDDEN_PATTERNS = [
    r"\bsounds? like\b",
    r"\bmight be\b",
    r"\bprobably\b",
    r"\byou have\b",
    r"\byou are diagnosed with\b",
    r"\bit could be\b",
    r"\bmost likely\b",
    r"\bdiagnosis\s+is\b",
]
_FORBIDDEN_RE = re.compile("|".join(_FORBIDDEN_PATTERNS), re.IGNORECASE)

_GLOBAL_CONSTRAINTS = """\
Hard constraints (must follow all):
1. Never produce a differential diagnosis or guess the condition.
2. Never use these phrases: "sounds like", "might be", "probably",
   "you have", "you are diagnosed with", "it could be", "most likely".
3. Never name specific medications or doses. Never recommend specific
   tests by name as things the user "needs"; tests belong to Stage 2.
4. Frame output as a summary the user reads TO a doctor, not as your
   assessment of them.
5. Output only the bullet summary, no preamble, disclaimers, or sign-off.
"""


def _load_templates() -> list[dict]:
    with open(_TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)

    templates = data.get("templates")
    if not templates:
        raise RuntimeError(f"no templates found in {_TEMPLATES_PATH}")

    required = {"id", "label", "framework", "keywords", "slot_questions", "summary_prompt"}
    for t in templates:
        missing = required - set(t.keys())
        if missing:
            raise RuntimeError(f"intake template {t.get('id', '?')} missing keys: {missing}")
        if len(t["slot_questions"]) != 5:
            raise RuntimeError(
                f"intake template {t['id']} must have exactly 5 slot_questions, "
                f"got {len(t['slot_questions'])}"
            )

    ids = [t["id"] for t in templates]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"duplicate template ids in {_TEMPLATES_PATH}: {ids}")
    if "other" not in ids:
        raise RuntimeError("intake_templates.yaml must define an 'other' fallback template")

    return templates


TEMPLATES: list[dict] = _load_templates()
TEMPLATES_BY_ID: dict[str, dict] = {t["id"]: t for t in TEMPLATES}
TEMPLATE_IDS: list[str] = [t["id"] for t in TEMPLATES]

print(f"[intake] loaded {len(TEMPLATES)} templates: {TEMPLATE_IDS}")


def _keyword_match(question: str) -> Optional[dict]:
    q = question.lower()
    for t in TEMPLATES:
        for kw in t["keywords"]:
            if kw.lower() in q:
                return t
    return None


def _llm_classify(question: str, groq_client: Any, groq_model: str) -> str:
    """Ask the LLM to pick a bucket id. Returns 'other' on any error or
    unexpected output. Bounded output, max_tokens=10."""
    prompt = (
        "Classify the user's complaint into exactly one of these buckets. "
        "Respond with ONLY the bucket id, nothing else.\n\n"
        f"Buckets: {', '.join(TEMPLATE_IDS)}\n\n"
        f"Complaint: {question}\n\nBucket:"
    )
    try:
        resp = groq_client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        if answer in TEMPLATES_BY_ID:
            return answer
        print(f"[intake] LLM classifier returned unknown bucket {answer!r}, using 'other'")
    except Exception as exc:
        print(f"[intake] LLM classifier failed: {exc}")
    return "other"


def select_template(
    question: str,
    groq_client: Any = None,
    groq_model: str = "",
) -> dict:
    """Pick an intake template for a complaint.

    Order:
      1. Keyword match against the keyword list of each template.
      2. LLM fallback if groq_client is available and no keyword matched.
      3. 'other' as the final fallback.
    """
    match = _keyword_match(question)
    if match is not None:
        return match
    if groq_client is not None:
        bucket = _llm_classify(question, groq_client, groq_model)
        return TEMPLATES_BY_ID.get(bucket, TEMPLATES_BY_ID["other"])
    return TEMPLATES_BY_ID["other"]


def compose_questions(template: dict) -> str:
    """Render the 5 slot questions as a single assistant message."""
    lines = [
        "Thanks for sharing. A few questions so you can describe this clearly to a doctor:",
        "",
    ]
    for i, q in enumerate(template["slot_questions"], start=1):
        lines.append(f"{i}. {q}")
    lines.append("")
    lines.append(
        "*(I'm not diagnosing — just helping you build a clear picture for your doctor.)*"
    )
    return "\n".join(lines)


def _redact_forbidden(text: str) -> str:
    """Drop any line containing a forbidden phrase. Bullet summaries are
    line-oriented, so line-level filtering is both safer and preserves
    formatting better than sentence-splitting."""
    lines = text.split("\n")
    safe = [ln for ln in lines if not _FORBIDDEN_RE.search(ln)]
    return "\n".join(safe).strip()


_FALLBACK_SUMMARY = (
    "I wasn't able to compose a summary right now. Please describe your "
    "symptoms directly to your doctor, covering: when it started, where it "
    "is in your body, what makes it better or worse, and anything else "
    "you've noticed alongside it."
)

_UNSAFE_FALLBACK_SUMMARY = (
    "I wasn't able to compose a safe summary for this complaint. Please "
    "describe your symptoms directly to your doctor."
)


def compose_summary(
    template: dict,
    user_answers: str,
    *,
    groq_client: Any = None,
    groq_model: str = "",
    cohere_client: Any = None,
    cohere_model: str = "command-r-08-2024",
    max_tokens: int = 250,
) -> str:
    """Generate the structured summary. Groq primary, Cohere fallback.

    Applies forbidden-phrase redaction on output. If redaction empties the
    result, returns a safe fallback message rather than silently passing a
    diagnosis-like statement.
    """
    system_prompt = template["summary_prompt"].strip() + "\n\n" + _GLOBAL_CONSTRAINTS
    user_prompt = (
        f"User's answers to the intake questions:\n\n{user_answers.strip()}\n\n"
        "Compose the structured summary now. Output only the summary text."
    )

    raw: Optional[str] = None

    if groq_client is not None:
        try:
            resp = groq_client.chat.completions.create(
                model=groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content
        except Exception as exc:
            print(f"[intake] Groq summary failed, falling back to Cohere: {exc}")

    if raw is None and cohere_client is not None:
        try:
            resp = cohere_client.chat(
                model=cohere_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
            )
            raw = resp.message.content[0].text
        except Exception as exc:
            print(f"[intake] Cohere summary also failed: {exc}")

    if raw is None:
        return _FALLBACK_SUMMARY

    redacted = _redact_forbidden(raw)
    if not redacted:
        return _UNSAFE_FALLBACK_SUMMARY
    return redacted
