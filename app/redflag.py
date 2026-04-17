"""Deterministic red-flag engine (docs/IMPROVEMENTS.md §4.2, Week 6 Stage 0).

Hand-authored YAML rules run before retrieval. If any rule matches the
user message, we return a Nepal-contextual emergency template and the LLM
is never invoked. The engine is intentionally biased toward false
positives — missing an emergency is unacceptable, over-firing on a
non-emergency is recoverable.

Matching model (simple on purpose):
  - Case-insensitive substring match on the user message.
  - Trigger DSL: 'all_of' / 'any_of'. 'all_of' is the usual top-level
    shape (body part + symptom + modifier). 'any_of' is used when any
    single signal is enough (stroke FAST signs, active seizure).
  - Conditions nest. A sub-condition can be a string or another dict.

Fail-closed: if a rule or template can't be loaded at startup, we raise.
A silently broken red-flag screen is worse than a crash.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


RULES_PATH = Path(__file__).parent / "redflag_rules.yaml"
TEMPLATES_PATH = Path(__file__).parent / "response_templates.yaml"


@dataclass(frozen=True)
class RedFlagHit:
    rule_id: str
    category: str
    urgency: str
    message: str


_rules: Optional[List[Dict[str, Any]]] = None
_templates: Optional[Dict[str, Dict[str, Any]]] = None


def _load() -> None:
    global _rules, _templates
    with RULES_PATH.open() as fh:
        rules_doc = yaml.safe_load(fh) or {}
    with TEMPLATES_PATH.open() as fh:
        templates_doc = yaml.safe_load(fh) or {}
    rules = rules_doc.get("rules") or []
    templates = templates_doc.get("templates") or {}
    for rule in rules:
        tpl = rule.get("response_template")
        if tpl not in templates:
            raise RuntimeError(
                f"redflag: rule {rule.get('id')!r} references unknown template {tpl!r}"
            )
    _rules = rules
    _templates = templates


def _match(cond: Any, text: str) -> bool:
    if isinstance(cond, str):
        return cond.lower() in text
    if isinstance(cond, dict):
        if "all_of" in cond:
            return all(_match(c, text) for c in cond["all_of"])
        if "any_of" in cond:
            return any(_match(c, text) for c in cond["any_of"])
    return False


def check(message: str) -> Optional[RedFlagHit]:
    """Return the first matching RedFlagHit, or None.

    First-match wins. Rules are evaluated in file order, so the YAML
    authors more-specific rules (pregnancy_seizure) before more-general
    ones (seizure_active).
    """
    if _rules is None or _templates is None:
        _load()
    if not message or not message.strip():
        return None
    text = message.lower()
    for rule in _rules:
        triggers = rule.get("triggers") or {}
        if _match(triggers, text):
            template = _templates[rule["response_template"]]
            return RedFlagHit(
                rule_id=rule["id"],
                category=rule.get("category", "unknown"),
                urgency=template.get("urgency", "emergency"),
                message=str(template["message"]).strip(),
            )
    return None


def all_rule_ids() -> List[str]:
    if _rules is None:
        _load()
    return [r["id"] for r in _rules]
