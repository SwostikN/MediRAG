"""Phase 2c scorer — end-to-end measurement for the Failure-B fix
(meta-question clarification).

For each gold case in eval/gold/meta_question.jsonl we:
  1. Call app.meta_question.is_meta_question against the history +
     question.
  2. If expected_stage == "clarification": assert detector fires,
     then call compose_clarification against the PRIOR assistant text
     using the real Groq client (from env) — mirrors the production
     path in app/RAG.py exactly (same function, same model).
  3. If expected_stage == "routine": assert detector does NOT fire.
     No composer call is made — those cases verify we correctly fall
     through to fresh retrieval.
  4. Validate the emitted clarification text:
       - No regex in forbidden_new_claims may match.
       - For expected_behavior == "say_not_specified" the text must
         contain a "did not specify" or equivalent admission phrase.
       - For expected_behavior == "answer_from_prior" the text must
         NOT contain the "did not specify" admission (otherwise the
         composer is over-refusing a case the prior answer actually
         covered).

Outputs a per-case JSON report to
eval/baselines/meta_question_<YYYY-MM-DD>.json.

Usage
-----
    # Direct function-call mode (preferred — cheap, deterministic).
    # Requires GROQ_API_KEY in env (or a .env the app loads).
    python eval/score_meta_question.py

    # Dry-run: skip Groq, only check the detector + gold self-consistency.
    python eval/score_meta_question.py --dry-run

Exit codes
----------
  0  all cases passed all their assertions
  1  one or more cases failed an assertion (hallucination or detector miss)
  2  gold file unreadable / no GROQ client available in live mode
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env so GROQ_API_KEY / GROQ_MODEL are visible without exporting.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from app.meta_question import is_meta_question, last_assistant_turn  # noqa: E402
from app.stages.clarification import compose_clarification  # noqa: E402

GOLD_PATH = ROOT / "eval" / "gold" / "meta_question.jsonl"
BASELINES_DIR = ROOT / "eval" / "baselines"


# Phrases that count as an honest "I did not cover that" admission. The
# production prompt instructs the model to say "My previous answer did
# not specify that"; we accept close paraphrases because the model is
# not prompted verbatim.
_NOT_SPECIFIED_PATTERNS = [
    re.compile(r"did\s+not\s+specify", re.IGNORECASE),
    re.compile(r"didn['\u2019]?t\s+specify", re.IGNORECASE),
    re.compile(r"did\s+not\s+(cover|mention|address|say)", re.IGNORECASE),
    re.compile(r"didn['\u2019]?t\s+(cover|mention|address|say)", re.IGNORECASE),
    re.compile(r"not\s+specified\s+(in|above|earlier|in\s+my)", re.IGNORECASE),
    re.compile(r"my\s+previous\s+answer\s+did\s+not", re.IGNORECASE),
    re.compile(r"ask\s+it\s+as\s+a\s+new\s+question", re.IGNORECASE),
    re.compile(r"rephrase", re.IGNORECASE),
]


def _said_not_specified(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _NOT_SPECIFIED_PATTERNS)


def _forbidden_hits(text: str, patterns: list[str]) -> list[str]:
    hits: list[str] = []
    if not text:
        return hits
    for pat in patterns or []:
        try:
            if re.search(pat, text, re.IGNORECASE):
                hits.append(pat)
        except re.error:
            # Treat a malformed regex as a literal substring match so a
            # typo in gold does not silently swallow violations.
            if pat.lower() in text.lower():
                hits.append(pat)
    return hits


def _load_gold() -> list[dict]:
    if not GOLD_PATH.exists():
        print(f"[gold] missing {GOLD_PATH}", file=sys.stderr)
        sys.exit(2)
    items: list[dict] = []
    for n, line in enumerate(GOLD_PATH.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        try:
            items.append(json.loads(s))
        except json.JSONDecodeError as exc:
            print(f"[gold] line {n}: {exc}", file=sys.stderr)
            sys.exit(2)
    return items


def _get_groq_client():
    try:
        from groq import Groq  # type: ignore
    except Exception:
        return None, None
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, None
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    try:
        return Groq(api_key=api_key), model
    except Exception:
        return None, None


@dataclass
class CaseResult:
    case_id: str
    expected_stage: str
    expected_behavior: str
    detector_fired: bool
    detector_ok: bool
    clarification_text: Optional[str] = None
    forbidden_hits: list[str] = field(default_factory=list)
    said_not_specified: Optional[bool] = None
    behavior_ok: Optional[bool] = None
    passed: bool = False
    notes: str = ""


def score_case(case: dict, *, groq_client: Any, groq_model: Optional[str], dry_run: bool) -> CaseResult:
    case_id = case["case_id"]
    expected_stage = case["expected_stage"]
    expected_behavior = case["expected_behavior"]
    question = case["question"]
    history = case.get("history") or []
    forbidden = case.get("forbidden_new_claims") or []

    fired = is_meta_question(question, history)
    detector_ok = (
        fired if expected_stage == "clarification" else (not fired)
    )

    res = CaseResult(
        case_id=case_id,
        expected_stage=expected_stage,
        expected_behavior=expected_behavior,
        detector_fired=fired,
        detector_ok=detector_ok,
        notes=case.get("notes", ""),
    )

    # Routine cases only validate the detector.
    if expected_stage == "routine":
        res.behavior_ok = True  # N/A
        res.passed = detector_ok
        return res

    # Clarification cases: if detector missed, we already failed.
    if not fired:
        res.passed = False
        return res

    # Compose the clarification.
    prior = last_assistant_turn(history) or ""
    if dry_run or groq_client is None:
        res.clarification_text = None
        res.behavior_ok = None
        # In dry-run we can at least pass detector-only assertions.
        res.passed = detector_ok
        return res

    text = compose_clarification(
        prior_answer=prior,
        user_question=question,
        groq_client=groq_client,
        groq_model=groq_model,
        max_tokens=200,
    )
    res.clarification_text = text

    hits = _forbidden_hits(text, forbidden)
    res.forbidden_hits = hits

    not_spec = _said_not_specified(text)
    res.said_not_specified = not_spec

    if expected_behavior == "say_not_specified":
        res.behavior_ok = not_spec
    elif expected_behavior == "answer_from_prior":
        # Must answer, must not over-refuse by saying "did not specify"
        # when the prior genuinely did specify it.
        res.behavior_ok = (not not_spec) and bool(text.strip())
    else:
        # Defensive: unknown expected_behavior on a clarification case.
        res.behavior_ok = False

    res.passed = bool(detector_ok and res.behavior_ok and not hits)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Groq; validate only the detector + gold shape.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional explicit output path (otherwise baselines/meta_question_<date>.json).",
    )
    args = ap.parse_args()

    gold = _load_gold()
    if not gold:
        print("[gold] no cases loaded", file=sys.stderr)
        return 2

    groq_client, groq_model = (None, None)
    if not args.dry_run:
        groq_client, groq_model = _get_groq_client()
        if groq_client is None:
            print(
                "[groq] no client available — set GROQ_API_KEY or pass --dry-run",
                file=sys.stderr,
            )
            return 2

    results: list[CaseResult] = []
    for case in gold:
        try:
            r = score_case(
                case,
                groq_client=groq_client,
                groq_model=groq_model,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            r = CaseResult(
                case_id=case.get("case_id", "?"),
                expected_stage=case.get("expected_stage", "?"),
                expected_behavior=case.get("expected_behavior", "?"),
                detector_fired=False,
                detector_ok=False,
                passed=False,
                notes=f"exception: {exc}",
            )
        results.append(r)

        status = "PASS" if r.passed else "FAIL"
        print(
            f"[{status}] {r.case_id}  stage={r.expected_stage}  "
            f"behavior={r.expected_behavior}  detector_fired={r.detector_fired}  "
            f"hits={len(r.forbidden_hits)}  not_spec={r.said_not_specified}"
        )

    n = len(results)
    passed = sum(1 for r in results if r.passed)
    det_miss = sum(1 for r in results if not r.detector_ok)
    fb_viol = sum(1 for r in results if r.forbidden_hits)
    beh_miss = sum(
        1
        for r in results
        if r.expected_stage == "clarification"
        and r.behavior_ok is False
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold_path": str(GOLD_PATH),
        "dry_run": bool(args.dry_run),
        "groq_model": groq_model if not args.dry_run else None,
        "totals": {
            "n": n,
            "passed": passed,
            "failed": n - passed,
            "detector_mismatches": det_miss,
            "forbidden_claim_violations": fb_viol,
            "behavior_mismatches": beh_miss,
        },
        "cases": [asdict(r) for r in results],
    }

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_path = BASELINES_DIR / f"meta_question_{date_str}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[report] wrote {out_path}")
    print(
        f"[report] n={n} passed={passed} failed={n - passed} "
        f"detector_miss={det_miss} forbidden_hits={fb_viol} behavior_miss={beh_miss}"
    )

    return 0 if passed == n else 1


if __name__ == "__main__":
    sys.exit(main())
