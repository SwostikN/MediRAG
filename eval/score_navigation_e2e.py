"""End-to-end two-turn navigation scorer.

Why this exists: the harness-level eval/gold/navigation.jsonl uses /query,
but navigation is only composed as the *second* intake turn (after
intake_summary is persisted). A single /query on a fresh session therefore
never triggers compose_recommendation — it falls through to routine
retrieval and scores the wrong codepath.

This scorer drives the real two-turn flow:
    1. POST /session/start         → session_id (current_stage=intake)
    2. POST /query (turn 1)        → slot questions; ignore
    3. POST /query (turn 2)        → summary + '\n---\n' + nav_block

Red-flag cases terminate at turn 2 returning stage=redflag directly —
that's also a valid navigation outcome (ED escalation) and is scored
against the same hints.

For turn 2 we re-send the original query as the "slot answers" payload.
The intake summary composer is tolerant of unstructured input; terse
queries yield partial summaries, but the navigation block still renders
against whatever the summary captured — much closer to production than
the no-coverage fallback the single-turn probe produces today.

Usage:
    EVAL_USER_ID=... COHERE_DISABLED=1 \\
      python eval/score_navigation_e2e.py --server-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

EVAL_DIR = Path(__file__).resolve().parent
GOLD_PATH = EVAL_DIR / "gold" / "navigation.jsonl"
BASELINES_DIR = EVAL_DIR / "baselines"


def load_gold() -> list[dict]:
    items: list[dict] = []
    for n, line in enumerate(GOLD_PATH.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        try:
            items.append(json.loads(s))
        except json.JSONDecodeError as exc:
            print(f"[gold] line {n}: {exc}", file=sys.stderr)
    return items


def faithfulness_proxy(text: str, hints: list[str]) -> Optional[float]:
    if not hints:
        return None
    lower = (text or "").lower()
    hits = sum(1 for h in hints if h.lower() in lower)
    return hits / len(hints)


def extract_nav_block(answer: str) -> str:
    """Return the portion after the first '---' separator, or the whole
    answer if no separator (red-flag responses, routine fallback)."""
    if "\n---\n" in answer:
        return answer.split("\n---\n", 1)[1]
    return answer


@dataclass
class CaseResult:
    case_id: str
    query: str
    stage_turn1: Optional[str]
    stage_turn2: Optional[str]
    terminated_at_turn1: bool
    answer: str
    hint_recall: Optional[float]
    error: Optional[str] = None


@dataclass
class Aggregate:
    n: int = 0
    hint_recall_sum: float = 0.0
    hint_recall_count: int = 0
    errors: list[str] = field(default_factory=list)
    redflag_terminated: int = 0
    two_turn_completed: int = 0
    per_case: list[CaseResult] = field(default_factory=list)

    def add(self, r: CaseResult) -> None:
        self.n += 1
        if r.error:
            self.errors.append(f"{r.case_id}: {r.error}")
        if r.terminated_at_turn1 and r.stage_turn1 == "redflag":
            self.redflag_terminated += 1
        if not r.terminated_at_turn1 and r.stage_turn2 == "intake":
            self.two_turn_completed += 1
        if r.hint_recall is not None:
            self.hint_recall_sum += r.hint_recall
            self.hint_recall_count += 1
        self.per_case.append(r)

    def summary(self) -> dict[str, Any]:
        mean = (self.hint_recall_sum / self.hint_recall_count) if self.hint_recall_count else None
        return {
            "n": self.n,
            "errors": len(self.errors),
            "hint_recall_mean": mean,
            "hint_recall_count": self.hint_recall_count,
            "redflag_terminated": self.redflag_terminated,
            "two_turn_completed": self.two_turn_completed,
            "error_detail": self.errors,
        }


def score_case(case: dict, server_url: str, timeout: float) -> CaseResult:
    case_id = case["id"]
    query = case["query"]
    hints = case.get("expected_output_hints") or []
    base = server_url.rstrip("/")

    try:
        r = requests.post(f"{base}/session/start", json={"current_stage": "intake"}, timeout=timeout)
        r.raise_for_status()
        session_id = r.json()["session_id"]
    except Exception as exc:
        return CaseResult(case_id, query, None, None, False, "", None, error=f"session/start: {exc}")

    try:
        r1 = requests.post(
            f"{base}/query",
            json={"question": query, "session_id": session_id},
            timeout=timeout,
        )
        r1.raise_for_status()
        b1 = r1.json()
    except Exception as exc:
        return CaseResult(case_id, query, None, None, False, "", None, error=f"query turn1: {exc}")

    stage1 = b1.get("stage")
    if stage1 == "redflag":
        answer = b1.get("answer") or ""
        return CaseResult(
            case_id=case_id,
            query=query,
            stage_turn1=stage1,
            stage_turn2=None,
            terminated_at_turn1=True,
            answer=answer,
            hint_recall=faithfulness_proxy(answer, hints),
        )

    try:
        r2 = requests.post(
            f"{base}/query",
            json={"question": query, "session_id": session_id},
            timeout=timeout,
        )
        r2.raise_for_status()
        b2 = r2.json()
    except Exception as exc:
        return CaseResult(case_id, query, stage1, None, False, "", None, error=f"query turn2: {exc}")

    answer = b2.get("answer") or ""
    nav_block = extract_nav_block(answer)
    return CaseResult(
        case_id=case_id,
        query=query,
        stage_turn1=stage1,
        stage_turn2=b2.get("stage"),
        terminated_at_turn1=False,
        answer=answer,
        hint_recall=faithfulness_proxy(nav_block, hints),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server-url", default="http://127.0.0.1:8000")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--limit", type=int, default=0, help="limit to first N rows (0 = all)")
    p.add_argument("--out", default=str(BASELINES_DIR / "navigation_e2e.json"))
    p.add_argument("--label", default="baseline")
    p.add_argument("--per-case-delay", type=float, default=0.0)
    args = p.parse_args()

    items = load_gold()
    if args.limit > 0:
        items = items[: args.limit]
    print(f"[nav-e2e] loaded {len(items)} cases; server={args.server_url}")

    agg = Aggregate()
    t0 = time.time()
    for i, case in enumerate(items, 1):
        elapsed = time.time() - t0
        print(f"[{i}/{len(items)}] {case['id']} ({elapsed:.1f}s) — {case['query'][:60]}", flush=True)
        r = score_case(case, args.server_url, args.timeout)
        if r.error:
            print(f"   ERROR {r.error}")
        else:
            recall = "—" if r.hint_recall is None else f"{r.hint_recall:.2f}"
            path = "redflag" if r.terminated_at_turn1 else f"{r.stage_turn1}→{r.stage_turn2}"
            print(f"   path={path:<20} hint_recall={recall}")
        agg.add(r)
        if args.per_case_delay > 0 and i < len(items):
            time.sleep(args.per_case_delay)

    summary = agg.summary()
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()
    summary["label"] = args.label
    summary["server_url"] = args.server_url
    summary["per_case"] = [
        {
            "id": c.case_id,
            "query": c.query,
            "path": "redflag" if c.terminated_at_turn1 else f"{c.stage_turn1}->{c.stage_turn2}",
            "hint_recall": c.hint_recall,
            "error": c.error,
            "answer": c.answer or "",
        }
        for c in agg.per_case
    ]

    print()
    print("=" * 72)
    print(f"Navigation e2e — {args.label}")
    print("=" * 72)
    print(f"n={summary['n']}  errors={summary['errors']}")
    print(f"redflag_terminated={summary['redflag_terminated']}  two_turn_completed={summary['two_turn_completed']}")
    mean = summary["hint_recall_mean"]
    print(f"hint_recall_mean={'—' if mean is None else f'{mean:.3f}'} over {summary['hint_recall_count']} scored")
    for e in summary["error_detail"]:
        print(f"  - {e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSnapshot → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
