"""Adversarial safety scorer — blocking CI gate.

Runs every prompt in eval/gold/must_refuse.jsonl against the live /query
endpoint, then for each response checks that NONE of the row's
`forbidden_patterns` (regex) appear in the answer. Any match is a hard
FAIL and the script exits non-zero.

Unlike the faithfulness harness, this gate is categorical: a single
forbidden-pattern hit in a single response is enough to block a merge.
That's the point — these are the stances the product is contractually
forbidden from taking (drug doses, patient-specific diagnoses, prescription
writing, self-harm facilitation, etc.).

Usage:
    python eval/score_must_refuse.py --server-url http://127.0.0.1:8000
    python eval/score_must_refuse.py --server-url http://127.0.0.1:8000 --out eval/results/must_refuse_$(date +%s).json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
GOLD_PATH = EVAL_DIR / "gold" / "must_refuse.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"[error] {path}:{line_no} invalid JSON — {exc}", file=sys.stderr)
            sys.exit(2)
    return rows


def run_query(server_url: str, query: str, timeout: int = 180) -> dict[str, Any]:
    import requests  # type: ignore

    try:
        resp = requests.post(
            f"{server_url.rstrip('/')}/query",
            json={"question": query},
            timeout=timeout,
        )
    except Exception as exc:
        return {"error": f"request failed: {exc}"}
    if resp.status_code != 200:
        return {"error": f"http {resp.status_code}: {resp.text[:200]}"}
    try:
        return resp.json()
    except Exception:
        return {"error": "non-json response"}


def check_row(row: dict[str, Any], answer: str) -> list[str]:
    """Return list of forbidden patterns that MATCHED (empty list = pass)."""
    violations: list[str] = []
    for pat in row.get("forbidden_patterns", []):
        try:
            if re.search(pat, answer, flags=re.IGNORECASE):
                violations.append(pat)
        except re.error as exc:
            violations.append(f"[invalid regex: {pat!r} — {exc}]")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="MediRAG must-refuse safety gate")
    parser.add_argument("--server-url", required=True, help="FastAPI base URL (e.g. http://127.0.0.1:8000)")
    parser.add_argument("--out", default=None, help="Optional path to write JSON results")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--min-interval", type=float, default=7.0,
                        help="Min seconds between queries — Cohere Trial key caps at 10/min; 7s keeps us under")
    args = parser.parse_args()

    rows = load_rows(GOLD_PATH)
    print(f"Loaded {len(rows)} adversarial prompts from {GOLD_PATH.name}")
    print(f"Target server: {args.server_url}")
    print("-" * 72)

    per_row: list[dict[str, Any]] = []
    fails: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    t0 = time.time()

    last_t = 0.0
    for i, row in enumerate(rows, start=1):
        # pace to stay under Cohere 10/min trial cap
        gap = time.time() - last_t
        if gap < args.min_interval:
            time.sleep(args.min_interval - gap)
        last_t = time.time()
        elapsed = time.time() - t0
        print(f"[{i}/{len(rows)}] {row['id']} ({row['category']}) — {elapsed:.1f}s", flush=True)
        resp = run_query(args.server_url, row["query"], timeout=args.timeout)
        if "error" in resp:
            errors.append({"id": row["id"], "error": resp["error"]})
            per_row.append({"id": row["id"], "status": "error", "error": resp["error"]})
            print(f"    ERROR: {resp['error']}", flush=True)
            continue
        answer = resp.get("answer") or ""
        violations = check_row(row, answer)
        if violations:
            fails.append({
                "id": row["id"],
                "category": row["category"],
                "query": row["query"],
                "violations": violations,
                "answer_snippet": answer[:400],
            })
            per_row.append({"id": row["id"], "status": "FAIL", "violations": violations})
            print(f"    FAIL: {len(violations)} forbidden pattern(s) matched", flush=True)
            for v in violations:
                print(f"       - {v}", flush=True)
        else:
            per_row.append({"id": row["id"], "status": "pass"})

    print("-" * 72)
    print(f"Total: {len(rows)}  Pass: {len(rows) - len(fails) - len(errors)}  "
          f"FAIL: {len(fails)}  Error: {len(errors)}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server_url": args.server_url,
            "gold_file": str(GOLD_PATH),
            "n_total": len(rows),
            "n_pass": len(rows) - len(fails) - len(errors),
            "n_fail": len(fails),
            "n_error": len(errors),
            "fails": fails,
            "errors": errors,
            "per_row": per_row,
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote results to {out_path}")

    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
