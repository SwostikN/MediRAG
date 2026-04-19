"""Static validator for must_refuse.jsonl.

Run in CI to catch schema drift and regex-syntax errors before the live
safety scorer would surface them at /query time.

Checks, per row:
  - required keys present: id, category, query, must_refuse,
    forbidden_patterns, expected_output_hints, notes
  - forbidden_patterns is a non-empty list of valid regex strings
  - id is unique across the file
  - category is non-empty

Exit code: 0 if clean, 1 if any violation.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

GOLD_PATH = Path(__file__).resolve().parent / "gold" / "must_refuse.jsonl"

REQUIRED = {
    "id",
    "category",
    "query",
    "must_refuse",
    "forbidden_patterns",
    "expected_output_hints",
    "notes",
}


def main() -> int:
    errors: list[str] = []
    seen_ids: set[str] = set()

    if not GOLD_PATH.exists():
        print(f"[error] missing gold file: {GOLD_PATH}", file=sys.stderr)
        return 1

    for line_no, raw in enumerate(GOLD_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON — {exc}")
            continue

        rid = row.get("id", f"?line{line_no}")
        for key in REQUIRED:
            if key not in row:
                errors.append(f"{rid}: missing required key {key!r}")

        if row.get("id") in seen_ids:
            errors.append(f"{rid}: duplicate id")
        seen_ids.add(row.get("id"))

        if not row.get("category"):
            errors.append(f"{rid}: empty category")

        patterns = row.get("forbidden_patterns")
        if not isinstance(patterns, list) or not patterns:
            errors.append(f"{rid}: forbidden_patterns must be a non-empty list")
        else:
            for pat in patterns:
                if not isinstance(pat, str):
                    errors.append(f"{rid}: forbidden_pattern not a string: {pat!r}")
                    continue
                try:
                    re.compile(pat)
                except re.error as exc:
                    errors.append(f"{rid}: invalid regex {pat!r} — {exc}")

    print(f"Checked {len(seen_ids)} rows from {GOLD_PATH.name}")
    if errors:
        print(f"{len(errors)} validation error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
