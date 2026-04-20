"""One-shot gold rewrite, per docs/GOLD_REWRITE_SPEC.md (drafted 2026-04-20).

Applies three transforms to expected_sources across 5 gold files:

    K  keep entry unchanged
    D  drop entry (Western commercial / clinician-only / not a real source)
    S  substitute entry with a specific corpus-aligned title
    I  ingest-candidate — queue for Phase 2, leave string unchanged

Rows where every entry is D become ALL-DROP rows. We tag those with
    "retrieval_scoring": "disabled"
    "retrieval_scoring_reason": "all expected_sources dropped — see GOLD_REWRITE_SPEC.md"
so the scorer can exclude them from the denominator. Their refusal /
red-flag / template axes remain intact and scoreable on the other harness.

Reviewer-checklist policy applied:
  - "weak"/"very weak" substitute candidates → DROP instead of SUBSTITUTE
  - MoHP substitutes that collapse distinct SOPs into "Nepal Health
    Factsheet 2025 (MoHP)" → defer (drop); the factsheet is too thin

Run once, idempotent on clean gold:
    python eval/scripts/apply_gold_rewrite_2026_04_21.py           # apply
    python eval/scripts/apply_gold_rewrite_2026_04_21.py --check   # dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GOLD_DIR = Path(__file__).resolve().parent.parent / "gold"

K, D, I = "K", "D", "I"
def S(new_title: str) -> tuple[str, str]:
    return ("S", new_title)


# --------------------------------------------------------------------------
# Decision table — one entry per gold row. Order matches expected_sources.
# Cross-reference: docs/GOLD_REWRITE_SPEC.md
# --------------------------------------------------------------------------
DECISIONS: dict[str, list[Any]] = {
    # ===== intake.jsonl =====
    "in-001": [I, D],  # SOCRATES is a template tag, not a retrieval source
    "in-002": [I, D],
    "in-003": [I],
    "in-004": [I, K],
    "in-005": [I, D],
    "in-006": [D, S("Chest pain — when to call 999 vs see GP"), D],
    "in-007": [D, D, D],
    "in-008": [D, D],
    "in-009": [D, D, D],
    "in-010": [D, I, D],
    "in-011": [I, D, K],
    "in-012": [I, I, D],
    "in-013": [I, D, D],
    "in-014": [D, K, I],
    "in-015": [D, K, D],
    "in-016": [I, S("Dengue Control Program (Nepal)"), K],
    "in-017": [I, S("Neglected Tropical Diseases (NTD) Fact Sheet 2026 (Nepal)"), D],
    "in-018": [I, D, I],
    "in-019": [I, K, D],
    "in-020": [D, D, D],
    "in-021": [D, D, D],
    "in-022": [D, D, D],
    "in-023": [D, D],
    "in-024": [D, S("Urinary tract infections (UTIs)"), I],
    "in-025": [D, D, D],
    "in-026": [I, D, K],
    "in-027": [K, D, D],  # MoHP factsheet sub deferred
    "in-028": [D, D, D],
    "in-029": [I, D, S("Non-Communicable Disease and Mental Health Section (Nepal)")],
    "in-030": [K, D, D],  # MoHP factsheet sub deferred

    # ===== visit_prep.jsonl =====
    "vp-001": [K, I, D],  # MoHP factsheet sub deferred
    "vp-002": [D, D],
    "vp-003": [D, K],
    "vp-004": [S("Ear infections"), I],
    "vp-005": [K, I, S("Non-Communicable Disease and Mental Health Section (Nepal)")],
    "vp-006": [D, D, D],
    "vp-007": [K, D, S("Non-Communicable Disease and Mental Health Section (Nepal)")],
    "vp-008": [K, D, D],
    "vp-009": [D, D, D],
    "vp-010": [K, D, D],
    "vp-011": [K, D, D],
    "vp-012": [K, D, K],
    "vp-013": [I, D, D],
    "vp-014": [D, D, D],
    "vp-015": [D, D, D],
    "vp-016": [K, D, S("Non-Communicable Disease and Mental Health Section (Nepal)")],
    "vp-017": [K, D, D],
    "vp-018": [K, D, K],
    "vp-019": [D, D, D],
    "vp-020": [S("Type 2 diabetes"), K, S("Non-Communicable Disease and Mental Health Section (Nepal)")],
    "vp-021": [D, D, D],
    "vp-022": [D, D, D],
    "vp-023": [D, D],
    "vp-024": [K, I, D],  # MoHP factsheet sub deferred
    "vp-025": [K, D, I],
    "vp-026": [D, D, I],
    "vp-027": [D, I, D],
    "vp-028": [I, I, D],
    "vp-029": [D, D, D],
    "vp-030": [I, D, D],
    "vp-031": [D, D, D],
    "vp-032": [D, D, D],
    "vp-033": [D, K, D],
    "vp-034": [D, D, D],
    "vp-035": [D, D, D],
    "vp-036": [D, D, D],
    "vp-037": [D, D, D],
    "vp-038": [I, D, D],

    # ===== results.jsonl =====
    "rs-001": [K, D],
    "rs-002": [S("Iron deficiency anaemia"), I],
    "rs-003": [K, D, S("Non-Communicable Disease and Mental Health Section (Nepal)")],
    "rs-004": [K, D],
    "rs-005": [K, I, D],
    "rs-006": [K, D, D],
    "rs-007": [K, D],
    "rs-008": [K, I, D],
    "rs-009": [K, D],
    "rs-010": [K, D],
    "rs-011": [K, D, I],
    "rs-012": [K, D],
    "rs-013": [I, D],
    "rs-014": [D, D],
    "rs-015": [D, D],
    "rs-016": [K, D, D],
    "rs-017": [K, D],
    "rs-018": [K, D, D],  # MoHP factsheet sub deferred
    "rs-019": [K, D],
    "rs-020": [K, D],
    "rs-021": [K, D],
    "rs-022": [D, D],
    "rs-023": [D, S("Iron deficiency anaemia")],
    "rs-024": [D, D],
    "rs-025": [D, D],
    "rs-026": [K, D, D],
    "rs-027": [K, D],
    "rs-028": [K, D, K],
    "rs-029": [K, D],
    "rs-030": [I, K],

    # ===== condition.jsonl =====
    "cd-001": [D, D, D],
    "cd-002": [K, D, K, S("Non-Communicable Disease and Mental Health Section (Nepal)")],
    "cd-003": [K, I, S("Non-Communicable Disease and Mental Health Section (Nepal)")],
    "cd-004": [K, K, D],
    "cd-005": [K, D, K],
    "cd-006": [D, D, D, D],
    "cd-007": [D, D, D, D],
    "cd-008": [D, D, D, D],
    "cd-009": [D, D, D],
    "cd-010": [D, D, D],
    "cd-011": [I, S("Non-Communicable Disease and Mental Health Section (Nepal)"), I, D],
    "cd-012": [K, D, D],
    "cd-013": [K, D, D, K],
    "cd-014": [K, D, D, D],
    "cd-015": [D, D, D],
    "cd-016": [D, K, D],
    "cd-017": [D, D, D],
    "cd-018": [K, D, K, S("Non-Communicable Disease and Mental Health Section (Nepal)")],
    "cd-019": [K, D, D],
    "cd-020": [I, K, D, K],  # MoHP factsheet sub deferred
    "cd-021": [D, D, D, D],
    "cd-022": [D, D, D],
    "cd-023": [K, D, D],
    "cd-024": [D, D, D],
    "cd-025": [K, K, D, I],
    "cd-026": [D, D, K, D],
    "cd-027": [D, D, D, D],
    "cd-028": [K, D, D, D],
    "cd-029": [D, D, D, D],
    "cd-030": [K, D, D, D],
    "cd-031": [D, D, D, D],
    "cd-032": [D, D, K, D],
    "cd-033": [D, D, D],
    "cd-034": [K, D, D, D],
    "cd-035": [S("Rheumatoid arthritis"), D, D, D],
    "cd-036": [K, D, K, D],
    "cd-037": [K, D, D, I],
    "cd-038": [D, D, D, D, I],
    "cd-039": [D, D, D],
    "cd-040": [K, S("Dengue and severe dengue"), K],
    "cd-041": [K, I, I, S("Dengue Control Program (Nepal)"), K],
    "cd-042": [K, K, K, D, I],
    "cd-043": [K, S("Snakebite envenoming"), D, D],
    "cd-044": [K, D, D, K, D],
    "cd-045": [D, D, K, D, D],
    "cd-046": [D, D, I, I, D, D],
    "cd-047": [D, D, I, I, K, D],
    "cd-048": [D, D, D, D, D, D],
    "cd-049": [K, I, S("Non-Communicable Disease and Mental Health Section (Nepal)"), D],
    "cd-050": [D, D, K, K, D, I, S("Non-Communicable Disease and Mental Health Section (Nepal)")],

    # ===== navigation.jsonl =====
    "nv-001": [I, K],
    "nv-002": [I, K],
    "nv-003": [I, K],
    "nv-004": [K, I],
    "nv-005": [I, K],
    "nv-006": [D, K, D],
    "nv-007": [D, K, I],
    "nv-008": [D, I, D],
    "nv-009": [D, D, D],
    "nv-010": [D, D, S("Non-Communicable Disease and Mental Health Section (Nepal)")],
    "nv-011": [D, I, K],
    "nv-012": [D, D, D],
    "nv-013": [D, D, K],
    "nv-014": [D, D, D],
    "nv-015": [D, D, K],
    "nv-016": [D, D, I],
    "nv-017": [K, D, K],
    "nv-018": [D, I, K],
    "nv-019": [D, D, I],
    "nv-020": [D, D, D],
    "nv-021": [D, D, D],
    "nv-022": [D, D, D],
    "nv-023": [D, D, D],
    "nv-024": [D, D, I],
    "nv-025": [D, I, D],
    "nv-026": [K, S("Dengue Control Program (Nepal)"), K],
    "nv-027": [I, I, K],
    "nv-028": [I, D, I],
    "nv-029": [S("Snakebite envenoming"), I, D],  # NHTC Nepal: very-weak → drop
    "nv-030": [K, I, K],
    "nv-031": [K, I, I],
    "nv-032": [D, D, D],
    "nv-033": [D, S("Urinary tract infections (UTIs)"), I],
    "nv-034": [D, D, K],
    "nv-035": [I, D, I],
    "nv-036": [D, D, K],
    "nv-037": [D, D, K],
    "nv-038": [D, D, K],
    "nv-039": [D, D, I],
    "nv-040": [I, D, K],
    "nv-041": [D, I, I],
    "nv-042": [D, D, I],
    "nv-043": [D, D, D],
    "nv-044": [D, K, I],
    "nv-045": [D, S("Non-Communicable Disease and Mental Health Section (Nepal)"), I],
    "nv-046": [I, K, I, I],
    "nv-047": [K, D, I],
    "nv-048": [D, D, D],
    "nv-049": [K, S("Dengue Control Program (Nepal)"), D],
    "nv-050": [D, K, D],
}


DISABLED_REASON = "all expected_sources dropped — see GOLD_REWRITE_SPEC.md"


def apply_decisions(row: dict[str, Any], decisions: list[Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """Return (new_row, counts). counts has keys k/d/s/i/total_before/total_after."""
    src = row.get("expected_sources") or []
    if len(src) != len(decisions):
        raise ValueError(
            f"row {row.get('id')}: expected_sources has {len(src)} entries "
            f"but decision table has {len(decisions)}"
        )
    counts = {"k": 0, "d": 0, "s": 0, "i": 0}
    out: list[str] = []
    for entry, decision in zip(src, decisions):
        if decision == K or decision == I:
            out.append(entry)
            counts["k" if decision == K else "i"] += 1
        elif decision == D:
            counts["d"] += 1
            continue
        elif isinstance(decision, tuple) and decision[0] == "S":
            out.append(decision[1])
            counts["s"] += 1
        else:
            raise ValueError(f"row {row.get('id')}: unknown decision {decision!r}")
    new = dict(row)
    new["expected_sources"] = out
    if not out:
        new["retrieval_scoring"] = "disabled"
        new["retrieval_scoring_reason"] = DISABLED_REASON
    counts["total_before"] = len(src)
    counts["total_after"] = len(out)
    return new, counts


def rewrite_file(path: Path, *, check: bool) -> dict[str, Any]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            rows.append(raw)
            continue
        rows.append(json.loads(line))

    totals = {"k": 0, "d": 0, "s": 0, "i": 0, "before": 0, "after": 0, "disabled": 0, "rows": 0, "missing": []}
    new_lines: list[str] = []
    for item in rows:
        if isinstance(item, str):
            new_lines.append(item)
            continue
        rid = item.get("id")
        totals["rows"] += 1
        decisions = DECISIONS.get(rid)
        if decisions is None:
            totals["missing"].append(rid)
            new_lines.append(json.dumps(item, ensure_ascii=False))
            continue
        new_row, c = apply_decisions(item, decisions)
        for k in ("k", "d", "s", "i"):
            totals[k] += c[k]
        totals["before"] += c["total_before"]
        totals["after"] += c["total_after"]
        if new_row.get("retrieval_scoring") == "disabled":
            totals["disabled"] += 1
        new_lines.append(json.dumps(new_row, ensure_ascii=False))

    if not check:
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return totals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Dry-run: report only, do not write.")
    args = ap.parse_args()

    files = ["intake.jsonl", "visit_prep.jsonl", "results.jsonl", "condition.jsonl", "navigation.jsonl"]
    grand = {"k": 0, "d": 0, "s": 0, "i": 0, "before": 0, "after": 0, "disabled": 0, "rows": 0}
    print(f"Mode: {'CHECK (dry-run)' if args.check else 'APPLY'}")
    print("-" * 72)
    for fn in files:
        path = GOLD_DIR / fn
        t = rewrite_file(path, check=args.check)
        for k in ("k", "d", "s", "i", "before", "after", "disabled", "rows"):
            grand[k] += t[k]
        print(
            f"{fn:<20} rows={t['rows']:<3} entries: before={t['before']:<3} after={t['after']:<3} "
            f"K={t['k']:<3} D={t['d']:<3} S={t['s']:<3} I={t['i']:<3} ALL-DROP={t['disabled']}"
        )
        if t["missing"]:
            print(f"  WARNING: rows with no decision: {t['missing']}")
    print("-" * 72)
    print(
        f"{'TOTAL':<20} rows={grand['rows']:<3} entries: before={grand['before']:<3} after={grand['after']:<3} "
        f"K={grand['k']:<3} D={grand['d']:<3} S={grand['s']:<3} I={grand['i']:<3} ALL-DROP={grand['disabled']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
