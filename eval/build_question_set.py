"""Build the frozen 200-question evaluation set for the paper (Phase 2.1).

WHY THIS EXISTS
---------------
The ablation in Phase 3 runs 7 arms over one question set. That set must be fixed,
stratified, reproducible, and mixed internal/external:

  * fixed        -- every arm sees the identical questions, or the comparison is noise
  * stratified   -- results must be reportable per question type, not just in aggregate
  * reproducible -- a seed and input checksums, so the set can be rebuilt exactly
  * external     -- reviewers discount a system evaluated only on gold its authors wrote

STRATA (200 total)

    consumer_symptom      40   coverage.jsonl (in-scope) + K-QA
    condition_education   40   condition.jsonl
    lab_explainer         25   results.jsonl
    nepal_navigation      30   navigation_stage2.jsonl
    adversarial           45   must_refuse.jsonl + coverage.jsonl (adversarial)
    emergency_redflag     20   redflag.jsonl (positives)

EXTERNAL ANCHOR
---------------
K-QA (Manes et al. 2024, MIT licence) -- real patient questions with clinician-written
`Must_have` and `Nice_to_have` statements, so it scores hallucination AND comprehensiveness
and feeds the safety-access frontier directly.

Correction to the plan: the plan cites "1,212 real patient questions". That is K-QA's total
question count; only **201** carry the annotated must-have/nice-to-have answers, and only
those are usable here. The external share of this set is sized accordingly.

TWO SHAPES OF ITEM
------------------
Most strata carry a `query` -- a raw user question. `navigation_stage2.jsonl` does not: it
carries an `intake_summary`, because Stage 2 routes an already-completed intake rather than
a free-text question. Items therefore declare `input_kind` ("query" or "intake_summary")
and the Phase 3 runner must branch on it. Collapsing the two would silently mis-run 30 of
the 200 items.

DETERMINISM
-----------
Fixed seed, inputs sorted by id before sampling, and SHA-256 of every source file recorded
in the manifest. Rebuilding on unchanged inputs reproduces the set exactly; if a source
file changes, the checksum in the manifest will not match and the set must be rebuilt and
re-frozen deliberately.

USAGE
    .venv\\Scripts\\python.exe eval/build_question_set.py
    .venv\\Scripts\\python.exe eval/build_question_set.py --check   # report, write nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "eval" / "gold"
KQA = ROOT / "eval" / "external" / "kqa" / "questions_w_answers.jsonl"
OUT = ROOT / "eval" / "question_set_v1.jsonl"
MANIFEST = ROOT / "eval" / "question_set_v1.manifest.json"

SEED = 20260917
TARGET = {
    "consumer_symptom": 40,
    "condition_education": 40,
    "lab_explainer": 25,
    "nepal_navigation": 30,
    "adversarial": 45,
    "emergency_redflag": 20,
}


def load(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"missing input: {path}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def take(rows: list[dict], n: int, rng: random.Random, key="id") -> list[dict]:
    """Deterministic sample: sort by id first so input order cannot affect the result."""
    pool = sorted(rows, key=lambda r: str(r.get(key, "")))
    if len(pool) <= n:
        return pool
    return rng.sample(pool, n)


def build() -> tuple[list[dict], dict]:
    rng = random.Random(SEED)
    items: list[dict] = []
    provenance: dict[str, dict] = {}

    def add(stratum, rows, *, input_kind="query", input_field="query", extra=None):
        for r in rows:
            item = {
                "id": f"qs-{len(items)+1:03d}",
                "stratum": stratum,
                "input_kind": input_kind,
                "input": r.get(input_field),
                "source_file": r["__src"],
                "source_id": r.get("id"),
            }
            for k in (extra or []):
                if k in r:
                    item[k] = r[k]
            items.append(item)

    def tag(rows, src):
        for r in rows:
            r["__src"] = src
        return rows

    # ── adversarial (45): all 40 must_refuse + 5 adversarial coverage rows ──
    mr = tag(load(GOLD / "must_refuse.jsonl"), "must_refuse.jsonl")
    cov = tag(load(GOLD / "coverage.jsonl"), "coverage.jsonl")
    cov_adv = [r for r in cov if str(r.get("category", "")).startswith("adversarial")]
    cov_in = [r for r in cov if not str(r.get("category", "")).startswith("adversarial")]
    adv = mr + take(cov_adv, TARGET["adversarial"] - len(mr), rng)
    add("adversarial", adv, extra=["category", "must_refuse", "forbidden_patterns",
                                   "expected_output_hints", "expected_refusal_bucket"])
    provenance["adversarial"] = {"must_refuse.jsonl": len(mr),
                                 "coverage.jsonl": len(adv) - len(mr)}

    # ── consumer_symptom (40): in-scope coverage rows, topped up from K-QA ──
    internal = take(cov_in, min(len(cov_in), TARGET["consumer_symptom"]), rng)
    add("consumer_symptom", internal, extra=["category", "should_answer"])
    need = TARGET["consumer_symptom"] - len(internal)
    kqa_used = 0
    if need > 0:
        if not KQA.exists():
            sys.exit(f"K-QA not found at {KQA}. External anchor is required.")
        kq = load(KQA)
        for i, r in enumerate(kq):
            r["id"] = f"kqa-{i+1:04d}"
            r["__src"] = "kqa/questions_w_answers.jsonl"
        picked = take(kq, need, rng)
        for r in picked:
            items.append({
                "id": f"qs-{len(items)+1:03d}",
                "stratum": "consumer_symptom",
                "input_kind": "query",
                "input": r.get("Question"),
                "source_file": r["__src"],
                "source_id": r["id"],
                "must_have": r.get("Must_have"),
                "nice_to_have": r.get("Nice_to_have"),
                "external": True,
            })
        kqa_used = len(picked)
    provenance["consumer_symptom"] = {"coverage.jsonl": len(internal),
                                      "K-QA (external)": kqa_used}

    # ── condition_education (40) ──
    cond = tag(load(GOLD / "condition.jsonl"), "condition.jsonl")
    sel = take(cond, TARGET["condition_education"], rng)
    add("condition_education", sel,
        extra=["expected_topics", "expected_sources", "must_refuse", "retrieval_scoring"])
    provenance["condition_education"] = {"condition.jsonl": len(sel)}

    # ── lab_explainer (25) ──
    res = tag(load(GOLD / "results.jsonl"), "results.jsonl")
    sel = take(res, TARGET["lab_explainer"], rng)
    add("lab_explainer", sel,
        extra=["expected_markers", "expected_output_hints", "expected_sources",
               "must_refuse", "retrieval_scoring"])
    provenance["lab_explainer"] = {"results.jsonl": len(sel)}

    # ── nepal_navigation (30) — intake_summary, NOT a raw query ──
    nav = tag(load(GOLD / "navigation_stage2.jsonl"), "navigation_stage2.jsonl")
    sel = take(nav, TARGET["nepal_navigation"], rng)
    add("nepal_navigation", sel, input_kind="intake_summary", input_field="intake_summary",
        extra=["intent_bucket", "expected_tier_id", "expected_urgency_band",
               "expected_escalation_triggers", "expected_sources", "must_refuse"])
    provenance["nepal_navigation"] = {"navigation_stage2.jsonl": len(sel)}

    # ── emergency_redflag (20) — positives only ──
    rf = tag(load(GOLD / "redflag.jsonl"), "redflag.jsonl")
    pos = [r for r in rf if r.get("must_escalate")]
    sel = take(pos, TARGET["emergency_redflag"], rng)
    add("emergency_redflag", sel,
        extra=["expected_rule_id", "expected_urgency", "expected_output_hints", "must_refuse"])
    provenance["emergency_redflag"] = {"redflag.jsonl (positives)": len(sel)}

    sources = ["must_refuse.jsonl", "coverage.jsonl", "condition.jsonl", "results.jsonl",
               "navigation_stage2.jsonl", "redflag.jsonl"]
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "total": len(items),
        "target": TARGET,
        "actual": dict(Counter(i["stratum"] for i in items)),
        "provenance": provenance,
        "external_items": sum(1 for i in items if i.get("external")),
        "input_kinds": dict(Counter(i["input_kind"] for i in items)),
        "source_checksums": {s: sha256(GOLD / s)[:16] for s in sources}
                            | ({"kqa/questions_w_answers.jsonl": sha256(KQA)[:16]}
                               if KQA.exists() else {}),
        "notes": [
            "nepal_navigation items carry input_kind='intake_summary', not a raw query. "
            "The Phase 3 runner MUST branch on input_kind.",
            "K-QA supplies the external anchor. Its annotated subset is 201 rows, not the "
            "1,212 figure quoted in the plan (that is K-QA's total question count).",
            "Rebuilding with unchanged inputs reproduces this set exactly. A checksum "
            "mismatch means a source changed and the set must be re-frozen deliberately.",
        ],
    }
    return items, manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    items, manifest = build()

    print(f"{'stratum':<22}{'built':>7}{'target':>8}   sources")
    ok = True
    for stratum, target in TARGET.items():
        got = manifest["actual"].get(stratum, 0)
        ok &= got == target
        src = ", ".join(f"{k}={v}" for k, v in manifest["provenance"][stratum].items())
        flag = "" if got == target else "  <-- SHORT"
        print(f"{stratum:<22}{got:>7}{target:>8}   {src}{flag}")
    print(f"{'TOTAL':<22}{len(items):>7}{sum(TARGET.values()):>8}")
    print(f"\nexternal (K-QA) items : {manifest['external_items']}")
    print(f"input kinds           : {manifest['input_kinds']}")

    missing = [i["id"] for i in items if not i.get("input")]
    if missing:
        print(f"\nITEMS WITH NO INPUT TEXT: {missing}")
        ok = False

    if not args.check:
        with OUT.open("w", encoding="utf-8", newline="\n") as fh:
            for it in items:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"\nwrote {OUT.relative_to(ROOT)} and {MANIFEST.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
