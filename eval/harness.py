"""MediRAG eval harness (Week 2 skeleton per IMPROVEMENTS.md §8, §7.2).

Loads the hand-written gold QA set, validates schema, optionally runs each
query against the current pipeline, and writes a baseline JSON snapshot.
Recall@5 and the faithfulness proxy are computed where possible; for Week 2
the corpus has not yet been ingested (Week 3), so retrieval metrics are
expected to be null. The harness captures that fact honestly — baseline
numbers for retrieval / faithfulness only become non-trivial after Week 3.

Usage:
    python eval/harness.py
    python eval/harness.py --server-url http://127.0.0.1:8000
    python eval/harness.py --out eval/baselines/baseline_v1.json
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

EVAL_DIR = Path(__file__).resolve().parent
GOLD_DIR = EVAL_DIR / "gold"
BASELINES_DIR = EVAL_DIR / "baselines"

STAGES = ["redflag", "intake", "navigation", "visit_prep", "results", "condition"]

COMMON_REQUIRED = {"id", "stage", "query", "must_refuse"}
STAGE_REQUIRED: dict[str, set[str]] = {
    "redflag": {"subtype", "must_escalate", "expected_urgency", "expected_rule_id", "expected_output_hints"},
    "intake": {"expected_template", "expected_sources", "expected_output_hints"},
    "navigation": {"expected_care_tier", "expected_urgency", "expected_nepal_level", "expected_sources", "expected_output_hints"},
    "visit_prep": {"expected_sources", "expected_output_hints"},
    "results": {"expected_markers", "expected_sources", "expected_output_hints"},
    "condition": {"expected_topics", "expected_sources"},
}


@dataclass
class GoldItem:
    raw: dict[str, Any]

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def stage(self) -> str:
        return self.raw["stage"]

    @property
    def query(self) -> str:
        return self.raw["query"]

    @property
    def hints(self) -> list[str]:
        return list(
            self.raw.get("expected_output_hints")
            or self.raw.get("expected_topics")
            or []
        )

    @property
    def expected_sources(self) -> list[str]:
        return list(self.raw.get("expected_sources") or [])


@dataclass
class StageResult:
    n: int = 0
    retrieval_attempted: int = 0
    generation_attempted: int = 0
    recall_at_5_sum: float = 0.0
    faithfulness_sum: float = 0.0
    pipeline_errors: list[str] = field(default_factory=list)

    def recall_at_5(self) -> Optional[float]:
        if self.retrieval_attempted == 0:
            return None
        return self.recall_at_5_sum / self.retrieval_attempted

    def faithfulness(self) -> Optional[float]:
        if self.generation_attempted == 0:
            return None
        return self.faithfulness_sum / self.generation_attempted


def load_gold() -> list[GoldItem]:
    items: list[GoldItem] = []
    for stage in STAGES:
        path = GOLD_DIR / f"{stage}.jsonl"
        if not path.exists():
            print(f"[warn] missing gold file: {path}", file=sys.stderr)
            continue
        for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[error] {path}:{line_no} invalid JSON — {exc}", file=sys.stderr)
                continue
            items.append(GoldItem(raw=obj))
    return items


def validate(items: list[GoldItem]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for item in items:
        raw = item.raw
        for field_name in COMMON_REQUIRED:
            if field_name not in raw:
                errors.append(f"{raw.get('id', '?')}: missing required field '{field_name}'")
        stage = raw.get("stage")
        if stage not in STAGES:
            errors.append(f"{raw.get('id', '?')}: unknown stage '{stage}'")
            continue
        for field_name in STAGE_REQUIRED[stage]:
            if field_name not in raw:
                errors.append(f"{raw['id']}: stage={stage} missing '{field_name}'")
        item_id = raw.get("id")
        if item_id in seen_ids:
            errors.append(f"duplicate id: {item_id}")
        seen_ids.add(item_id)
    return errors


def faithfulness_proxy(answer: str, hints: list[str]) -> Optional[float]:
    if not hints:
        return None
    lower = (answer or "").lower()
    hits = sum(1 for h in hints if h.lower() in lower)
    return hits / len(hints)


_TOKEN_RE = __import__("re").compile(r"[a-z0-9]+")
_STOPWORDS = {"the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "info"}


def _tokenize(s: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(s.lower()) if t not in _STOPWORDS}


def recall_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int = 5) -> Optional[float]:
    """Recall@k via token-set overlap.

    A gold expected_source counts as retrieved if any of the top-k retrieved
    sources share >=60% of its non-stopword tokens. Substring match was too
    strict — it missed e.g. gold "NHS type 2 diabetes" against retrieved
    "Type 2 diabetes NHS https://...".
    """
    if not expected_sources:
        return None
    retrieved_token_sets = [_tokenize(s) for s in retrieved_sources[:k]]
    hits = 0
    for exp in expected_sources:
        exp_tok = _tokenize(exp)
        if not exp_tok:
            continue
        for r_tok in retrieved_token_sets:
            if len(exp_tok & r_tok) / len(exp_tok) >= 0.6:
                hits += 1
                break
    return hits / len(expected_sources)


def run_pipeline(query: str, server_url: Optional[str]) -> Optional[dict[str, Any]]:
    """Try to run the query through the current pipeline.

    Today the /query endpoint requires a PDF to have been uploaded first and
    returns the final answer only — no retrieved-source list — so Recall@5
    cannot be computed yet. This function is structured so that when the
    pipeline is updated in Week 3+ to return retrieved chunk metadata, the
    harness can start filling real numbers without further changes.
    """
    if not server_url:
        return None
    try:
        import requests  # type: ignore
    except ImportError:
        return {"error": "requests not installed"}
    try:
        resp = requests.post(
            f"{server_url.rstrip('/')}/query",
            json={"question": query},
            timeout=180,
        )
    except Exception as exc:
        return {"error": f"request failed: {exc}"}
    if resp.status_code != 200:
        return {"error": f"http {resp.status_code}: {resp.text[:200]}"}
    try:
        body = resp.json()
    except Exception:
        return {"error": "non-json response"}
    raw_sources = body.get("sources") or []
    flat: list[str] = []
    for s in raw_sources:
        if isinstance(s, str):
            flat.append(s.lower())
        elif isinstance(s, dict):
            parts = [s.get("title") or "", s.get("source") or "", s.get("source_url") or ""]
            flat.append(" ".join(p for p in parts if p).lower())
    return {
        "answer": body.get("answer"),
        "retrieved_sources": flat,
    }


def run_eval(items: list[GoldItem], server_url: Optional[str]) -> dict[str, StageResult]:
    per_stage: dict[str, StageResult] = {s: StageResult() for s in STAGES}
    runnable = [it for it in items if server_url and it.stage != "redflag"]
    progress_total = len(runnable)
    progress_i = 0
    t0 = time.time()
    for item in items:
        result = per_stage[item.stage]
        result.n += 1
        if server_url is None:
            continue
        if item.stage == "redflag":
            continue
        progress_i += 1
        elapsed = time.time() - t0
        print(
            f"[{progress_i}/{progress_total}] {item.stage}::{item.id} "
            f"({elapsed:.1f}s elapsed) — {item.query[:60]}",
            flush=True,
        )
        pipeline = run_pipeline(item.query, server_url)
        if pipeline is None:
            continue
        if "error" in pipeline:
            print(f"  error: {pipeline['error']}", flush=True)
            result.pipeline_errors.append(f"{item.id}: {pipeline['error']}")
            continue
        retrieved = pipeline.get("retrieved_sources") or []
        if retrieved:
            r = recall_at_k(retrieved, item.expected_sources, k=5)
            if r is not None:
                result.retrieval_attempted += 1
                result.recall_at_5_sum += r
        answer = pipeline.get("answer") or ""
        f = faithfulness_proxy(answer, item.hints)
        if f is not None:
            result.generation_attempted += 1
            result.faithfulness_sum += f
    return per_stage


def print_summary(items: list[GoldItem], errors: list[str], per_stage: dict[str, StageResult], server_url: Optional[str]) -> None:
    print("=" * 72)
    print("MediRAG eval — Week 2 baseline")
    print("=" * 72)
    print(f"Total gold pairs: {len(items)}")
    print(f"Validation errors: {len(errors)}")
    for err in errors:
        print(f"  - {err}")
    print(f"Pipeline: {server_url or 'disabled (no --server-url given)'}")
    print()
    header = f"{'stage':<14} {'n':>3}  {'recall@5':>10}  {'faithfulness':>13}  {'pipeline-errs':>14}"
    print(header)
    print("-" * len(header))
    for stage in STAGES:
        r = per_stage[stage]
        recall = r.recall_at_5()
        faith = r.faithfulness()
        recall_s = "—" if recall is None else f"{recall:.3f}"
        faith_s = "—" if faith is None else f"{faith:.3f}"
        print(f"{stage:<14} {r.n:>3}  {recall_s:>10}  {faith_s:>13}  {len(r.pipeline_errors):>14}")
    print()
    print("Notes:")
    print("  - Corpus not yet ingested (Week 3 task) → Recall@5 expected to be '—'.")
    print("  - Red-flag engine is deterministic YAML (Week 6); not evaluated here yet.")
    print("  - Faithfulness proxy here is hint-keyword-overlap, not RAGAS entailment.")


def write_baseline(items: list[GoldItem], errors: list[str], per_stage: dict[str, StageResult], server_url: Optional[str], out_path: Path) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_pairs": len(items),
        "gold_file_counts": {s: sum(1 for it in items if it.stage == s) for s in STAGES},
        "validation_errors": errors,
        "pipeline": {"server_url": server_url, "available": server_url is not None},
        "per_stage": {
            s: {
                "n": per_stage[s].n,
                "recall_at_5": per_stage[s].recall_at_5(),
                "faithfulness_proxy": per_stage[s].faithfulness(),
                "retrieval_attempted": per_stage[s].retrieval_attempted,
                "generation_attempted": per_stage[s].generation_attempted,
                "pipeline_errors": per_stage[s].pipeline_errors,
            }
            for s in STAGES
        },
        "notes": (
            "Week 2 baseline. Corpus not yet ingested (Week 3 task); retrieval "
            "metrics are '—' until then. Red-flag engine is deterministic YAML "
            "and lands in Week 6 — its gold items are validated but not run. "
            "Faithfulness proxy is hint-keyword-overlap; RAGAS entailment is "
            "Week 10 work."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Baseline written to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="MediRAG eval harness")
    parser.add_argument("--server-url", default=None, help="FastAPI base URL; if omitted, pipeline is not run")
    parser.add_argument("--out", default=str(BASELINES_DIR / "baseline_v1.json"), help="Path to write the baseline JSON")
    args = parser.parse_args()

    items = load_gold()
    errors = validate(items)
    per_stage = run_eval(items, args.server_url)
    print_summary(items, errors, per_stage, args.server_url)
    write_baseline(items, errors, per_stage, args.server_url, Path(args.out))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
