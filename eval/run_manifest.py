"""Provenance block for every evaluation result file (paper phase, 0.5).

WHY THIS EXISTS
---------------
Three failures in this project's own history all trace to results that did not record the
conditions they were produced under:

  1. `llama-3.3-70b-versatile` was decommissioned in Sept 2026. Every result in the repo
     came from it, and nothing recorded that — so the numbers cannot be reproduced or even
     properly attributed.
  2. `eval/baselines/baseline_v4_filtered.json` and `baseline_v5_full.json` both contain
     "Week 2 baseline" data with n=30. The filenames describe runs that were never stored.
  3. The headline hallucination A/B was quota-truncated at 10 of 20 planned cases. The JSON
     recorded `quota_exhausted: true`, which is the one thing that saved it from being
     quoted as a complete result.

The rule the plan sets: no number enters the manuscript unless it carries a manifest
naming the code, the models, the thresholds and the corpus that produced it.

USAGE
    from eval.run_manifest import build_manifest
    result = {"run_manifest": build_manifest(notes="stage-2 rerun"), ...}

Then reject, in review, any result file whose `run_manifest` is missing or has
`"dirty_worktree": true`.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip()
    except Exception:
        return ""


def _corpus_fingerprint() -> dict:
    """Identify the corpus by its exported snapshot, not by a live row count.

    A live count changes the moment anything is ingested; the snapshot checksum pins the
    exact text and embeddings a number was computed against.
    """
    snap = ROOT / "eval" / "corpus_snapshot" / "snapshot.json"
    if not snap.exists():
        return {"available": False,
                "warning": "no corpus snapshot — run eval/export_corpus.py"}
    try:
        data = json.loads(snap.read_text(encoding="utf-8"))
        return {
            "available": True,
            "documents": data["counts"]["documents"],
            "chunks": data["counts"]["chunks"],
            "documents_sha256": data["files"]["documents.jsonl.gz"]["sha256"][:16],
            "chunks_sha256": data["files"]["chunks.jsonl.gz"]["sha256"][:16],
            "exported_at": data.get("exported_at"),
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "error": str(exc)[:200]}


def _thresholds() -> dict:
    """Read the live threshold values rather than restating them here.

    Imported lazily and defensively: the manifest must still be produced when a scorer
    runs without the heavy app dependencies installed.
    """
    out: dict = {}
    try:
        from app import RAG  # noqa: PLC0415
        out.update({
            "RERANK_REFUSAL_THRESHOLD": RAG.RERANK_REFUSAL_THRESHOLD,
            "RERANK_CONTEXT_MIN": RAG.RERANK_CONTEXT_MIN,
            "MAIN_QUERY_TEMPERATURE": RAG.MAIN_QUERY_TEMPERATURE,
            "GEN_TEMPERATURE_EFFECTIVE": RAG.GEN_TEMPERATURE,
            "INLINE_CITATIONS_ENABLED": RAG.INLINE_CITATIONS_ENABLED,
            "FUSION_DRIFT_ENABLED": RAG.FUSION_DRIFT_ENABLED,
            "QUERY_REWRITE_ENABLED": RAG.QUERY_REWRITE_ENABLED,
            "GROQ_MODEL": RAG.GROQ_MODEL,
            "RERANK_MODEL": RAG.RERANK_MODEL,
        })
    except Exception as exc:
        out["app_import_error"] = str(exc)[:200]
    try:
        from app import guardrails  # noqa: PLC0415
        out.update({
            "_NLI_REDACT_BELOW": guardrails._NLI_REDACT_BELOW,
            "_NLI_SOFTEN_BELOW": guardrails._NLI_SOFTEN_BELOW,
            "_NLI_HARD_CLAIM_MIN": guardrails._NLI_HARD_CLAIM_MIN,
        })
    except Exception as exc:
        out["guardrails_import_error"] = str(exc)[:200]
    return out


def build_manifest(notes: str | None = None, **extra) -> dict:
    """Assemble the provenance block. Pass anything run-specific via **extra."""
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "sha": sha or "unknown",
            "tag": _git("describe", "--tags", "--exact-match") or None,
            "nearest_tag": _git("describe", "--tags", "--abbrev=0") or None,
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None,
            "dirty_worktree": dirty,
        },
        "determinism": {
            "EVAL_DETERMINISTIC": os.getenv("EVAL_DETERMINISTIC", "0"),
            "EVAL_SEED": os.getenv("EVAL_SEED"),
        },
        "backend": {
            "LLM_BACKEND": os.getenv("LLM_BACKEND", "groq"),
            "model_weight_hash": os.getenv("LOCAL_MODEL_SHA256"),
        },
        "thresholds": _thresholds(),
        "corpus": _corpus_fingerprint(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "notes": notes,
    }
    if dirty:
        manifest["WARNING"] = (
            "Uncommitted changes were present. This result is NOT reproducible from the "
            "recorded SHA and must not be cited. Commit, then re-run."
        )
    manifest.update(extra)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build_manifest(notes="self-test"), indent=2, default=str))
