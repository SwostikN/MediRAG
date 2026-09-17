"""Evaluation determinism controls (paper phase, 0.5).

WHY THIS EXISTS
---------------
The hallucination A/B in `eval/baselines/hallucination_compare.json` was run at
temperature 0.15 with no seed, and identical configurations were observed to flip 2-3
gold items between runs (docs/DOCUMED.md, Hallucination-Zero sprint §5.2). A result that
moves when nothing changed cannot carry a paper. Every number in the manuscript must come
from a run whose sampling is pinned and recorded.

WHY IT IS ITS OWN MODULE
------------------------
Generation happens in `app/RAG.py` AND in the stage modules (`app/stages/*.py`,
`app/query_rewrite.py`, `app/intent_gate.py`). `RAG.py` imports the stages, so the stages
cannot import `RAG.py` back. A shared leaf module is the only place all of them can reach.

This matters more than it looks: a non-deterministic query rewrite changes what is
retrieved, which changes the context, which changes the answer. Pinning only the final
generation call would leave the run irreproducible through the back door.

PRODUCTION IS UNAFFECTED. Without EVAL_DETERMINISTIC set, every call site keeps the exact
temperature it had before.

HONEST LIMITS
-------------
Temperature 0 is necessary but not sufficient on a hosted API: batching, load balancing
and model-version drift still perturb outputs, and a provider can retire a model outright
(as happened here — llama-3.3-70b-versatile returned 404 in Sept 2026, invalidating every
prior result). Real reproducibility needs local, hash-pinned open weights. `run_manifest`
records which backend produced a number so the distinction is visible in the data rather
than assumed.

USAGE
    EVAL_DETERMINISTIC=1 EVAL_SEED=20260917 .venv\\Scripts\\python.exe -m uvicorn app.RAG:app
"""
from __future__ import annotations

import os

EVAL_DETERMINISTIC: bool = os.getenv("EVAL_DETERMINISTIC", "0") not in (
    "0", "false", "False", "")

EVAL_SEED: int = int(os.getenv("EVAL_SEED", "20260917"))


def eval_temp(production_default: float) -> float:
    """Return 0.0 in deterministic eval mode, else the production temperature.

    Wrap EVERY generation call site with this, including auxiliary ones (query rewrite,
    intake slot-filling, tier classification). Any unpinned call reintroduces variance
    into the whole pipeline.
    """
    return 0.0 if EVAL_DETERMINISTIC else production_default


def seed_kwargs() -> dict:
    """Seed kwarg for backends that accept one (OpenAI-compatible: Groq, llama.cpp,
    Ollama). Empty when not in eval mode, so production calls are untouched. Cohere's
    chat API has no seed parameter — for that backend temperature 0 is the floor, which
    is exactly why run_manifest records the backend alongside every number.
    """
    return {"seed": EVAL_SEED} if EVAL_DETERMINISTIC else {}
