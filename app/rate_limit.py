"""In-process sliding-window rate limiter for MediRAG endpoints.

Windows are kept in memory as per-(identity, endpoint) timestamp deques.
Good enough for single-worker uvicorn; swap to Redis when scaling
horizontally (one key per deque, ZSET with score = ts).

Identity is whatever the caller passes — today that's:
  - /upload, /upload/resolve, DELETE /upload/{id}: the user_id from the
    form / body (header-trust, same as the rest of the app).
  - /query, /query/stream: session_id if present, else a shared "anon"
    bucket. Once P0.1 (JWT auth) lands, swap to jwt.sub everywhere.

Each endpoint may have multiple overlapping windows (per-minute caps
to stop bursts; per-day caps to stop slow drains). A request is
rejected if ANY window is over. On success, the timestamp is recorded
to all applicable windows (one append; windows share the deque).

Global windows cap total traffic across all users — safety net against
a single compromised account. Implemented with identity "__global__".
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Deque, Dict, List, Tuple

from fastapi import HTTPException


# (identity, endpoint) -> deque of monotonic timestamps (seconds)
_BUCKETS: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)
_LOCK = Lock()

# Per-user limits: list of (max_requests, window_seconds). A request
# passes only if it's under EVERY window.
#
# Sizing notes:
#   - /query: 20/min handles a thoughtful 5-10 questions/session with
#     3x headroom; 200/day is generous for a real user and an obvious
#     wall for a scraper.
#   - /upload: uploads are heavy (PyMuPDF + MedCPT embed + Groq/Cohere
#     for research papers). 5/hour covers "I had 3 reports to upload";
#     20/day stops anyone slow-draining storage or token budget.
_PER_USER_LIMITS: Dict[str, List[Tuple[int, int]]] = {
    "query":  [(20, 60), (200, 86400)],
    "upload": [(5, 3600), (20, 86400)],
}

# Global (cross-user) safety net.
_GLOBAL_LIMITS: Dict[str, Tuple[int, int]] = {
    "upload": (100, 86400),
}


def _prune(dq: Deque[float], now: float, window: int) -> None:
    cutoff = now - window
    while dq and dq[0] < cutoff:
        dq.popleft()


def check(identity: str, endpoint: str) -> None:
    """Raise HTTPException(429) if any rate-limit window is exceeded.

    Side effect on success: appends `now` to the (identity, endpoint)
    bucket (and to the global bucket if the endpoint has one). Prunes
    expired timestamps out of the front of each bucket it touches.

    Fail-open on empty identity so unauthenticated health checks etc.
    are not blocked. Upstream endpoints already 400 on missing
    identity before calling this.
    """
    if not identity:
        return

    limits = _PER_USER_LIMITS.get(endpoint, [])
    global_limit = _GLOBAL_LIMITS.get(endpoint)

    now = time.monotonic()
    with _LOCK:
        dq = _BUCKETS[(identity, endpoint)]
        for limit, window in limits:
            _prune(dq, now, window)
            if len(dq) >= limit:
                retry_after = int(window - (now - dq[0])) + 1
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Rate limit hit: max {limit} {endpoint} "
                        f"requests per {window}s. Please retry in "
                        f"{retry_after}s."
                    ),
                    headers={"Retry-After": str(retry_after)},
                )

        if global_limit is not None:
            g_limit, g_window = global_limit
            g_dq = _BUCKETS[("__global__", endpoint)]
            _prune(g_dq, now, g_window)
            if len(g_dq) >= g_limit:
                retry_after = int(g_window - (now - g_dq[0])) + 1
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"MediRAG has hit its daily {endpoint} cap. "
                        f"Please try again in {retry_after}s."
                    ),
                    headers={"Retry-After": str(retry_after)},
                )

        # Record the request on all applicable buckets.
        dq.append(now)
        if global_limit is not None:
            _BUCKETS[("__global__", endpoint)].append(now)
