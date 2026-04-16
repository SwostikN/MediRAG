"""HTTP fetcher for corpus sources.

Returns raw bytes + content-type guess. Respects a simple timeout and a
polite User-Agent. Retries are the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

_UA = "MediRAG-ingest/0.1 (+contact: developer; educational use)"


@dataclass
class FetchResult:
    url: str
    status: int
    content_type: str
    body: bytes
    error: Optional[str] = None

    @property
    def is_pdf(self) -> bool:
        return "pdf" in self.content_type.lower() or self.url.lower().endswith(".pdf")

    @property
    def is_html(self) -> bool:
        return "html" in self.content_type.lower()


def fetch(url: str, timeout: float = 30.0) -> FetchResult:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": _UA})
    except Exception as exc:
        return FetchResult(url=url, status=0, content_type="", body=b"", error=str(exc))
    ctype = resp.headers.get("Content-Type", "")
    if resp.status_code >= 400:
        return FetchResult(
            url=url,
            status=resp.status_code,
            content_type=ctype,
            body=b"",
            error=f"http {resp.status_code}",
        )
    return FetchResult(url=url, status=resp.status_code, content_type=ctype, body=resp.content)
