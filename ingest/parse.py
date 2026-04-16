"""Parse fetched bytes into clean text + basic metadata.

HTML via trafilatura (with a BeautifulSoup fallback). PDF via PyMuPDF.
Returns (text, metadata) where metadata keys align with what the
`documents` table accepts: title, publication_date (ISO), page_count.
"""

from __future__ import annotations

import io
import re
from typing import Optional

import pymupdf
import trafilatura
from bs4 import BeautifulSoup


_PDF_DATE_RE = re.compile(r"D:(\d{4})(\d{2})(\d{2})")


def _parse_pdf_date(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    m = _PDF_DATE_RE.match(raw)
    if not m:
        return None
    y, mo, d = m.groups()
    return f"{y}-{mo}-{d}"


def parse_pdf(body: bytes) -> tuple[str, dict]:
    with pymupdf.open(stream=body, filetype="pdf") as doc:
        text_parts: list[str] = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text_parts.append(page_text)
        raw_meta = doc.metadata or {}
        metadata = {
            "title": (raw_meta.get("title") or "").strip() or None,
            "publication_date": _parse_pdf_date(raw_meta.get("creationDate")),
            "page_count": doc.page_count,
        }
    return "\n".join(text_parts).strip(), metadata


def parse_html(body: bytes, url: str) -> tuple[str, dict]:
    html_str = body.decode("utf-8", errors="replace")
    text: Optional[str] = None
    meta: dict = {}

    extracted = trafilatura.extract(
        html_str,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )
    if extracted:
        text = extracted.strip()

    try:
        soup = BeautifulSoup(html_str, "lxml")
    except Exception:
        soup = BeautifulSoup(html_str, "html.parser")

    if not text:
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split()).strip()

    if soup.title and soup.title.string:
        meta["title"] = soup.title.string.strip()
    for selector, key in [
        ('meta[property="article:published_time"]', "publication_date"),
        ('meta[name="article:published_time"]', "publication_date"),
        ('meta[name="date"]', "publication_date"),
        ('meta[property="article:modified_time"]', "last_revised_date"),
    ]:
        tag = soup.select_one(selector)
        if tag and tag.get("content"):
            val = tag["content"][:10]
            if re.match(r"\d{4}-\d{2}-\d{2}", val):
                meta.setdefault(key, val)

    return text or "", meta
