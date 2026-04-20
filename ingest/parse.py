"""Parse fetched bytes into clean text + basic metadata.

HTML via trafilatura (with a BeautifulSoup fallback). PDF via PyMuPDF.
Returns (text, metadata) where metadata keys align with what the
`documents` table accepts: title, publication_date (ISO), page_count.
"""

from __future__ import annotations

import io
import re
from typing import Optional, Union

import pymupdf
import trafilatura
from bs4 import BeautifulSoup


_PDF_DATE_RE = re.compile(r"D:(\d{4})(\d{2})(\d{2})")

# ─── Phase 4b: publication-date extraction ──────────────────────────────
#
# `extract_publication_date` is a best-effort utility that tries HTML
# meta tags first, falls back to text-pattern scanning, and always
# returns either an ISO-8601 YYYY-MM-DD string or None. It never throws
# — any exception is swallowed and returns None.

_META_DATE_SELECTORS = [
    ('meta[property="article:published_time"]', "content"),
    ('meta[name="article:published_time"]', "content"),
    ('meta[property="article:modified_time"]', "content"),
    ('meta[name="date"]', "content"),
    ('meta[name="DC.date"]', "content"),
    ('meta[name="DC.date.issued"]', "content"),
    ('meta[name="DC.Date"]', "content"),
    ('meta[name="dcterms.issued"]', "content"),
    ('meta[name="citation_publication_date"]', "content"),
    ('meta[name="pubdate"]', "content"),
    ('meta[itemprop="datePublished"]', "content"),
    ('meta[itemprop="dateModified"]', "content"),
    ('time[datetime]', "datetime"),
]

_ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
# Human date phrases we scan for in the first 500 chars of visible body.
# "Published: March 2, 2025" / "Last updated 2 Mar 2025" / "Revised: 2025-03-02"
_TEXT_DATE_CUES = re.compile(
    r"(?:last\s+updated|published|revised|reviewed|issued)\s*[:\-]?\s*"
    r"(?P<d>"
    r"\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}"
    r"|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}"
    r")",
    re.IGNORECASE,
)
_MONTHS = {m: i for i, m in enumerate(
    ["january","february","march","april","may","june",
     "july","august","september","october","november","december"], start=1
)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})


def _normalise_iso(raw: str) -> Optional[str]:
    """Take any stringy date (ISO prefix, '2 Mar 2025', 'March 2, 2025') →
    YYYY-MM-DD. Returns None if unparsable or impossible."""
    if not raw:
        return None
    s = raw.strip()
    # ISO-prefix form ("2025-03-02T12:34:56Z" or "2025-3-2").
    m = _ISO_DATE_RE.search(s)
    if m:
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{y:04d}-{mo:02d}-{d:02d}"
        except (ValueError, TypeError):
            return None
    # "2 Mar 2025" / "March 2, 2025".
    parts = re.findall(r"[A-Za-z]+|\d+", s)
    if len(parts) >= 3:
        # Try month-name detection across the first few tokens.
        month = day = year = None
        for tok in parts[:4]:
            key = tok.lower()
            if key in _MONTHS and month is None:
                month = _MONTHS[key]
            elif tok.isdigit():
                val = int(tok)
                if val >= 1900 and year is None:
                    year = val
                elif 1 <= val <= 31 and day is None:
                    day = val
        if month and day and year:
            return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def extract_publication_date(
    html_or_text: Union[str, bytes],
    url_hint: Optional[str] = None,
) -> Optional[str]:
    """Return an ISO-8601 (YYYY-MM-DD) publication date or None.

    Strategy:
      1. Try HTML meta tags / <time datetime="..."> first.
      2. Fall back to pattern-scan on the first ~500 chars of visible body
         text ("Last updated:", "Published:", "Revised:", ISO-like dates).

    Never throws. `url_hint` is accepted for forward compatibility (e.g.
    for domain-specific heuristics we may add later, such as preferring the
    `datePublished` microdata over `dateModified` for NHS pages) but is
    unused in the current implementation.
    """
    del url_hint  # reserved for future domain heuristics
    try:
        if isinstance(html_or_text, bytes):
            html_or_text = html_or_text.decode("utf-8", errors="replace")
        if not html_or_text:
            return None

        # 1) Meta-tag path. Only worth it if this looks like HTML.
        if "<" in html_or_text and ">" in html_or_text:
            try:
                soup = BeautifulSoup(html_or_text, "lxml")
            except Exception:
                soup = BeautifulSoup(html_or_text, "html.parser")
            for selector, attr in _META_DATE_SELECTORS:
                tag = soup.select_one(selector)
                if tag and tag.get(attr):
                    iso = _normalise_iso(str(tag[attr]))
                    if iso:
                        return iso
            # No meta hit → fall through with visible-text body.
            visible = " ".join(soup.get_text(separator=" ").split())
        else:
            visible = html_or_text

        # 2) Text-pattern fallback on the first 500 chars of visible body.
        head = visible[:500]
        m = _TEXT_DATE_CUES.search(head)
        if m:
            iso = _normalise_iso(m.group("d"))
            if iso:
                return iso
        # Plain ISO-looking date in the head as last resort.
        m2 = _ISO_DATE_RE.search(head)
        if m2:
            iso = _normalise_iso(m2.group(0))
            if iso:
                return iso
        return None
    except Exception:
        # Absolute contract: never throws.
        return None


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

    # Phase 4b: if the narrow selector set above didn't populate a date,
    # run the broader `extract_publication_date` util (meta + body-text
    # fallback). doc_publication_date is written to the DB by the ingest
    # driver via documents.publication_date; back-filling existing 131 docs
    # is out of scope — see followup note in ingest/run.py.
    if "publication_date" not in meta:
        iso = extract_publication_date(html_str, url_hint=url)
        if iso:
            meta["publication_date"] = iso
    meta.setdefault("doc_publication_date", meta.get("publication_date"))

    return text or "", meta
