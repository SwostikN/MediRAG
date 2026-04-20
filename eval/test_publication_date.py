"""Unit tests for ingest.parse.extract_publication_date (Phase 4b).

The util must:
  - Return YYYY-MM-DD or None.
  - Never throw.
  - Try meta tags first, fall back to first-500-char body text scan.

Run:
    python -m pytest eval/test_publication_date.py -v
    # or without pytest:
    python eval/test_publication_date.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.parse import extract_publication_date  # noqa: E402


def test_meta_article_published_time():
    html = """
    <html><head>
      <meta property="article:published_time" content="2024-06-15T10:30:00Z">
    </head><body><p>Hi.</p></body></html>
    """
    assert extract_publication_date(html) == "2024-06-15"


def test_meta_dc_date():
    html = """
    <html><head>
      <meta name="DC.date" content="2023-01-07">
    </head><body></body></html>
    """
    assert extract_publication_date(html) == "2023-01-07"


def test_time_datetime_tag():
    html = '<html><body><time datetime="2025-02-28">Feb 28</time></body></html>'
    assert extract_publication_date(html) == "2025-02-28"


def test_meta_name_date():
    html = '<meta name="date" content="2022-11-11T00:00:00">'
    assert extract_publication_date(html) == "2022-11-11"


def test_body_last_updated_iso():
    html = "<html><body><p>Last updated: 2025-03-02. More prose.</p></body></html>"
    assert extract_publication_date(html) == "2025-03-02"


def test_body_published_month_name():
    html = "<html><body><p>Published: March 2, 2025</p></body></html>"
    assert extract_publication_date(html) == "2025-03-02"


def test_body_revised_dmy():
    html = "<html><body><p>Revised 2 Mar 2025 by editorial team.</p></body></html>"
    assert extract_publication_date(html) == "2025-03-02"


def test_plain_text_with_iso_in_first_500_chars():
    txt = "This article was published 2024-09-09. " + "x" * 100
    assert extract_publication_date(txt) == "2024-09-09"


def test_no_date_returns_none():
    html = "<html><body><p>No dates anywhere in this document.</p></body></html>"
    assert extract_publication_date(html) is None


def test_invalid_date_in_meta_returns_none_or_valid():
    # Garbage meta value should not crash; may fall through to body scan.
    html = '<meta property="article:published_time" content="not-a-date"><body>No dates.</body>'
    assert extract_publication_date(html) is None


def test_never_throws_on_bytes():
    # Bytes input should be accepted, not raise.
    result = extract_publication_date(b"<html><body>Published: 2024-01-01</body></html>")
    assert result == "2024-01-01"


def test_never_throws_on_none_like_input():
    # Empty string must return None, not raise.
    assert extract_publication_date("") is None


if __name__ == "__main__":
    # Minimal runner so this file works without pytest installed.
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
        except Exception:
            failed += 1
            print(f"  ERROR {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
