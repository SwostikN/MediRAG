"""Download gpt-oss-20b weights for local, quota-free, reproducible inference.

WHY THIS EXISTS
---------------
`llama-3.3-70b-versatile` was decommissioned by Groq in September 2026, invalidating every
result in this repository. Hosted models can be withdrawn at any time, and no amount of
careful record-keeping survives that. Open weights on local disk cannot be withdrawn.

`gpt-oss-20b` is the replacement because it is simultaneously:
  * served on Groq's free tier  -> fast bulk evaluation at zero cost
  * released as open weights    -> the SAME model runs locally, hash-pinned, forever

The local copy is what makes the paper's reproducibility claim real: seeds actually work,
there is no daily quota, and the exact weights can be archived alongside the results.

WHICH FILE
----------
`ggml-org/gpt-oss-20b-GGUF / gpt-oss-20b-MXFP4.gguf` (12.11 GB, Apache-2.0, ungated).
MXFP4 is gpt-oss's NATIVE format, so this is the reference conversion rather than a
re-quantisation. ggml-org is the llama.cpp maintainers' own org.

WHERE
-----
Outside the repository by default (`~/models/`). 12 GB must never risk entering git.

USAGE
    .venv\\Scripts\\python.exe eval/download_local_model.py
    .venv\\Scripts\\python.exe eval/download_local_model.py --verify   # hash an existing file

Resumable: re-running continues a partial download via an HTTP Range request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import requests

REPO = "ggml-org/gpt-oss-20b-GGUF"
FILENAME = "gpt-oss-20b-MXFP4.gguf"
URL = f"https://huggingface.co/{REPO}/resolve/main/{FILENAME}"
DEST_DIR = Path(os.getenv("LOCAL_MODEL_DIR", str(Path.home() / "models")))
DEST = DEST_DIR / FILENAME
META = DEST_DIR / (FILENAME + ".meta.json")
CHUNK = 1 << 20  # 1 MiB


def human(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    last = 0.0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
            done += len(block)
            if time.time() - last > 5:
                print(f"  hashing {100*done/total:5.1f}%", flush=True)
                last = time.time()
    return h.hexdigest()


def download() -> int:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    head = requests.head(URL, allow_redirects=True, timeout=60)
    head.raise_for_status()
    total = int(head.headers.get("content-length", 0))
    etag = (head.headers.get("etag") or "").strip('"')

    have = DEST.stat().st_size if DEST.exists() else 0
    if have == total and total > 0:
        print(f"already complete: {DEST} ({human(total)})")
        return finalize(total, etag)
    if have > total:
        print(f"local file larger than remote ({have} > {total}); restarting")
        DEST.unlink()
        have = 0

    # shutil.disk_usage is cross-platform; os.statvfs is POSIX-only and absent on Windows.
    need = total - have
    free = shutil.disk_usage(DEST_DIR).free
    if free < need * 1.05:
        print(f"insufficient disk space: need ~{human(need * 1.05)}, free {human(free)}")
        return 1

    print(f"source : {REPO}/{FILENAME}")
    print(f"dest   : {DEST}")
    print(f"size   : {human(total)}"
          + (f"   resuming from {human(have)} ({100*have/total:.1f}%)" if have else ""))

    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(URL, headers=headers, stream=True, timeout=(30, 300)) as r:
        if have and r.status_code != 206:
            print(f"server refused resume (HTTP {r.status_code}); restarting from 0")
            have = 0
            r.close()
            with requests.get(URL, stream=True, timeout=(30, 300)) as r2:
                r2.raise_for_status()
                have = _stream(r2, total, 0)
        else:
            r.raise_for_status()
            have = _stream(r, total, have)

    final = DEST.stat().st_size
    if total and final != total:
        print(f"\nINCOMPLETE: {human(final)} of {human(total)}. Re-run to resume.")
        return 1
    print(f"\ndownload complete: {human(final)}")
    return finalize(total, etag)


def _stream(resp, total: int, done: int) -> int:
    start = time.time()
    last = 0.0
    mode = "ab" if done else "wb"
    with DEST.open(mode) as fh:
        for chunk in resp.iter_content(CHUNK):
            if not chunk:
                continue
            fh.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last > 10:
                rate = done / max(now - start, 1e-6) / 1e6
                pct = 100 * done / total if total else 0
                eta = (total - done) / max(done / max(now - start, 1e-6), 1) / 60
                print(f"  {pct:5.1f}%  {human(done)} / {human(total)}  "
                      f"{rate:6.1f} MB/s  eta {eta:5.1f} min", flush=True)
                last = now
    return done


def finalize(total: int, etag: str) -> int:
    print("hashing for the run manifest (this takes a minute)...")
    digest = sha256_file(DEST)
    META.write_text(json.dumps({
        "repo": REPO, "filename": FILENAME, "url": URL,
        "bytes": DEST.stat().st_size, "sha256": digest, "etag": etag,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "why": "Open-weight twin of the Groq-hosted openai/gpt-oss-20b. Pinned by hash so "
               "results stay reproducible if the hosted model is withdrawn, as "
               "llama-3.3-70b-versatile was in Sept 2026.",
    }, indent=2), encoding="utf-8")
    print(f"\nsha256: {digest}")
    print(f"meta  : {META}")
    print("\nRecord this in the environment so run_manifest picks it up:")
    print(f'  setx LOCAL_MODEL_SHA256 {digest[:16]}...')
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="hash an already-downloaded file")
    args = ap.parse_args()
    if args.verify:
        if not DEST.exists():
            sys.exit(f"not found: {DEST}")
        print(f"sha256: {sha256_file(DEST)}")
        raise SystemExit(0)
    raise SystemExit(download())
