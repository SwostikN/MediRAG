# Paper Worklog

A running record of every change made during the research-paper phase of MediRAG.

**Why this file exists.** You need to be able to explain and defend every change in this
project yourself — in a viva, to a supervisor, and to reviewers. Working code you cannot
justify is not useful to you. So each entry answers three questions in plain language:

- **What changed** — which file, and what is different now
- **Why** — the problem it solves
- **Impact on the project** — what this changes for you and for the paper, *not* what it
  changes in the code

Newest entries first. Companion to `docs/DOCUMED.md` (which covers the build phase) and
to the plan at `~/.claude/plans/i-have-a-folder-virtual-kahan.md`.

---

## 2026-09-17 — Phase 0.1: Backed up the corpus and fixed the bug that lost the manifests

### 1. Exported the whole corpus to local disk

**What changed:** New file `eval/export_corpus.py`, which downloads every row of the
`documents` and `chunks` tables from Supabase and saves them as compressed files in
`eval/corpus_snapshot/`. Run it with `--verify` to re-check an existing export.

Result of the first run:

| | |
|---|---|
| Documents | 1,009 |
| Chunks | 3,816 |
| Orphan chunks (pointing at a missing document) | 0 |
| Chunks missing an embedding | 0 |
| Embedding dimensions | 768 (all sampled) |
| Total size on disk | 12.7 MB compressed |

**Why:** In early September the Supabase project hosting this corpus stopped resolving
entirely — it looked deleted. It has since come back, but that was a genuine near-miss:
the corpus existed in exactly one place, and every evaluation number in the paper depends
on it. A corpus that lives in one cloud account is one billing lapse or one accidental
click away from taking the project with it.

**Impact on the project:** The corpus is no longer a single point of failure. If Supabase
disappears again it now costs an afternoon to restore instead of ending the project. This
also becomes a paper artifact — reviewers and other researchers can be given the exact
corpus the results were computed on, which is what "reproducible" actually requires.

**Two details worth knowing:**

- The `tsv` column is deliberately *not* exported. It is a generated column — Postgres
  computes it automatically from `content` — so it cannot be inserted back and would only
  waste space.
- The embeddings are 768-dimensional, which means they come from **MedCPT**, a model that
  runs locally, not from Cohere's paid embedding API. **This is good news for the
  zero-budget plan:** restoring or extending the corpus needs no paid embedding service at
  all. The plan assumed Cohere embeddings here; that assumption was wrong and the reality
  is cheaper.

**Files touched:** `eval/export_corpus.py` (new), `eval/corpus_snapshot/` (new)
**Verify with:** `.venv\Scripts\python.exe eval/export_corpus.py --verify`

---

### 2. Found and fixed the actual reason the corpus manifests were missing

**What changed:** Two lines in `.gitignore`. The pattern `MANIFEST` became `/MANIFEST`,
and an explicit re-include for `ingest/manifest/` was added.

**Why:** The `ingest/manifest/*.jsonl` files list the 224 source URLs that define which
documents make up the corpus. Without them the corpus cannot be rebuilt from scratch.
They were missing from this machine, and `git log` showed they had **never been committed
in the entire history of the project**.

The reason turned out not to be an oversight. Line 55 of `.gitignore` said `MANIFEST` —
a standard Python convention, meant for a single setuptools file at the top of a repo.
But a git ignore pattern with no slash in it matches at *any* depth, and Windows git runs
with `core.ignorecase = true`. So `MANIFEST` also matched the **directory**
`ingest/manifest/`, and git silently excluded all ten files from every commit you ever
made. No warning, no error — `git status` simply never mentioned them.

**Impact on the project:** This is the difference between a fix and a real fix. Copying
the manifest folder back from the old machine would have looked like it worked, and then
the files would have vanished again at the next machine change — exactly as before.
Anchoring the pattern to the repo root (`/MANIFEST`, which is all setuptools ever meant)
plus an explicit re-include means the corpus provenance is now permanently tracked and no
future ignore rule can quietly swallow it.

Confirmed working: `git add --dry-run` now stages all 10 manifest files, while a
root-level `MANIFEST` file would still be correctly ignored.

**Worth remembering for the paper:** this is a small, concrete example of the kind of
silent reproducibility failure the Methods section should argue against. Nothing errored.
Nothing was flagged. The project simply lost the ability to rebuild its own dataset, and
nobody found out until someone changed machines.

**Files touched:** `.gitignore`
**Verify with:** `git add --dry-run ingest/manifest/` (expect 10 files)

---

### 3. Reverted an accidental change from the audit

**What changed:** `eval/baselines/baseline_v1.json` restored to its committed version.

**Why:** During the audit I ran `eval/harness.py` to check that the gold files were valid.
That script has a side effect that is not obvious from its name — it rewrites
`baseline_v1.json` every time it runs. It regenerated the file with 329 gold pairs instead
of the committed 309, because the gold sets have grown since that baseline was recorded.

**Impact on the project:** None now — the original file is back. Flagging it because the
side effect is easy to trigger by accident, and because it is a symptom of a wider problem
the plan already addresses: the baseline files in this repo do not reliably describe what
they contain (`baseline_v4_filtered.json` and `baseline_v5_full.json` both hold "Week 2"
data with n=30). All pre-`paper-v1` baselines should be treated as uncitable and archived.

**Files touched:** `eval/baselines/baseline_v1.json` (reverted, no net change)

---

### 4. Produced the plan document

**What changed:** New files `docs/build_paper_plan_docx.py` and
`docs/MediRAG_Paper_Plan.docx`. Installed `python-docx` into `.venv`.

**Why:** You asked for the full plan as a readable document. `python-docx` was already
imported by `docs/build_viva_docx.py` but had never been installed in this virtual
environment.

**Impact on the project:** No effect on how the system behaves. The document is generated
by the script, so **edit the script and re-run it** rather than editing the `.docx` by
hand — otherwise changes are lost on the next build.

**Files touched:** `docs/build_paper_plan_docx.py` (new), `docs/MediRAG_Paper_Plan.docx` (new)
**Verify with:** `.venv\Scripts\python.exe docs/build_paper_plan_docx.py`
