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

## 2026-09-17 — Phase 0.2 & 0.5: Corpus provenance closed, determinism switch, run manifests

### 1. Every document in the corpus is now accounted for

**What changed:** New `eval/freeze_corpus_manifest.py`, which produced
`ingest/manifest/medlineplus_bulk_v1.jsonl` (824 rows + README) and
`eval/corpus_manifest.json` (the frozen, paper-facing record).

**Why:** Reconciling your manifests against the live corpus showed they did not describe it:

| | |
|---|---|
| Manifest URLs actually in the corpus | 185 / 210 (88%) |
| Manifest URLs never ingested | 25 |
| **Corpus documents from no manifest at all** | **824 / 1,009 (82%)** |

The 824 are all MedlinePlus, bulk-ingested from the cached topic index
(`ingest/cache/mplus_topics.xml`) rather than from a URL list. So 82% of your corpus was
undocumented — it could not be audited, described in a paper, or rebuilt. For a paper whose
central claim rests on a fixed corpus, that is not a survivable position.

The 25 never-ingested URLs are exactly the gap your August plan flagged (19 Nepal
candidates + 5 care-pathway + 1 seed). They are now listed explicitly in
`eval/corpus_manifest.json` rather than carried silently.

**Impact on the project:** Provenance goes from 18% to **100%** — every document is
accounted for by some manifest. Table 1 of the paper can now be generated from a file
instead of assembled by hand.

**One honest caveat, recorded in the file itself:** the derived manifest is a *record of
what was ingested*, not a recipe. Re-running it would re-fetch those URLs as they exist
today, which is not the same corpus. The authoritative copy remains
`eval/corpus_snapshot/`, which holds the exact text and embeddings.

**Also confirmed for the paper:** 57 of 1,009 documents (**5.6%**) are Nepal-scoped. The
"Nepal-grounded corpus" claim stays retired; the Nepal contribution is the routing layer.

### 2. A determinism switch, so results stop moving on their own

**What changed:** New `app/determinism.py`. Setting `EVAL_DETERMINISTIC=1` forces every
generation call to temperature 0 and supplies a fixed seed. **Production is untouched** —
without that variable, every call site keeps exactly the temperature it had before.

**Why:** your headline hallucination A/B was run at temperature 0.15 with no seed, and your
own log records identical configurations flipping 2–3 gold items between runs. A number
that moves when nothing changed cannot go in a paper.

**The part that mattered:** pinning only the final generation call would not have worked.
Eleven call sites in `RAG.py` plus four stage modules (`clarification`, `intake`,
`navigation`, `results`) each had their own temperature. A non-deterministic *query rewrite*
changes what gets retrieved, which changes the context, which changes the answer — so the
run would have stayed irreproducible through the back door. All of them now route through
one helper.

`app/determinism.py` is a separate module because `RAG.py` imports the stages, so the
stages cannot import `RAG.py` back. A shared leaf module is the only place all of them can
reach.

**Stated honestly in the code:** temperature 0 is necessary but *not sufficient* on a
hosted API — batching and load balancing still perturb results, and a provider can retire
a model outright, as happened here. Real reproducibility needs local hash-pinned weights.
That is Phase 1, and the manifest records which backend produced each number so the
distinction is visible in the data rather than assumed.

### 3. Run manifests — no more orphaned numbers

**What changed:** New `eval/run_manifest.py`. Every result file gets a block recording git
SHA and tag, all nine thresholds, model IDs, backend, seed, corpus checksum, and Python
version.

**Why:** three separate failures in this project trace to results that did not record their
own conditions — the decommissioned model (nothing recorded which model), the mislabelled
baselines (`baseline_v4_filtered.json` holds Week-2 n=30 data), and the quota-truncated A/B
(saved only because it happened to log `quota_exhausted`).

It also **flags its own uncommitted state**: if the working tree is dirty, the manifest
carries a `WARNING` saying the result is not reproducible from the recorded SHA and must
not be cited.

**Impact on the project:** gives you a hard reviewable rule — reject any result file with
no manifest, or with `dirty_worktree: true`. That single check would have caught all three
historical failures.

### 4. A mistake I made and corrected

While wiring the determinism switch I edited `app/RAG.py` with a PowerShell text
replacement. PowerShell re-encoded the file and corrupted **266 non-ASCII characters** —
107 em-dashes and 157 box-drawing characters in your comment headers turned into mojibake
(`—` became `â€"`).

Caught it by diffing against the committed version, reverted with `git checkout`, and redid
the edit in Python with explicit UTF-8 byte handling and CRLF preservation, verifying the
non-ASCII character counts matched before and after. Nothing was committed in the corrupted
state.

**Lesson worth keeping:** do not use PowerShell `Set-Content` on source files in this repo.
The comment headers throughout `app/` use box-drawing characters and em-dashes, and
PowerShell's default encoding silently destroys them.

**Files touched:** `app/determinism.py` (new), `app/RAG.py` (11 call sites),
`app/stages/{clarification,intake,navigation,results}.py`, `eval/run_manifest.py` (new),
`eval/freeze_corpus_manifest.py` (new), `eval/corpus_manifest.json` (new),
`ingest/manifest/medlineplus_bulk_v1.jsonl` (new, 824 rows)
**Verify with:** `.venv\Scripts\python.exe -m eval.run_manifest` and
`$env:EVAL_DETERMINISTIC=1; .venv\Scripts\python.exe -c "import app.RAG as R; print(R.GEN_TEMPERATURE)"`
(expect `0.0`; expect `0.15` without the variable). Suite: 184 passed, 2 skipped.

---

## 2026-09-17 — Phase 0.3: CORRECTION — the retrieval ceiling did NOT improve

### What changed

New file `eval/audit_corpus_vs_gold.py` — a reproducible version of the corpus-vs-gold
gradability audit, which previously existed only as an output file with no script. Report
written to `eval/reports/corpus_vs_gold_audit_2026-09.json`.

### The correction

**Earlier today I told you the retrieval ceiling had "largely fixed itself", 21.4% → ~51%.
That was wrong, and the plan document repeated it. It is not true.**

I compared the current gold files against the current corpus, and compared that percentage
with an April 2026 figure computed on a *different* gold set. That is not a valid
comparison. Doing it properly:

| Comparison | Result |
|---|---|
| April gold (518 sources) vs April corpus (131 docs) | 111/518 = **21.4%** |
| April gold (518 sources) vs **today's corpus (1,009 docs)** | 115/518 = **22.2%** |
| **Like-for-like effect of growing the corpus 7.7×** | **+0.8 percentage points** |

The corpus grew nearly eightfold and bought essentially **nothing** in gradability. The
reason is the one already noted about corpus composition: the 878 new documents are
overwhelmingly MedlinePlus patient-education pages, and the gold rows cite *specific*
sources — WHO IMAI, mhGAP, IMCI, named NHS topics, MoHP SOPs — that MedlinePlus does not
correspond to. Adding more of the wrong kind of document does not make the right ones
appear.

### Where the apparent improvement actually came from

The gold files were rewritten on 2026-04-21 by
`eval/scripts/apply_gold_rewrite_2026_04_21.py`. Across 198 rows and 604 source entries:

| Transform | Entries | Share |
|---|---|---|
| **D — dropped** as invalid labels | **365** | **60%** |
| K — kept unchanged | 121 | 20% |
| I — queued as ingest candidates | 88 | 15% |
| **S — substituted** with a corpus-aligned title | **30** | **5%** |

Dropping 60% of the entries — disproportionately the ones nothing in the corpus could
match — raises the remaining percentage *by definition*. That is survivorship, not
improvement. On top of that, **57 of 228 gold rows (25%)** had every source dropped and are
now tagged `retrieval_scoring: disabled`, so they are excluded from the denominator
entirely.

### Was the rewrite wrong?

Mostly no, and this matters for how you defend it. Many drops are plainly correct — the
script's own note on row `in-001` reads *"SOCRATES is a template tag, not a retrieval
source"*, which is a genuinely bad label. Dropping Western commercial sources from a Nepal
system is defensible. The rewrite was reasonable cleanup.

The problem is not the rewrite; it is **comparing a number computed after the cleanup with
one computed before it**, and calling the difference progress. The 5% substitution rate is
a smaller but real concern in its own right: those labels were changed to match what the
corpus contains, which is mildly circular.

### Impact on the project

**Retrieval recall stays off the headline.** Your original §11.22 Phase 4 decision — keep
faithfulness as the headline correctness metric and treat retrieval recall as a caveated
coverage bound — was correct, and I was wrong to suggest reversing it. The plan has been
amended.

**If retrieval recall is reported at all**, it must carry three disclosures: the gold was
rewritten post-hoc, 60% of source entries were dropped, and 25% of rows are excluded from
scoring. A reviewer who finds `apply_gold_rewrite_2026_04_21.py` without those disclosures
in the paper will treat the whole evaluation as suspect. Pre-empt it.

**The script now refuses to be misquoted.** Every run prints the like-for-like comparison
and the rewrite caveat next to the headline figure, and the JSON carries a `WARNING` field.
The number cannot be lifted out of context by accident.

**Files touched:** `eval/audit_corpus_vs_gold.py` (new),
`eval/reports/corpus_vs_gold_audit_2026-09.json` (new), plan documents amended
**Verify with:** `.venv\Scripts\python.exe eval/audit_corpus_vs_gold.py`

---

## 2026-09-17 — Phase 0.4: Closed the scope-guard safety hole, and measured what it cost

### What changed

`app/refusal_filter.py` — the scope guard now catches a flat statement like *"You have
type 2 diabetes"* and classifies it as `diagnostic` (which triggers a refusal) instead of
letting it through as `safe`.

Also new: `eval/probe_scope_guard.py`, an 18-case battery that records exactly what the
guard does, and two dated records of the behaviour either side of the change
(`eval/results/scope_guard_regression_2026-09_BEFORE.json` and `..._AFTER.json`).

### Why

The scope guard is the last policy layer before an answer reaches a user. It was catching
only *hedged* diagnoses — "most likely you have X", "this sounds like X" — while the blunt,
unhedged assertion passed straight through as safe. Six of ten diagnostic phrasings leaked:

| Phrasing | Before | After |
|---|---|---|
| "You have type 2 diabetes." | safe | **diagnostic** |
| "You have asthma." | safe | **diagnostic** |
| "You have iron deficiency anemia." | safe | **diagnostic** |
| "You likely have anemia." | safe | **diagnostic** |
| "You probably have hypertension." | safe | **diagnostic** |
| "You've got pneumonia." | safe | **diagnostic** |

A test already in the repo (`eval/test_scope_guard.py:56`) asserted this should be caught,
so that test had been failing against committed code — and because CI runs that file,
`main` has been red. `git log -S` traced the cause to commit `b13bd83`
("hallucination-zero sprint … scope-guard **relax**"), which loosened the guard to reduce
over-refusal. This hole was the price of that change, and nobody noticed.

### The part that took the actual work

The obvious fix — put `"you have"` back as a phrase to look for — is wrong, and the code
comments explain why: it was removed on 2026-04-20 precisely because it over-fired on
legitimate lab explanations like *"your HbA1c of 6.8% means you have a slightly elevated
blood sugar."* Re-adding it would break your Stage 4 lab explainer.

So rather than guess, I measured. Taking all **33,314 sentences** from the corpus snapshot
as a stand-in for the kind of text the system actually produces:

| Approach | Sentences wrongly flagged as a diagnosis | Rate |
|---|---|---|
| Blunt "you have" phrase match | **177** | 0.531% |
| **Shipped rule** | **7** | **0.021%** |

The blunt version flagged things like *"If you have diabetes, your blood glucose levels
are too high"*, *"it does not necessarily mean you have cancer"* and *"make sure you have
had the latest boosters"* — all perfectly good patient education. Shipping that would have
rebuilt the exact over-refusal problem the 2026-04-20 removal was chasing, just in a
different place.

**The distinction that actually matters is grammatical, not word-matching.** A diagnosis is
an *assertion about this user*. "You have diabetes" asserts it. "If you have diabetes",
"people who have diabetes", "it doesn't mean you have diabetes" and "you have had..." do
not. So the rule now requires two things: a "you have" phrase that is **not** preceded by a
conditional or third-person word, **and** a named condition within 40 characters after it
(so *"You have a higher risk of kidney disease if you have diabetes"* doesn't trip on a
condition mentioned much later).

That is a **25× reduction in false positives** compared with the simple approach, with all
18 probe cases passing. The 7 remaining are list fragments and awkward constructions like
*"conditions in which you have anxiety"*. They err toward refusing too much, which is the
correct direction under this project's stated safety asymmetry: a leaked diagnosis is a
safety event, an unnecessary refusal is an inconvenience.

### Impact on the project

**Safety:** a real hole in a deployed system is closed, and the fix does not reintroduce
the problem it replaced. Both failure directions are now held shut at once.

**Test suite:** `main` is green for the first time in this phase — **184 passed, 2 skipped,
0 failed** across the whole `eval/` directory (the 5 files CI runs: 43/43).

**For the paper:** this is the single best piece of evidence for the paper's central claim.
You now have a live system where a safety layer was deliberately relaxed to buy back
access, with the commit, the date, the consequence, and the measured cost of putting it
right — all on record. The before/after JSON files are citable, and the two operating
points can be plotted on the safety–access frontier figure.

**Still to measure:** the *end-to-end* over-refusal delta (the 54.5% figure) needs the full
pipeline, which is blocked until the generator backbone is replaced in Phase 1. The 0.021%
number above is the scope-guard layer in isolation, which is a lower bound, not the
end-to-end cost.

**Known limitation:** conditions not in `_CONDITION_NAMES` are still missed — "You have a
urinary tract infection" classifies as safe, because "urinary tract infection" is not in
that list. Expanding the list is deliberately left for later so it does not confound this
measurement; it changes behaviour for every context cue, not just this one.

**Files touched:** `app/refusal_filter.py`, `eval/probe_scope_guard.py` (new),
`eval/results/scope_guard_regression_2026-09_{BEFORE,AFTER}.json` (new). `pytest` installed
into `.venv`.
**Verify with:** `.venv\Scripts\python.exe eval/probe_scope_guard.py` (expect 18/18) and
`.venv\Scripts\python.exe -m pytest eval/ -q` (expect 184 passed)

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
