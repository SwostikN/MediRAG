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

## 2026-09-17 — Phase 2: The 200-question set, and the retrieval cache the ablation runs on

### 1. The frozen question set

**What changed:** New `eval/build_question_set.py` produced `eval/question_set_v1.jsonl`
(200 items) and a manifest recording the seed and a checksum of every source file.

| Stratum | n | Where from |
|---|---|---|
| consumer_symptom | 40 | `coverage.jsonl` (20) + **K-QA** (20, external) |
| condition_education | 40 | `condition.jsonl` |
| lab_explainer | 25 | `results.jsonl` |
| nepal_navigation | 30 | `navigation_stage2.jsonl` |
| adversarial | 45 | `must_refuse.jsonl` (40) + `coverage.jsonl` (5) |
| emergency_redflag | 20 | `redflag.jsonl` positives |

**Why it has to be frozen:** the ablation runs 7 arms over this set. If the arms see
different questions, the differences between them are noise rather than measurement.

**Verified deterministic** — rebuilding produces a byte-identical file
(`sha256 a8e7e611…`). No duplicate questions, no empty inputs.

**External anchor:** K-QA (Manes et al. 2024, MIT licence) — real patient questions with
clinician-written must-have and nice-to-have statements. It matters because reviewers
discount a system measured only against gold its own authors wrote.

**A correction to the plan:** the plan cites K-QA as "1,212 real patient questions". That
is the total question count; only **201 carry the annotated answers**, and only those are
usable. The external share was sized to what actually exists.

### 2. Two input shapes — a trap worth knowing about

Most items are a raw question. `navigation_stage2` items are **not**: they carry an
`intake_summary`, because Stage 2 routes an already-completed intake rather than free text.
Items therefore declare `input_kind`, and the runner must branch on it. Treating all 200 as
raw questions would silently mis-run 30 of them and the failure would look like poor
navigation accuracy rather than a bug.

### 3. The retrieval cache — and a second correction to the plan

**What changed:** New `eval/build_retrieval_cache.py`, writing
`eval/results/retrieval_cache_v1.jsonl`. It retrieves once per question per retrieval
config and stores the full ranked rows including the chunk text.

**Why:** retrieval is the expensive, rate-limited part — MedCPT encoding, a Supabase hybrid
query, and a Cohere rerank capped at roughly 10 requests/minute on the trial tier.
Re-retrieving for every arm would make **Cohere, not the LLM, the bottleneck**.

More importantly it would make the comparison *wrong*. If arms A4 and A6 each retrieve
separately, any difference between them mixes guardrail effects with retrieval jitter. The
arms must see byte-identical context so the only thing varying is the layer under test.
Caching is what makes this a controlled experiment.

**The plan said "retrieve once per question and reuse across all arms". That is not right:**

| Arms | Retrieval |
|---|---|
| A0 | none at all (context-free upper bound) |
| A1 | dense-only, no rerank — **its own config** |
| A2–A6 | hybrid + rerank — one shared config |

So it is one retrieval per *(question, config)* — **two configs, not one, and not seven**.
A1 exists precisely to show what reranking buys; giving it A2's context would erase the
very thing it measures.

**Confirmed the two configs actually differ** on the smoke test — only 2–4 of the top 6
chunks overlap between dense and hybrid+rerank. If they had matched, A1 vs A2 would have
measured nothing.

**The abstention gate needs no extra retrieval.** It is a threshold on `rerank_score`, so
arms A3+ are a pure function of the cached rows. One smoke-test question already lands at
0.285, below your 0.4 threshold — so the gate will fire on it, computed straight from cache.

**Resumable by design:** one line per item, flushed immediately, and anything already
cached is skipped on restart. A rate-limit stall costs only the item in flight. Quota
exhaustion has already truncated one experiment in this project's history.

**Smoke test:** 3 questions × 2 configs = 6 retrievals, 0 errors, ~7.8 s/item.

### Impact on the project

Phase 3 can now run as a controlled experiment rather than a correlated one, and the
Cohere rate limit is paid once (about 25 minutes for the full 400 retrievals) instead of
once per arm.

**What is not built yet:** the arm runner itself — the part that takes cached context,
toggles the guardrail layers per arm, generates, and scores. That is the next piece.

**Files touched:** `eval/build_question_set.py` (new), `eval/question_set_v1.jsonl` (new),
`eval/question_set_v1.manifest.json` (new), `eval/build_retrieval_cache.py` (new),
`eval/external/kqa/` (new, downloaded)
**Verify with:** `.venv\Scripts\python.exe eval/build_question_set.py --check` and
`.venv\Scripts\python.exe eval/build_retrieval_cache.py --limit 3`

---

## 2026-09-17 — Phase 1.1: Backend switch built, local inference measured — and rejected for bulk

### What changed

`app/RAG.py` gained an `LLM_BACKEND` switch with three settings: `groq` (default), `local`,
`cohere`. Downloaded llama.cpp's prebuilt CPU binaries and the `gpt-oss-20b` weights, ran
the model locally, and measured it against Groq on identical prompts. Results recorded in
`eval/results/backend_benchmark_2026-09-17.json`.

### Why the switch is built the way it is

`local` reuses the **same Groq client class**, just pointed at a different address.
`llama-server` speaks the same OpenAI protocol Groq does, so switching backends is a URL
change — no second client to maintain, no separate code path through the guardrails that
could drift out of sync.

`cohere` needed no new code at all: every call site already checks "is there a Groq client?"
and falls back to Cohere when there isn't. Setting it to nothing routes everything down the
path that was already there.

Also changed the default model from `llama-3.3-70b-versatile` to `openai/gpt-oss-20b`. The
old default now returns 404, so any machine that didn't explicitly set the model would have
broken on startup.

### The measurement, and the answer

You asked whether local gives a better result, and to pick the best way forward if not.

**It does not.** Same behaviour, dramatically slower.

| | Measured | The plan had estimated |
|---|---|---|
| Reading the prompt (prefill) | **14.59 tok/s** | 30–60 |
| Writing the answer (decode) | **5.45 tok/s** | 8–14 |

Both roughly 2–3× worse than I estimated when writing the plan. Worth noting the software
was *not* the problem: llama.cpp automatically selected the AVX-512 build for your Xeon, so
this is the fast path for this hardware, not a misconfiguration. The limit is having 5 CPU
cores and no GPU.

End-to-end on an identical prompt (4 real corpus chunks, temperature 0, same seed):

| | Local | Groq |
|---|---|---|
| Time | **103.6 seconds** | **0.56 seconds** |
| Result | correctly refused | correctly refused |

**180× slower for the same decision.** Both correctly refused to answer, because the
chunks I sampled didn't cover the question — so quality is equivalent, speed is not.

What that means for the evaluation:

| Work | Groq | Local |
|---|---|---|
| 2,600 generations | **24 minutes** | 3.6 days |
| 5,500 generations | **54 minutes** | 7.6 days |

And the local figures assume nothing else runs, nothing fails, and nothing needs redoing.

### Decision

**Groq runs all bulk evaluation.** There is no case for spending days of wall-clock to get
the same answers you can have in under an hour.

**Local is kept, but for a much smaller job than the plan assumed:**
- a **50–100 prompt agreement check** (1.5–3 hours, runs overnight) showing the results
  hold under seed-exact local inference;
- **archival insurance** — weights pinned by hash that no vendor can withdraw. This is
  exactly the failure that already destroyed every number this project had;
- a slow-but-it-finishes fallback if free-tier quota blocks a run.

**The plan was wrong about one use I've now removed:** it proposed local as a "quota
overflow valve". At ~2 minutes per generation it cannot absorb a real backlog. Phase 4.1's
latency measurements must also come from Groq — CPU speed would completely swamp the
guardrail overhead those numbers are meant to isolate.

### One finding that affects the paper's wording

Local and Groq are **not token-identical**, even at temperature 0 with the same seed — the
two inference stacks differ numerically. So the reproducibility claim must be *behavioural*
comparability, never token-level equivalence. Worth getting right in Methods, because it is
the kind of overclaim a reviewer checks.

### Impact on the project

The pipeline can now switch backends with one environment variable, which is what makes the
Phase 3 ablation runner able to fail over mid-experiment. The compute schedule in the plan
has been corrected from estimates to measurements — the bulk evaluation is about an hour of
Groq time, not days of local grinding, which materially de-risks Phase 3.

**Files touched:** `app/RAG.py`, `eval/results/backend_benchmark_2026-09-17.json` (new),
plan §3 corrected. Downloaded: `~/models/gpt-oss-20b-MXFP4.gguf` (11.27 GiB, sha256
`27cd6c43…`), `~/llamacpp/` (llama.cpp b11013 CPU binaries). Neither is in the repo.
**Verify with:** `set LLM_BACKEND=local` then `.venv\Scripts\python.exe -c "import app.RAG"`
(expect `LLM backend: local`). Benchmark command is in the JSON artifact.

---

## 2026-09-17 — Phase 1.0: Replaced the dead backend, and caught what the swap would have broken

### What changed

New Groq API key in `.env`, and `GROQ_MODEL` switched from the decommissioned
`llama-3.3-70b-versatile` to **`openai/gpt-oss-20b`**. New `app/llm_compat.py`, and every
`max_tokens` in `app/` now routes through it (20 call sites across 9 files). Local weights
downloading via new `eval/download_local_model.py`.

### Why gpt-oss-20b specifically

It is the only candidate that is **both** served on Groq's free tier **and** released as
open weights. That means the same model runs two ways: fast and free on Groq for bulk
evaluation, and locally — hash-pinned, no quota, real seeds — for anything that must be
reproducible. Switching the hosted model is therefore not a detour away from going local;
it is the first half of it.

### The trap this swap was hiding

`gpt-oss-20b` is a **reasoning model**. It emits hidden chain-of-thought tokens before any
visible text, and those tokens are charged against `max_tokens`. Measured on Groq:

| `max_tokens` | reasoning tokens used | content returned |
|---|---|---|
| 40 | 38 | `''` — empty |
| 120 | 118 | `''` — empty |
| 200 | 147 | `'Pleurisy'` |

**It needs roughly 150 tokens of headroom before it will say anything at all.** Your
codebase is full of deliberately tight budgets, and every one of them would have returned
an empty string — not an error:

| File | Budget | What it does | Would have become |
|---|---|---|---|
| `app/intent_gate.py` | 2 | **First-layer gate** | always empty |
| `app/stages/intake.py` | 10 | Slot filling | always empty |
| `app/RAG.py` | 40 | Topic derivation | always empty |
| `app/intent.py` | 50 | Intent classification | always empty |
| `app/query_rewrite.py` | 80 | Retrieval rewrite | always empty |
| `app/RAG.py` | 320 | Stage answers | badly truncated |

Every one of those call sites treats an empty string as "no result" and quietly falls back.
**Nothing would have raised an error.** The pipeline would have run, produced plausible
output, and silently lost its intent gate, its query rewriting and its slot filling — and
the evaluation numbers would have looked like a model-quality finding rather than a
configuration bug.

I checked whether the API's `reasoning_effort` parameter could avoid this. It is accepted
but had no effect on the token count at any setting, so headroom is the only remedy.

### The fix

`app/llm_compat.py` provides `gen_max_tokens(visible_budget, model)`. Call sites now state
what they actually want to **see** — 2 tokens of YES/NO, 700 tokens of answer — and the
helper adds a 512-token reasoning allowance only when the model is a reasoning family.
Cohere paths (`command-r-08-2024`) are untouched and get exactly what they asked for.

The allowance is 512 against a worst observed 147, roughly 3.5× headroom. That asymmetry
is deliberate: too much headroom wastes a little free quota, too little silently disables
a safety layer.

### Two bugs my own edit introduced, caught before commit

The first pass wired the model argument by scanning surrounding code, and got three call
sites wrong — `intent_gate.py` and `intake.py` referenced a module-level `GROQ_MODEL` that
does not exist in those files (the model arrives as a function parameter), and `intent.py`
uses Cohere's `MODEL`, not Groq's.

Your own test suite caught the first one immediately (`test_intent_gate` failed with
`name 'GROQ_MODEL' is not defined`). I then wrote a static check that parses every file and
verifies each `gen_max_tokens` model argument actually resolves in its enclosing scope, and
fixed the other two. That check is worth keeping in mind as a pattern: a wrong name here
fails loudly, but a wrong *value* would not have.

### Verified live on the new model

The three paths that would have silently broken, tested against real Groq:

```
query_rewrite   (was max_tokens=80) -> 'pleuritic chest pain, chest pain on inspiration, ...'
intent_gate     (was max_tokens=2)  -> 'condition'   (correct)
intake classify (was max_tokens=10) -> 'pain'
```

Suite: **184 passed, 2 skipped.** Encoding verified clean across all of `app/`.

### Impact on the project

The pipeline has a working generator again — `paper-v1` deliberately did not. More
importantly, the swap is now *safe*: a future model change runs through one helper instead
of 20 scattered literals.

**For the paper:** this is a second concrete example of the fragility argument, and a better
one than the decommissioning itself. The model was replaced with a supposedly equivalent
one, and six subsystems would have silently stopped working while still producing
plausible output. That is exactly the failure mode that makes hosted-model evaluation
untrustworthy, and it is why the local hash-pinned copy matters.

### Housekeeping

The new Groq key was pasted into the chat transcript, so treat it as exposed and rotate it
once the paper runs are done. The old key is retained, commented, in `.env`. `.env` remains
gitignored and was not committed.

**Files touched:** `.env` (not committed), `app/llm_compat.py` (new),
`eval/download_local_model.py` (new), `app/RAG.py`, `app/intent.py`, `app/intent_gate.py`,
`app/query_rewrite.py`, `app/stages/{clarification,intake,navigation,results}.py`
**Verify with:** `.venv\Scripts\python.exe -m pytest eval/ -q` (expect 184 passed)

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
