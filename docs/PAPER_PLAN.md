# DocuMed AI — Research Paper Plan

Written 2026-08-07. Basis: full read of `app/`, `eval/`, `ingest/`, `docs/DOCUMED.md`
(8,603 lines), all 24 baseline/result JSONs, and 10 gold files (454 rows).

---

## 1. Verdict

**Yes, there is a paper here — but not the one you currently think you have.**

You believe you have "a solid model to reduce hallucination in medical analysis."
What you actually have is a **deployed safety-cascade system with three
under-evaluated novel mechanisms and one fatally circular evaluation**. The
engineering is genuinely strong and unusually well-documented. The empirical
claim is not established.

The single most important sentence in this document:

> Your headline hallucination result rests on **n=10 questions and 12 claim
> sentences**, scored by **the same DeBERTa-NLI model that the guardrail uses to
> filter** (`eval/baselines/hallucination_compare.json`). A reviewer will reject
> on that alone, in the first paragraph of Review 1.

Everything below is about converting a strong system into a defensible paper.
Estimate: **12–16 weeks of consistent part-time work**. Not two weeks.

---

## 2. Asset inventory — what you actually have

### Genuinely strong (rare in submitted papers)

| Asset | Evidence | Why it matters |
|---|---|---|
| A **deployed** system, not a notebook | FastAPI + React + Supabase, live pilot path | Most RAG-safety papers evaluate a script. Deployment is a differentiator. |
| Deterministic pre-LLM emergency screen | `app/redflag_rules.yaml`, 45 rules; 69/69 recall, 0 FP on gold | Clean, citable, falsifiable result. The LLM is *never* in the emergency path — that is a real architectural claim. |
| Adversarial refusal gate | `must_refuse.jsonl`, 40 rows / 18 categories; **33/40 vanilla → 40/40 system** | Your **strongest existing number**. Categorical semantics, no judge circularity. |
| Sentence-buffered streaming guardrail | `guardrails.py:681` (`process_streaming_chunk`, "Option C") | The most novel single mechanism in the repo. Under-explored in the literature. |
| Citation-binding NLI | `guardrails.py:279` (`parse_inline_citations`) + `_process_sentence` | Verifies against the chunks the model *claims* to cite, not max-over-all. Catches "right claim, wrong citation." |
| Fusion-drift detection | `guardrails.py:441` (`_fusion_drift_check`) | Catches compound claims where each half is grounded but the conjunction is not. Genuinely under-studied failure mode. |
| Nepal / LMIC grounding | `nepal_care_tiers.yaml`, MoHP/EDCD/WHO-SEARO corpus | Under-served deployment setting. Reviewers in health informatics weight this heavily. |
| Safety-asymmetry framing | `docs/DOCUMED.md` §7B, memory `feedback_clinical_safety` | Principled: down-tier errors are safety events, up-tier errors are inconvenience. This belongs in the paper as a stated design axiom. |
| 454 labelled gold rows + 183 unit tests + CI | `eval/gold/`, `.github/workflows/eval.yml` | Reproducibility artefact. Most papers ship less. |
| Full engineering provenance | `docs/DOCUMED.md` | You can answer any "why this threshold?" question with a dated log entry. Rare and valuable. |

### Present but not paper-grade

| Asset | Problem |
|---|---|
| Hallucination A/B | n=10, quota-truncated, circular judge, unfair baseline |
| Retrieval recall | 78.6% of gold `expected_sources` are ungradable (§11.22) — you already documented this |
| Stage 2 navigation eval | n=30, tier accuracy 0.533, high variance across runs |
| Coverage / over-refusal | n=40, ±2–3 items of decoding noise per run |
| Stages 3 (visit_prep) and 5 (condition) | **Not implemented** (`docs/DOCUMED.md` §11.24) — cannot be claimed |

---

## 3. The six blockers — what kills the paper today

Fix all six or do not submit.

### B1. Circular evaluation (fatal)
`cross-encoder/nli-deberta-v3-base` is both the **filter** and the **judge**.
The system redacts sentences the model scores low, then you measure quality with
that same model. You are grading the exam you wrote the answers to.
→ **Fix:** human gold + an independent second judge (§5, Phase 1).

### B2. Sample size (fatal)
10 questions. 12 claim sentences in the treatment arm. Nothing survives peer
review at that n.
→ **Fix:** 200 questions × 5 arms × 3 seeds (§5, Phase 1).

### B3. The baseline arm is unfair *against you*
"Vanilla" = same Llama 3.3 70B, **no retrieval**, scored against chunks it never
saw. Of course it fails entailment. This measures "did the answer coincidentally
match unseen documents," not hallucination. A reviewer will read this as either
naive or as stacking the deck.
→ **Fix:** proper ablation ladder where every arm is scored against the same
context, plus a human-labelled arbiter (§5, Phase 1).

### B4. No utility axis
Any filter reaches zero hallucinations by refusing everything. Your over-refusal
rate is **54.5%** (`coverage_post_redflag_chest_tingle.json`). Reporting safety
without utility is the classic reviewer trap.
→ **Fix:** hallucination-rate × answer-rate Pareto curve. This is also your best
figure (§6, Fig. 2).

### B5. Two of your three novel mechanisms are unevaluated
Fusion-drift and citation-binding are shipped and env-gated
(`FUSION_DRIFT_ENABLED`, `INLINE_CITATIONS_ENABLED`) but **have no dedicated
eval**. You cannot claim a contribution you have not measured. Right now they are
code, not results.
→ **Fix:** targeted diagnostic sets (§5, Phase 2).

### B6. Non-determinism unquantified
`MAIN_QUERY_TEMPERATURE = 0.15`, unpinned; you documented ±2–3 item flips on
identical configs (§5.2 of the Hallucination-Zero sprint). Every number needs
mean ± CI over seeds.
→ **Fix:** deterministic eval mode + 3–5 seed reporting (§5, Phase 0).

### Secondary (declare as limitations, do not need fixing)
- Retrieval-recall gradability ceiling of 21.4% — report honestly as a coverage
  bound, keep faithfulness as the headline correctness metric (your §11.22
  Phase 4 decision was correct).
- Stages 3/5 unimplemented — remove from the architecture claim, list as future work.
- Single generator backbone (Groq Llama 3.3 70B) — note as a generalisation limit.

---

## 4. The contribution claim

Be honest with yourself about what is and is not novel.

**Not novel** (do not claim): RAG for medical QA; hybrid dense+BM25 with RRF;
cross-encoder reranking; NLI-based faithfulness checking; claim decomposition;
abstention thresholds. All established.

**Defensible contributions**, ranked by strength:

1. **Streaming-safe sentence-level guardrailing.** Faithfulness filtering under a
   token-streaming UX, where you may never display text you later retract.
   Batch-mode work (FActScore, RARR, SelfCheckGPT) does not address this. You have
   the mechanism; you need the latency/safety numbers.
2. **Fusion-drift detection.** Compound-claim hallucination where each conjunct is
   grounded but the conjunction is not. Needs its own diagnostic set to become a claim.
3. **Citation-binding as a guardrail, not just an eval.** ALCE and Liu et al.
   *measure* attribution post-hoc; you *enforce* it at generation time by
   restricting the entailment premise to cited chunks. That is a real delta.
4. **Cascaded asymmetric safety floors for consumer health in an LMIC.** The
   systems contribution: deterministic rules before the LLM, abstention before
   generation, entailment after generation, policy filter last — with the failure
   asymmetry stated as a design axiom and measured in both directions.
5. **A Nepal-grounded adversarial safety benchmark.** 40 rows now; at 150–200 with
   clinician review it is a standalone dataset contribution.

**One-sentence claim to defend:**
> A four-layer cascade — deterministic pre-LLM emergency rules, retrieval
> abstention, sentence-level entailment verification with citation binding and
> fusion-drift detection, and a policy filter — reduces unsupported clinical
> claims by X% relative to a strong RAG baseline at a measured cost of Y
> percentage points of answer rate, and does so under streaming without ever
> displaying retracted text.

Fill X and Y from Phase 1. Do not write the paper before you have them.

---

## 5. The plan

### Phase 0 — Freeze and instrument (Week 1)

Cheap, unblocks everything.

1. `git tag paper-v1` on the frozen SHA. **Every number in the paper comes from
   this tag or a later tagged revision.** Record the tag in each result JSON.
2. Add `EVAL_DETERMINISTIC=1` → forces `temperature=0` and a fixed seed in both
   generator call sites in `app/RAG.py` (~L1480, ~L1860). Production stays at 0.15.
3. Add `--seed` and `--n-seeds` to `score_hallucination_compare.py`,
   `score_coverage.py`, `score_must_refuse.py`. Report mean ± 95% CI.
4. Add a `run_manifest` block to every result JSON: git SHA, model IDs, all
   thresholds (`RERANK_REFUSAL_THRESHOLD`, `_NLI_REDACT_BELOW`,
   `_NLI_SOFTEN_BELOW`, `_NLI_HARD_CLAIM_MIN`), env flags, corpus doc/chunk count.
5. Top up the two partial manifests (§I.3: 5 care-pathway + 19 nepal-candidates
   URLs), then snapshot exact corpus stats. Freeze the corpus for the paper.
6. **Start the ethics/IRB conversation now** if you want any human-subject data
   (§ Phase 4). NHRC ethical review in Nepal takes ~6–12 weeks. This is the
   longest lead-time item in the entire plan and it costs nothing to start today.

### Phase 1 — Build an evaluation that can carry a paper (Weeks 2–5)

This is the bottleneck. Budget most of your effort here.

**1.1 Question set — target n=200, stratified**

| Stratum | n | Source |
|---|---|---|
| Consumer symptom | 40 | your `coverage.jsonl` + K-QA subset |
| Condition education | 40 | `condition.jsonl` |
| Lab explainer | 25 | `results.jsonl` |
| Nepal care navigation | 30 | `navigation_stage2.jsonl` |
| Adversarial (must-refuse) | 45 | `must_refuse.jsonl`, expanded |
| Emergency red-flag | 20 | `redflag.jsonl` positives |

Pull the external portion from **K-QA** (Manes et al. 2024 — 1,212 real patient
questions with clinician-written must-have/nice-to-have statements). It is the
single best external anchor for your use case: it measures hallucination *and*
comprehensiveness, so it feeds the Pareto framing directly. HealthSearchQA
(Singhal et al. 2023) is a reasonable second source.

Do **not** use MIRAGE/MedQA/MedMCQA. They are exam-style MCQ; your system
deliberately refuses diagnostic reasoning and would score near-floor for correct
reasons. Say this explicitly in the paper — it is a good limitation paragraph.

**1.2 The ablation ladder — 7 arms, shared retrieval**

One runner script. Retrieve once per question, reuse the identical context across
all arms so the comparison is fair.

| Arm | Config |
|---|---|
| A0 | LLM only, no retrieval (context-free upper bound on hallucination) |
| A1 | Naive RAG: dense-only, no rerank, no gate, no guardrail |
| A2 | + hybrid (dense + BM25 + RRF) + Cohere rerank |
| A3 | + abstention gate (`RERANK_REFUSAL_THRESHOLD`) |
| A4 | + claim classifier + NLI verifier (tiered redact/soften/keep) |
| A5 | + citation binding (`INLINE_CITATIONS_ENABLED=1`) |
| A6 | + fusion drift (`FUSION_DRIFT_ENABLED=1`) — **full system** |

Every arm × 200 questions × 3 seeds. Quota-aware: this is ~4,200 generations.
Groq TPD has bitten you twice already (`project_stage2_rerun` memory). Plan a
multi-day schedule with checkpoint/resume in the runner, and log
`quota_exhausted` per case as you already do.

**1.3 Human ground truth — the anchor (non-negotiable for a medical venue)**

- Stratified sample: **300 sentences** across arms A1/A4/A6 and strata.
- **2 independent annotators**, blind to arm. Recruit MBBS interns or final-year
  students — IOM Maharajgunj, BPKIHS, KUSMS, Patan Academy. Pay them properly.
- Label per sentence: `supported` / `unsupported` / `contradicted` / `not-a-claim`,
  given the retrieved context. Plus a sentence-level harm rating (none / minor /
  potentially harmful) — harm-weighted hallucination rate is a much more
  interesting metric than raw count for a clinical venue.
- Report **Cohen's κ**. Adjudicate disagreements with a third rater or discussion.
- Write the annotation guideline as a standalone appendix document. It is a paper
  artefact and reviewers ask for it.

**1.4 Independent automatic judge**

Add a second judge that is *not* the guardrail model. Options, in order of preference:
- **MiniCheck** (Tang et al. 2024) — purpose-built, cheap, fast.
- **AlignScore** (Zha et al. 2023).
- LLM-as-judge (Claude or GPT-4-class) with a strict entailment rubric and
  few-shot examples.

Then report **judge-vs-human agreement** for all three judges (guardrail-NLI,
independent judge, human). This is what converts your NLI from "the metric" into
"a validated proxy" — and it is a small contribution in its own right.

**1.5 Threshold sweep → Pareto curve**

Sweep `_NLI_SOFTEN_BELOW` ∈ {0.3, 0.4, 0.5, 0.6, 0.7} × `RERANK_REFUSAL_THRESHOLD`
∈ {0.3, 0.35, 0.4, 0.45, 0.5}. Plot hallucination rate vs. answer rate. Mark your
shipped operating point. This single figure defuses B4 and turns 54.5%
over-refusal from an embarrassment into a *stated design choice on a measured
frontier*.

### Phase 2 — Make the novel mechanisms measurable (Weeks 6–7)

**2.1 Fusion-drift diagnostic set (~80 items)**
Build a controlled set: 40 compound claims where the conjunction *is* supported by
a single chunk, 40 where each conjunct is supported by a different chunk but the
conjunction is not. Construct semi-synthetically from your own corpus, then have a
clinician verify every label. Report precision/recall/F1 of `_fusion_drift_check`
on this set. Without this, fusion drift is code, not a contribution.

**2.2 Citation-attribution eval (ALCE-style)**
On the A5 vs A4 comparison, measure:
- Citation **precision**: of sentences carrying `[N]`, what fraction are entailed
  by chunk N specifically?
- Citation **recall**: what fraction of claim sentences carry any marker?
- **Mis-binding rate**: sentence is entailed by *some* chunk but not by its cited
  chunk. This is the failure mode citation binding exists to catch — quantify how
  often it fires.

**2.3 Streaming latency measurement**
Your comment claims "~1–2s of sentence-level latency" — unmeasured. Instrument
`/query/stream` and report distributions (not means):
- time-to-first-token
- **time-to-first-*safe*-sentence** (your actual UX metric)
- total response latency
- per-sentence NLI overhead
Compare batch guardrail vs streaming guardrail vs no guardrail. On real Nepal
network conditions if you can — that detail lands well with reviewers.

**2.4 Red-flag engine: the clean result**
Formalise what you have. 45 rules, 131 gold rows (73 positive / 58 negative),
69/69 recall, 0 FP. Report sensitivity, specificity, and — critically — a
**dangerous-direction error rate** (emergency case not escalated). Add negative
controls to test over-triggering. This is your most rigorous existing result;
present it as such.

### Phase 3 — External comparison (Week 8)

Reviewers will ask "compared to what?" Answer with at least three:

1. **GPT-4-class or Claude with a "cite your sources" prompt** over your same
   corpus and retrieval. This is the honest strong baseline. Expect it to be
   competitive on fluency and worse on refusal discipline — say so either way.
2. **Off-the-shelf RAG** (LlamaIndex/LangChain defaults, generic embeddings, no
   guardrail). The "what most people build" baseline.
3. **Self-RAG** (Asai et al. 2024) if you can run the released checkpoint. It is
   the closest conceptual competitor — self-reflective retrieval with critique
   tokens. If you cannot run it, you must still **differentiate from it in Related
   Work**: Self-RAG trains critique tokens into the generator; you apply an
   external, model-agnostic post-hoc cascade that works with any API backbone and
   under streaming. That distinction is your defence.

### Phase 4 — Deployment evidence (Weeks 9–11, parallel — optional but strong)

Two tiers, pick by ethics timeline:

**Tier 1 (fast, likely exempt or expedited): clinician expert review.**
50 system responses, 3 Nepali clinicians, Likert on safety / factual accuracy /
usefulness / appropriateness of care-tier routing, plus free-text harm flags. No
patient data, no human subjects in the research sense. Usually the fastest path to
credible clinical validation. **Confirm with NHRC regardless.**

**Tier 2 (slow, high value): pilot user study.**
20–30 real users, task-completion + trust + comprehension measures. Requires NHRC
ethical approval — hence the Phase 0 item. Only pursue if the approval is already
moving.

If neither lands in time: cut Phase 4, submit without it, and add it to the
journal extension. Do not delay the whole paper on ethics timelines.

### Phase 5 — Write (Weeks 12–14)

Draft in the order: Results → Methods → Related Work → Discussion → Intro →
Abstract. Never write the abstract first; you do not yet know what the numbers say.

---

## 6. Paper skeleton

**Working title:** *Cascaded Abstention and Sentence-Level Entailment Guardrails
for Streaming Medical Retrieval-Augmented Generation: Design and Evaluation of a
Health Navigator for Nepal*

| § | Content |
|---|---|
| 1. Introduction | Consumer health LLMs in LMICs; hallucination as a safety problem; the asymmetry axiom; contributions list (4 bullets, each tied to a table) |
| 2. Related work | RAG for medicine; attribution & citation; NLI faithfulness; abstention; safety in consumer health. **Must explicitly differentiate from Self-RAG and CRAG.** |
| 3. System | Cascade architecture; corpus construction; Nepal care-tier model; red-flag rules; retrieval; guardrail layers with exact thresholds and their provenance |
| 4. Evaluation design | Gold sets; ablation ladder; human annotation protocol + κ; judge validation; deterministic protocol |
| 5. Results | Main ablation table; Pareto; adversarial safety; red-flag; fusion-drift diagnostic; citation attribution; latency |
| 6. Discussion | The utility cost of safety; what the 54.5% over-refusal actually buys; why exam benchmarks are the wrong target; LMIC deployment lessons |
| 7. Limitations | Single backbone; corpus coverage ceiling (21.4% gradability); English-only; no long-term outcome data; stages 3/5 unimplemented; small-corpus scale |
| 8. Ethics & safety statement | Refusal boundaries; no PHI; regulatory positioning (Nepal DDA, FDA SaMD non-device rationale); annotator compensation |

### Figures

1. **Cascade architecture** — four layers, with the hard-exit path for emergencies
   shown as bypassing the LLM entirely. Make this one excellent; it is your
   thesis in a picture.
2. **Pareto: hallucination rate vs. answer rate** — threshold sweep, shipped
   operating point marked. Your most persuasive figure.
3. **Ablation waterfall** — marginal contribution of each layer A1→A6.
4. **Judge validation** — agreement of guardrail-NLI / independent judge / human,
   with κ. Defuses B1 visually.
5. **Latency CDFs** — no-guardrail vs batch vs streaming; time-to-first-safe-sentence.

### Tables

1. Corpus composition (source, authority tier, docs, chunks, country scope)
2. **Main results** — 7 arms × {hallucination rate (human), hallucination rate
   (independent judge), answer rate, citation precision, harm-weighted rate,
   latency p50/p95}
3. Adversarial refusal by category — system vs. all baselines, 18 categories
4. Red-flag: sensitivity / specificity / dangerous-direction errors
5. Care-tier routing accuracy + directional error breakdown (up-tier vs down-tier)
6. Fusion-drift diagnostic: precision / recall / F1

---

## 7. Related work you must engage

Verify every bibliographic detail before submission — this list is from memory and
is a starting map, not a verified bibliography.

**Medical LLMs / RAG:** Singhal et al. 2023 (Med-PaLM, *Nature*); Xiong et al.
2024 (MedRAG / MIRAGE); Manes et al. 2024 (K-QA).

**Attribution & citation:** Gao et al. 2023 (ALCE); Liu, Zhang & Liang 2023
(Evaluating Verifiability in Generative Search Engines); Rashkin et al. (AIS
framework).

**Faithfulness / hallucination detection:** Min et al. 2023 (FActScore);
Manakul et al. 2023 (SelfCheckGPT); Zha et al. 2023 (AlignScore); Tang et al.
2024 (MiniCheck); Es et al. 2024 (RAGAS); Ji et al. 2023 (hallucination survey).

**Self-correcting RAG — your closest competitors:** Asai et al. 2024 (Self-RAG);
Yan et al. 2024 (CRAG); Gao et al. 2023 (RARR).

**Consumer health safety:** Semigran et al. 2015 (*BMJ*, symptom-checker
accuracy); Schmieding et al. 2022 (*JMIR*). Both already in your project memory —
they are the reason your scope is a navigator, and they belong in the Intro.

**Governance:** WHO 2021, *Ethics and governance of artificial intelligence for
health*; Nepal MoHP Standard Treatment Protocols; NHRC research guidelines.

---

## 8. Venue strategy

Be realistic: **this is not a NeurIPS/ICML/ACL-main paper.** The algorithmic
novelty is thin — the strength is systems design, safety engineering, and
deployment in an under-served setting. Target venues that value exactly that.

| Venue | Fit | Notes |
|---|---|---|
| **JMIR / JMIR Med Inform** | **Best primary target** | Rolling submission (no deadline), digital-health deployment + eval is squarely their scope, LMIC angle valued. APC applies — budget for it. |
| **AMIA Annual Symposium** | Strong | Clinical informatics systems papers; typically ~March deadline for a November meeting. |
| **ACL/EMNLP Findings or Industry Track** (via ARR) | Good, if Phase 2 lands | Rolling ARR cycles. Needs the fusion-drift + citation-binding evals to be rigorous. |
| **ML4H / CHIL** | Good | Safety guardrails for medical LLMs; ML4H typically ~Sept deadline, CHIL ~Feb. |
| **PLOS Global Public Health** | Good for a second paper | If you later foreground the Nepal health-equity angle. |
| **EMNLP System Demonstrations** | Safe fallback | Much lower bar; you have a live demo. Good insurance. |

**Verify all deadlines yourself** — cycles shift and I cannot confirm 2026/2027
dates.

**Recommended sequence:**
1. **arXiv preprint** as soon as Phase 2 completes (~Week 8). Timestamps the
   contribution; costs nothing; JMIR permits preprints. Check dual-submission
   rules for any venue you target.
2. **JMIR submission** ~Week 14 (rolling, so no deadline pressure).
3. **AMIA 2027 in parallel** with the Nepal-deployment framing if the timing works.

**On splitting into two papers:** eventually yes — a systems/NLP paper on the
guardrail cascade, and a health-informatics paper on the Nepal navigator with
pilot data. Do **not** split now. You do not have enough results for one paper yet,
let alone two. Write the guardrail paper with Nepal as the deployment setting;
split later if the pilot generates enough independent material.

---

## 9. Risks and how to kill them

| Risk | Mitigation |
|---|---|
| **Groq TPD exhaustion mid-run** — already burned you twice | Checkpoint/resume in the runner; multi-day schedule; consider a paid tier for the eval window; log per-case quota state (you already do) |
| **Annotator recruitment stalls** | Start outreach in Week 1, not Week 4. Two annotators + one adjudicator. Have a backup institution. |
| **Ethics approval blocks the pilot** | Phase 4 is optional by design. Tier-1 clinician review is the fallback. Never let this gate the paper. |
| **Results are weaker than hoped** — the cascade may buy less than expected | This is *fine* and possibly more interesting. A rigorous negative or modest result with a clean Pareto curve is publishable; an unsupported strong claim is not. Decide now that you will report what you find. |
| **Scope creep back into product work** | Freeze at `paper-v1`. Product bugs go to a branch. Paper numbers come from the tag. |
| **Reviewer: "this is just RAG + NLI"** | Your defence is Phase 2: streaming-safe guardrailing, fusion drift, citation binding — three mechanisms with three dedicated evals. Without Phase 2 you have no defence. |

---

## 10. What to do this week

1. `git tag paper-v1` and push.
2. Add `EVAL_DETERMINISTIC` + seed control + `run_manifest` to the three main scorers.
3. Email NHRC about ethics-review scope for (a) clinician expert review and
   (b) a 20–30 user pilot. Ask which needs full review.
4. Email two medical schools about paid annotator recruitment.
5. Download K-QA; map 60 of its questions onto your strata.
6. Top up the two partial ingest manifests; freeze and record corpus stats.

Do not write a single sentence of the paper until Phase 1 produces numbers.
