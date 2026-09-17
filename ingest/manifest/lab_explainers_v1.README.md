# lab_explainers_v1

Corpus manifest for **Stage 4 (lab-results explainer)**. Feeds the shared
library with per-marker explainer content so that when
`_handle_lab_report` queries `_retrieve_ranked("TSH")`,
`_retrieve_ranked("HbA1c")` etc., the rerank path has strong on-topic
material to return (rerank_score > 0.4 to clear the refusal gate).

## Why separate from seed_v1 / care_pathway_v1

- `seed_v1` = condition-level patient-ed (hypertension, diabetes, ...).
- `care_pathway_v1` = emergency / when-to-see-a-doctor pages.
- `lab_explainers_v1` = **test-level** explainer pages. A lab report
  question (“what does my TSH mean?”) needs a page whose subject IS the
  test, not a condition page where the test is mentioned in passing.

Keeping it in a separate manifest means we can re-ingest / version it
without touching the other two.

## Source selection

Two tiers, both patient-ed, both high-authority:

- **NHS (authority_tier=1)** — UK state-level patient education. Used
  for the “Diagnosis” subpages of the relevant conditions, which is
  where NHS talks about the actual blood tests. Very stable URLs.
- **Testing.com (authority_tier=2)** — formerly Lab Tests Online,
  maintained by the American Association for Clinical Chemistry.
  Authoritative test-specific pages where the entire page is about one
  marker / panel (stronger rerank signal per marker than NHS’s
  condition pages).

Tier 2 for Testing.com (not 1) because NHS is a government source with
a national clinical remit, while Testing.com is a professional
association’s consumer site — both high quality, but the user’s
authority ranking already assigns the highest tier to WHO/NHS-style
sources.

## Marker coverage

The 16 canonical markers in `app.stages.results._MARKER_ALIASES` are
covered by the URLs below (most pages cover multiple markers):

| Marker                 | Covered by                                               |
| ---------------------- | -------------------------------------------------------- |
| TSH, FT4, FT3          | NHS hypothyroidism/hyperthyroidism + Testing.com TSH, FT4|
| HbA1c, FBS             | NHS type-2-diabetes/diagnosis + Testing.com HbA1c        |
| Hb                     | NHS iron-deficiency + Testing.com CBC                    |
| Ferritin               | NHS iron-deficiency + Testing.com ferritin               |
| LDL, HDL, TG, Total C. | NHS high-cholesterol/diagnosis + Testing.com lipid panel |
| ALT, AST               | Testing.com liver panel                                  |
| Creatinine             | NHS kidney-disease + Testing.com creatinine              |
| Vitamin D              | NHS vitamins-and-minerals/vitamin-d                      |
| Vitamin B12            | NHS vitamin-b12-or-folate-deficiency-anaemia/diagnosis   |

## How to ingest

Dry-run first to flush dead URLs without writing to Supabase:

    python -m ingest.run \
      --manifest ingest/manifest/lab_explainers_v1.jsonl \
      --dry-run

Then the real run:

    python -m ingest.run \
      --manifest ingest/manifest/lab_explainers_v1.jsonl

`find_document_by_url` already skips previously-ingested URLs, so this
is safe to re-run.

## Reference ranges

All ranges surfaced in the UI come from the **uploaded lab report
itself** (the lab’s own reference ranges, printed on the PDF). The
corpus is used ONLY for the explainer prose around those ranges —
“what does it mean when TSH is high”, not “what is the normal range”.
Lab-specific reference ranges vary by assay and population, and we
deliberately defer to the ordering lab rather than baking a single
global range into the corpus.
