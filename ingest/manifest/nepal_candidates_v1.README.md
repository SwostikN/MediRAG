# Nepal candidates v1 — user review required before ingestion

**File:** `nepal_candidates_v1.jsonl` (73 URLs, English-only, `country_scope=['NP']`)

These URLs were extracted by crawling the 17 landing pages you provided,
skipping login portals / dashboards / navigation chrome / Nepali-language
pages, and filtering for content-bearing endpoints (program pages,
guidelines, fact sheets, PDFs, reports, OPD/service info).

## Distribution

| Source        | n  | Authority tier | Note                                    |
|---------------|----|----------------|-----------------------------------------|
| EDCD          | 29 | 1              | Disease control programs (biggest win)  |
| DoHS          |  9 | 1              | Annual Health Reports + fact sheets     |
| NPHL          |  7 | 2              | National Public Health Laboratory       |
| WHO Nepal     |  7 | 1              | Nepal-specific WHO publications         |
| MoHP          |  6 | 1              | National health policy                  |
| DDA           |  5 | 2              | Drug regulator (Essential Drug List)    |
| CSH           |  5 | 3              | Civil Service Hospital (care-nav)       |
| BPKIHS        |  4 | 3              | BP Koirala Institute (care-nav)         |
| Bir Hospital  |  1 | 3              | Only English homepage survived filter   |

## Authority-tier key

- **Tier 1:** National policy / disease control authorities (MoHP, EDCD,
  DoHS, WHO Nepal). Highest-weight sources for care-navigation and
  disease information.
- **Tier 2:** Specialized national bodies (NPHL for lab guidelines,
  DDA for drug regulation).
- **Tier 3:** Teaching hospitals (factual care-navigation info:
  departments, OPD schedules, service fees).
- **Tier 4:** Research bodies / journals (none in v1 — NHRC is down).

## Coverage wins (what this unblocks for the eval)

Current gold-set stages stuck at floor (see DOCUMED §4.3):
- `intake`: 0.000 Recall@5 — expects care-pathway / symptom-intake sources.
- `navigation`: 0.100 — expects Nepal care-tier routing (Health Post →
  PHCC → District Hospital → Zonal → Central).

With this corpus, `intake` and `navigation` should find hits in:
- DoHS `National Directory of Patient Referrals, 2082`
- MoHP `Hospital Service Improvement Procedures, 2082`
- EDCD section pages (per-disease symptom/epidemiology info)
- BPKIHS / CSH / Bir Hospital OPD + department pages

## Known gaps (failed endpoints — flagged, not fixed)

| URL                            | Failure mode                      | Action                     |
|--------------------------------|-----------------------------------|----------------------------|
| `http://nhrc.gov.np`           | Malformed HTTP response (server)  | Drop v1; retry later       |
| `https://tuth.edu.np`          | ECONNREFUSED                      | Drop v1; site may be down  |
| `https://pahs.edu.np`          | Empty response                    | Drop v1                    |
| `https://www.unicef.org/nepal` | 403 (bot detection)               | User-curated URLs only     |
| `http://vaccine.mohp.gov.np`   | 500 error                         | Drop v1                    |

## Review before ingestion

Before running `python -m ingest.run --manifest ingest/manifest/nepal_candidates_v1.jsonl`:

1. **Skim titles.** Drop any you don't want (prefix the line with `//`
   or delete it).
2. **Spot-check 3–5 URLs** in a browser — make sure they actually load
   substantive English content. Some `mohp.gov.np/content/XXX/*`
   URLs may redirect to a CDN or require JavaScript to render content;
   if they return a title-only shell, the ingester will log `short_doc`
   and skip them.
3. **Validate domain tags.** The `domains` field drives pre-retrieval
   filtering (Week 5 §4.4 in IMPROVEMENTS.md). Adjust if a tag feels
   off — e.g., if you want all BPKIHS pages tagged `["care-navigation",
   "teaching-hospital"]`, edit in bulk.

## Ingestion (once you've approved)

```bash
python -m ingest.run --manifest ingest/manifest/nepal_candidates_v1.jsonl
```

Run after Week 5's pre-retrieval filter migration (006) lands so the
new `authority_tier` / `country_scope` / `domains` tags are actually
used at retrieval time.

## What's NOT in this manifest (deliberately)

- Login-gated portals (EWARS, IHMIS, HMIS dashboards, QR vaccine cert,
  ambulance booking, labreport.merodoctor.com) — auth required + these
  are transactional systems, not reference content.
- Tool/form downloads (thesis templates, ADR reporting forms,
  research proposal forms) — not patient-facing.
- Nepali-language documents — permanent non-goal (IMPROVEMENTS.md
  §8.Week 5 scope lock).
- World Bank Nepal — the crawl returned zero health-specific pages;
  their Nepal presence is fiscal/development, not clinical.
- DDA category landing pages with no specific PDFs yet — the top-5
  included here are the most patient-relevant categories; drill-down
  to specific PDFs once you review them.
