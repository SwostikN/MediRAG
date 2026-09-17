# Phase 2 ingest manifest — WHO + MoHP expansion

Drafted 2026-04-21 from `docs/GOLD_REWRITE_SPEC.md` Phase 2 shortlist. Targets the 88 INGEST-CANDIDATE entries held out of scoring during the 2026-04-20 gold rewrite.

## Phase 2-A — ready for live ingest (22 URLs, ~165 chunks)

All URLs fetched + parsed cleanly in dry-run on 2026-04-21. See [phase2_who_mohp_v1.jsonl](./phase2_who_mohp_v1.jsonl).

**22 substantial WHO fact sheets** (5,000–13,000 chars each) — meningitis, sepsis, suicide, HIV, cervical cancer, diarrhoeal disease (ORS), household air pollution (CO), road traffic injuries, drowning, adolescent mental health, anxiety, epilepsy, immunization coverage, pneumonia, stillbirth, adolescent health, STIs, family planning, preterm birth, blindness, deafness, dementia.

### Why not the 5 WHO publication landing pages

The earlier draft also included 5 `/publications/i/item/` landing pages (mhGAP 2.0, ANC recommendations, IMCI chart booklet, HEARTS technical package, IMAI chronic HIV care). Dropped because:

1. **Convention break** — zero other manifests (231 docs across seed_v1 / primary_care_v1 / lab_explainers_v1 / care_pathway_v1 / nepal_candidates_v1) use `/publications/` URLs. The existing corpus is full-content only.
2. **Hallucination risk** — each landing page yields only 1,300–2,200 chars (WHO-authored abstract + metadata). That is enough for Phase 3A inline-citation binding to emit a `[src:N]` marker pointing at a chunk with no clinical content behind it. Given the Phase 1-3 sprint was built specifically to fix unsupported-claim hallucinations, introducing thin-anchor chunks works against that effort.
3. **Impact gap is small** — token-level title matches on "WHO mhGAP" / "WHO IMCI" / etc. would make the scorer flip ~5 gold entries, but user-visible behavior wouldn't improve because the retrieval behind those matches is hollow. Phase 2-B should land the actual iris.who.int PDFs.

**Cost**: no Cohere quota. MedCPT-only. Duplicate detection via `skip_existing=True` on `find_document_by_url`. Safe to re-run.

**How to ingest** (requires user approval):
```
python -m ingest.run --manifest ingest/manifest/phase2_who_mohp_v1.jsonl
```

## Phase 2-B — needs URL discovery (12 URLs, 0 chunks)

These failed dry-run and need human URL verification before they can be ingested. Do **not** add to the Phase 2-A manifest until a working URL is confirmed.

### WHO — HTTP 404 (URL path wrong or page retired)
| Target | Broken URL | Likely fix |
|---|---|---|
| Newborn mortality | `who.int/news-room/fact-sheets/detail/newborn-death-and-illness` | try `newborn-mortality` slug |
| Chronic kidney disease | `who.int/news-room/fact-sheets/detail/chronic-kidney-disease` | may be under NCD topic page; no standalone fact sheet |
| Nutrition in emergencies | `who.int/news-room/fact-sheets/detail/nutrition-in-complex-emergencies` | try `malnutrition` or `childhood-nutrition` |
| WHO PEN primary-care package | `/publications/i/item/package-of-essential-noncommunicable...` | URL slug is stale; search iris.who.int for "PEN" |
| ORS / WHO-UNICEF joint statement | `/publications/i/item/9789240021723` | ISBN probably wrong; verify on iris |
| SEARO snakebite guidelines | `/publications/i/item/9789241549219` | ISBN probably wrong; SEARO regional office URL |

### Nepal gov — HTTP 404 / 500 / SSL
| Target | Broken URL | Triage |
|---|---|---|
| MoHP English portal | `mohp.gov.np/en` | try `mohp.gov.np/` (root) or `mohp.gov.np/en/` with trailing slash |
| DoHS Patient Referral Directory | `giwmscdnone.gov.np/content/86/...` | URL in existing nepal_candidates_v1 also 404s; site may have moved |
| EDCD Rabies Control Section | `edcd.gov.np/section/rabies-control-section` | 500 — server error; retry, or try different slug |
| EDCD Immunization Section | `edcd.gov.np/section/immunization-section` | 500 — retry, or the section may be named "epi" / "national-immunization-program" |
| HRA Altitude guidelines | `himalayanrescue.org.np/altitude-sickness/` | HRA site structure changed; try root or "aguidelines" |
| NCASC (HIV/STD) | `ncasc.gov.np/` | SSL cert mismatch on prod. Works in browser (cert error ignored); fetch.py needs verify=False escape hatch |

### Still missing from the spec shortlist (never attempted — too fragmented)
- WHO IMAI district clinician manual (multiple subchapters)
- WHO IMCI young-infant, ear, diarrhoea subchapters
- WHO mhGAP humanitarian intervention guide
- WHO scrub typhus, leptospirosis (no standalone fact sheets; publications only)
- WHO stroke guidance (no standalone fact sheet)
- WHO pediatric anemia thresholds
- WHO child development milestones
- MoHP NCD PEN protocol (clinician-facing PDF, not the EDCD landing page)
- MoHP National Immunization Schedule (PDF)
- MoHP dengue/cholera/safe-motherhood SOPs (specific PDFs)
- FPAN — `fpan.org/` landing was 781 chars, below useful threshold; need an interior page

## Impact expectation

With Phase 2-A ingested, the following gold INGEST entries should flip to matchable (conservative lower bound, per spec section 11.22):
- **Mental health** (suicide, mhGAP references): ~5 rows — cd-037, cd-038, in-029, nv-017, nv-040
- **Immunization / child health** (EPI, IMCI-adjacent): ~3 rows — vp-028, cd-046, nv-025
- **STIs / sexual health**: ~2 rows — nv-046
- **Environmental / CO**: ~2 rows — in-012, nv-018
- **Meningitis / sepsis**: ~3 rows — in-011, nv-011, nv-019
- **Cervical cancer**: 1 row — cd-047
- **Diarrhoea / ORS**: ~2 rows — in-026, nv-035

Rough estimate: **15–20 of the 88 INGEST entries flip to matchable** post Phase 2-A. Phase 2-B (iris PDFs + Nepal gov) is needed to cover the remaining ~60–70.
