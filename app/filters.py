"""Pre-retrieval filter builder (docs/IMPROVEMENTS.md §4.4).

Week 5 Phase 2 added the filter mechanism with static defaults.
Week 5 Phase 3 makes it intent-driven: if an intent dict is passed,
stage- and domain-specific overrides are applied on top of the Phase 2
base. When intent is None (classifier failed, or caller opts out),
the function returns the Phase 2 default.
"""
from typing import Any, Dict, Optional


def build_filter(
    question: str,
    intent: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Return filter kwargs for match_chunks_hybrid_filtered.

    Base (Phase 2):
      filter_country_scope=['NP','global']
          Nepal + international reference docs (WHO/NHS tag as 'global' or
          'uk','global'). NULL country_scope also admitted via SQL clause.
      filter_min_authority_tier=3
          Tier 1-3 only — excludes research and user-uploaded PDFs.

    Intent overrides (Phase 3, narrowed after v5 ablation):
      stage='condition' with a specific domain →
          filter_domains=[domain]. Condition queries are disease-centric
          and benefit from domain narrowing.
      stage='navigation' →
          filter_max_age_years=10. Care-pathway guidance is time-sensitive.
          Nepal docs without a publication_date are still admitted
          (null-permissive in SQL).

    intake / visit_prep / results keep the base filter. v5 data showed
    domain filtering on `results` regressed faithfulness (−74%); lab-value
    questions need the reranker's full candidate pool, not a pre-narrowed
    one. intake/visit_prep gold docs are cross-cutting.
    """
    base = {
        "filter_domains": None,
        "filter_country_scope": ["NP", "global"],
        "filter_min_authority_tier": 3,
        "filter_max_age_years": None,
    }
    if not intent:
        return base

    stage = intent.get("stage")
    domain = intent.get("domain")

    if stage == "condition" and domain and domain != "general":
        base["filter_domains"] = [domain]

    if stage == "navigation":
        base["filter_max_age_years"] = 10

    return base
