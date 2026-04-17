"""Pre-retrieval filter builder (docs/IMPROVEMENTS.md §4.4, Week 5 Phase 2).

Phase 2 scope is mechanism + a conservative default. Phase 3 will replace
the body of build_filter() with an intent-classifier-driven policy that
sets (stage, domain) per query.
"""
from typing import Any, Dict


def build_filter(question: str) -> Dict[str, Any]:
    """Return filter kwargs for match_chunks_hybrid_filtered.

    Defaults:
      filter_country_scope=['NP','global']
          Nepal-specific + international reference docs. The seed
          manifest tags WHO as ['global'] and NHS as ['uk','global'],
          so both tokens are required to keep the international corpus
          in the retrieval pool. Any doc tagged with only another
          country (e.g. ['in']) is excluded. NULL country_scope is
          also admitted via the glocal clause in the SQL.
      filter_min_authority_tier=3
          Tier 1-3 only. Excludes tier 4 (research) and tier 5
          (user-uploaded PDFs) — user uploads must not leak across
          accounts as sources.
      filter_domains=None, filter_max_age_years=None
          Not restricted in Phase 2.
    """
    return {
        "filter_domains": None,
        "filter_country_scope": ["NP", "global"],
        "filter_min_authority_tier": 3,
        "filter_max_age_years": None,
    }
