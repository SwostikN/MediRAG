-- Pre-retrieval metadata filter (docs/IMPROVEMENTS.md §4.4, Week 5 Phase 2).
--
-- Extends 005's RRF hybrid with optional metadata filters. 005 is kept
-- as-is so ablation snapshots (v3.5 unfiltered vs v4 filtered) stay
-- reproducible by routing to the two functions.
--
-- Filter semantics:
--   filter_domains               — require d.domains && filter_domains
--   filter_country_scope         — "glocal": match OR d.country_scope IS NULL
--                                  (NULL country_scope = global doc)
--   filter_min_authority_tier    — UPPER bound on authority_tier
--                                  (lower tier int = higher authority: 1=best)
--   filter_max_age_years         — age bound; NULL publication_date kept
--                                  (many WHO/NHS docs lack a pub_date)
--
-- All filter args default NULL → no-op. Same output shape as 005.

create or replace function public.match_chunks_hybrid_filtered(
  query_embedding            vector(768),
  query_text                 text,
  match_count                int default 30,
  candidate_count            int default 50,
  rrf_k                      int default 60,
  filter_domains             text[] default null,
  filter_country_scope       text[] default null,
  filter_min_authority_tier  int default null,
  filter_max_age_years       int default null
)
returns table (
  chunk_id             uuid,
  doc_id               uuid,
  ord                  int,
  content              text,
  section_heading      text,
  rrf_score            float,
  similarity           float,
  bm25_rank            int,
  doc_title            text,
  doc_source           text,
  doc_source_url       text,
  doc_authority_tier   int,
  doc_type             text,
  doc_publication_date date
)
language sql
stable
as $$
  with
  eligible as (
    select c.chunk_id, c.embedding, c.tsv
    from public.chunks c
    join public.documents d on d.doc_id = c.doc_id
    where c.embedding is not null
      and d.retracted = false
      and (filter_domains is null or d.domains && filter_domains)
      and (
        filter_country_scope is null
        or d.country_scope is null
        or d.country_scope && filter_country_scope
      )
      and (filter_min_authority_tier is null
           or d.authority_tier <= filter_min_authority_tier)
      and (
        filter_max_age_years is null
        or d.publication_date is null
        or d.publication_date >= current_date - make_interval(years => filter_max_age_years)
      )
  ),
  dense as (
    select
      chunk_id,
      row_number() over (order by embedding <=> query_embedding) as r,
      1 - (embedding <=> query_embedding) as sim
    from eligible
    order by embedding <=> query_embedding
    limit candidate_count
  ),
  lex as (
    select
      chunk_id,
      row_number() over (
        order by ts_rank_cd(tsv, plainto_tsquery('english', query_text)) desc
      ) as r
    from eligible
    where tsv @@ plainto_tsquery('english', query_text)
    limit candidate_count
  ),
  fused as (
    select chunk_id, sum(1.0 / (rrf_k + r)) as rrf
    from (
      select chunk_id, r from dense
      union all
      select chunk_id, r from lex
    ) u
    group by chunk_id
    order by rrf desc
    limit match_count
  )
  select
    f.chunk_id,
    c.doc_id,
    c.ord,
    c.content,
    c.section_heading,
    f.rrf::float as rrf_score,
    d.sim::float as similarity,
    l.r::int     as bm25_rank,
    doc.title             as doc_title,
    doc.source            as doc_source,
    doc.source_url        as doc_source_url,
    doc.authority_tier    as doc_authority_tier,
    doc.doc_type          as doc_type,
    doc.publication_date  as doc_publication_date
  from fused f
  join public.chunks c     on c.chunk_id = f.chunk_id
  join public.documents doc on doc.doc_id = c.doc_id
  left join dense d on d.chunk_id = f.chunk_id
  left join lex   l on l.chunk_id = f.chunk_id
  order by f.rrf desc;
$$;

grant execute on function public.match_chunks_hybrid_filtered(
  vector, text, int, int, int, text[], text[], int, int
) to anon, authenticated, service_role;
