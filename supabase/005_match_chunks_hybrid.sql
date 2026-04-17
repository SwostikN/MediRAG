-- Hybrid retrieval: dense (MedCPT cosine) + lexical (BM25-ish via tsvector)
-- fused with Reciprocal Rank Fusion.
--
-- Reference: Cormack, Clarke, Büttcher, 2009. Reciprocal Rank Fusion
--   outperforms Condorcet and Individual Rank Learning Methods. SIGIR.
--
--   RRF(d) = Σ_i  1 / (k + rank_i(d)),   k = 60 (Cormack et al. default)
--
-- Called via PostgREST: POST /rest/v1/rpc/match_chunks_hybrid.
-- Week 5 will add the §4.4 pre-retrieval filter (domain / population /
-- authority / freshness); for now the candidate set is
-- "all non-retracted chunks with embeddings."

create or replace function public.match_chunks_hybrid(
  query_embedding vector(768),
  query_text      text,
  match_count     int default 30,
  candidate_count int default 50,
  rrf_k           int default 60
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
    where c.embedding is not null and d.retracted = false
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
  join public.chunks c   on c.chunk_id   = f.chunk_id
  join public.documents doc on doc.doc_id = c.doc_id
  left join dense d on d.chunk_id = f.chunk_id
  left join lex   l on l.chunk_id = f.chunk_id
  order by f.rrf desc;
$$;

grant execute on function public.match_chunks_hybrid(vector, text, int, int, int)
  to anon, authenticated, service_role;
