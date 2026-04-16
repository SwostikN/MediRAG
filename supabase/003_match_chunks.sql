-- match_chunks: dense (cosine) similarity retrieval over chunks.
-- Called from Python via Supabase REST: POST /rest/v1/rpc/match_chunks.
-- Hybrid (dense + BM25 via RRF) lands in Week 4 as match_chunks_hybrid.

create or replace function public.match_chunks(
  query_embedding vector(768),
  match_count     int default 50
)
returns table (
  chunk_id        uuid,
  doc_id          uuid,
  ord             int,
  content         text,
  section_heading text,
  similarity      float,
  doc_title       text,
  doc_source      text,
  doc_source_url  text,
  doc_authority_tier int,
  doc_type        text,
  doc_publication_date date
)
language sql
stable
as $$
  select
    c.chunk_id,
    c.doc_id,
    c.ord,
    c.content,
    c.section_heading,
    1 - (c.embedding <=> query_embedding) as similarity,
    d.title             as doc_title,
    d.source            as doc_source,
    d.source_url        as doc_source_url,
    d.authority_tier    as doc_authority_tier,
    d.doc_type          as doc_type,
    d.publication_date  as doc_publication_date
  from public.chunks c
  join public.documents d on d.doc_id = c.doc_id
  where c.embedding is not null
    and d.retracted = false
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

grant execute on function public.match_chunks(vector, int) to anon, authenticated, service_role;
