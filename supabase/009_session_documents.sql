-- Week 8: per-session uploaded documents (lab reports, research papers,
-- other) + session-private chunks for retrieval. Patient lab values must
-- never reach the shared `chunks` corpus, so this lives in a separate
-- pair of tables with strict per-user RLS.
--
-- Safe to re-run: uses IF NOT EXISTS / DROP POLICY IF EXISTS.

-- 1. attached_documents pointer on chat_sessions
--    Cheap quick-reference list shown in the sidebar / chat header. Source
--    of truth for the actual chunks is session_documents below.
alter table public.chat_sessions
  add column if not exists attached_documents jsonb not null default '[]'::jsonb;

-- 2. session_documents — one row per uploaded file, scoped to a session
create table if not exists public.session_documents (
  id            uuid primary key default gen_random_uuid(),
  session_id    uuid not null references public.chat_sessions(id) on delete cascade,
  user_id       uuid not null references public.user_profiles(id) on delete cascade,
  filename      text not null,
  doc_type      text not null check (doc_type in ('lab_report','research_paper','other')),
  content_hash  text,
  page_count    int,
  byte_size     int,
  uploaded_at   timestamptz not null default timezone('utc', now())
);

create index if not exists session_documents_session_idx
  on public.session_documents (session_id, uploaded_at desc);
create index if not exists session_documents_user_idx
  on public.session_documents (user_id, uploaded_at desc);
-- Hash dedup within a session (re-upload of the same PDF returns the same row)
create unique index if not exists session_documents_session_hash_uniq
  on public.session_documents (session_id, content_hash)
  where content_hash is not null;

-- 3. session_chunks — content + embedding, FK to session_documents
--    Mirrors public.chunks shape so the same retrieval/rerank pipeline
--    can merge corpus chunks and session chunks before reranking.
create table if not exists public.session_chunks (
  chunk_id        uuid primary key default gen_random_uuid(),
  session_doc_id  uuid not null references public.session_documents(id) on delete cascade,
  ord             int not null,
  content         text not null,
  section_heading text,
  token_count     int,
  embedding       vector(768),
  tsv             tsvector generated always as (to_tsvector('english', content)) stored
);

create index if not exists session_chunks_doc_ord_idx
  on public.session_chunks (session_doc_id, ord);
create index if not exists session_chunks_embedding_idx
  on public.session_chunks using ivfflat (embedding vector_cosine_ops) with (lists = 50);
create index if not exists session_chunks_tsv_idx
  on public.session_chunks using gin (tsv);

-- 4. RLS — both tables are private to the session owner
alter table public.session_documents enable row level security;
alter table public.session_chunks    enable row level security;

drop policy if exists "session_documents_select_own" on public.session_documents;
create policy "session_documents_select_own" on public.session_documents
  for select
  using (user_id = auth.uid());

drop policy if exists "session_documents_modify_own" on public.session_documents;
create policy "session_documents_modify_own" on public.session_documents
  for all
  using (user_id = auth.uid())
  with check (user_id = auth.uid());

drop policy if exists "session_chunks_select_own" on public.session_chunks;
create policy "session_chunks_select_own" on public.session_chunks
  for select
  using (
    exists (
      select 1
      from public.session_documents d
      where d.id = session_chunks.session_doc_id
        and d.user_id = auth.uid()
    )
  );

drop policy if exists "session_chunks_modify_own" on public.session_chunks;
create policy "session_chunks_modify_own" on public.session_chunks
  for all
  using (
    exists (
      select 1
      from public.session_documents d
      where d.id = session_chunks.session_doc_id
        and d.user_id = auth.uid()
    )
  )
  with check (
    exists (
      select 1
      from public.session_documents d
      where d.id = session_chunks.session_doc_id
        and d.user_id = auth.uid()
    )
  );

-- 5. match_session_chunks — RRF hybrid retrieval restricted to ONE session.
--    Same shape as public.match_chunks_hybrid (005) but joined to
--    session_documents so chunks from other sessions / users are
--    invisible. RLS still applies; the explicit p_session_id is a
--    second line of defence + the join target for ordering.
create or replace function public.match_session_chunks(
  p_session_id      uuid,
  query_embedding   vector(768),
  query_text        text,
  match_count       int default 10,
  candidate_count   int default 30,
  rrf_k             int default 60
)
returns table (
  chunk_id        uuid,
  session_doc_id  uuid,
  ord             int,
  content         text,
  section_heading text,
  rrf_score       float,
  similarity      float,
  bm25_rank       int,
  doc_filename    text,
  doc_type        text
)
language sql stable
as $$
  with dense as (
    select
      c.chunk_id,
      c.session_doc_id,
      c.ord,
      c.content,
      c.section_heading,
      1 - (c.embedding <=> query_embedding) as similarity,
      row_number() over (order by c.embedding <=> query_embedding) as dense_rank
    from public.session_chunks c
    join public.session_documents d on d.id = c.session_doc_id
    where d.session_id = p_session_id
      and c.embedding is not null
    order by c.embedding <=> query_embedding
    limit candidate_count
  ),
  lex as (
    select
      c.chunk_id,
      row_number() over (
        order by ts_rank_cd(c.tsv, plainto_tsquery('english', query_text)) desc
      ) as bm25_rank
    from public.session_chunks c
    join public.session_documents d on d.id = c.session_doc_id
    where d.session_id = p_session_id
      and c.tsv @@ plainto_tsquery('english', query_text)
    limit candidate_count
  ),
  fused as (
    select
      coalesce(dense.chunk_id, lex.chunk_id) as chunk_id,
      coalesce(1.0 / (rrf_k + dense.dense_rank), 0.0)
        + coalesce(1.0 / (rrf_k + lex.bm25_rank), 0.0) as rrf_score,
      dense.similarity,
      lex.bm25_rank
    from dense
    full outer join lex on dense.chunk_id = lex.chunk_id
  )
  select
    f.chunk_id,
    c.session_doc_id,
    c.ord,
    c.content,
    c.section_heading,
    f.rrf_score,
    f.similarity,
    f.bm25_rank,
    d.filename as doc_filename,
    d.doc_type
  from fused f
  join public.session_chunks c    on c.chunk_id = f.chunk_id
  join public.session_documents d on d.id = c.session_doc_id
  where d.session_id = p_session_id
  order by f.rrf_score desc
  limit match_count;
$$;
