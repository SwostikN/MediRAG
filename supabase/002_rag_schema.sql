-- MediRAG RAG schema (per docs/IMPROVEMENTS.md §4.1).
-- Additive to create_users.sql. Safe to re-run.

create extension if not exists pgcrypto;
create extension if not exists vector;

-- ---------------------------------------------------------------------------
-- documents: curated corpus metadata (WHO / NHS / CDC / MoHP / society / ...)
-- ---------------------------------------------------------------------------
create table if not exists public.documents (
  doc_id            uuid primary key default gen_random_uuid(),
  title             text not null,
  source            text not null,
  source_url        text,
  authority_tier    int  not null check (authority_tier between 1 and 5),
  doc_type          text not null check (doc_type in ('patient-ed','clinical-guideline','reference')),
  publication_date  date,
  last_revised_date date,
  language          text default 'en',
  domains           text[],
  population        text[],
  country_scope     text[],
  retracted         boolean not null default false,
  ingested_at       timestamptz not null default timezone('utc', now())
);

create index if not exists documents_domains_idx       on public.documents using gin (domains);
create index if not exists documents_country_scope_idx on public.documents using gin (country_scope);
create index if not exists documents_population_idx    on public.documents using gin (population);
create index if not exists documents_authority_idx     on public.documents (authority_tier);
create index if not exists documents_retracted_idx     on public.documents (retracted);

-- ---------------------------------------------------------------------------
-- chunks: chunked content + dense + lexical indexes
-- ---------------------------------------------------------------------------
create table if not exists public.chunks (
  chunk_id        uuid primary key default gen_random_uuid(),
  doc_id          uuid not null references public.documents(doc_id) on delete cascade,
  ord             int  not null,
  content         text not null,
  section_heading text,
  token_count     int,
  embedding       vector(768),
  tsv             tsvector generated always as (to_tsvector('english', content)) stored
);

create index if not exists chunks_doc_id_ord_idx on public.chunks (doc_id, ord);
create index if not exists chunks_embedding_idx  on public.chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);
create index if not exists chunks_tsv_idx        on public.chunks using gin (tsv);

-- ---------------------------------------------------------------------------
-- citations: per-message retrieved chunks + per-chunk score breakdown
-- ---------------------------------------------------------------------------
create table if not exists public.citations (
  id              uuid primary key default gen_random_uuid(),
  message_id      uuid not null references public.chat_messages(id) on delete cascade,
  chunk_id        uuid not null references public.chunks(chunk_id),
  rank            int,
  rerank_score    float,
  freshness_score float,
  authority_score float,
  final_score     float,
  created_at      timestamptz not null default timezone('utc', now())
);

create index if not exists citations_message_id_rank_idx on public.citations (message_id, rank);
create index if not exists citations_chunk_id_idx        on public.citations (chunk_id);

-- ---------------------------------------------------------------------------
-- user_reports: uploaded patient lab PDFs (Stage 4 — Results explainer)
-- ---------------------------------------------------------------------------
create table if not exists public.user_reports (
  id               uuid primary key default gen_random_uuid(),
  user_id          uuid not null references public.user_profiles(id) on delete cascade,
  filename         text,
  extracted_values jsonb,
  uploaded_at      timestamptz not null default timezone('utc', now())
);

create index if not exists user_reports_user_id_uploaded_at_idx
  on public.user_reports (user_id, uploaded_at desc);

-- ---------------------------------------------------------------------------
-- RLS
-- documents / chunks / citations are server-managed corpus + audit tables.
-- Anon clients can read corpus; only service role writes.
-- citations are also read-restricted — they reference chat messages, which
-- are private — so we mirror the chat_messages owner-based policy.
-- user_reports are private to the owner (mirrors chat_sessions).
-- ---------------------------------------------------------------------------
alter table public.documents    enable row level security;
alter table public.chunks       enable row level security;
alter table public.citations    enable row level security;
alter table public.user_reports enable row level security;

drop policy if exists "Documents are readable by everyone" on public.documents;
create policy "Documents are readable by everyone"
on public.documents
for select
using (retracted = false);

drop policy if exists "Chunks are readable by everyone" on public.chunks;
create policy "Chunks are readable by everyone"
on public.chunks
for select
using (true);

drop policy if exists "Citations are viewable by message owner" on public.citations;
create policy "Citations are viewable by message owner"
on public.citations
for select
using (
  exists (
    select 1
    from public.chat_messages m
    join public.chat_sessions s on s.id = m.session_id
    where m.id = citations.message_id
      and s.user_id = auth.uid()
  )
);

drop policy if exists "User reports are viewable by owner" on public.user_reports;
create policy "User reports are viewable by owner"
on public.user_reports
for select
using (auth.uid() = user_id);

drop policy if exists "User reports are insertable by owner" on public.user_reports;
create policy "User reports are insertable by owner"
on public.user_reports
for insert
with check (auth.uid() = user_id);

drop policy if exists "User reports are deletable by owner" on public.user_reports;
create policy "User reports are deletable by owner"
on public.user_reports
for delete
using (auth.uid() = user_id);
