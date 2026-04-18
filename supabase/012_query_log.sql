-- P2.11: per-response audit log.
--
-- One row per /query (or /query/stream) response. Stores the minimum
-- needed to reproduce a conversation when a user complains about what
-- MediRAG said:
--   - the query
--   - which composer stage ran
--   - which chunks were retrieved (chunk_ids only — the chunk text
--     itself lives in public.chunks and public.session_chunks)
--   - a hash of the prompt sent to the LLM (for drift detection; full
--     prompts are not stored because they embed retrieved chunk text
--     and bloat the table)
--   - the final answer shown to the user
--   - citations shown, as jsonb
--   - refusal / red-flag flags
--
-- NOT stored: the full prompt, the LLM provider's raw response,
-- per-chunk rerank scores. Those are reproducible from the logged
-- chunk_ids + the current corpus; keeping them here would quadruple
-- table size for marginal value.
--
-- Retention: keep indefinitely for now (rows are small, ~1 KB each).
-- Revisit when volume grows past ~10M rows or when DPA posture
-- requires a TTL.
--
-- Safe to re-run.

create table if not exists public.query_log (
  id                      uuid primary key default gen_random_uuid(),
  user_id                 text,
  session_id              text,
  logged_at               timestamptz not null default timezone('utc', now()),

  -- Which composer ran. Free text rather than enum so new stages
  -- ('classifier', 'lab_report', ...) don't need a migration.
  stage                   text,

  -- User-facing fields
  query_text              text,
  response_text           text,
  citations               jsonb,

  -- Retrieval provenance. UUIDs stored as text[] because session_chunks
  -- and chunks live in different tables; one column keeps both kinds.
  retrieved_chunk_ids     text[],
  prompt_hash             text,

  -- Safety outcome flags
  refusal_triggered       boolean not null default false,
  refusal_reason          text,
  red_flag_fired          boolean not null default false,
  red_flag_rule_id        text
);

create index if not exists query_log_user_time_idx
  on public.query_log (user_id, logged_at desc);
create index if not exists query_log_session_time_idx
  on public.query_log (session_id, logged_at desc);
create index if not exists query_log_red_flag_idx
  on public.query_log (logged_at desc) where red_flag_fired;
create index if not exists query_log_refusal_idx
  on public.query_log (logged_at desc) where refusal_triggered;

-- RLS: server-side writes only. No user-facing reads yet; admin
-- debugging goes through the service role key directly.
alter table public.query_log enable row level security;
