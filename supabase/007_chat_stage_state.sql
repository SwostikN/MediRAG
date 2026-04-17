-- Week 7A: extend chat_sessions / chat_messages with stage-aware fields
-- and enforce per-user row-level security.
--
-- Safe to re-run: uses IF NOT EXISTS, DROP POLICY IF EXISTS, and idempotent
-- ENABLE RLS. Does not modify existing rows.

-- 1. Stage-1+ columns on chat_sessions
alter table public.chat_sessions
  add column if not exists current_stage  text not null default 'intake',
  add column if not exists intent_bucket  text,
  add column if not exists intake_summary text;

-- Allowed values for current_stage. Extend as later stages ship.
alter table public.chat_sessions
  drop constraint if exists chat_sessions_current_stage_check;
alter table public.chat_sessions
  add constraint chat_sessions_current_stage_check
  check (current_stage in ('intake', 'navigation', 'visit_prep', 'results', 'condition', 'closed'));

-- 2. Per-message metadata on chat_messages
alter table public.chat_messages
  add column if not exists stage    text,
  add column if not exists red_flag jsonb,
  add column if not exists sources  jsonb;

-- 3. Row-level security — users see/write only their own data
alter table public.chat_sessions enable row level security;
alter table public.chat_messages enable row level security;

drop policy if exists "chat_sessions_select_own" on public.chat_sessions;
drop policy if exists "chat_sessions_modify_own" on public.chat_sessions;
drop policy if exists "chat_messages_select_own" on public.chat_messages;
drop policy if exists "chat_messages_modify_own" on public.chat_messages;

create policy "chat_sessions_select_own" on public.chat_sessions
  for select
  using (user_id = auth.uid());

create policy "chat_sessions_modify_own" on public.chat_sessions
  for all
  using (user_id = auth.uid())
  with check (user_id = auth.uid());

create policy "chat_messages_select_own" on public.chat_messages
  for select
  using (
    exists (
      select 1 from public.chat_sessions s
      where s.id = chat_messages.session_id and s.user_id = auth.uid()
    )
  );

create policy "chat_messages_modify_own" on public.chat_messages
  for all
  using (
    exists (
      select 1 from public.chat_sessions s
      where s.id = chat_messages.session_id and s.user_id = auth.uid()
    )
  )
  with check (
    exists (
      select 1 from public.chat_sessions s
      where s.id = chat_messages.session_id and s.user_id = auth.uid()
    )
  );

-- 4. Indexes for the sidebar query and the per-session message fetch
create index if not exists chat_sessions_user_updated_idx
  on public.chat_sessions (user_id, updated_at desc);

create index if not exists chat_messages_session_created_idx
  on public.chat_messages (session_id, created_at);
