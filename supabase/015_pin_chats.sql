-- Pin chats feature. Adds a nullable pinned_at timestamp to
-- chat_sessions. NULL = not pinned; NOT NULL = pinned at that time.
-- Sidebar orders pinned sessions first (DESC by pinned_at, most
-- recent pin at top), then unpinned (DESC by updated_at).
--
-- RLS inherits from the existing chat_sessions policies (owner can
-- UPDATE their own row) — no new policy needed.

alter table if exists public.chat_sessions
  add column if not exists pinned_at timestamptz;

-- Partial index over pinned rows only. Unpinned rows (the majority)
-- stay out of the index so inserts/updates on non-pin fields don't
-- pay the index maintenance cost.
create index if not exists chat_sessions_pinned_at_idx
  on public.chat_sessions (pinned_at desc)
  where pinned_at is not null;
