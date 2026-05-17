-- Persist the parsed lab-marker list with each Stage-4 assistant message
-- so the marker table re-renders on session reload. The array is also
-- present in user_lab_markers, but keyed to the upload, not the message —
-- joining back would need timestamp disambiguation when a session has
-- multiple uploads. Storing the resolved list as JSONB on the message
-- itself is simpler and matches how `sources` is already shaped.
--
-- Safe to re-run: IF NOT EXISTS.

alter table public.chat_messages
  add column if not exists markers jsonb;
