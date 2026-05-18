-- Persist the citation list with each assistant message so the SOURCES
-- chip strip re-renders on session reload. Previously sources lived only
-- on the streamed React state (pendingSources -> setMessages); switching
-- to another chat and back dropped them because loadConversationMessages
-- restores from chat_messages, which had no sources column.
--
-- Stored as JSONB to match the shape already used in-flight by the
-- /query/stream `sources` event and by upload_document for lab reports.
--
-- Safe to re-run: IF NOT EXISTS.

alter table public.chat_messages
  add column if not exists sources jsonb;
