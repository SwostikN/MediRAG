-- Week 8 follow-up: per-user, per-marker lab values extracted from
-- uploaded reports.
--
-- Why a separate table instead of a JSONB column on session_documents?
--   - Longitudinal queries: "show me TSH over the last 6 months" is one
--     SQL line with this shape; awkward with JSONB.
--   - Per-marker indexes (user_id, marker_name, taken_at) make trend
--     pulls fast even at scale.
--   - Trends are the natural follow-up feature; retrofitting JSONB →
--     relational later is a real data migration.
--
-- Provenance: every marker FKs to session_documents.id so we can always
-- trace a value back to the exact uploaded PDF (and therefore the exact
-- session) it came from.
--
-- Safe to re-run: uses IF NOT EXISTS / DROP POLICY IF EXISTS.

create table if not exists public.user_lab_markers (
  id              uuid primary key default gen_random_uuid(),
  user_id         uuid not null references public.user_profiles(id) on delete cascade,
  session_doc_id  uuid not null references public.session_documents(id) on delete cascade,
  -- Canonical name from app/stages/results.py _MARKER_ALIASES
  -- (TSH, FT4, HbA1c, LDL, etc.). Kept as text rather than enum so the
  -- app can add new markers without a migration.
  marker_name     text not null,
  value           numeric not null,
  unit            text not null,
  -- Reference range printed on the report, verbatim. NULL when the
  -- report omitted one. Stored as text because lab formats vary
  -- ("0.4 - 4.0", "<5.7", ">40").
  reference_range text,
  -- "low" | "normal" | "high" | "unknown" — derived in the app from
  -- value vs reference_range. Persisted (denormalised) so trend
  -- queries don't have to re-parse the range string.
  status          text not null check (status in ('low','normal','high','unknown')),
  -- When the sample was collected, parsed from the report when
  -- available; falls back to upload time. Required for time-series.
  taken_at        timestamptz not null default timezone('utc', now()),
  created_at      timestamptz not null default timezone('utc', now())
);

create index if not exists user_lab_markers_user_marker_taken_idx
  on public.user_lab_markers (user_id, marker_name, taken_at desc);
create index if not exists user_lab_markers_session_doc_idx
  on public.user_lab_markers (session_doc_id);

-- RLS: a marker is visible only to the user it belongs to.
alter table public.user_lab_markers enable row level security;

drop policy if exists "user_lab_markers_select_own" on public.user_lab_markers;
create policy "user_lab_markers_select_own" on public.user_lab_markers
  for select
  using (user_id = auth.uid());

drop policy if exists "user_lab_markers_modify_own" on public.user_lab_markers;
create policy "user_lab_markers_modify_own" on public.user_lab_markers
  for all
  using (user_id = auth.uid())
  with check (user_id = auth.uid());
