-- Week 8 follow-up (Gap 1): hold the extracted plaintext for documents
-- the classifier couldn't confidently bucket as lab_report or
-- research_paper.
--
-- Why this column instead of re-uploading?
--   When the user clicks one of the disambiguation buttons ("Treat as
--   lab report" / "Treat as research paper") on the assistant message,
--   we need to re-run the marker parser or the chunking pipeline on
--   the SAME bytes the user already uploaded. We don't store the
--   original bytes (PDFs are large + we don't need them at rest), so
--   we cache the extracted text instead. PDF bytes ≫ extracted text in
--   size for typical lab reports, so this column stays small.
--
-- Lifecycle:
--   - /upload 'other' branch sets extracted_text and leaves doc_type
--     as 'other'.
--   - /upload/resolve reads extracted_text, runs the chosen handler,
--     updates doc_type to the user's pick, and clears extracted_text
--     to null (no need to keep it once the doc has been processed).
--
-- Safe to re-run.

alter table public.session_documents
  add column if not exists extracted_text text;
