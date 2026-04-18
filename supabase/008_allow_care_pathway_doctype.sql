-- Week 7B: extend the documents.doc_type CHECK constraint to allow a
-- new 'care_pathway' value for sources that primarily encode WHEN to
-- escalate (NHS "when to see a GP", WHO IMAI referral criteria, MoHP
-- STG escalation thresholds), as opposed to general patient education.
--
-- The Stage 2 navigation retrieval applies a soft score boost to chunks
-- with this doc_type so condition-specific 102-escalation triggers
-- surface in the LLM context. See app/RAG.py _retrieve_ranked.

ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_doc_type_check;

ALTER TABLE documents ADD CONSTRAINT documents_doc_type_check
  CHECK (doc_type IN ('patient-ed', 'clinical-guideline', 'reference', 'care_pathway'));
