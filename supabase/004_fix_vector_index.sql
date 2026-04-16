-- Replace the ivfflat index with HNSW.
--
-- Why: the 002 migration created `chunks_embedding_idx` as ivfflat with
-- lists=100. That setting is tuned for large corpora (>100k rows). On the
-- Week 3 seed corpus (~300 chunks) each ivfflat list holds ~3 rows, and
-- the default probes=1 scans one near-empty list per query — retrieval
-- returns zero rows even for a direct hit. HNSW has no probes/lists
-- tuning and works well from ~100 rows up to ~1M+, so we don't need to
-- revisit this as the corpus grows.

drop index if exists public.chunks_embedding_idx;

create index if not exists chunks_embedding_hnsw_idx
  on public.chunks
  using hnsw (embedding vector_cosine_ops);
