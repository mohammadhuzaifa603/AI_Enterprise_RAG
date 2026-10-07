CREATE EXTENSION IF NOT EXISTS vector;

-- PostgreSQL migration/index note:
-- chunks.embedding is declared as VECTOR(EMBEDDING_DIM) by SQLAlchemy when
-- DATABASE_URL points at PostgreSQL. Create the ANN index after the table exists:
-- CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw
-- ON chunks USING hnsw (embedding vector_cosine_ops);
