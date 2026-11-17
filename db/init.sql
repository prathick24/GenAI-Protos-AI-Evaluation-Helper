-- AI Evaluation Helper — database bootstrap.
--
-- Run once against the `aeh` database to enable the pgvector extension.
-- The 17 tables are created idempotently by the application at startup
-- (SQLAlchemy `Base.metadata.create_all`), so the ORM remains the single
-- source of truth for the schema.
--
--   createdb -U postgres aeh
--   psql -U postgres -d aeh -f db/init.sql

CREATE EXTENSION IF NOT EXISTS vector;
