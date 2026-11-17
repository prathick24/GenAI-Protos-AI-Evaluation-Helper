# AI Evaluation Helper

A locally-runnable, multi-agent AI assistant for teachers that automates two
high-volume tasks:

1. **Question paper generation** — teacher uploads a syllabus, selects the
   in-scope chapters/topics, and sets a blueprint (marks, types, difficulty).
   Questions are generated from the **raw** syllabus text (retrieved per
   unit/topic), never from summaries.
2. **Answer sheet evaluation** — teacher uploads student answer sheets (typed or
   scanned handwritten PDFs). The system OCRs (if scanned), segments answers,
   grades each against a rubric with cited evidence, independently verifies the
   marks, writes per-student feedback, and produces a marksheet with class
   analytics. The teacher reviews and can override any mark before approving.

Full design: [design/SOLDEF_AI_Evaluation_Helper_AEH-101.md](design/SOLDEF_AI_Evaluation_Helper_AEH-101.md).
Data model: [design/er_diagram.mmd](design/er_diagram.mmd).

## Requirements

- Python 3.12 (managed via [uv](https://docs.astral.sh/uv/))
- PostgreSQL 18 with the `pgvector` extension (local)
- A free-tier, OpenAI-compatible LLM API key (optional — a stub provider runs
  without network access)

## Setup

```powershell
# 1. Create the venv (pins Python 3.12)
uv venv --python 3.12

# 2. Install dependencies (add --extra ml for embeddings/OCR/pdf,
#    --extra ui for the Streamlit frontend)
uv sync --extra dev --extra ml --extra ui

# 3. Configure environment
Copy-Item .env.sample .env
# then edit .env (database URL, LLM key)

# 4. Create the database + pgvector extension
#    psql is at: C:\Program Files\PostgreSQL\18\bin\psql.exe
#    (or run `make db-init`, which creates the `aeh` database if missing)
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -c "CREATE DATABASE aeh"
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -d aeh -f db/init.sql
```

The 17 tables are created idempotently by the backend at startup, so the ORM
(`src/repositories/schema/`) is the single source of truth for the schema.

## Run

```powershell
# Backend (FastAPI)
uv run uvicorn src.main:app --reload --port 8000

# Frontend (Streamlit) — separate terminal
uv run streamlit run frontend/app.py
```

Generation with validation repairs can take several minutes. The UI waits 600s
by default; override it with `AEH_REQUEST_TIMEOUT` (seconds) if needed.

### Handwritten answer sheets

The default local OCR (`OCR_PROVIDER=local`, RapidOCR) is tuned for printed
text. For handwriting, set `OCR_PROVIDER=cloud` and point the vision settings at
an OpenAI-compatible vision model. The defaults use a local Ollama server:

```powershell
ollama pull qwen2.5vl:3b
```

Relevant settings: `OCR_VISION_BASE_URL`, `OCR_VISION_API_KEY`,
`OCR_VISION_MODEL`, `OCR_VISION_API` (`ollama` or `openai`), and
`OCR_VISION_TIMEOUT_SECONDS` if needed.

- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/health
- Ready: http://localhost:8000/api/v1/ready

## Layout

```
api/        OpenAPI contracts (internal + external LLM)
db/         Database init SQL (pgvector extension)
design/     ER diagram + Solution Document
src/        Backend (routes -> services -> agents -> repositories -> models)
frontend/   Streamlit app
tests/      Tests
```

## Status

Foundation (contracts, config, DB models, app skeleton) in progress. Feature
implementation proceeds per the FRs in the Solution Document.
