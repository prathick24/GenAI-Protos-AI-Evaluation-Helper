.PHONY: venv sync sync-ml run run-ui lint format test db-init

# Create the virtual environment pinned to Python 3.12
venv:
	uv venv --python 3.12

# Install base + dev dependencies
sync:
	uv sync --extra dev

# Install ML dependencies too (sentence-transformers, OCR, pdf)
sync-ml:
	uv sync --extra dev --extra ml

# Run the FastAPI backend
run:
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Run the Streamlit frontend
run-ui:
	uv run streamlit run frontend/app.py

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

# Create the database (if missing) and enable pgvector (psql must be on PATH)
# psql lives at "C:\Program Files\PostgreSQL\18\bin\psql.exe" on Windows.
db-init:
	psql -U postgres -tc "SELECT 1 FROM pg_database WHERE datname='aeh'" | findstr /C:"1" >nul || psql -U postgres -c "CREATE DATABASE aeh"
	psql -U postgres -d aeh -f db/init.sql
