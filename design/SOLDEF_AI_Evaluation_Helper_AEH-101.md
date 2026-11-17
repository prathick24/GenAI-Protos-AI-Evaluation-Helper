## SOLDEF — AI Evaluation Helper

### <u>Project Details</u>

- **Project ID:** AEH
- **Project Name:** AI Evaluation Helper

### <u>Story Details</u>

- **Story ID:** AEH-101
- **Story Name:** Multi-Agent Exam Evaluation & Question Paper Generation
- **Story Description:**
  Build a locally-runnable, multi-agent AI assistant for teachers that automates two high-volume tasks: (1) **generating exam question papers** from a teacher-defined syllabus scope and a marks/difficulty blueprint, and (2) **evaluating student answer sheets** (typed or scanned handwritten PDFs) per-question against a rubric, awarding every mark with cited evidence, verifying marks independently, writing per-student feedback, and producing a marksheet with class analytics. The teacher always reviews and can override any mark before results are finalized. Everything must be explainable, traceable, and feasible on free-tier resources.

### <u>Table of Contents</u>

- [Section 1: Functional Requirements](#section-1-functional-requirements)
- [Section 2: Non Functional Requirements](#section-2-non-functional-requirements)
- [Section 3: In Scope and Out Scope](#section-3-in-scope-and-out-scope)
- [Section 4: Solution Diagrams](#section-4-solution-diagrams)

---

### <u>Section 1: Functional Requirements</u>

#### <u>1.1 Overview</u>

The application is a local, single-user teacher workstation with two capabilities sharing one multi-agent backend.

**Capability A — Question Paper Generation.** A teacher uploads a syllabus (PDF/text), ticks the **in-scope chapters/topics**, and sets a **blueprint** (total marks, question types, marks per type, difficulty mix, topic weightage). The syllabus is parsed into **chunks that preserve the original wording** and tagged with structured metadata (`unit_no`, `unit_name`, `topic_name`, `chunk_index`, `page_no`). Every chunk is embedded locally (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim) and stored in PostgreSQL/pgvector. Paper generation is **scope-driven**: the system retrieves the **raw** chunks for the target unit/topic (by metadata filter), and when a unit is too large for one prompt it loops **sub-topic by sub-topic** using the syllabus's own headings. Questions are always written from **raw text — never from a summary** — and each question records the syllabus chunk it was grounded in (`source_chunk_id`). A validation pass checks syllabus coverage, duplication, ambiguity, mark totals, and difficulty distribution.

**Capability B — Answer Sheet Evaluation.** A teacher creates an `EXAM` (linked to an approved question paper) and uploads student answer sheets (typed or scanned handwritten PDFs). For typed PDFs the text is extracted directly; for scanned sheets a **pluggable OCR service** (local-first, cloud-optional) performs image cleanup, layout analysis, and handwriting recognition. A parsing agent segments the sheet into per-question answers (`ANSWERS`). A grading agent scores each answer against the question's `model_answer` and `RUBRIC_CRITERIA`, producing `awarded_marks`, a per-criterion `rubric_scores` breakdown, `evidence` quoted from the student's own answer, `reasoning`, and a `confidence` value (`MARKS`). An **independent verification agent** re-checks the marks (`verifier_status`, `verifier_notes`). A feedback agent writes per-student `FEEDBACK`. A marksheet and class analytics are derived from `MARKS` and `ANSWER_SHEETS`. The teacher reviews and may override any mark (`is_overridden`, `original_marks` preserved) before approving the result.

**Shared multi-agent design.** Specialized agents handle syllabus ingestion, blueprint planning, question generation, paper validation, OCR/segmentation, rubric building, grading, verification, feedback, and the chat assistant. Agents never touch the database directly: the flow is **Route → Service → Agent → Validate → Repository → Data Store**. LLM output is validated against typed contracts (Pydantic) before persistence. Long-running work (grading many sheets) is paced by a **per-student queue** with explicit status transitions so it is resumable, observably progressable, and safe against free-tier rate limits.

**Human-in-the-loop.** No mark or paper is final until the teacher approves it. The AI assists; the teacher decides.

**Free-tier and privacy.** LLM calls go to a free-tier OpenAI-compatible endpoint (open-source model, default `gpt-oss-120b`). Embeddings and OCR default to local execution (zero cost). The application runs locally (FastAPI + Streamlit + local PostgreSQL 18 with pgvector); no cloud infrastructure is required for the core demo.

#### <u>1.2 Requirement Details</u>

- **FR-001: Syllabus Ingestion and Indexing**
- **FR-002: Question Paper Generation**
- **FR-003: Question Paper Validation**
- **FR-004: Answer Sheet Ingestion and OCR**
- **FR-005: Answer Segmentation and Question Alignment**
- **FR-006: Rubric-Based Grading with Evidence**
- **FR-007: Independent Mark Verification**
- **FR-008: Feedback, Marksheet and Class Analytics**
- **FR-009: Human-in-the-Loop Review, Override and Approval**
- **FR-010: Multi-Agent Chat Assistant**
- **FR-011: Traceability, Cost and Governance**
- **FR-012: Streamlit Frontend**

---

##### <u>1.2.1 FR-001: Syllabus Ingestion and Indexing</u>

##### Description:
The teacher uploads a syllabus document. The system extracts text (or OCRs it if scanned), identifies chapters/units and their sub-topics from the document's headings, splits the content into chunks that **preserve the original wording**, embeds each chunk locally, and persists everything idempotently.

##### Inputs:
- Syllabus file (`.txt` / text PDF / scanned PDF). Max size configurable (default 20 MB).
- Teacher-selected **scope**: which units/topics are examinable (e.g. `["1","2","3"]`).

##### Process:
1. Extract text: typed PDF/TXT → direct extraction; scanned → OCR service (FR-004 mechanism, reused).
2. Split by headings into **unit → topic (sub-topic)** blocks; fall back to fixed-size splitting when headings are absent.
3. Emit `SYLLABUS_CHUNKS` rows: `unit_no`, `unit_name`, `topic_name`, `chunk_index`, `page_no`, `raw_text`.
4. Embed each chunk locally (`all-MiniLM-L6-v2`, 384-dim) → `embedding`.
5. Persist `SYLLABUS` (with `scope` jsonb, `input_type`, `ocr_status`, `status`, `version`) and its chunks. Idempotent on `(syllabus_id, chunk_index)`.

##### Storage:
- Tables `SYLLABUS` and `SYLLABUS_CHUNKS`; vector column `vector(384)`; cosine operator `<=>`.
- Re-uploading the same filename creates a new `version` and deactivates the prior record (`is_active = false`).

##### Acceptance Criteria:
- All in-scope units/topics are captured as chunks; every chunk has a `unit_no`, `topic_name`, `chunk_index`, and **verbatim** `raw_text`.
- Embeddings stored as `vector(384)`; cosine search works via `<=>`.
- Filtering by `unit_no`/`topic_name` returns the complete, correctly-ordered chunk set for that scope.
- Re-running ingestion does not create duplicates.

---

##### <u>1.2.2 FR-002: Question Paper Generation</u>

##### Description:
Given a syllabus scope and a blueprint, the system generates a complete, balanced question paper. Questions are written from **raw syllabus text** retrieved per unit/topic — never from summaries — so nothing important is lost.

##### Inputs:
- `syllabus_id` and the selected scope (units/topics).
- Blueprint jsonb: `total_marks`, `duration_minutes`, `question_types` (MCQ/short/long), `marks_per_type`, `difficulty_mix` (easy/medium/hard), `topic_weightage`.

##### Retrieval (scope-driven, two modes):
- **Whole-unit (exact):** when the target is a unit, filter `WHERE syllabus_id = :id AND unit_no = :unit ORDER BY chunk_index` → returns **all** chunks of that unit (complete, not similarity-ranked).
- **Topic/sub-topic (exact):** filter by `topic_name` for finer targets; sub-topic loop splits an oversized unit into its sections.
- **Semantic (assist):** when a target is described loosely, embed the phrase and retrieve top-K by `<=>`. Used only to find sub-parts, never to replace completeness checks.

##### Oversized-unit handling:
- If a unit's raw chunks exceed the prompt budget, generate **section-by-section** (sub-topic loop). If a section is still too large, split one heading level further or by page size. **Raw text is always used; no summarization occurs in the generation path.**

##### Generation:
- LLM task `paper_generation`, temperature 0.2–0.4.
- Prompt contract per prompt-engineering standards (Role, Mission, Context = retrieved raw chunks only, Inputs, Constraints, Rules, Critical Rules, Output Contract, Error Handling).
- Strict typed JSON output; each generated question includes `question_text`, `question_type`, `marks`, `options`/`correct_option` (for MCQs), `model_answer`, `topic`, `difficulty`, and `source_chunk_id`.
- `model_answer` and `RUBRIC_CRITERIA` are produced alongside each question.

##### Output:
- `QUESTION_PAPERS` row (`status = draft`, `source = generated`, `syllabus_id`, `blueprint`) with `QUESTIONS` and their `RUBRIC_CRITERIA`.
- Streaming variant (`/question-papers/generate/stream`) emits per-unit progress as NDJSON.

##### Acceptance Criteria:
- Generated questions sum to `total_marks` and match the blueprint's type/difficulty distribution.
- Every question has a `model_answer`, a rubric, and a `source_chunk_id` traceable to a real syllabus chunk.
- A large unit is covered fully via the sub-topic loop (no section silently skipped).

---

##### <u>1.2.3 FR-003: Question Paper Validation</u>

##### Description:
The draft paper is validated before approval.

##### Checks:
- **Syllabus coverage:** every in-scope topic has at least the blueprint-required number of questions (deterministic against the topic list — no LLM needed).
- **Mark distribution:** totals and per-type marks match the blueprint (deterministic).
- **Difficulty mix:** matches the blueprint (deterministic).
- **Duplication:** near-duplicate detection across questions (deterministic string/embedding similarity + LLM judgement for phrasing).
- **Ambiguity / clarity:** LLM judgement on vague or unanswerable phrasing.
- **Source grounding:** each question's `source_chunk_id` resolves to a real chunk (deterministic).

##### Output:
- `QUESTIONS.validation_status` per question; a paper-level validation report. Failures may trigger regeneration of the affected items.
- Paper `status` transitions `draft → validated → approved`.

##### Acceptance Criteria:
- Deterministic checks (coverage, marks, difficulty, grounding) always run and are authoritative.
- A paper failing any check cannot move to `approved` until corrected.
- The coverage report is reproducible from stored data (no hidden state).

---

##### <u>1.2.4 FR-004: Answer Sheet Ingestion and OCR</u>

##### Description:
The teacher uploads one or more student answer sheets for an exam. Typed PDFs are text-extracted; scanned handwritten sheets are OCR'd.

##### OCR service (pluggable):
- Provider interface with a **local-first default** (e.g. PaddleOCR / TrOCR) and an optional **cloud vision** provider (handwriting). Selected via configuration.
- Sub-steps: image cleanup (deskew/de-noise/contrast) → layout analysis (locate question regions) → handwriting recognition → post-processing (line breaks, low-confidence tagging).

##### Storage:
- `ANSWER_SHEETS` row: `exam_id`, `student_id`, `file_path`, `input_type` (`typed`/`scanned`), `ocr_status` (`pending`/`processing`/`done`/`failed`), `sheet_status` (`uploaded`/`queued`/`grading`/`graded`/`reviewed`/`approved`), `ocr_confidence` jsonb.
- Files stored on the filesystem; DB stores only `file_path`.

##### Failure handling:
- OCR failure does not block the batch; the sheet is flagged `ocr_status = failed` with a reason, and can be retried.

##### Acceptance Criteria:
- Typed sheets are extracted without OCR; scanned sheets are OCR'd and tagged with `input_type`/`ocr_status`.
- Every uploaded sheet has a status lifecycle enabling resumption.
- Per-file errors are reported without failing the rest of the batch.

---

##### <u>1.2.5 FR-005: Answer Segmentation and Question Alignment</u>

##### Description:
Each answer sheet is segmented into per-question answers and aligned to the questions of the exam's paper.

##### Process:
1. Parse the sheet layout (question numbering / OCR regions) into candidate answer blocks.
2. Align each block to a `QUESTIONS` row via the question's `order_index` / number.
3. Persist `ANSWERS`: `answer_sheet_id`, `question_id`, `answer_text` (raw/OCR text, verbatim), `is_answered`, `ocr_confidence`.
4. Unmatched or unreadable regions are flagged for teacher attention rather than guessed.

##### Acceptance Criteria:
- Every answer is linked to a real `question_id` from the correct exam's paper.
- `answer_text` is the student's verbatim content (no AI rewriting).
- Unanswered questions are recorded with `is_answered = false` (not silently dropped).

---

##### <u>1.2.6 FR-006: Rubric-Based Grading with Evidence</u>

##### Description:
Each answer is graded against the question's `model_answer` and `RUBRIC_CRITERIA`, producing a mark with cited evidence and a per-criterion breakdown.

##### Process:
1. For each answer, load the question's rubric criteria and model answer.
2. Grading agent scores each criterion → `rubric_scores` (jsonb), sums to `awarded_marks` (capped at `max_marks`).
3. `evidence` must quote the specific part of the student's `answer_text` that justifies the mark; `reasoning` explains the award.
4. `confidence` records the agent's certainty; `graded_by` = `agent`.
5. Persist one `MARKS` row per answer (unique per `answer_id`).

##### Pacing:
- Grading runs through a **per-student queue** (status-based); the LLM rate limiter paces calls (see NFR 2.2.3). One student's sheet is graded per unit of work.

##### Prompt rules:
- Use ONLY the provided rubric + student answer. Never invent evidence. If the answer is blank/irrelevant, award 0 with explicit reasoning.
- Temperature 0.0–0.2.

##### Acceptance Criteria:
- Every mark has a non-empty `evidence` that is a substring of the student's `answer_text` (or an explicit "no relevant content" rationale for 0).
- `awarded_marks` never exceeds `max_marks`; `rubric_scores` sum to `awarded_marks`.
- Grading is resumable; re-running does not duplicate marks (unique on `answer_id`).

---

##### <u>1.2.7 FR-007: Independent Mark Verification</u>

##### Description:
A separate verification agent independently re-checks each mark for consistency before the teacher sees it.

##### Process:
- The verifier receives the question, model answer, rubric, the student's answer, and the awarded mark — but not the grader's hidden chain-of-thought.
- It returns an agreement verdict: `verifier_status` (`agreed`/`flagged`/`adjusted`) plus `verifier_notes`.
- Disagreements are surfaced to the teacher (never silently overwritten).

##### Acceptance Criteria:
- Every graded answer has a `verifier_status`.
- Flagged marks are visible in the review UI with the verifier's reasoning.
- Verification is logged as its own `AGENT_RUNS` entry.

---

##### <u>1.2.8 FR-008: Feedback, Marksheet and Class Analytics</u>

##### Description:
Per-student qualitative feedback and class-level results are produced from the verified marks.

##### Process:
- Feedback agent writes `FEEDBACK`: `overall_comments`, `strengths`, `improvements`, grounded in the marks obtained.
- `ANSWER_SHEETS.total_awarded_marks` is computed from `MARKS`.
- Marksheet: per-student per-question marks. Analytics: class average, distribution, per-topic strength/weakness, hardest/easiest questions (deterministic aggregates over `MARKS`).

##### Acceptance Criteria:
- Feedback references the student's actual performance (no generic filler).
- Marksheet totals reconcile with the sum of `MARKS`.
- Analytics are computed deterministically from stored data.

---

##### <u>1.2.9 FR-009: Human-in-the-Loop Review, Override and Approval</u>

##### Description:
The teacher is the final authority. Any mark or feedback can be reviewed, overridden, and then approved.

##### Process:
- Review UI shows each answer with OCR text, evidence, rubric breakdown, verifier status, and confidence.
- Override: teacher sets a new mark → `is_overridden = true`, `original_marks` preserves the AI's value, `graded_by = teacher`.
- Approval: sheet `sheet_status → approved`; exam results finalize only after approval.

##### Acceptance Criteria:
- Every AI mark is visible with its evidence and can be overridden.
- Overrides preserve the original AI mark for audit.
- No result is finalized without explicit teacher approval.

---

##### <u>1.2.10 FR-010: Multi-Agent Chat Assistant</u>

##### Description:
A conversational surface where the teacher can ask questions about a paper, an exam, or the syllabus, and instruct the agents.

##### Process:
- `CONVERSATIONS` + `CONVERSATION_MESSAGES` record each turn, including which `agent_name` responded, `tool_calls`, and `tokens`.
- Retrieval-grounded answers with citations to the syllabus/papers; responses use ONLY provided context.
- If context is insufficient, reply that it could not be found and state what is covered (no guessing).
- Stateless HTTP endpoint; conversation display is owned by the frontend.

##### Acceptance Criteria:
- Turns are persisted with agent attribution, tool calls, and token counts.
- Answers are grounded and cite their source; unanswerable questions return a scope reply.
- No raw LLM errors or internal traces leak to the client.

---

##### <u>1.2.11 FR-011: Traceability, Cost and Governance</u>

##### Description:
A durable record of what the backend actually did — every agent task, every LLM call, and every failure.

##### Tables:
| Table | Rows |
|-------|------|
| `AGENT_RUNS` | One row per agent task (agent_name, task, status, started_at/ended_at, input/output summaries) |
| `MODEL_COST` | One row per LLM call (task, agent_name, model, provider, tokens, cost_usd, latency_ms, success, error_code, correlation_id) |
| `ERROR_LOGS` | One row per recorded failure (level, component, message, exception_type, endpoint, method, correlation_id, detail) |

##### Behaviour:
- Recording is **best-effort**: each write uses its own short-lived session; a storage failure is logged and never aborts the primary request or streaming contract.
- The LLM client reports every call (tokens, latency, success) through an injected usage recorder; failed calls are recorded too.
- Traceability is inspected directly in the database.

##### Acceptance Criteria:
- Every agent task, LLM call, and recorded error is persisted.
- Regardless of DB state, primary request behaviour is unchanged — recording can only add a log line.
- Cost and latency are attributable per task/agent.

---

##### <u>1.2.12 FR-012: Streamlit Frontend</u>

##### Description:
A single-page Streamlit app calling only the FastAPI backend. No DB or LLM access from the frontend.

##### Views:
| View | Content |
|------|---------|
| Syllabus | Upload syllabus; select scope; view extracted units/topics |
| Generate Paper | Set blueprint; generate; view questions + rubric; validate; approve |
| Exams & Sheets | Create exam; upload answer sheets; grade (paced); view progress |
| Review | Per-answer OCR text, evidence, rubric scores, verifier status; override; approve |
| Marksheet & Analytics | Per-student marksheet; class analytics |
| Assistant | Chat with grounded answers + citations |

##### Behaviour:
- Spinner/loading states during backend calls; graceful errors (stale connection, 5xx, rate-limit) with an explicit Retry action.
- A failed/partial stream is cached in session state so reruns do not silently re-stream it.
- Long jobs show live status from `sheet_status`; UI never grows unbounded (fixed-height scroll containers where relevant).

##### Acceptance Criteria:
- All views render from backend data only.
- Evidence, citations, and verifier status are displayed for every mark.
- The app does not crash on backend unavailability; it shows a clear, actionable error.

---

#### <u>1.3 Project Artifacts</u>

| File | Purpose |
|------|---------|
| `template.md` | Requirement brief (the complete use case) |
| `design/er_diagram.mmd` | ER diagram (17 tables incl. syllabus chunks) |
| `design/SOLDEF_AI_Evaluation_Helper_AEH-101.md` | This Solution Document |
| `api/openapi.yaml` | Internal API contract |
| `api/external-api.yaml` | External API (LLM provider) contract |
| `datas/` (runtime) | Uploaded syllabi and answer sheets (or configurable storage path) |

#### <u>1.4 Dependencies</u>

| Package | Version (pin) | Purpose |
|---------|---------------|---------|
| Python | >= 3.11 | Runtime |
| `fastapi` | latest stable | Backend framework |
| `uvicorn[standard]` | latest stable | ASGI server |
| `streamlit` | latest stable | Frontend |
| `sqlalchemy[asyncio]` | 2.x | Async ORM |
| `asyncpg` | 0.30.x | Async PostgreSQL driver |
| `pgvector` | 0.3.x | pgvector bindings |
| `sentence-transformers` | latest stable | Local embeddings (`all-MiniLM-L6-v2`, 384-dim) |
| `pypdf` / `pdfplumber` | latest stable | Text PDF extraction |
| `paddleocr` / `transformers` (TrOCR) | latest stable | Default local OCR/HTR provider |
| `httpx` | latest stable | LLM API client (OpenAI-compatible) |
| `pydantic` | 2.x | DTOs/validation |
| `pydantic-settings` | latest stable | `.env` config |
| `tenacity` | latest stable | Retry/backoff |
| `ruff` | latest stable | Lint + format |
| `pytest`, `pytest-asyncio` | latest stable | Tests |

Note: LLM key from `LLM_API_KEY` env var (`.env`, never committed). PostgreSQL 18 with the `vector` extension is required locally.

---

### <u>Section 2: Non Functional Requirements</u>

### 2.1 Infrastructure and Deployment

#### <u>2.1.1 Overview</u>

The application runs locally (not deployed). It requires: PostgreSQL 18 with the pgvector extension, a Python 3.11+ virtual environment (managed with `uv`), an LLM API key (free tier), and the sentence-transformers model downloaded on first run. Deployment is `make run` (FastAPI/uvicorn) plus `make run-ui` (Streamlit). No cloud infrastructure is used.

#### <u>2.1.2 Requirement Details</u>

##### NFR-001: Local Database Setup
- PostgreSQL 18 with `CREATE EXTENSION vector`.
- `DATABASE_URL` in `.env` (default `postgresql+asyncpg://postgres:postgres@localhost:5432/aeh`).
- All 17 tables created idempotently at startup or via an init script.
- Vector columns `vector(384)`; cosine distance operator `<=>`.

##### NFR-002: Configuration
- `.env` + `.env.sample`; keys: `LLM_API_KEY`, `LLM_MODEL` (default `gpt-oss-120b`), `LLM_BASE_URL`, `DATABASE_URL`, `EMBEDDING_MODEL` (default `all-MiniLM-L6-v2`), `OCR_PROVIDER` (default `local`), `TOP_K` (default 5), `GRADE_CONCURRENCY` (default 1), `LOG_LEVEL`.
- Validated at startup via Pydantic Settings; missing `LLM_API_KEY`/`DATABASE_URL` raises a clear startup error.
- `.env` listed in `.gitignore`.

##### NFR-003: Layer Architecture
- Strict layering: Routes → Services → Agents → Repositories → Data Store.
- `src/` layout: `routes/`, `services/`, `agents/`, `repositories/` (+`schema/`), `models/`, `client/` (LLM, OCR), `middleware/`, `utils/` (+`exceptions/`), `prompt/`, `settings.py`, `main.py`.
- DI via `src/services/dependencies.py`.

### 2.2 Architecture and System Design

#### <u>2.2.1 Security and Compliance</u>
- LLM API key never hardcoded; loaded from environment via Pydantic Settings, excluded from git.
- All SQL via SQLAlchemy ORM / parameterized statements (no SQL injection).
- No secrets, PII leaks, or raw stack traces returned to the client; LLM errors logged server-side, surfaced as generic errors (fail-closed).
- Input validation with Pydantic DTOs on every route; uploaded files validated (type/size).

#### <u>2.2.2 System Performance</u>
- Retrieval bounded by `LIMIT :top_k` where similarity is used; metadata-filter retrieval is bounded by scope size → constant-time behaviour.
- Deterministic LLM results cached by task signature → repeat requests served from memory without LLM cost.
- Local embeddings (singleton model) — no embedding API round-trip.
- Raw chunk size capped before prompting; sub-topic loop keeps each call within token budget.
- Grading paced by a queue with configurable `GRADE_CONCURRENCY`.

#### <u>2.2.3 Availability and Reliability</u>
- LLM client retry: exponential backoff (e.g. 5 attempts, 2s→10s) on 5xx; 429s honor `Retry-After`/reset headers (sleep-and-retry when < 60s, raise a rate-limit error otherwise — no blind backlog re-fills).
- Per-student queue with status rows → grading is resumable and observable.
- Graceful degradation: if the LLM is down/rate-limited, the API returns a structured error and the frontend offers Retry.
- Idempotent ingestion; readiness endpoint fails (503) if DB/pgvector/tables are unavailable.

#### <u>2.2.4 Cost Efficiency</u>
- Free-tier LLM (open-source model) — $0 at demo scale.
- Embeddings and default OCR run locally — zero cost.
- Scope-driven retrieval + result caching avoid wasted tokens.
- Total stack cost: $0 (free-tier LLM, local PG, local embeddings, local OCR).

#### <u>2.2.5 Traceability and Observability</u>
- Structured JSON logging with fields: timestamp, level, logger, message, `correlation_id`/`task_id`, model, latency, token usage, outcome.
- Log LLM events (request, completion, failure with retry count/error class) and guardrail events (verification decisions, "not mentioned" decisions, cache hits/misses).
- Logs are privacy-safe: no raw API keys, no full prompt/response bodies by default.

---

### <u>Section 3: In Scope and Out Scope</u>

#### <u>3.1 In Scope Details</u>
- Syllabus upload, text extraction/OCR, heading-based chunking, local embedding, idempotent storage.
- Teacher-defined exam scope (units/topics).
- Scope-driven question paper generation from **raw** syllabus text, with the sub-topic loop for oversized units.
- Question paper validation (coverage, marks, difficulty, duplication, ambiguity, source grounding).
- Answer sheet upload (typed + scanned), pluggable OCR, per-question segmentation and alignment.
- Rubric-based grading with verbatim evidence and per-criterion scores.
- Independent mark verification.
- Per-student feedback, marksheet, and class analytics.
- Human-in-the-loop review, override (`is_overridden`, `original_marks`), and approval.
- Multi-agent chat assistant with grounded, cited answers.
- Traceability: `AGENT_RUNS`, `MODEL_COST`, `ERROR_LOGS` (best-effort).
- FastAPI backend with versioned routes (`/api/v1`), health/ready endpoints, and NDJSON streaming for long jobs.
- Streamlit frontend consuming the API.
- Structured logging, deterministic-result caching.
- Artifacts: `requirements.md` (if provided), `er_diagram.mmd`, `openapi.yaml`, `external-api.yaml`.

#### <u>3.2 Out Scope Details</u>
- Authentication / multi-user support (local single-user demo).
- Cloud deployment / Docker images.
- Fine-tuning of any model.
- Speech-to-text / audio ingestion.
- Real-time collaborative grading.
- Automatic re-indexing on syllabus file change (re-upload creates a new version instead).
- HNSW/IVFFlat index tuning (default scan is sufficient at this scale — explain in demo, do not implement).
- Persistent backend chat session memory (conversation display lives in the frontend; turns are recorded for traceability only).
- Payments, notifications, or external LMS integration.

---

### <u>Section 4: Solution Diagrams</u>

#### <u>4.1 UI/UX Design Diagram</u>

**Diagram Location:** Not applicable — diagram sections are skipped per SD generation convention. UI is described in FR-012 (views: Syllabus, Generate Paper, Exams & Sheets, Review, Marksheet & Analytics, Assistant).

#### <u>4.2 Sequence Diagrams</u>

**Flow: Question Paper Generation (`POST /api/v1/question-papers/generate`)**

1. Teacher selects `syllabus_id`, scope, and blueprint; frontend POSTs to `/api/v1/question-papers/generate`.
2. Route validates the request via Pydantic DTO → 422 on invalid input.
3. Service loads the syllabus and creates a `QUESTION_PAPERS` row (`status = draft`, `syllabus_id`, `blueprint`).
4. For each blueprint item, the service resolves the target unit/topic and retrieves **raw** chunks (metadata filter; sub-topic loop if the unit exceeds the prompt budget).
5. Generation agent writes questions from the raw chunks; each question carries `source_chunk_id`.
6. Service validates LLM output against the typed contract, persists `QUESTIONS` + `RUBRIC_CRITERIA` transactionally.
7. Validation agent runs coverage/marks/difficulty/duplication/ambiguity/grounding checks → `validation_status`.
8. Route returns the draft paper (or streams per-unit progress as NDJSON). Teacher reviews and approves → `status = approved`.

**Flow: Answer Sheet Evaluation (`POST /api/v1/exams/{id}/answer-sheets/upload` → paced grading)**

1. Teacher uploads sheet(s) for an exam; route validates file type/size.
2. Service creates `ANSWER_SHEETS` rows (`sheet_status = uploaded`).
3. If scanned → OCR service runs (cleanup → layout → handwriting recognition) → `ocr_status = done`.
4. Segmentation agent parses the sheet into `ANSWERS` aligned to `QUESTIONS` (verbatim text).
5. The sheet enters the **per-student queue** (`sheet_status = queued → grading`).
6. For each answer: grading agent scores against rubric + model answer → `MARKS` (evidence, rubric_scores, reasoning, confidence).
7. Verification agent re-checks each mark → `verifier_status`.
8. Feedback agent writes `FEEDBACK`; `total_awarded_marks` computed; `sheet_status = graded`.
9. Teacher reviews, overrides if needed, and approves → `sheet_status = approved`. Marksheet/analytics available.
10. Throughout: `AGENT_RUNS` per agent task, `MODEL_COST` per LLM call, `ERROR_LOGS` per failure (best-effort).

**Flow: Assistant Q&A (`POST /api/v1/chat/ask`)**

1. Frontend POSTs the question; route validates via DTO.
2. Service retrieves grounded context (syllabus/papers/exam as relevant) — bounded top-K.
3. LLM (temperature ≤ 0.2) answers using ONLY the provided context, with citations; insufficient context → scope reply.
4. Turn persisted to `CONVERSATIONS`/`CONVERSATION_MESSAGES` with agent attribution and tokens.
5. Route returns `{question, answer, citations}` → frontend renders.

#### <u>4.3 Architectural Diagram</u>

```
                          ┌───────────────────────────────┐
                          │      Teacher (Browser)        │
                          └──────────────┬────────────────┘
                                         │
                                         ▼
                          ┌───────────────────────────────┐
                          │  Frontend: Streamlit (app.py) │
                          │  Syllabus | Generate | Exams   │
                          │  Review | Marksheet | Assistant│
                          └──────────────┬────────────────┘
                                         │  HTTP /api/v1
                                         ▼
                          ┌───────────────────────────────┐
                          │  Backend: FastAPI (main.py)   │
                          │  Routes → Services → Agents   │
                          │             → Repositories     │
                          └───┬────────┬────────┬─────────┘
                              │        │        │
              ┌───────────────▼──┐  ┌──▼──────────────┐  ┌▼──────────────────┐
              │ sentence-transf. │  │ OCR Service     │  │ LLM Client         │
              │ (local embeds)   │  │ (local/cloud)   │  │ (free-tier, OSS)   │
              │ all-MiniLM-L6-v2 │  │ Paddle/TrOCR    │  └─┬──────────────────┘
              └────────┬─────────┘  └────────┬────────┘    │ HTTPS
                       │                     │             ▼
                       │                     │   ┌──────────────────────┐
                       ▼                     ▼   │  LLM API (cloud,     │
              ┌─────────────────────────────────┐│  open-source model)  │
              │  PostgreSQL 18 + pgvector       │└──────────────────────┘
              │  ├─ syllabus / syllabus_chunks  │
              │  ├─ question_papers / questions │
              │  ├─ exams / answer_sheets ...   │
              │  └─ agent_runs / model_cost     │
              └─────────────────────────────────┘
```

Key decisions:
- **Raw-text grounding** → syllabus chunks keep original wording; questions are never generated from summaries.
- **Metadata-filter retrieval first, similarity as assist** → completeness for coverage-driven generation.
- **Sub-topic loop** → oversized units are generated section-by-section, keeping every LLM call small.
- **Embeddings + OCR local by default** → zero cost; only LLM calls hit the network.
- **Per-student paced queue** → resumable, observable, and free-tier rate-limit safe.
- **PostgreSQL + pgvector** → free, local, SQL-native retrieval.
