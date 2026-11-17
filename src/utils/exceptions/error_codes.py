"""Centralized application error codes (UPPER_SNAKE_CASE).

Codes are grouped by domain so a failure can be attributed at a glance:
``AEH-0xxx`` generic, ``AEH-1xxx`` syllabus, ``AEH-2xxx`` question papers,
``AEH-3xxx`` exams/answer sheets, ``AEH-4xxx`` LLM/external services.
"""

# ---- Generic (AEH-0xxx) ----
INTERNAL_ERROR = "AEH-0001"
VALIDATION_ERROR = "AEH-0002"
NOT_FOUND = "AEH-0003"
CONFLICT = "AEH-0004"
DATABASE_ERROR = "AEH-0005"

# ---- Syllabus (AEH-1xxx) ----
SYLLABUS_NOT_FOUND = "AEH-1001"
SYLLABUS_INGESTION_FAILED = "AEH-1002"
SYLLABUS_CHUNKS_NOT_FOUND = "AEH-1003"

# ---- Question papers (AEH-2xxx) ----
QUESTION_PAPER_NOT_FOUND = "AEH-2001"
QUESTION_NOT_FOUND = "AEH-2002"
PAPER_GENERATION_FAILED = "AEH-2003"
PAPER_VALIDATION_FAILED = "AEH-2004"

# ---- Exams / answer sheets (AEH-3xxx) ----
EXAM_NOT_FOUND = "AEH-3001"
STUDENT_NOT_FOUND = "AEH-3002"
ANSWER_SHEET_NOT_FOUND = "AEH-3003"
ANSWER_NOT_FOUND = "AEH-3004"
MARK_NOT_FOUND = "AEH-3005"
FILE_PROCESSING_ERROR = "AEH-3006"
OCR_ERROR = "AEH-3007"
SHEET_GRADING_FAILED = "AEH-3008"

# ---- LLM / external services (AEH-4xxx) ----
LLM_ERROR = "AEH-4001"
LLM_RATE_LIMITED = "AEH-4002"
LLM_TIMEOUT = "AEH-4003"
