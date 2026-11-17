"""SQLAlchemy ORM models mapped to the AI Evaluation Helper schema.

Importing this package registers every model on :class:`Base.metadata`, which
is the single source of truth for the 17 tables created at startup.
"""

from __future__ import annotations

from src.repositories.schema.base import AuditMixin, Base
from src.repositories.schema.chat import Conversation, ConversationMessage
from src.repositories.schema.exams import Exam
from src.repositories.schema.grading import Answer, AnswerSheet, Feedback, Mark
from src.repositories.schema.observability import AgentRun, ErrorLog, ModelCost
from src.repositories.schema.papers import Question, QuestionPaper, RubricCriterion
from src.repositories.schema.syllabus import Syllabus, SyllabusChunk
from src.repositories.schema.users import Student, User

__all__ = [
    "AgentRun",
    "Answer",
    "AnswerSheet",
    "AuditMixin",
    "Base",
    "Conversation",
    "ConversationMessage",
    "ErrorLog",
    "Exam",
    "Feedback",
    "Mark",
    "ModelCost",
    "Question",
    "QuestionPaper",
    "RubricCriterion",
    "Student",
    "Syllabus",
    "SyllabusChunk",
    "User",
]
