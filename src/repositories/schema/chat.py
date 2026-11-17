"""Conversations and their messages for the multi-agent chat assistant."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.repositories.schema.base import AuditMixin, Base

if TYPE_CHECKING:
    from src.repositories.schema.users import User


class Conversation(Base, AuditMixin):
    """A chat conversation, optionally scoped to an exam."""

    __tablename__ = "conversations"

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    exam_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("exams.id"), nullable=True
    )
    title: Mapped[str | None] = mapped_column(String(300), nullable=True)

    user: Mapped[User | None] = relationship(foreign_keys="Conversation.user_id", lazy="selectin")
    messages: Mapped[list[ConversationMessage]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Conversation id={self.id} title={self.title!r}>"


class ConversationMessage(Base, AuditMixin):
    """A single turn in a conversation, attributed to the responding agent."""

    __tablename__ = "conversation_messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    conversation: Mapped[Conversation] = relationship(back_populates="messages", lazy="selectin")

    def __repr__(self) -> str:
        return f"<ConversationMessage id={self.id} role={self.role!r} agent={self.agent_name!r}>"
