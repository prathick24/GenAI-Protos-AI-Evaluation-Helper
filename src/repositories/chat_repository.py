"""Data-access layer for conversations and their messages (FR-010)."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.schema import Conversation, ConversationMessage


async def get_conversation(
    session: AsyncSession, conversation_id: uuid.UUID
) -> Conversation | None:
    """Return an active conversation by id, or ``None``."""

    stmt = select(Conversation).where(
        Conversation.id == conversation_id, Conversation.is_active.is_(True)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def create_conversation(
    session: AsyncSession,
    *,
    title: str | None = None,
    exam_id: uuid.UUID | None = None,
    user_id: uuid.UUID | None = None,
) -> Conversation:
    """Persist a new conversation and flush so its id is available."""

    conversation = Conversation(user_id=user_id, exam_id=exam_id, title=title)
    session.add(conversation)
    await session.flush()
    return conversation


async def add_message(
    session: AsyncSession,
    conversation: Conversation,
    *,
    role: str,
    content: str | None = None,
    agent_name: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    tokens: int | None = None,
) -> ConversationMessage:
    """Persist one conversation turn and flush so its id is available."""

    message = ConversationMessage(
        conversation_id=conversation.id,
        role=role,
        content=content,
        agent_name=agent_name,
        tool_calls=tool_calls,
        tokens=tokens,
    )
    session.add(message)
    await session.flush()
    return message
