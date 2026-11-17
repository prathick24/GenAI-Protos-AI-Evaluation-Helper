"""Paper chat router agent (FR-010).

Turns a teacher's free-text instruction about a paper into a structured
:class:`ChangePlan`. The LLM is used only for this interpretation step and for
the conversational reply; the executors in the chat service apply the plan.
When no LLM is configured the router returns an explanatory reply with no
operations, so the endpoint still behaves predictably offline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from src.client import llm
from src.models.chat import ChangeOperation, ChangePlan
from src.prompt import paper_chat as prompt
from src.repositories.observability_repository import default_usage_recorder
from src.settings import CHAT_ROUTER_TEMPERATURE
from src.utils.exceptions import AppError
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.repositories.schema import QuestionPaper

logger = get_logger(__name__)

_AGENT_NAME = "paper_chat_router"

_OFFLINE_REPLY = (
    "I can't interpret paper changes without an LLM configured. "
    "Set an LLM API key to enable chat-driven edits."
)
_FALLBACK_REPLY = "Done — I applied the requested change to the paper."


class PaperChatRouterAgent:
    """Interprets a teacher's message as a plan of paper changes."""

    async def plan(self, *, paper: QuestionPaper, message: str) -> ChangePlan:
        """Return the change plan for ``message`` against ``paper``."""

        if not llm.is_available():
            logger.warning(
                "No LLM key configured; the chat router cannot interpret instructions",
                extra={"task": _AGENT_NAME},
            )
            return ChangePlan(reply=_OFFLINE_REPLY)

        system, user = prompt.build_prompt(paper=paper, message=message)
        try:
            result = await llm.complete_json(
                system=system,
                user=user,
                task=_AGENT_NAME,
                temperature=CHAT_ROUTER_TEMPERATURE,
                sink=default_usage_recorder(),
            )
            payload = llm.extract_json_object(result.text)
        except AppError as exc:
            logger.warning(
                "Chat router could not parse the model output",
                extra={"task": _AGENT_NAME, "error": exc.error_code},
            )
            return ChangePlan(reply=_OFFLINE_REPLY)

        logger.info(
            "Paper chat router LLM call",
            extra={
                "task": _AGENT_NAME,
                "model": result.model,
                "total_tokens": result.total_tokens,
                "latency_ms": result.latency_ms,
            },
        )
        return _parse_plan(payload)


def _parse_plan(payload: dict[str, object]) -> ChangePlan:
    """Leniently map the raw payload onto a plan, ignoring unusable operations."""

    operations: list[ChangeOperation] = []
    raw = payload.get("operations")
    if isinstance(raw, list):
        for item in raw:
            operation = _parse_operation(item)
            if operation is not None:
                operations.append(operation)

    reply = str(payload.get("reply") or "").strip() or _FALLBACK_REPLY
    return ChangePlan(reply=reply, operations=operations)


def _parse_operation(item: object) -> ChangeOperation | None:
    """Validate one raw operation, returning ``None`` when it is unusable."""

    if not isinstance(item, dict):
        return None
    cleaned = {key: value for key, value in item.items() if value is not None}
    try:
        return ChangeOperation.model_validate(cleaned)
    except ValidationError:
        logger.warning(
            "Discarding an operation the model returned in an unusable shape",
            extra={"task": _AGENT_NAME, "operation": cleaned.get("op")},
        )
        return None


__all__ = ["PaperChatRouterAgent"]
