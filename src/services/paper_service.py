"""Business logic for question paper generation (FR-002) and validation (FR-003).

Generation flow: load the syllabus -> create a draft QUESTION_PAPERS row ->
resolve the in-scope units -> retrieve their *verbatim* chunks (splitting
oversized units into sections) -> generate and validate questions per section ->
persist the QUESTIONS and their RUBRIC_CRITERIA transactionally.

Validation flow: load the paper -> run the deterministic checks (marks,
question-type and difficulty distribution, source grounding, syllabus coverage,
near-duplicate detection) -> ask the agent for the advisory ambiguity/duplication
review -> persist ``validation_status`` and move the paper to ``validated`` when
every check passes.
"""

from __future__ import annotations

import math
import re
import uuid
from collections import defaultdict
from dataclasses import replace

from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.paper_generation_agent import (
    ExistingQuestion,
    GeneratedQuestion,
    GenerationRequest,
    PaperGenerationAgent,
    QuestionSpec,
    RevisionRequest,
    RevisionTarget,
)
from src.agents.paper_validation_agent import PaperValidationAgent, ValidationTarget
from src.models.paper import (
    Blueprint,
    DifficultyMix,
    GeneratePaperRequest,
    GeneratePaperResponse,
    PaperValidationReport,
    QuestionPaperRead,
    QuestionValidationResult,
    ValidationCheck,
)
from src.repositories import paper_repository as paper_repo
from src.repositories import syllabus_repository as syllabus_repo
from src.repositories.schema import (
    Question,
    QuestionPaper,
    RubricCriterion,
    Syllabus,
    SyllabusChunk,
)
from src.settings import (
    MAX_PAPER_PROMPT_CHARS,
    PAPER_DUPLICATE_SIMILARITY_THRESHOLD,
    PAPER_REPAIR_MAX_ATTEMPTS,
    Difficulty,
    PaperStatus,
    SyllabusStatus,
    ValidationStatus,
)
from src.utils.exceptions import ConflictError, NotFoundError, RequestValidationError
from src.utils.exceptions import error_codes as code
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PaperService:
    """Orchestrates scope-driven question paper generation and validation."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        agent: PaperGenerationAgent | None = None,
        validation_agent: PaperValidationAgent | None = None,
    ) -> None:
        self._session = session
        self._agent = agent or PaperGenerationAgent()
        self._validation_agent = validation_agent or PaperValidationAgent()

    async def generate(self, payload: GeneratePaperRequest) -> GeneratePaperResponse:
        """Generate a question paper and validate it in a single step.

        The paper and its questions are always persisted; the paper is moved to
        ``validated`` when every FR-003 check passes, and left as ``draft`` (with
        the failing questions marked) otherwise. When the first validation fails,
        the failing questions are sent back to the generator to be rewritten in
        place and re-validated, up to ``PAPER_REPAIR_MAX_ATTEMPTS`` times. Both
        the paper and the final validation report are returned so the caller can
        gate on the outcome.
        """

        syllabus = await syllabus_repo.get(self._session, payload.syllabus_id)
        if syllabus is None or not syllabus.is_active:
            raise NotFoundError(
                f"Syllabus {payload.syllabus_id} not found",
                error_code=code.SYLLABUS_NOT_FOUND,
            )
        if syllabus.status != SyllabusStatus.READY.value:
            raise ConflictError(
                "The syllabus is not ready for generation yet.",
                error_code=code.CONFLICT,
            )

        chunks = await syllabus_repo.list_chunks(self._session, syllabus.id)
        if not chunks:
            raise NotFoundError(
                "The syllabus has no indexed content to generate from.",
                error_code=code.SYLLABUS_CHUNKS_NOT_FOUND,
            )

        units = _ordered_units(chunks, payload.scope)
        if not units:
            raise RequestValidationError("None of the requested units exist in the syllabus.")

        # The scope is stored alongside the blueprint so validation can later
        # recompute the coverage report from persisted data alone.
        blueprint = payload.blueprint.model_dump(mode="json")
        blueprint["scope"] = payload.scope

        paper = QuestionPaper(
            syllabus_id=syllabus.id,
            title=payload.title,
            subject=syllabus.subject,
            class_section=syllabus.class_section,
            duration_minutes=payload.blueprint.duration_minutes,
            total_marks=payload.blueprint.total_marks,
            source="generated",
            status=PaperStatus.DRAFT.value,
            blueprint=blueprint,
        )
        await paper_repo.add_paper(self._session, paper)

        specs = _plan_questions(payload.blueprint)
        allocation = _allocate(specs, units, payload.blueprint.topic_weightage)
        questions = await self._generate_questions(paper, syllabus, units, allocation)
        await paper_repo.add_questions(self._session, questions)

        report = await self._validate_paper(paper, questions)

        attempts = 0
        while not report.passed and attempts < PAPER_REPAIR_MAX_ATTEMPTS:
            attempts += 1
            repaired = await self._repair(paper, questions, report, chunks)
            if not repaired:
                break
            logger.info(
                "Question paper repaired",
                extra={"paper_id": str(paper.id), "attempt": attempts},
            )
            report = await self._validate_paper(paper, questions)

        logger.info(
            "Question paper generated",
            extra={
                "paper_id": str(paper.id),
                "questions": len(questions),
                "passed": report.passed,
                "repair_attempts": attempts,
            },
        )
        return GeneratePaperResponse(
            paper=QuestionPaperRead.from_model(paper, questions),
            validation_report=report,
        )

    async def generate_specs(
        self, paper: QuestionPaper, specs: list[QuestionSpec]
    ) -> list[Question]:
        """Generate questions for ``specs`` across the paper's in-scope units.

        The returned rows are not yet persisted and start at order index 0. Used
        when a chat-driven blueprint change needs additional questions.
        """

        if not specs or paper.syllabus_id is None:
            return []
        syllabus = await syllabus_repo.get(self._session, paper.syllabus_id)
        if syllabus is None:
            return []
        chunks = await syllabus_repo.list_chunks(self._session, paper.syllabus_id)
        units = _ordered_units(chunks, _stored_scope(paper))
        if not units:
            return []

        weights = (paper.blueprint or {}).get("topic_weightage")
        allocation = _allocate(specs, units, weights)
        return await self._generate_questions(paper, syllabus, units, allocation)

    async def list(self) -> list[QuestionPaperRead]:
        """Return all active question papers with their questions."""

        papers = await paper_repo.list_papers(self._session)
        return [QuestionPaperRead.from_model(paper) for paper in papers]

    async def get(self, paper_id: uuid.UUID) -> QuestionPaperRead:
        """Return a single question paper or raise :class:`NotFoundError`."""

        paper = await paper_repo.get_paper(self._session, paper_id)
        if paper is None or not paper.is_active:
            raise NotFoundError(
                f"Question paper {paper_id} not found",
                error_code=code.QUESTION_PAPER_NOT_FOUND,
            )
        return QuestionPaperRead.from_model(paper)

    async def validate(self, paper_id: uuid.UUID) -> PaperValidationReport:
        """Re-run FR-003 validation on a persisted paper and record the outcome.

        Generation already validates inline; this method remains as an internal
        (non-route) entry point for re-validating an existing paper, e.g. after a
        chat-driven edit.
        """

        paper = await paper_repo.get_paper(self._session, paper_id)
        if paper is None or not paper.is_active:
            raise NotFoundError(
                f"Question paper {paper_id} not found",
                error_code=code.QUESTION_PAPER_NOT_FOUND,
            )
        if paper.status == PaperStatus.APPROVED.value:
            raise ConflictError(
                "An approved paper can no longer be re-validated.",
                error_code=code.CONFLICT,
            )
        return await self._validate_paper(paper, list(paper.questions))

    async def _validate_paper(
        self, paper: QuestionPaper, questions: list[Question]
    ) -> PaperValidationReport:
        """Run every FR-003 check over ``questions`` and persist the outcome.

        The deterministic checks (marks, distributions, source grounding,
        syllabus coverage, near-duplicate detection) are authoritative and
        always run. The ambiguity/duplication wording review is advisory and is
        skipped when no LLM is configured.
        """

        questions = sorted(questions, key=lambda item: item.order_index)
        blueprint = Blueprint.model_validate(paper.blueprint) if paper.blueprint else None

        chunk_ids: set[uuid.UUID] | None = None
        units: list[tuple[str, list[SyllabusChunk]]] = []
        if paper.syllabus_id is not None:
            chunks = await syllabus_repo.list_chunks(self._session, paper.syllabus_id)
            chunk_ids = {chunk.id for chunk in chunks}
            units = _ordered_units(chunks, _stored_scope(paper))

        issues: dict[uuid.UUID, list[str]] = defaultdict(list)
        checks = [
            _check_marks_total(blueprint, questions),
            _check_type_distribution(blueprint, questions),
            _check_difficulty_mix(blueprint, questions),
            _check_grounding(questions, chunk_ids, issues),
            _check_coverage(questions, units, issues),
            _check_duplication(questions, issues),
            await self._check_advisory(paper, questions, issues),
        ]

        passed = all(check.passed for check in checks)
        paper.status = PaperStatus.VALIDATED.value if passed else PaperStatus.DRAFT.value
        for question in questions:
            question.validation_status = (
                ValidationStatus.FAILED.value
                if issues.get(question.id)
                else ValidationStatus.PASSED.value
            )
        await self._session.commit()

        logger.info(
            "Question paper validated",
            extra={
                "paper_id": str(paper.id),
                "passed": passed,
                "failed_checks": [check.name for check in checks if not check.passed],
                "flagged_questions": {
                    question.order_index + 1: issues[question.id]
                    for question in questions
                    if issues.get(question.id)
                },
            },
        )
        return PaperValidationReport(
            paper_id=paper.id,
            passed=passed,
            paper_status=paper.status,
            checks=checks,
            questions=[
                QuestionValidationResult(
                    question_id=question.id,
                    order_index=question.order_index,
                    status=question.validation_status or ValidationStatus.PENDING.value,
                    issues=issues.get(question.id, []),
                )
                for question in questions
            ],
        )

    async def _repair(
        self,
        paper: QuestionPaper,
        questions: list[Question],
        report: PaperValidationReport,
        chunks: list[SyllabusChunk],
    ) -> bool:
        """Rewrite the questions flagged by ``report``; return whether any changed.

        Failing questions are grouped by the syllabus chunk they were grounded in
        so each revision call sees the verbatim text it must stay faithful to.
        Only a question's wording and answer are replaced; its type, marks,
        difficulty, order and rubric are preserved so the paper stays aligned
        with the blueprint. Returns ``False`` when nothing could be rewritten
        (no flagged questions, ungrounded questions, or no LLM configured), which
        lets the caller stop retrying.
        """

        if paper.syllabus_id is None:
            return False

        flagged = {
            result.question_id: result.issues
            for result in report.questions
            if result.status == ValidationStatus.FAILED.value
        }
        if not flagged:
            return False

        by_id = {question.id: question for question in questions}
        by_chunk = {chunk.id: chunk for chunk in chunks}
        paper_questions = [
            ExistingQuestion(
                order_index=question.order_index,
                question_text=question.question_text,
                question_type=question.question_type,
                marks=int(question.marks),
                topic=question.topic,
            )
            for question in sorted(questions, key=lambda item: item.order_index)
        ]

        groups: dict[uuid.UUID, list[Question]] = defaultdict(list)
        for question_id in flagged:
            question = by_id.get(question_id)
            if question is None or question.source_chunk_id not in by_chunk:
                continue
            groups[question.source_chunk_id].append(question)
        if not groups:
            return False

        repaired = False
        for chunk_id, targets in groups.items():
            chunk = by_chunk[chunk_id]
            request = RevisionRequest(
                syllabus_id=paper.syllabus_id,
                subject=paper.subject,
                unit_no=chunk.unit_no,
                unit_name=chunk.unit_name,
                topic_name=None,
                chunks=[chunk],
                targets=[
                    RevisionTarget(
                        order_index=question.order_index,
                        question_text=question.question_text,
                        question_type=question.question_type,
                        marks=int(question.marks),
                        difficulty=question.difficulty or Difficulty.MEDIUM.value,
                        issues=flagged[question.id],
                    )
                    for question in targets
                ],
                paper_questions=paper_questions,
            )
            revised = await self._agent.revise(request)
            if not revised:
                continue
            for question, item in zip(targets, revised, strict=True):
                _apply_revision(question, item)
                repaired = True
        return repaired

    async def approve(self, paper_id: uuid.UUID) -> QuestionPaperRead:
        """Move a validated paper to ``approved`` so an exam can use it (FR-003)."""

        paper = await paper_repo.get_paper(self._session, paper_id)
        if paper is None or not paper.is_active:
            raise NotFoundError(
                f"Question paper {paper_id} not found",
                error_code=code.QUESTION_PAPER_NOT_FOUND,
            )
        if paper.status != PaperStatus.VALIDATED.value:
            raise ConflictError(
                "Only a validated paper can be approved.",
                error_code=code.CONFLICT,
            )

        paper.status = PaperStatus.APPROVED.value
        await self._session.commit()
        logger.info("Question paper approved", extra={"paper_id": str(paper.id)})
        return QuestionPaperRead.from_model(paper)

    async def _check_advisory(
        self,
        paper: QuestionPaper,
        questions: list[Question],
        issues: dict[uuid.UUID, list[str]],
    ) -> ValidationCheck:
        """Run the agent's advisory ambiguity/duplication review."""

        targets = [
            ValidationTarget(
                index=question.order_index,
                question_text=question.question_text,
                question_type=question.question_type,
                marks=int(question.marks),
                topic=question.topic,
                options=list(question.options) if question.options else None,
            )
            for question in questions
        ]
        reviews = await self._validation_agent.review(subject=paper.subject, questions=targets)
        if not reviews:
            return ValidationCheck(
                name="ambiguity",
                passed=True,
                summary="Advisory review skipped (no LLM configured).",
            )

        by_index = {question.order_index: question for question in questions}
        flagged = 0
        for review in reviews:
            target = by_index.get(review.index)
            if target is None:
                continue
            if review.ambiguous:
                flagged += 1
                note = "Wording flagged as ambiguous"
                issues[target.id].append(f"{note}: {review.notes}" if review.notes else note)
            duplicate = (
                by_index.get(review.duplicate_of) if review.duplicate_of is not None else None
            )
            if duplicate is not None and duplicate.order_index < target.order_index:
                flagged += 1
                issues[target.id].append(
                    f"Possible duplicate of question {duplicate.order_index + 1}"
                )
                issues[duplicate.id].append(
                    f"Possible duplicate of question {target.order_index + 1}"
                )

        return ValidationCheck(
            name="ambiguity",
            passed=flagged == 0,
            summary=f"{flagged} issue(s) raised across {len(reviews)} reviewed question(s).",
            details=None if flagged == 0 else {"flagged": flagged},
        )

    async def _generate_questions(
        self,
        paper: QuestionPaper,
        syllabus: Syllabus,
        units: list[tuple[str, list[SyllabusChunk]]],
        allocation: dict[str, list[QuestionSpec]],
    ) -> list[Question]:
        """Generate every allocated question, unit by unit and section by section."""

        questions: list[Question] = []
        order = 0
        for unit_key, unit_chunks in units:
            specs = allocation.get(unit_key) or []
            if not specs:
                continue

            sections = _merge_sections(
                _sectionise(unit_chunks, MAX_PAPER_PROMPT_CHARS), len(specs)
            )
            counts = _even_counts(len(specs), len(sections))
            cursor = 0
            for section, count in zip(sections, counts, strict=True):
                if count == 0:
                    continue
                section_specs = specs[cursor : cursor + count]
                cursor += count
                generated = await self._agent.generate(
                    GenerationRequest(
                        syllabus_id=syllabus.id,
                        subject=syllabus.subject,
                        unit_no=unit_key or None,
                        unit_name=_first_value(unit_chunks, "unit_name"),
                        topic_name=None,
                        chunks=section,
                        specs=section_specs,
                    )
                )
                for item in generated:
                    questions.append(_to_question(paper.id, item, order))
                    order += 1
        return questions


def _ordered_units(
    chunks: list[SyllabusChunk], scope: list[str] | None
) -> list[tuple[str, list[SyllabusChunk]]]:
    """Group chunks by unit (document order), optionally restricted to ``scope``.

    Chunks carrying no unit number are document front matter (title, subject,
    class) that precedes the first unit heading. When the syllabus has real
    units they are dropped, so questions are never generated from the header
    block; a syllabus with no unit headings keeps all its chunks.
    """

    selected = [chunk for chunk in chunks if chunk.unit_no] or chunks

    groups: dict[str, list[SyllabusChunk]] = {}
    for chunk in selected:
        groups.setdefault(chunk.unit_no or "", []).append(chunk)

    keys = list(groups)
    if scope:
        wanted = {item.strip() for item in scope if item.strip()}
        keys = [key for key in keys if key in wanted]
    return [(key, groups[key]) for key in keys]


def _plan_questions(blueprint: Blueprint) -> list[QuestionSpec]:
    """Turn the blueprint into an ordered list of required question specs."""

    specs: list[QuestionSpec] = []
    for type_spec in blueprint.question_types:
        specs.extend(
            QuestionSpec(
                question_type=type_spec.type,
                marks=type_spec.marks_each,
                difficulty=Difficulty.MEDIUM,
            )
            for _ in range(type_spec.count)
        )

    difficulties = _difficulty_sequence(len(specs), blueprint.difficulty_mix)
    return [
        replace(spec, difficulty=difficulty)
        for spec, difficulty in zip(specs, difficulties, strict=True)
    ]


def _difficulty_sequence(total: int, mix: DifficultyMix | None) -> list[Difficulty]:
    """Return a per-question difficulty list honouring the blueprint's mix."""

    if total <= 0:
        return []
    if mix is None:
        return [Difficulty.MEDIUM] * total

    counts = _largest_remainder(
        total,
        {Difficulty.EASY: mix.easy, Difficulty.MEDIUM: mix.medium, Difficulty.HARD: mix.hard},
    )
    sequence: list[Difficulty] = []
    for level in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD):
        sequence.extend([level] * counts[level])
    return sequence


def _allocate(
    specs: list[QuestionSpec],
    units: list[tuple[str, list[SyllabusChunk]]],
    weights: dict[str, float] | None,
) -> dict[str, list[QuestionSpec]]:
    """Distribute the required questions across units.

    Allocation follows ``weights`` (topic/unit weightage) when supplied, falling
    back to an even split. Questions are interleaved so each unit receives a mix
    of question types rather than one contiguous block.
    """

    keys = [key for key, _ in units]
    allocation: dict[str, list[QuestionSpec]] = {key: [] for key in keys}
    if not specs or not keys:
        return allocation

    if weights:
        weight_map = {key: max(weights.get(key, 0.0), 0.0) for key in keys}
    else:
        weight_map = {key: 1.0 for key in keys}

    counts = _largest_remainder(len(specs), weight_map)
    if sum(counts.values()) == 0:
        counts = _largest_remainder(len(specs), {key: 1.0 for key in keys})

    pending = [key for key in keys if counts[key] > 0]
    cursor = 0
    for spec in specs:
        while counts[pending[cursor]] == 0:
            cursor = (cursor + 1) % len(pending)
        key = pending[cursor]
        allocation[key].append(spec)
        counts[key] -= 1
        cursor = (cursor + 1) % len(pending)
    return allocation


def _sectionise(chunks: list[SyllabusChunk], budget: int) -> list[list[SyllabusChunk]]:
    """Split a unit's chunks into sections that each fit the prompt budget."""

    sections: list[list[SyllabusChunk]] = []
    current: list[SyllabusChunk] = []
    size = 0
    for chunk in chunks:
        text_len = len(chunk.raw_text or "")
        if current and size + text_len > budget:
            sections.append(current)
            current, size = [], 0
        current.append(chunk)
        size += text_len
    if current:
        sections.append(current)
    return sections


def _merge_sections(
    sections: list[list[SyllabusChunk]], target: int
) -> list[list[SyllabusChunk]]:
    """Merge adjacent sections until there are at most ``target`` of them.

    This guarantees every section can be given at least one question, so no part
    of an oversized unit is silently skipped.
    """

    if target <= 0 or len(sections) <= target:
        return sections

    total = len(sections)
    merged: list[list[SyllabusChunk]] = [[] for _ in range(target)]
    for index, section in enumerate(sections):
        band = min(target - 1, index * target // total)
        merged[band].extend(section)
    return [group for group in merged if group]


def _even_counts(total: int, buckets: int) -> list[int]:
    """Split ``total`` questions across ``buckets`` as evenly as possible."""

    if buckets <= 0:
        return []
    base, extra = divmod(total, buckets)
    return [base + 1 if index < extra else base for index in range(buckets)]


def _largest_remainder[K](total: int, weights: dict[K, float]) -> dict[K, int]:
    """Apportion ``total`` across weighted keys using the largest-remainder method."""

    positive = {key: max(value, 0.0) for key, value in weights.items()}
    weight_total = sum(positive.values())
    if total <= 0 or not positive or weight_total <= 0:
        return {key: 0 for key in weights}

    raw = {key: total * value / weight_total for key, value in positive.items()}
    floors = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(floors.values())
    ranked = sorted(raw, key=lambda key: raw[key] - floors[key], reverse=True)
    for key in ranked[:remaining]:
        floors[key] += 1
    return floors


def _first_value(chunks: list[SyllabusChunk], attribute: str) -> str | None:
    """Return the first non-empty value of ``attribute`` across ``chunks``."""

    for chunk in chunks:
        value = getattr(chunk, attribute, None)
        if value:
            return value
    return None


_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def _stored_scope(paper: QuestionPaper) -> list[str] | None:
    """Return the scope persisted alongside the paper's blueprint, if any."""

    scope = (paper.blueprint or {}).get("scope")
    if isinstance(scope, list):
        return [str(item) for item in scope]
    return None


def _check_marks_total(
    blueprint: Blueprint | None, questions: list[Question]
) -> ValidationCheck:
    """Check the paper's total marks against the blueprint."""

    if blueprint is None:
        return _no_blueprint("marks_total")
    actual = float(sum(question.marks for question in questions))
    expected = float(blueprint.total_marks)
    passed = math.isclose(actual, expected)
    return ValidationCheck(
        name="marks_total",
        passed=passed,
        summary=f"Total marks {'match' if passed else 'do not match'} the blueprint.",
        details={"expected": expected, "actual": actual},
    )


def _check_type_distribution(
    blueprint: Blueprint | None, questions: list[Question]
) -> ValidationCheck:
    """Check the question-type counts and per-type marks against the blueprint."""

    if blueprint is None:
        return _no_blueprint("question_types")

    actual_counts: dict[str, int] = defaultdict(int)
    actual_marks: dict[str, float] = defaultdict(float)
    for question in questions:
        actual_counts[question.question_type] += 1
        actual_marks[question.question_type] += float(question.marks)

    expected_counts = {spec.type.value: spec.count for spec in blueprint.question_types}
    expected_marks = {
        spec.type.value: float(spec.count * spec.marks_each) for spec in blueprint.question_types
    }
    passed = set(actual_counts) <= set(expected_counts) and all(
        actual_counts.get(kind, 0) == count
        and math.isclose(actual_marks.get(kind, 0.0), expected_marks[kind])
        for kind, count in expected_counts.items()
    )
    return ValidationCheck(
        name="question_types",
        passed=passed,
        summary=f"Question types {'match' if passed else 'do not match'} the blueprint.",
        details={"expected": expected_counts, "actual": dict(actual_counts)},
    )


def _check_difficulty_mix(
    blueprint: Blueprint | None, questions: list[Question]
) -> ValidationCheck:
    """Check the easy/medium/hard distribution against the blueprint's mix."""

    if blueprint is None:
        return _no_blueprint("difficulty_mix")
    mix = blueprint.difficulty_mix
    if mix is None:
        return ValidationCheck(
            name="difficulty_mix",
            passed=True,
            summary="No difficulty mix was specified.",
        )

    expected = _largest_remainder(
        len(questions),
        {Difficulty.EASY: mix.easy, Difficulty.MEDIUM: mix.medium, Difficulty.HARD: mix.hard},
    )
    actual = {
        level: sum(1 for question in questions if question.difficulty == level.value)
        for level in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
    }
    passed = actual == expected
    verdict = "matches" if passed else "does not match"
    return ValidationCheck(
        name="difficulty_mix",
        passed=passed,
        summary=f"Difficulty distribution {verdict} the blueprint.",
        details={
            "expected": {level.value: expected[level] for level in expected},
            "actual": {level.value: actual[level] for level in actual},
        },
    )


def _check_grounding(
    questions: list[Question],
    chunk_ids: set[uuid.UUID] | None,
    issues: dict[uuid.UUID, list[str]],
) -> ValidationCheck:
    """Check that every question cites a syllabus chunk that really exists."""

    if chunk_ids is None:
        return ValidationCheck(
            name="source_grounding",
            passed=False,
            summary="The paper is not linked to a syllabus, so grounding cannot be verified.",
        )

    unresolved = 0
    for question in questions:
        if question.source_chunk_id is None or question.source_chunk_id not in chunk_ids:
            unresolved += 1
            issues[question.id].append("source_chunk_id does not resolve to a syllabus chunk")
    passed = unresolved == 0
    return ValidationCheck(
        name="source_grounding",
        passed=passed,
        summary=(
            "Every question is grounded in a syllabus chunk."
            if passed
            else f"{unresolved} question(s) are not grounded in a syllabus chunk."
        ),
        details=None if passed else {"ungrounded": unresolved},
    )


def _check_coverage(
    questions: list[Question],
    units: list[tuple[str, list[SyllabusChunk]]],
    issues: dict[uuid.UUID, list[str]],
) -> ValidationCheck:
    """Check that every in-scope unit contributes at least one question."""

    if not units:
        return ValidationCheck(
            name="syllabus_coverage",
            passed=True,
            summary="There are no in-scope units to cover.",
        )

    chunk_unit = {chunk.id: key for key, chunks in units for chunk in chunks}
    counts: dict[str, int] = {key: 0 for key, _ in units}
    for question in questions:
        key = chunk_unit.get(question.source_chunk_id) if question.source_chunk_id else None
        if key is not None:
            counts[key] += 1

    uncovered = [key for key, count in counts.items() if count == 0]
    passed = not uncovered
    return ValidationCheck(
        name="syllabus_coverage",
        passed=passed,
        summary=(
            f"All {len(counts)} in-scope unit(s) have at least one question."
            if passed
            else f"{len(uncovered)} in-scope unit(s) have no question."
        ),
        details={"questions_per_unit": counts, "uncovered_units": uncovered},
    )


def _check_duplication(
    questions: list[Question], issues: dict[uuid.UUID, list[str]]
) -> ValidationCheck:
    """Check for near-duplicate question wording (deterministic)."""

    texts = [_words(question.question_text) for question in questions]
    pairs = [
        (i, j)
        for i in range(len(questions))
        for j in range(i + 1, len(questions))
        if _similarity(texts[i], texts[j]) >= PAPER_DUPLICATE_SIMILARITY_THRESHOLD
    ]
    for i, j in pairs:
        issues[questions[i].id].append(f"Near-duplicate of question {questions[j].order_index + 1}")
        issues[questions[j].id].append(f"Near-duplicate of question {questions[i].order_index + 1}")

    passed = not pairs
    return ValidationCheck(
        name="duplication",
        passed=passed,
        summary=(
            "No near-duplicate questions detected."
            if passed
            else f"{len(pairs)} near-duplicate pair(s) detected."
        ),
        details=None
        if passed
        else {"pairs": [[questions[i].order_index, questions[j].order_index] for i, j in pairs]},
    )


def _no_blueprint(name: str) -> ValidationCheck:
    """Build a failing check for a paper that carries no blueprint."""

    return ValidationCheck(
        name=name,
        passed=False,
        summary="The paper has no blueprint to validate against.",
    )


def _words(text: str) -> set[str]:
    """Return the lowercased word set of ``text``."""

    return set(_WORD_PATTERN.findall((text or "").lower()))


def _similarity(left: set[str], right: set[str]) -> float:
    """Return the Jaccard similarity of two word sets."""

    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _apply_revision(question: Question, item: GeneratedQuestion) -> None:
    """Replace a question's wording and answer in place, keeping its structure.

    Type, marks, difficulty, order index and rubric criteria are deliberately
    left untouched so the revised paper still matches the blueprint.
    """

    question.question_text = item.question_text
    question.options = item.options
    question.correct_option = item.correct_option
    question.model_answer = item.model_answer
    question.topic = item.topic
    question.source_chunk_id = item.source_chunk_id or question.source_chunk_id
    question.validation_status = ValidationStatus.PENDING.value


def _to_question(paper_id: uuid.UUID, item: GeneratedQuestion, order: int) -> Question:
    """Map a generated question onto a persistable :class:`Question` row."""

    question = Question(
        question_paper_id=paper_id,
        source_chunk_id=item.source_chunk_id,
        order_index=order,
        question_text=item.question_text,
        question_type=item.question_type,
        marks=item.marks,
        options=item.options,
        correct_option=item.correct_option,
        model_answer=item.model_answer,
        topic=item.topic,
        difficulty=item.difficulty,
        validation_status=ValidationStatus.PENDING.value,
    )
    question.rubric_criteria = [
        RubricCriterion(
            criterion_text=rubric.criterion_text,
            max_marks=rubric.max_marks,
            order_index=index,
        )
        for index, rubric in enumerate(item.rubrics)
    ]
    return question
