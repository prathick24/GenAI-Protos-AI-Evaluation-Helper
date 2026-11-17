"""Streamlit frontend for the AI Evaluation Helper (FR-012).

A single-page app that talks ONLY to the FastAPI backend over HTTP; it has no
database or LLM access. Every view renders backend data and surfaces errors with
an explicit retry action.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from collections.abc import Callable
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_BASE_URL = os.getenv("AEH_API_BASE_URL", "http://localhost:8000/api/v1")
# Generation with repairs can take several minutes; allow a long read timeout.
REQUEST_TIMEOUT = float(os.getenv("AEH_REQUEST_TIMEOUT", "600"))
QUESTION_TYPES = ("mcq", "short", "long")
_QUESTION_TYPE_ORDER = {"mcq": 0, "short": 1, "long": 2}
_QUESTION_TYPE_LABEL = {
    "mcq": "Multiple Choice",
    "short": "Short Answer",
    "long": "Long Answer",
}


class ApiError(Exception):
    """A failed backend call with a message safe to show the teacher."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ApiClient:
    """Thin HTTP client for the backend API."""

    def __init__(self, base_url: str) -> None:
        timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=10.0)
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def get_bytes(self, path: str, *, params: dict[str, Any] | None = None) -> bytes:
        try:
            response = self._client.get(path, params=params)
        except httpx.TimeoutException as exc:
            raise ApiError(
                f"Request timed out after {REQUEST_TIMEOUT:.0f}s. The server may still "
                "be working - refresh the list before retrying."
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"Backend unreachable at {API_BASE_URL}: {exc}") from exc

        if response.status_code >= 400:
            raise ApiError(_error_message(response), response.status_code)
        return response.content

    def post(
        self,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        return self._request("POST", path, json=json, files=files, data=data)

    def patch(self, path: str, *, json: dict[str, Any]) -> Any:
        return self._request("PATCH", path, json=json)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiError(
                f"Request timed out after {REQUEST_TIMEOUT:.0f}s. The server may still "
                "be working - refresh the list before retrying."
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"Backend unreachable at {API_BASE_URL}: {exc}") from exc

        if response.status_code >= 400:
            raise ApiError(_error_message(response), response.status_code)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()


def _error_message(response: httpx.Response) -> str:
    """Extract the backend's error message from a failed response."""

    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    if isinstance(payload, dict):
        message = payload.get("message") or payload.get("detail") or payload.get("error")
        if message:
            return f"{message} ({payload.get('error', response.status_code)})"
    return f"HTTP {response.status_code}"


def _run(action: Callable[[], Any], *, key: str) -> Any:
    """Run an API action, showing the error and a retry action on failure."""

    try:
        return action()
    except ApiError as exc:
        st.error(str(exc))
        if st.button("Retry", key=f"{key}_{time.time_ns()}"):
            st.rerun()
        return None


def _api() -> ApiClient:
    return ApiClient(API_BASE_URL)


def _section_letter(index: int) -> str:
    letters: list[str] = []
    value = index
    while True:
        value, remainder = divmod(value, 26)
        letters.append(chr(65 + remainder))
        if value == 0:
            break
        value -= 1
    return "".join(reversed(letters))


def _paper_sections(
    questions: list[dict[str, Any]],
) -> list[tuple[str, list[tuple[int, dict[str, Any]]]]]:
    """Group questions by marks then type, with continuous display numbers."""

    groups: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for question in questions:
        key = (float(question["marks"]), question["question_type"])
        groups.setdefault(key, []).append(question)

    ordered = sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            _QUESTION_TYPE_ORDER.get(item[0][1], len(_QUESTION_TYPE_ORDER)),
            item[0][1],
        ),
    )

    sections: list[tuple[str, list[tuple[int, dict[str, Any]]]]] = []
    number = 1
    for index, ((marks, question_type), rows) in enumerate(ordered):
        letter = _section_letter(index)
        marks_text = f"{marks:g} mark" if marks == 1 else f"{marks:g} marks"
        type_label = _QUESTION_TYPE_LABEL.get(
            question_type, question_type.replace("_", " ").title()
        )
        numbered: list[tuple[int, dict[str, Any]]] = []
        for question in sorted(rows, key=lambda item: item["order_index"]):
            numbered.append((number, question))
            number += 1
        sections.append((f"Section {letter} - {marks_text} ({type_label})", numbered))
    return sections


def _pdf_filename(paper: dict[str, Any], *, include_answers: bool) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", paper["title"]).strip("._-")
    suffix = "teacher" if include_answers else "student"
    return f"{(base or 'question-paper')[:80]}_{suffix}.pdf"


def _paper_pdf(
    api: ApiClient, paper: dict[str, Any], *, include_answers: bool
) -> bytes | None:
    suffix = "teacher" if include_answers else "student"
    cache_key = f"paper_pdf_{paper['id']}_{suffix}_{paper.get('updated_at', '')}"
    if cache_key not in st.session_state:
        content = _run(
            lambda: api.get_bytes(
                f"/question-papers/{paper['id']}/export.pdf",
                params={"include_answers": include_answers},
            ),
            key=cache_key,
        )
        if content:
            st.session_state[cache_key] = content
    return st.session_state.get(cache_key)


# --------------------------------------------------------------------------- #
# Syllabus
# --------------------------------------------------------------------------- #
def _syllabus_view(api: ApiClient) -> None:
    st.header("Syllabus")

    with st.form("syllabus_upload"):
        uploaded = st.file_uploader("Syllabus (PDF or TXT)", type=["pdf", "txt"])
        title = st.text_input("Title")
        subject = st.text_input("Subject")
        class_section = st.text_input("Class / section")
        scope = st.text_input("In-scope units (comma-separated, optional)")
        submitted = st.form_submit_button("Upload and index")

    if submitted:
        if uploaded is None:
            st.warning("Choose a syllabus file first.")
        else:
            files = [
                (
                    "file",
                    (
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type or "application/octet-stream",
                    ),
                )
            ]
            data = {
                "title": title or uploaded.name,
                "subject": subject or None,
                "class_section": class_section or None,
                "scope": scope or None,
            }
            result = _run(
                lambda: api.post("/syllabus/upload", files=files, data=data),
                key="syllabus_upload",
            )
            if result:
                st.success(f"Indexed {result['title']} ({result['status']}).")

    syllabi = _run(lambda: api.get("/syllabus"), key="syllabus_list")
    if not syllabi:
        st.info("No syllabi yet. Upload one above.")
        return

    for syllabus in syllabi:
        with st.expander(f"{syllabus['title']} - {syllabus.get('subject') or 'no subject'}"):
            st.caption(
                f"version {syllabus['version']} - {syllabus['input_type']} - "
                f"{syllabus['status']}"
            )
            topics = _run(
                lambda sid=syllabus["id"]: api.get(f"/syllabus/{sid}/topics"),
                key=f"topics_{syllabus['id']}",
            )
            if topics:
                st.dataframe(
                    [
                        {
                            "unit": unit["unit_no"],
                            "unit name": unit["unit_name"],
                            "topics": ", ".join(unit["topics"]),
                        }
                        for unit in topics
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


# --------------------------------------------------------------------------- #
# Generate paper
# --------------------------------------------------------------------------- #
def _generate_view(api: ApiClient) -> None:
    st.header("Generate Paper")
    syllabi = _run(lambda: api.get("/syllabus"), key="generate_syllabi")
    if not syllabi:
        st.info("Upload a syllabus before generating a paper.")
        return

    labels = {
        item["id"]: f"{item['title']} ({item.get('subject') or 'no subject'})"
        for item in syllabi
    }
    with st.form("generate_paper"):
        syllabus_id = st.selectbox(
            "Syllabus", options=list(labels), format_func=lambda key: labels[key]
        )
        title = st.text_input("Paper title", value="Midterm")
        duration = st.number_input("Duration (minutes)", min_value=1, value=90)

        st.markdown("**Question types** (set count to 0 to skip)")
        specs: list[dict[str, Any]] = []
        for kind in QUESTION_TYPES:
            left, middle, right = st.columns(3)
            count = left.number_input(f"{kind} count", min_value=0, value=1, key=f"count_{kind}")
            marks = middle.number_input(
                f"{kind} marks each", min_value=1, value=5, key=f"marks_{kind}"
            )
            right.caption(f"{count * marks} marks")
            if count > 0:
                specs.append({"type": kind, "count": int(count), "marks_each": int(marks)})

        st.markdown("**Difficulty mix** (leave all zero to skip)")
        easy, medium, hard = st.columns(3)
        easy_weight = easy.number_input("easy", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        medium_weight = medium.number_input(
            "medium", min_value=0.0, max_value=1.0, value=0.5, step=0.1
        )
        hard_weight = hard.number_input("hard", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

        submitted = st.form_submit_button("Generate and validate")

    if submitted:
        total = sum(spec["count"] * spec["marks_each"] for spec in specs)
        if not specs or total <= 0:
            st.error("Add at least one question type before generating.")
        else:
            payload: dict[str, Any] = {
                "syllabus_id": syllabus_id,
                "title": title,
                "blueprint": {
                    "total_marks": total,
                    "duration_minutes": int(duration),
                    "question_types": specs,
                },
            }
            weights = {
                "easy": float(easy_weight),
                "medium": float(medium_weight),
                "hard": float(hard_weight),
            }
            if sum(weights.values()) > 0:
                payload["blueprint"]["difficulty_mix"] = weights
            result = _run(
                lambda: api.post("/question-papers/generate", json=payload),
                key="generate_paper",
            )
            if result:
                st.session_state["generated_paper"] = result["paper"]
                st.session_state["validation_report"] = result["validation_report"]

    paper = st.session_state.get("generated_paper")
    report = st.session_state.get("validation_report")
    if paper:
        _render_paper(api, paper, report)


def _render_paper(api: ApiClient, paper: dict[str, Any], report: dict[str, Any] | None) -> None:
    st.subheader(f"{paper['title']} - {paper['total_marks']} marks ({paper['status']})")
    if report:
        st.success("Validation passed." if report["passed"] else "Validation needs attention.")
        st.dataframe(
            [
                {
                    "check": check["name"],
                    "passed": check["passed"],
                    "summary": check["summary"],
                }
                for check in report["checks"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with st.container(height=420):
        for label, questions in _paper_sections(paper["questions"]):
            st.markdown(f"#### {label}")
            for number, question in questions:
                with st.expander(
                    f"Q{number} [{question['question_type']}] "
                    f"{question['marks']:g} marks - {question.get('topic') or 'general'}"
                ):
                    st.write(question["question_text"])
                    if question.get("options"):
                        st.write("Options: " + ", ".join(question["options"]))
                    if question.get("model_answer"):
                        st.caption("Model answer: " + question["model_answer"])
                    if question.get("rubric_criteria"):
                        st.table(
                            [
                                {
                                    "criterion": item["criterion_text"],
                                    "marks": item["max_marks"],
                                }
                                for item in question["rubric_criteria"]
                            ]
                        )

    st.markdown("#### Downloads")
    student_pdf = _paper_pdf(api, paper, include_answers=False)
    teacher_pdf = _paper_pdf(api, paper, include_answers=True)
    student_col, teacher_col = st.columns(2)
    if student_pdf:
        student_col.download_button(
            "Download student PDF",
            data=student_pdf,
            file_name=_pdf_filename(paper, include_answers=False),
            mime="application/pdf",
            key=f"download_student_{paper['id']}",
            help="Questions only, with no answers or rubric.",
        )
    if teacher_pdf:
        teacher_col.download_button(
            "Download teacher PDF",
            data=teacher_pdf,
            file_name=_pdf_filename(paper, include_answers=True),
            mime="application/pdf",
            key=f"download_teacher_{paper['id']}",
            help="Question paper followed by the answer key and rubric.",
        )

    if paper["status"] == "validated":
        if st.button("Approve paper", key=f"approve_paper_{paper['id']}"):
            approved = _run(
                lambda: api.post(f"/question-papers/{paper['id']}/approve"),
                key="approve_paper",
            )
            if approved:
                st.session_state["generated_paper"] = approved
                st.success("Paper approved; it can now be used for an exam.")
    elif paper["status"] == "draft":
        st.info("This paper is still a draft; resolve the validation issues before approving.")


# --------------------------------------------------------------------------- #
# Exams and sheets
# --------------------------------------------------------------------------- #
def _exams_view(api: ApiClient) -> None:
    st.header("Exams & Sheets")
    papers = _run(
        lambda: api.get("/question-papers", params={"status": "approved"}),
        key="exams_papers",
    )
    approved = [paper for paper in (papers or []) if paper["status"] == "approved"]

    with st.form("create_exam"):
        if approved:
            paper_labels = {paper["id"]: paper["title"] for paper in approved}
            paper_id = st.selectbox(
                "Approved paper",
                options=list(paper_labels),
                format_func=lambda key: paper_labels[key],
            )
        else:
            paper_id = None
            st.info("Approve a question paper before creating an exam.")
        title = st.text_input("Exam title", value="Midterm exam")
        exam_date = st.date_input("Exam date")
        submitted = st.form_submit_button("Create exam")

    if submitted and paper_id:
        exam = _run(
            lambda: api.post(
                "/exams",
                json={
                    "title": title,
                    "question_paper_id": paper_id,
                    "exam_date": exam_date.isoformat(),
                },
            ),
            key="create_exam",
        )
        if exam:
            st.success(f"Created exam {exam['title']}.")

    exams = _run(lambda: api.get("/exams"), key="exams_list")
    if not exams:
        st.info("No exams yet.")
        return

    exam_labels = {exam["id"]: f"{exam['title']} ({exam['total_marks']} marks)" for exam in exams}
    exam_id = st.selectbox(
        "Exam", options=list(exam_labels), format_func=lambda key: exam_labels[key]
    )

    st.subheader("Answer sheets")
    uploaded = st.file_uploader(
        "Upload answer sheets",
        type=["txt", "pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    student_names: list[str] = []
    if uploaded:
        for index, item in enumerate(uploaded):
            student_names.append(
                st.text_input(
                    f"Student name for {item.name}",
                    key=f"student_name_{exam_id}_{index}",
                    placeholder="Name on the answer sheet",
                    help="Shown on the sheet list, review page and marksheet.",
                )
            )
    input_type = st.selectbox("Input type", options=["auto", "typed", "scanned"])
    if st.button("Upload sheets", key="upload_sheets") and uploaded:
        missing = [
            item.name
            for item, name in zip(uploaded, student_names, strict=True)
            if not name.strip()
        ]
        if missing:
            st.warning("Enter the student name for: " + ", ".join(missing))
        else:
            results: list[dict[str, Any]] = []
            progress = st.progress(0.0, text="Uploading sheets...")
            for index, (item, name) in enumerate(
                zip(uploaded, student_names, strict=True)
            ):
                uploaded_result = _run(
                    lambda item=item, name=name: api.post(
                        f"/exams/{exam_id}/answer-sheets/upload",
                        files=[
                            (
                                "files",
                                (
                                    item.name,
                                    item.getvalue(),
                                    item.type or "application/octet-stream",
                                ),
                            )
                        ],
                        data={"input_type": input_type, "student_name": name.strip()},
                    ),
                    key=f"upload_sheets_{index}",
                )
                if uploaded_result:
                    results.extend(uploaded_result)
                progress.progress(
                    (index + 1) / len(uploaded),
                    text=f"Uploaded {index + 1}/{len(uploaded)}",
                )
            if results:
                st.dataframe(results, use_container_width=True, hide_index=True)

    sheets = _run(
        lambda: api.get(f"/exams/{exam_id}/answer-sheets"), key="exam_sheets"
    )
    if not sheets:
        st.info("No sheets uploaded for this exam yet.")
        return
    sheets = sorted(sheets, key=lambda item: item["created_at"], reverse=True)

    with st.container(height=320):
        for sheet in sheets:
            label = sheet.get("student_name") or f"Unnamed sheet {sheet['id'][:8]}"
            left, middle, right = st.columns([3, 2, 2])
            left.write(f"**{label}**")
            left.caption(sheet["file_path"] or "no file")
            middle.write(f"status: **{sheet['sheet_status']}**")
            if right.button("Segment & grade", key=f"grade_{sheet['id']}"):
                _run(
                    lambda sid=sheet["id"]: api.post(f"/answer-sheets/{sid}/segment"),
                    key="segment",
                )
                graded = _run(
                    lambda sid=sheet["id"]: api.post(f"/answer-sheets/{sid}/grade"),
                    key="grade",
                )
                if graded:
                    st.session_state["last_job"] = {
                        "sheet": sheet["id"],
                        "student_name": label,
                        "status": graded["sheet_status"],
                        "total": graded["total_awarded_marks"],
                    }
                    st.rerun()

    job = st.session_state.get("last_job")
    if job:
        st.success(
            f"{job['student_name']} is {job['status']} with {job['total']:g} marks."
        )


# --------------------------------------------------------------------------- #
# Review
# --------------------------------------------------------------------------- #
def _review_view(api: ApiClient) -> None:
    st.header("Review")
    exams = _run(lambda: api.get("/exams"), key="review_exams")
    if not exams:
        st.info("Create an exam and upload sheets first.")
        return

    exam_labels = {exam["id"]: exam["title"] for exam in exams}
    exam_id = st.selectbox(
        "Exam", options=list(exam_labels), format_func=lambda key: exam_labels[key]
    )
    sheets = _run(lambda: api.get(f"/exams/{exam_id}/answer-sheets"), key="review_sheets")
    if not sheets:
        st.info("No sheets for this exam.")
        return
    sheets = sorted(sheets, key=lambda item: item["created_at"], reverse=True)

    sheet_labels = {
        sheet["id"]: (
            f"{sheet.get('student_name') or 'Unnamed sheet ' + sheet['id'][:8]}"
            f" - {sheet['sheet_status']}"
        )
        for sheet in sheets
    }
    sheet_id = st.selectbox(
        "Answer sheet", options=list(sheet_labels), format_func=lambda key: sheet_labels[key]
    )
    detail = _run(lambda: api.get(f"/answer-sheets/{sheet_id}"), key="review_detail")
    if not detail:
        return

    name = detail.get("student_name") or ""
    st.subheader(name or f"Unnamed sheet {detail['id'][:8]}")
    st.caption(
        f"status: **{detail['sheet_status']}** - total: "
        f"**{detail.get('total_awarded_marks') or 0:g}**"
    )
    with st.form(f"rename_sheet_{sheet_id}"):
        new_name = st.text_input("Student name", value=name)
        rename = st.form_submit_button("Save name")
    if rename:
        updated = _run(
            lambda: api.patch(
                f"/answer-sheets/{sheet_id}",
                json={"student_name": new_name.strip() or None},
            ),
            key="rename_sheet",
        )
        if updated:
            st.success("Student name saved.")
            st.rerun()

    feedback = detail.get("feedback")
    if feedback:
        st.info(
            f"**Feedback**\n\n{feedback.get('overall_comments') or ''}\n\n"
            f"**Strengths:** {feedback.get('strengths') or '-'}\n\n"
            f"**Improvements:** {feedback.get('improvements') or '-'}"
        )

    with st.container(height=520):
        for answer in detail["answers"]:
            mark = answer.get("mark")
            header = (
                f"Q{answer['order_index'] + 1} - "
                f"{'answered' if answer['is_answered'] else 'blank'}"
            )
            if mark:
                header += (
                    f" - {mark['awarded_marks']:g}/{mark['max_marks']:g}"
                    f" - verifier: {mark.get('verifier_status') or 'n/a'}"
                )
            with st.expander(header):
                st.write(answer.get("answer_text") or "_No answer text_")
                if not mark:
                    st.warning("This answer has not been marked yet.")
                    continue
                st.write(f"**Evidence:** {mark.get('evidence') or '-'}")
                st.write(f"**Reasoning:** {mark.get('reasoning') or '-'}")
                st.write(
                    f"**Graded by:** {mark.get('graded_by') or 'n/a'} - "
                    f"**overridden:** {mark.get('is_overridden')}"
                )
                if mark.get("original_marks") is not None:
                    st.caption(f"Original AI mark: {mark['original_marks']:g}")
                if mark.get("rubric_scores"):
                    st.table(
                        [
                            {
                                "criterion": item.get("criterion_text"),
                                "awarded": item.get("awarded_marks"),
                                "max": item.get("max_marks"),
                            }
                            for item in mark["rubric_scores"]
                        ]
                    )
                with st.form(f"override_{mark['id']}"):
                    awarded = st.number_input(
                        "New mark",
                        min_value=0.0,
                        max_value=float(mark["max_marks"]),
                        value=float(mark["awarded_marks"]),
                        step=0.5,
                    )
                    reason = st.text_input("Reason")
                    override = st.form_submit_button("Override mark")
                if override:
                    updated = _run(
                        lambda mark_id=mark["id"], value=float(awarded), note=reason or None: (
                            api.patch(
                                f"/marks/{mark_id}/override",
                                json={"awarded_marks": value, "reason": note},
                            )
                        ),
                        key="override_mark",
                    )
                    if updated:
                        st.success("Mark overridden.")
                        st.rerun()

    if detail["sheet_status"] != "approved":
        if st.button("Approve sheet", key=f"approve_sheet_{sheet_id}"):
            approved = _run(
                lambda: api.post(f"/answer-sheets/{sheet_id}/approve"),
                key="approve_sheet",
            )
            if approved:
                st.success("Sheet approved.")
                st.rerun()


# --------------------------------------------------------------------------- #
# Marksheet and analytics
# --------------------------------------------------------------------------- #
def _results_view(api: ApiClient) -> None:
    st.header("Marksheet & Analytics")
    exams = _run(lambda: api.get("/exams"), key="results_exams")
    if not exams:
        st.info("No exams yet.")
        return

    exam_labels = {exam["id"]: exam["title"] for exam in exams}
    exam_id = st.selectbox(
        "Exam", options=list(exam_labels), format_func=lambda key: exam_labels[key]
    )

    marksheet = _run(lambda: api.get(f"/exams/{exam_id}/marksheet"), key="marksheet")
    if marksheet:
        st.subheader(f"Marksheet ({marksheet['total_marks']} marks)")
        question_ids = sorted(
            {key for row in marksheet["rows"] for key in row["per_question"]}
        )
        st.dataframe(
            [
                {
                    "student": row["student_name"],
                    "total": row["total_awarded_marks"],
                    **{
                        f"Q{index + 1}": row["per_question"].get(key)
                        for index, key in enumerate(question_ids)
                    },
                }
                for row in marksheet["rows"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    analytics = _run(lambda: api.get(f"/exams/{exam_id}/analytics"), key="analytics")
    if not analytics:
        return

    left, middle, right = st.columns(3)
    left.metric("Average", f"{analytics['average']:g}")
    middle.metric("Highest", f"{analytics['highest']:g}")
    right.metric("Lowest", f"{analytics['lowest']:g}")

    st.subheader("Score distribution")
    distribution = analytics["distribution"]
    st.bar_chart(
        pd.DataFrame(
            {"students": list(distribution.values())},
            index=list(distribution.keys()),
        )
    )

    st.subheader("Topic strength")
    st.table(
        [
            {"topic": topic, "awarded ratio": ratio}
            for topic, ratio in analytics["per_topic_strength"].items()
        ]
    )


# --------------------------------------------------------------------------- #
# Assistant
# --------------------------------------------------------------------------- #
def _assistant_view(api: ApiClient) -> None:
    st.header("Assistant")
    exams = _run(lambda: api.get("/exams"), key="assistant_exams") or []
    scope = {"all material": None}
    scope.update({exam["title"]: exam["id"] for exam in exams})
    choice = st.selectbox("Scope", options=list(scope))

    messages = st.session_state.setdefault("assistant_messages", [])
    conversation_id = st.session_state.get("assistant_conversation")

    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            for citation in message.get("citations", []):
                st.caption(
                    f"{citation['source_type']} - "
                    f"{citation.get('topic_name') or citation.get('unit_no') or ''}: "
                    f"{citation['snippet']}"
                )

    question = st.chat_input("Ask about the syllabus or papers")
    if question:
        messages.append({"role": "user", "content": question, "citations": []})
        payload: dict[str, Any] = {"question": question}
        if scope[choice]:
            payload["exam_id"] = scope[choice]
        if conversation_id:
            payload["conversation_id"] = conversation_id
        result = _run(lambda: api.post("/chat/ask", json=payload), key="ask")
        if result:
            st.session_state["assistant_conversation"] = result["conversation_id"]
            messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result["citations"],
                }
            )
        else:
            messages.pop()
        st.rerun()

    if messages and st.button("Clear conversation", key="clear_chat"):
        st.session_state["assistant_messages"] = []
        st.session_state.pop("assistant_conversation", None)
        st.rerun()


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="AI Evaluation Helper", page_icon="📝", layout="wide")
    st.sidebar.title("AI Evaluation Helper")
    st.sidebar.caption(f"Backend: {API_BASE_URL}")
    view = st.sidebar.radio(
        "View",
        [
            "Syllabus",
            "Generate Paper",
            "Exams & Sheets",
            "Review",
            "Marksheet & Analytics",
            "Assistant",
        ],
    )
    st.session_state.setdefault("session_id", str(uuid.uuid4())[:8])
    st.sidebar.caption(f"Session {st.session_state['session_id']}")

    api = _api()
    views = {
        "Syllabus": _syllabus_view,
        "Generate Paper": _generate_view,
        "Exams & Sheets": _exams_view,
        "Review": _review_view,
        "Marksheet & Analytics": _results_view,
        "Assistant": _assistant_view,
    }
    views[view](api)


if __name__ == "__main__":
    main()
