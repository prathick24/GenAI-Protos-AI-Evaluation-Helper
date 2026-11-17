"""Users and students."""

from __future__ import annotations

from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.repositories.schema.base import AuditMixin, Base


class User(Base, AuditMixin):
    """A teacher account (single-user local demo)."""

    __tablename__ = "users"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="teacher")
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)


class Student(Base, AuditMixin):
    """A student who submits answer sheets."""

    __tablename__ = "students"
    __table_args__ = (UniqueConstraint("roll_no", "class_section", name="uq_students_roll_class"),)

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    roll_no: Mapped[str] = mapped_column(String(50), nullable=False)
    class_section: Mapped[str | None] = mapped_column(String(100), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
