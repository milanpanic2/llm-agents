from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.database.models import Base


class WFQTaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


_STATUS_VALUES = ", ".join(f"'{s.value}'" for s in WFQTaskStatus)


class WFQTask(Base):
    """Single source of truth for the transcription task table: schema (DDL via
    metadata.create_all), nullability, and the row shape workers receive."""

    __tablename__ = "transcription_tasks"
    __table_args__ = (
        CheckConstraint(f"status IN ({_STATUS_VALUES})", name="status_check"),
        Index("transcription_tasks_ctx_started_idx", "context_id", "started_at"),
        Index("transcription_tasks_pending_idx", "created_at",
              postgresql_where=text("status = 'pending'")),
        Index("transcription_tasks_running_idx", "started_at",
              postgresql_where=text("status = 'running'")),

    )

    id: Mapped[str] = mapped_column(Text, primary_key=True, server_default=text("gen_random_uuid()::text"))
    context_id: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default=WFQTaskStatus.PENDING.value)
    created_by: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    priority: Mapped[str] = mapped_column(String(50), nullable=False, server_default="standard")
