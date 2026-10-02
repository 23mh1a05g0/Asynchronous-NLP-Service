"""SQLAlchemy ORM models for NLP service."""

from datetime import datetime, timezone
from enum import Enum
from sqlalchemy import Column, String, Text, DateTime, JSON
from app.database import Base


class TaskType(str, Enum):
    """Supported NLP task types."""
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"


class TaskStatus(str, Enum):
    """Lifecycle statuses for NLP tasks."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def utc_now() -> datetime:
    """Return current UTC datetime without timezone offset for database storage."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class NLPTask(Base):
    """Database model for tracking asynchronous NLP task state and results."""

    __tablename__ = "nlp_tasks"

    id = Column(
        String(36),
        primary_key=True,
        index=True,
        nullable=False,
        doc="Unique UUID primary key for the task",
    )
    input_text = Column(
        Text,
        nullable=False,
        doc="Raw input text submitted for NLP processing",
    )
    task_type = Column(
        String(50),
        nullable=False,
        index=True,
        doc="Type of NLP task to execute",
    )
    status = Column(
        String(20),
        nullable=False,
        default=TaskStatus.PENDING.value,
        index=True,
        doc="Current lifecycle state (PENDING, PROCESSING, COMPLETED, FAILED)",
    )
    result = Column(
        JSON,
        nullable=True,
        doc="JSON structure storing completed NLP inference outputs",
    )
    error_message = Column(
        Text,
        nullable=True,
        doc="Detailed error message if task processing failed",
    )
    created_at = Column(
        DateTime,
        nullable=False,
        default=utc_now,
        doc="UTC timestamp when task record was created",
    )
    updated_at = Column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
        doc="UTC timestamp when task record was last updated",
    )

    def __repr__(self) -> str:
        return f"<NLPTask id={self.id} type={self.task_type} status={self.status}>"
