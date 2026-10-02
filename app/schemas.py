"""Pydantic schemas for request validation and response serialization."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
from app.models import TaskType, TaskStatus


class TaskCreateRequest(BaseModel):
    """Payload schema for submitting an asynchronous NLP task."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "I absolutely love this product!",
                    "task_type": "sentiment_analysis",
                },
                {
                    "text": "Elon Musk founded SpaceX in California.",
                    "task_type": "named_entity_recognition",
                },
            ]
        }
    )

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text content to be processed by the NLP model (1-5000 characters, non-whitespace).",
    )
    task_type: TaskType = Field(
        ...,
        description="NLP task type to execute: 'sentiment_analysis' or 'named_entity_recognition'.",
    )

    @field_validator("text")
    @classmethod
    def validate_not_whitespace_only(cls, value: str) -> str:
        """Ensure input text is not empty or composed solely of whitespace."""
        if not value or not value.strip():
            raise ValueError("Input text cannot be blank or contain only whitespace.")
        return value.strip()


class TaskResponse(BaseModel):
    """Immediate response payload returned upon accepting an NLP task."""

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "task_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "PENDING",
            }
        },
    )

    task_id: str = Field(..., description="Unique UUID assigned to the submitted task.")
    status: TaskStatus = Field(..., description="Current status of the task (PENDING).")


class TaskStatusResponse(BaseModel):
    """Status and result response payload for an NLP task query."""

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "task_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "COMPLETED",
                "result": {
                    "label": "POSITIVE",
                    "score": 0.9998,
                },
                "error_message": None,
                "updated_at": "2026-10-02T08:30:00Z",
            }
        },
    )

    task_id: str = Field(..., description="Unique task identifier UUID.")
    status: TaskStatus = Field(..., description="Current lifecycle state of the task.")
    result: Optional[Union[Dict[str, Any], List[Any]]] = Field(
        None, description="NLP inference output when completed; null otherwise."
    )
    error_message: Optional[str] = Field(
        None, description="Error details if the task failed; null otherwise."
    )
    updated_at: datetime = Field(
        ..., description="UTC timestamp when the task was last updated."
    )
