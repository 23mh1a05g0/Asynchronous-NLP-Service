"""Unit tests for Celery background tasks and state transitions."""

import uuid
from unittest.mock import MagicMock, patch
from sqlalchemy.orm import Session
from app.models import NLPTask, TaskStatus, TaskType
from app.tasks import process_text_task
from tests.conftest import TestingSessionLocal


def test_task_sentiment_success_transition(db_session: Session):
    """Test task execution flow: PENDING -> PROCESSING -> COMPLETED for sentiment analysis."""
    task_id = str(uuid.uuid4())
    task = NLPTask(
        id=task_id,
        input_text="I absolutely love this product!",
        task_type=TaskType.SENTIMENT_ANALYSIS.value,
        status=TaskStatus.PENDING.value,
    )
    db_session.add(task)
    db_session.commit()

    expected_result = {"label": "POSITIVE", "score": 0.98}

    mock_engine = MagicMock()
    mock_engine.analyze_sentiment.return_value = expected_result

    with patch("app.tasks.SessionLocal", side_effect=TestingSessionLocal), \
         patch("app.tasks.get_nlp_engine", return_value=mock_engine):
        task_output = process_text_task(task_id)

    assert task_output["status"] == "COMPLETED"
    assert task_output["task_id"] == task_id
    assert task_output["result"] == expected_result

    # Verify database state after task execution
    updated_task = db_session.query(NLPTask).filter(NLPTask.id == task_id).first()
    assert updated_task.status == TaskStatus.COMPLETED.value
    assert updated_task.result == expected_result
    assert updated_task.error_message is None
    mock_engine.analyze_sentiment.assert_called_once_with("I absolutely love this product!")


def test_task_ner_success_transition(db_session: Session):
    """Test task execution flow: PENDING -> PROCESSING -> COMPLETED for named entity recognition."""
    task_id = str(uuid.uuid4())
    task = NLPTask(
        id=task_id,
        input_text="Elon Musk founded SpaceX in California.",
        task_type=TaskType.NAMED_ENTITY_RECOGNITION.value,
        status=TaskStatus.PENDING.value,
    )
    db_session.add(task)
    db_session.commit()

    expected_entities = {
        "entities": [
            {"text": "Elon Musk", "label": "PERSON", "start": 0, "end": 9},
            {"text": "SpaceX", "label": "ORG", "start": 20, "end": 26},
            {"text": "California", "label": "GPE", "start": 30, "end": 40},
        ]
    }

    mock_engine = MagicMock()
    mock_engine.extract_entities.return_value = expected_entities

    with patch("app.tasks.SessionLocal", side_effect=TestingSessionLocal), \
         patch("app.tasks.get_nlp_engine", return_value=mock_engine):
        task_output = process_text_task(task_id)

    assert task_output["status"] == "COMPLETED"
    assert task_output["result"] == expected_entities

    updated_task = db_session.query(NLPTask).filter(NLPTask.id == task_id).first()
    assert updated_task.status == TaskStatus.COMPLETED.value
    assert updated_task.result == expected_entities
    assert updated_task.error_message is None
    mock_engine.extract_entities.assert_called_once_with("Elon Musk founded SpaceX in California.")


def test_task_failure_transition(db_session: Session):
    """Test task failure flow: PENDING -> PROCESSING -> FAILED when NLP engine raises exception."""
    task_id = str(uuid.uuid4())
    task = NLPTask(
        id=task_id,
        input_text="Malformed text causes inference crash",
        task_type=TaskType.SENTIMENT_ANALYSIS.value,
        status=TaskStatus.PENDING.value,
    )
    db_session.add(task)
    db_session.commit()

    mock_engine = MagicMock()
    mock_engine.analyze_sentiment.side_effect = RuntimeError("Inference tensor computation failed")

    with patch("app.tasks.SessionLocal", side_effect=TestingSessionLocal), \
         patch("app.tasks.get_nlp_engine", return_value=mock_engine):
        task_output = process_text_task(task_id)

    assert task_output["status"] == "FAILED"
    assert "Inference tensor computation failed" in task_output["error_message"]

    # Verify task state in database
    updated_task = db_session.query(NLPTask).filter(NLPTask.id == task_id).first()
    assert updated_task.status == TaskStatus.FAILED.value
    assert updated_task.result is None
    assert "Inference tensor computation failed" in updated_task.error_message


def test_nonexistent_task_gracefully_handled(db_session: Session):
    """Test that a task_id not present in the database does not cause an unhandled worker crash."""
    random_id = str(uuid.uuid4())
    with patch("app.tasks.SessionLocal", side_effect=TestingSessionLocal):
        task_output = process_text_task(random_id)

    assert task_output["status"] == "NOT_FOUND"
    assert task_output["task_id"] == random_id
