"""Unit tests for FastAPI endpoints."""

import uuid
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.models import NLPTask, TaskStatus, TaskType


def test_valid_sentiment_request(client: TestClient, db_session: Session):
    """Test 1: Valid sentiment analysis request returns 202 with task_id and PENDING status."""
    payload = {
        "text": "I absolutely love this product!",
        "task_type": "sentiment_analysis",
    }
    response = client.post("/api/nlp/process", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "PENDING"
    # Verify task UUID validity
    uuid_obj = uuid.UUID(data["task_id"])
    assert str(uuid_obj) == data["task_id"]

    # Verify database record
    task_in_db = db_session.query(NLPTask).filter(NLPTask.id == data["task_id"]).first()
    assert task_in_db is not None
    assert task_in_db.task_type == TaskType.SENTIMENT_ANALYSIS.value
    assert task_in_db.status == TaskStatus.PENDING.value
    assert task_in_db.input_text == payload["text"]


def test_valid_ner_request(client: TestClient, db_session: Session):
    """Test 2: Valid named entity recognition request returns 202 with PENDING status."""
    payload = {
        "text": "Elon Musk founded SpaceX in California.",
        "task_type": "named_entity_recognition",
    }
    response = client.post("/api/nlp/process", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "PENDING"

    task_in_db = db_session.query(NLPTask).filter(NLPTask.id == data["task_id"]).first()
    assert task_in_db is not None
    assert task_in_db.task_type == TaskType.NAMED_ENTITY_RECOGNITION.value
    assert task_in_db.status == TaskStatus.PENDING.value


def test_missing_text(client: TestClient):
    """Test 3: Missing 'text' attribute in request body triggers 422 Unprocessable Entity."""
    payload = {
        "task_type": "sentiment_analysis",
    }
    response = client.post("/api/nlp/process", json=payload)
    assert response.status_code == 422


def test_empty_text(client: TestClient):
    """Test 4: Empty string 'text' triggers 422 Unprocessable Entity."""
    payload = {
        "text": "",
        "task_type": "sentiment_analysis",
    }
    response = client.post("/api/nlp/process", json=payload)
    assert response.status_code == 422


def test_whitespace_only_text(client: TestClient):
    """Test 5: Whitespace-only string triggers 422 Unprocessable Entity."""
    payload = {
        "text": "   \n\t  ",
        "task_type": "sentiment_analysis",
    }
    response = client.post("/api/nlp/process", json=payload)
    assert response.status_code == 422


def test_unsupported_task_type(client: TestClient):
    """Test 6: Unsupported 'task_type' value triggers 422 Unprocessable Entity."""
    payload = {
        "text": "Translate this to French.",
        "task_type": "language_translation",
    }
    response = client.post("/api/nlp/process", json=payload)
    assert response.status_code == 422


def test_excessively_long_text(client: TestClient):
    """Test 7: Text exceeding 5000 characters triggers 422 Unprocessable Entity."""
    payload = {
        "text": "A" * 5001,
        "task_type": "sentiment_analysis",
    }
    response = client.post("/api/nlp/process", json=payload)
    assert response.status_code == 422


def test_post_returns_202_and_task_id(client: TestClient):
    """Test 8 & 9: POST returns 202 Accepted status code and a valid task_id."""
    payload = {
        "text": "FastAPI is amazingly fast!",
        "task_type": "sentiment_analysis",
    }
    response = client.post("/api/nlp/process", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert len(data["task_id"]) == 36


def test_initial_status_is_pending(client: TestClient, db_session: Session):
    """Test 10: Task initial state is strictly PENDING in both HTTP response and database."""
    payload = {
        "text": "Antigravity helps developers build scalable systems.",
        "task_type": "named_entity_recognition",
    }
    response = client.post("/api/nlp/process", json=payload)
    task_id = response.json()["task_id"]

    task_record = db_session.query(NLPTask).filter(NLPTask.id == task_id).first()
    assert task_record.status == "PENDING"
    assert task_record.result is None
    assert task_record.error_message is None


def test_get_existing_task(client: TestClient, db_session: Session):
    """Test 11: GET /api/nlp/status/{task_id} returns 200 with accurate task details."""
    task_id = str(uuid.uuid4())
    task = NLPTask(
        id=task_id,
        input_text="Sample input text",
        task_type=TaskType.SENTIMENT_ANALYSIS.value,
        status=TaskStatus.PROCESSING.value,
        result=None,
    )
    db_session.add(task)
    db_session.commit()

    response = client.get(f"/api/nlp/status/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
    assert data["status"] == "PROCESSING"
    assert data["result"] is None
    assert "updated_at" in data


def test_get_nonexistent_task_returns_404(client: TestClient):
    """Test 12: GET /api/nlp/status/{task_id} returns 404 for unknown task ID."""
    unknown_id = str(uuid.uuid4())
    response = client.get(f"/api/nlp/status/{unknown_id}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_celery_dispatch_failure_marks_failed(client: TestClient, db_session: Session):
    """Test 13: Transaction safety: Celery dispatch failure marks task as FAILED in DB and returns 503."""
    with patch("app.main.process_text_task.delay", side_effect=Exception("RabbitMQ connection refused")):
        payload = {
            "text": "This should fail queuing.",
            "task_type": "sentiment_analysis",
        }
        response = client.post("/api/nlp/process", json=payload)
        assert response.status_code == 503
        assert "could not be dispatched" in response.json()["detail"].lower()

        # Verify task is recorded as FAILED rather than phantom PENDING
        failed_task = db_session.query(NLPTask).first()
        assert failed_task is not None
        assert failed_task.status == TaskStatus.FAILED.value
        assert "RabbitMQ connection refused" in failed_task.error_message


def test_health_check_endpoint(client: TestClient):
    """Test 14: /health endpoint returns 200 and reachable status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["database"] == "reachable"
