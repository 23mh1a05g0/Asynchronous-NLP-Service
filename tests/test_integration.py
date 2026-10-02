"""End-to-end integration tests for Asynchronous NLP Service.

Tests the full lifecycle:
Client POST -> FastAPI -> Database -> Celery -> Worker -> NLP Engine -> Database -> Client GET.
"""

import os
import time
import pytest
import httpx
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.models import NLPTask, TaskStatus
from app.nlp_engine import get_nlp_engine
from app.tasks import process_text_task
from tests.conftest import TestingSessionLocal


@pytest.mark.integration
def test_end_to_end_nlp_pipeline_sentiment(client: TestClient, db_session: Session):
    """End-to-End Pipeline Test: Sentiment Analysis.

    Flow:
    1. Client submits POST request to /api/nlp/process.
    2. API records PENDING state in Database and assigns task_id.
    3. Celery process_text_task executes asynchronously with real NLP engine.
    4. Worker updates status to PROCESSING -> COMPLETED in Database.
    5. Client queries GET /api/nlp/status/{task_id} and verifies final result.
    """
    input_text = "I absolutely love this product!"

    # 1. Submit POST request
    post_response = client.post(
        "/api/nlp/process",
        json={"text": input_text, "task_type": "sentiment_analysis"},
    )
    assert post_response.status_code == 202
    data = post_response.json()
    task_id = data["task_id"]
    assert data["status"] == "PENDING"

    # Verify initial task state in Database
    task_record = db_session.query(NLPTask).filter(NLPTask.id == task_id).first()
    assert task_record is not None
    assert task_record.status == TaskStatus.PENDING.value

    # 2. Celery Worker execution (using real NLP engine)
    from unittest.mock import patch
    with patch("app.tasks.SessionLocal", side_effect=TestingSessionLocal):
        task_output = process_text_task(task_id)

    assert task_output["status"] == "COMPLETED"

    # 3. Query GET /api/nlp/status/{task_id}
    status_response = client.get(f"/api/nlp/status/{task_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()

    assert status_data["task_id"] == task_id
    assert status_data["status"] == "COMPLETED"
    assert status_data["error_message"] is None
    assert status_data["result"] is not None

    result = status_data["result"]
    assert "label" in result
    assert "score" in result
    assert result["label"] in ["POSITIVE", "NEGATIVE"]
    assert isinstance(result["score"], (float, int))
    assert 0.0 <= result["score"] <= 1.0


@pytest.mark.integration
def test_end_to_end_nlp_pipeline_ner(client: TestClient, db_session: Session):
    """End-to-End Pipeline Test: Named Entity Recognition.

    Flow:
    1. Client submits POST request for NER.
    2. Real NLP model extracts entities with text, label, start, end.
    3. Final GET endpoint validates extracted entities.
    """
    input_text = "Elon Musk founded SpaceX in California."

    # 1. Submit POST request
    post_response = client.post(
        "/api/nlp/process",
        json={"text": input_text, "task_type": "named_entity_recognition"},
    )
    assert post_response.status_code == 202
    task_id = post_response.json()["task_id"]

    # 2. Worker executes task with real NLP engine
    from unittest.mock import patch
    with patch("app.tasks.SessionLocal", side_effect=TestingSessionLocal):
        task_output = process_text_task(task_id)

    assert task_output["status"] == "COMPLETED"

    # 3. Check status via GET endpoint
    status_response = client.get(f"/api/nlp/status/{task_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()

    assert status_data["status"] == "COMPLETED"
    entities = status_data["result"].get("entities", [])
    assert isinstance(entities, list)
    assert len(entities) > 0

    entity_texts = [ent["text"] for ent in entities]
    assert any("Elon Musk" in t for t in entity_texts)
    assert any("SpaceX" in t for t in entity_texts)

    for ent in entities:
        assert "text" in ent
        assert "label" in ent
        assert "start" in ent
        assert "end" in ent
        assert ent["start"] < ent["end"]


@pytest.mark.live_docker
def test_live_docker_stack_e2e():
    """Live full-stack integration test against running Docker Compose services.

    Requires:
        docker compose up -d
        TEST_TARGET_URL environment variable set (default: http://localhost:8000)
    """
    target_url = os.environ.get("TEST_TARGET_URL", "http://localhost:8000")

    try:
        health_resp = httpx.get(f"{target_url}/health", timeout=5.0)
        if health_resp.status_code != 200:
            pytest.skip(f"Docker stack at {target_url} is not responding with 200 OK.")
    except Exception:
        pytest.skip(f"Live Docker stack at {target_url} is not accessible.")

    # Submit task to live API
    submit_resp = httpx.post(
        f"{target_url}/api/nlp/process",
        json={
            "text": "Antigravity and FastAPI provide exceptional developer experience!",
            "task_type": "sentiment_analysis",
        },
        timeout=10.0,
    )
    assert submit_resp.status_code == 202
    task_id = submit_resp.json()["task_id"]

    # Poll status until Celery worker completes the task via RabbitMQ
    max_wait_seconds = 45
    poll_interval = 2
    elapsed = 0
    final_status = None
    result_payload = None

    while elapsed < max_wait_seconds:
        time.sleep(poll_interval)
        elapsed += poll_interval

        status_resp = httpx.get(f"{target_url}/api/nlp/status/{task_id}", timeout=5.0)
        assert status_resp.status_code == 200
        data = status_resp.json()
        final_status = data["status"]

        if final_status in ["COMPLETED", "FAILED"]:
            result_payload = data
            break

    assert final_status == "COMPLETED", f"Task did not complete within {max_wait_seconds}s. Last payload: {result_payload}"
    assert result_payload["result"] is not None
    assert result_payload["result"]["label"] in ["POSITIVE", "NEGATIVE"]
