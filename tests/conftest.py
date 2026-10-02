"""Pytest fixtures for unit and integration testing."""

from typing import Generator
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from app.database import Base, get_db
from app.main import app

# Shared in-memory SQLite engine for fast and isolated unit tests
TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine,
)


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """Create a fresh database schema for each test and yield an isolated session."""
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def mock_celery_delay() -> Generator[MagicMock, None, None]:
    """Mock process_text_task.delay in app.main so RabbitMQ is not called in unit tests."""
    with patch("app.main.process_text_task.delay") as mock_delay:
        mock_delay.return_value = MagicMock(id="mocked-celery-id")
        yield mock_delay


@pytest.fixture(scope="function")
def client(
    db_session: Session, mock_celery_delay: MagicMock
) -> Generator[TestClient, None, None]:
    """Provide a TestClient with overridden database dependency and mocked Celery dispatch."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
