"""FastAPI application entrypoint, routes, and task dispatching."""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.config import get_settings
from app.database import Base, engine, get_db
from app.models import NLPTask, TaskStatus
from app.schemas import TaskCreateRequest, TaskResponse, TaskStatusResponse
from app.tasks import process_text_task

settings = get_settings()

# Configure logging format and level
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown hooks."""
    logger.info("Starting up %s (env: %s)...", settings.APP_NAME, settings.APP_ENV)
    # Ensure database schema is created on application startup
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema validated and initialized successfully.")
    except Exception as exc:
        logger.warning(
            "Could not initialize database schema during immediate startup: %s. "
            "Tables will be verified when database is reachable.",
            exc,
        )
    yield
    logger.info("Shutting down %s...", settings.APP_NAME)


app = FastAPI(
    title="Asynchronous NLP Service",
    description=(
        "Production-grade Asynchronous NLP API powered by FastAPI, Celery, "
        "RabbitMQ, and MySQL. Decouples computationally expensive NLP workloads "
        "from HTTP response cycles."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get(
    "/",
    tags=["Root"],
    summary="Root Service Status",
    description="Returns service metadata and helpful links to API documentation.",
)
def read_root():
    """Service status and quick links."""
    return {
        "service": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "status": "healthy",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Health Check",
    description="Checks the operational status of the service and database connection.",
    status_code=status.HTTP_200_OK,
)
def health_check(db: Session = Depends(get_db)):
    """Perform health checks on database and API server."""
    health_status = {"status": "ok", "database": "unknown"}
    try:
        db.execute(text("SELECT 1"))
        health_status["database"] = "reachable"
    except Exception as exc:
        logger.error("Health check failed on database: %s", exc)
        health_status["database"] = "unreachable"
        health_status["status"] = "degraded"
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status,
        )
    return health_status


@app.post(
    "/api/nlp/process",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["NLP Processing"],
    summary="Submit an Asynchronous NLP Task",
    description=(
        "Accepts an NLP processing request (sentiment_analysis or named_entity_recognition), "
        "records the task as PENDING in MySQL, dispatches it to RabbitMQ via Celery, "
        "and immediately returns HTTP 202 Accepted with a unique task ID."
    ),
    responses={
        202: {
            "description": "Task successfully accepted and queued for processing.",
            "model": TaskResponse,
        },
        422: {
            "description": "Validation error: invalid task_type, missing text, or whitespace-only input.",
        },
        503: {
            "description": "Service Unavailable: Message broker dispatch failed.",
        },
    },
)
def submit_nlp_task(
    payload: TaskCreateRequest,
    db: Session = Depends(get_db),
):
    """Submit an NLP task for asynchronous execution."""
    task_id = str(uuid.uuid4())

    # Step 1: Create initial database record in PENDING status
    new_task = NLPTask(
        id=task_id,
        input_text=payload.text,
        task_type=payload.task_type.value,
        status=TaskStatus.PENDING.value,
    )
    db.add(new_task)
    db.commit()
    db.refresh(new_task)
    logger.info("Task created: task_id=%s, type=%s", task_id, payload.task_type.value)

    # Step 2: Queue the task with Celery via RabbitMQ
    try:
        process_text_task.delay(task_id)
        logger.info("Task queued: task_id=%s", task_id)
    except Exception as exc:
        # Transaction safety handling: if queuing fails, do not leave phantom PENDING task
        logger.error(
            "Transaction safety alert: Celery/RabbitMQ dispatch failed for task_id=%s: %s",
            task_id,
            exc,
        )
        try:
            new_task.status = TaskStatus.FAILED.value
            new_task.error_message = f"Failed to dispatch task to message broker: {str(exc)}"
            db.commit()
        except Exception as db_exc:
            logger.error("Failed to update status to FAILED in DB: %s", db_exc)

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task could not be dispatched to message broker. Please try again later.",
        )

    # Step 3: Return HTTP 202 Accepted immediately
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
    )


@app.get(
    "/api/nlp/status/{task_id}",
    response_model=TaskStatusResponse,
    status_code=status.HTTP_200_OK,
    tags=["NLP Processing"],
    summary="Get NLP Task Status and Result",
    description="Retrieve the real-time lifecycle status (PENDING, PROCESSING, COMPLETED, FAILED) and result of an NLP task.",
    responses={
        200: {
            "description": "Current status and result of the specified task.",
            "model": TaskStatusResponse,
        },
        404: {
            "description": "Task ID not found in database.",
        },
    },
)
def get_task_status(
    task_id: str,
    db: Session = Depends(get_db),
):
    """Retrieve status, results, and timestamps for a specific task ID."""
    task = db.query(NLPTask).filter(NLPTask.id == task_id).first()
    if not task:
        logger.warning("Task status requested for nonexistent ID: %s", task_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID '{task_id}' was not found.",
        )

    return TaskStatusResponse(
        task_id=task.id,
        status=TaskStatus(task.status),
        result=task.result,
        error_message=task.error_message,
        updated_at=task.updated_at,
    )
