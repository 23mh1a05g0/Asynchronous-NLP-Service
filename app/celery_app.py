"""Celery application configuration and worker process lifecycle hooks."""

import logging
import os
from celery import Celery
from celery.signals import worker_process_init, worker_ready
from app.config import get_settings
from app.nlp_engine import get_nlp_engine

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize Celery app instance
celery_instance = Celery(
    "nlp_service",
    broker=settings.CELERY_BROKER_URL,
    include=["app.tasks"],
)

# Celery production configuration
celery_instance.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_default_queue=settings.CELERY_TASK_DEFAULT_QUEUE,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=int(os.environ.get("CELERY_CONCURRENCY", 2)),
    broker_connection_retry_on_startup=True,
)


@worker_process_init.connect
def on_worker_process_init(**kwargs) -> None:
    """Pre-load heavy NLP models once when each Celery worker process initializes.

    This ensures tasks do not incur model loading overhead on every execution.
    """
    pid = os.getpid()
    logger.info("Initializing NLP models for Celery worker process [PID: %d]...", pid)
    try:
        nlp_engine = get_nlp_engine()
        nlp_engine.load_models()
        logger.info("NLP models successfully cached in memory for process [PID: %d].", pid)
    except Exception as exc:
        logger.exception(
            "Critical error preloading NLP models in worker process [PID: %d]: %s",
            pid,
            exc,
        )
        # Note: Do not exit process completely here to allow retry or fallback on demand


@worker_ready.connect
def on_worker_ready(**kwargs) -> None:
    """Log when Celery worker is ready to receive tasks."""
    logger.info("Celery worker is ready and waiting for NLP tasks from RabbitMQ.")
