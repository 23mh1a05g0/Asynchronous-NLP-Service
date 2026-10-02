"""Celery background tasks for asynchronous NLP processing."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from app.celery_app import celery_instance
from app.database import SessionLocal
from app.models import NLPTask, TaskStatus, TaskType, utc_now
from app.nlp_engine import get_nlp_engine

logger = logging.getLogger(__name__)


@celery_instance.task(
    bind=True,
    name="app.tasks.process_text_task",
    max_retries=3,
    default_retry_delay=5,
    autoretry_for=(OperationalError,),
    retry_backoff=True,
    retry_jitter=True,
)
def process_text_task(self, task_id: str) -> Dict[str, Any]:
    """Execute asynchronous NLP processing on a queued task.

    State lifecycle:
        PENDING -> PROCESSING -> COMPLETED | FAILED

    Args:
        task_id: UUID string corresponding to an NLPTask in the database.

    Returns:
        Dict indicating status and task_id.
    """
    logger.info("Task received: task_id=%s", task_id)
    db = SessionLocal()

    try:
        # 1. Fetch task record from database
        task: NLPTask = db.query(NLPTask).filter(NLPTask.id == task_id).first()
        if not task:
            logger.warning(
                "Task not found in database, discarding task: task_id=%s",
                task_id,
            )
            return {"status": "NOT_FOUND", "task_id": task_id}

        # 2. Transition state: PENDING -> PROCESSING
        task.status = TaskStatus.PROCESSING.value
        task.updated_at = utc_now()
        db.commit()
        logger.info(
            "Processing task: task_id=%s, task_type=%s",
            task_id,
            task.task_type,
        )

        # 3. Execute NLP model inference
        engine = get_nlp_engine()
        input_text = task.input_text
        task_type = task.task_type

        if task_type == TaskType.SENTIMENT_ANALYSIS.value:
            result = engine.analyze_sentiment(input_text)
        elif task_type == TaskType.NAMED_ENTITY_RECOGNITION.value:
            result = engine.extract_entities(input_text)
        else:
            raise ValueError(f"Unsupported task_type encountered in task execution: '{task_type}'")

        logger.info("NLP processing completed: task_id=%s", task_id)

        # 4. Transition state: PROCESSING -> COMPLETED
        task.status = TaskStatus.COMPLETED.value
        task.result = result
        task.error_message = None
        task.updated_at = utc_now()
        db.commit()

        logger.info("Completed task: task_id=%s", task_id)
        return {
            "status": TaskStatus.COMPLETED.value,
            "task_id": task_id,
            "result": result,
        }

    except OperationalError as op_err:
        logger.warning(
            "Operational database error on task_id=%s. Attempting retry (%d/%d): %s",
            task_id,
            self.request.retries + 1,
            self.max_retries,
            op_err,
        )
        db.rollback()
        raise self.retry(exc=op_err)

    except Exception as exc:
        logger.exception("Failed task: task_id=%s, error=%s", task_id, exc)
        try:
            db.rollback()
            # Fetch task again to avoid stale object state
            task = db.query(NLPTask).filter(NLPTask.id == task_id).first()
            if task:
                task.status = TaskStatus.FAILED.value
                task.error_message = str(exc)
                task.updated_at = utc_now()
                db.commit()
                logger.info(
                    "Recorded FAILED status in database for task_id=%s",
                    task_id,
                )
        except SQLAlchemyError as db_err:
            logger.error(
                "Failed to persist failure status for task_id=%s: %s",
                task_id,
                db_err,
            )
        return {
            "status": TaskStatus.FAILED.value,
            "task_id": task_id,
            "error_message": str(exc),
        }

    finally:
        # Guarantee session closure to avoid connection pool exhaustion
        db.close()
