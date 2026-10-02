"""Application configuration management using pydantic-settings."""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # General Application Settings
    APP_NAME: str = "Asynchronous NLP Service"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False

    # Database Settings
    DATABASE_URL: str = (
        "mysql+pymysql://nlp_user:nlp_password@localhost:3306/nlp_db"
    )
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_RECYCLE: int = 1800
    DB_POOL_PRE_PING: bool = True

    # Celery & RabbitMQ Settings
    CELERY_BROKER_URL: str = "amqp://guest:guest@localhost:5672//"
    CELERY_TASK_DEFAULT_QUEUE: str = "nlp_tasks"

    # NLP Models
    SPACY_MODEL: str = "en_core_web_sm"
    TRANSFORMERS_MODEL: str = "distilbert-base-uncased-finetuned-sst-2-english"


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
