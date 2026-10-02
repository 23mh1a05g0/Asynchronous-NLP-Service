# Multi-purpose production Dockerfile for FastAPI API and Celery Worker
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

WORKDIR /app

# Install system build dependencies and health check utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python package dependencies with layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Pre-download spaCy language model and Hugging Face sentiment model
# so the containers do not require external network downloads at runtime
RUN python -m spacy download en_core_web_sm && \
    python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"

# Create non-root user and grant permissions
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/.cache && \
    chown -R appuser:appuser /app

# Copy application source code
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# Default command runs the FastAPI server; overridden in docker-compose for the Celery worker
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
