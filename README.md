# Asynchronous NLP Service with FastAPI, Celery, and RabbitMQ

A production-grade, distributed backend service engineered to decouple computationally expensive Natural Language Processing (NLP) inference from synchronous HTTP request-response lifecycles.

---

## 1. Project Overview

The **Asynchronous NLP Service** demonstrates how to build a scalable and resilient distributed architecture for AI/NLP workloads. When an API client submits unstructured text for heavy processing (such as sentiment analysis or named entity recognition), the FastAPI server accepts the payload, records it in MySQL, dispatches an execution job to RabbitMQ via Celery, and immediately returns an HTTP `202 Accepted` status with a unique task ID.

The computationally intensive NLP model inference is executed asynchronously by distributed Celery workers running on CPU or GPU worker nodes. The API never blocks, guaranteeing sub-millisecond response latency and high availability even under high concurrent load.

---

## 2. Problem Statement

Synchronous machine learning endpoints suffer from fundamental architectural bottlenecks:
- **Thread Exhaustion**: Standard HTTP servers (e.g., Uvicorn/Gunicorn workers) hold network sockets and worker threads open for the entire duration of model inference (often hundreds of milliseconds to multiple seconds).
- **Client Timeouts**: High latency causes HTTP gateway timeouts (HTTP 504) across reverse proxies like Nginx, Cloudflare, or AWS ALB.
- **Resource Starvation**: Spike traffic can overwhelm memory and CPU resources, causing the entire API to crash and drop incoming connections.
- **Lack of Resilience**: If an inference worker crashes mid-request in a synchronous service, the client receives an HTTP 500 error with no recovery mechanism or retry queue.

---

## 3. Real-World Use Case

Consider a customer experience platform analyzing millions of customer support tickets, product reviews, and enterprise communications:
- Users or third-party webhooks submit large volumes of reviews and tickets.
- The service instantly acknowledges receipt, assigning a tracking UUID.
- Celery worker pools process NLP tasks at a sustainable, controlled rate based on system capacity.
- Downstream applications poll or listen for completion to trigger workflows, sentiment dashboards, or automated routing.

---

## 4. Why Asynchronous Processing is Needed

Asynchronous processing segregates the **I/O-bound API tier** from the **Compute-bound Worker tier**:

1. **Horizontal Independent Scaling**: Scale the FastAPI API tier (handling HTTP connections) independently from the Celery worker tier (handling heavy CPU/GPU matrix computations).
2. **Backpressure Management**: RabbitMQ queues buffer sudden spikes in traffic without dropping requests or running out of memory.
3. **Resilience & Fault Tolerance**: If a worker fails, RabbitMQ's acknowledgment protocol (`task_acks_late=True`) ensures the task is re-queued and retried by another worker.
4. **Predictable Latency**: API endpoints respond consistently in less than 20 milliseconds regardless of text length or model complexity.

---

## 5. System Architecture

```
Client
   |
   | HTTP Request (POST /api/nlp/process)
   v
FastAPI API (Port 8000)
   |
   +------------------> MySQL 8.0 (Port 3306)
   |                     |
   |                     | Store task: PENDING
   |
   +------------------> Celery Producer
                          |
                          v
                    RabbitMQ Broker (Port 5672)
                          |
                          v
                    Celery Worker Pool
                          |
                          v
                 NLP Engine (spaCy + Transformers)
                          |
                          v
                    MySQL 8.0 (Update: COMPLETED / FAILED)
```

### Flow Diagram

```mermaid
flowchart LR
    Client[Client App] -->|POST /api/nlp/process| FastAPI[FastAPI Service]
    FastAPI -->|1. Insert PENDING task| MySQL[(MySQL Database)]
    FastAPI -->|2. Dispatch task| Celery[Celery Producer]
    Celery -->|3. AMQP Message| RabbitMQ{RabbitMQ Broker}
    FastAPI -->|4. Return 202 Accepted| Client
    RabbitMQ -->|5. Deliver task| Worker[Celery Worker]
    Worker -->|6. Execute inference| NLP[NLP Engine]
    NLP -->|7. Update result & COMPLETED| MySQL
    Client -->|GET /api/nlp/status/:id| FastAPI
    FastAPI -->|Read status & result| MySQL
```

---

## 6. Technology Stack

| Technology | Purpose | Rationale |
|---|---|---|
| **Python 3.10+** | Core Language | Modern type hinting, async features, and broad AI library support |
| **FastAPI** | Web Framework | High-performance asynchronous ASGI framework with automated OpenAPI docs |
| **Uvicorn** | ASGI Web Server | Lightning-fast asynchronous HTTP server implementation |
| **Pydantic v2** | Data Validation | High-speed Rust-based schema validation and serialization |
| **pydantic-settings**| Config Management | Type-safe environment variable management |
| **Celery** | Distributed Task Queue | Industry-standard distributed task execution with retry protocols |
| **RabbitMQ** | Message Broker | Robust AMQP message broker with persistent queues and management UI |
| **MySQL 8.0** | Relational Database | ACID-compliant persistence for task status, metadata, and JSON results |
| **SQLAlchemy 2.0** | ORM & Engine | Connection pooling, declarative mapping, and transactional safety |
| **PyMySQL + Cryptography** | Database Driver | Pure-Python MySQL driver with modern authentication support |
| **spaCy** | Named Entity Recognition | Ultra-fast, industrial-grade CPU NER inference (`en_core_web_sm`) |
| **HuggingFace Transformers** | Sentiment Analysis | State-of-the-art transformer sentiment classification (`distilbert`) |
| **Pytest + HTTPX** | Testing Framework | Modern test runner and asynchronous HTTP client for unit and integration testing |
| **Docker & Docker Compose**| Containerization | Reproducible multi-container orchestration with health checks |

---

## 7. Folder Structure

```
.
├── app/
│   ├── __init__.py           # Package version and export
│   ├── config.py             # Settings via pydantic-settings
│   ├── database.py           # SQLAlchemy engine, sessionmaker, and get_db dependency
│   ├── models.py             # NLPTask ORM model, TaskType, and TaskStatus enums
│   ├── schemas.py            # Pydantic schemas (TaskCreateRequest, TaskResponse, etc.)
│   ├── nlp_engine.py         # Singleton NLP inference engine (spaCy + Transformers)
│   ├── celery_app.py         # Celery instance, configuration, and worker_process_init
│   ├── tasks.py              # Celery background tasks (process_text_task)
│   └── main.py               # FastAPI application, routes, and error handling
├── tests/
│   ├── __init__.py           # Test package
│   ├── conftest.py           # Pytest fixtures (in-memory SQLite, mocked Celery, client)
│   ├── test_api.py           # Unit tests for all API endpoints and validation rules
│   ├── test_tasks.py         # Unit tests for Celery task lifecycle and error states
│   └── test_integration.py   # End-to-end integration tests (pipeline and live Docker)
├── .env.example              # Environment variables template
├── .gitignore                # Git exclusions
├── Dockerfile                # Production multi-stage Docker build
├── docker-compose.yml        # Multi-service stack (db, rabbitmq, api, worker)
├── pytest.ini                # Pytest configuration and registered markers
├── requirements.txt          # Python dependency specifications
└── README.md                 # Complete system documentation
```

---

## 8. Environment Variables

All settings are configured via environment variables. See [`.env.example`](file:///.env.example):

| Variable | Description | Default (Local / Docker) |
|---|---|---|
| `APP_ENV` | Application environment (`development`, `production`) | `development` |
| `LOG_LEVEL` | Python logging verbosity (`DEBUG`, `INFO`, `WARNING`) | `INFO` |
| `MYSQL_ROOT_PASSWORD` | MySQL root account password | `nlp_root_password` |
| `MYSQL_DATABASE` | MySQL database name | `nlp_db` |
| `MYSQL_USER` | MySQL non-root application user | `nlp_user` |
| `MYSQL_PASSWORD` | MySQL application user password | `nlp_password` |
| `DATABASE_URL` | SQLAlchemy connection string | `mysql+pymysql://nlp_user:nlp_password@db:3306/nlp_db` |
| `CELERY_BROKER_URL` | RabbitMQ AMQP broker URI | `amqp://guest:guest@rabbitmq:5672//` |
| `CELERY_CONCURRENCY` | Number of worker child processes | `2` |
| `SPACY_MODEL` | spaCy model for NER | `en_core_web_sm` |
| `TRANSFORMERS_MODEL` | HuggingFace model for sentiment | `distilbert-base-uncased-finetuned-sst-2-english` |

---

## 9. Critical Architectural Decisions

### 9.1 Model Preloading via `worker_process_init`

**The Problem**:
Loading heavy NLP models (spaCy language graphs and Transformer weights) inside the task execution body incurs severe overhead (1 to 5 seconds per task) and rapidly fragments worker memory.

**The Solution**:
In [`app/celery_app.py`](file:///app/celery_app.py), we attach a handler to Celery's `worker_process_init` signal:
```python
@worker_process_init.connect
def on_worker_process_init(**kwargs) -> None:
    nlp_engine = get_nlp_engine()
    nlp_engine.load_models()
```
When a Celery worker process is spawned, it loads the spaCy and Hugging Face transformer models into memory **once**. All subsequent tasks dispatched to that process execute inference immediately against the warm in-memory model.

### 9.2 Transaction Safety & Broker Dispatch Failures

**The Problem**:
What happens if the MySQL record is created with status `PENDING`, but the RabbitMQ broker is unreachable when `process_text_task.delay(task_id)` is called?
Silently returning `202 Accepted` leaves a "ghost" task in the database that will never execute.

**The Solution**:
In [`app/main.py`](file:///app/main.py), Celery dispatch is wrapped in a guarded transaction block:
1. The database transaction is committed first.
2. If `process_text_task.delay(task_id)` throws an exception (e.g. RabbitMQ connection failure):
   - The error is logged with stack trace.
   - The task record in MySQL is updated to `status = FAILED` with a descriptive `error_message`.
   - The API immediately raises an `HTTP 503 Service Unavailable` exception informing the client that queuing failed.

---

## 10. Task State Machine

Every NLP task transitions through an explicit state machine:

```
[ POST /api/nlp/process ]
            |
            v
       +---------+
       | PENDING |  (Stored in MySQL, enqueued to RabbitMQ)
       +---------+
            |
            | Celery worker consumes task
            v
      +------------+
      | PROCESSING | (Worker executing NLP model)
      +------------+
            |
     +------+------+
     |             |
     v             v
+-----------+  +--------+
| COMPLETED |  | FAILED | (Stored with error_message)
+-----------+  +--------+
```

- **`PENDING`**: Task created by FastAPI, stored in MySQL, and sent to RabbitMQ.
- **`PROCESSING`**: Celery worker picked up the task from RabbitMQ and is running model inference.
- **`COMPLETED`**: Inference succeeded; `result` contains JSON output and `error_message` is null.
- **`FAILED`**: Inference or system error occurred; `error_message` details the failure reason.

---

## 11. API Documentation & Endpoints

Interactive Swagger UI documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Endpoints Overview

#### 1. Submit NLP Task
- **Endpoint**: `POST /api/nlp/process`
- **Status Code**: `202 Accepted`
- **Headers**: `Content-Type: application/json`

**Request Body (Sentiment Analysis)**:
```json
{
  "text": "I absolutely love this product!",
  "task_type": "sentiment_analysis"
}
```

**Request Body (Named Entity Recognition)**:
```json
{
  "text": "Elon Musk founded SpaceX in California.",
  "task_type": "named_entity_recognition"
}
```

**Validation Constraints**:
- `text`: Required, string, 1 to 5000 characters, non-blank (rejects whitespace-only).
- `task_type`: Required, must be `"sentiment_analysis"` or `"named_entity_recognition"`.
- Violations return `HTTP 422 Unprocessable Entity`.

**Response (`202 Accepted`)**:
```json
{
  "task_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "PENDING"
}
```

---

#### 2. Get Task Status
- **Endpoint**: `GET /api/nlp/status/{task_id}`
- **Status Code**: `200 OK` (or `404 Not Found`)

**Response (`COMPLETED` Sentiment Analysis)**:
```json
{
  "task_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "COMPLETED",
  "result": {
    "label": "POSITIVE",
    "score": 0.9998
  },
  "error_message": null,
  "updated_at": "2026-10-02T09:15:30"
}
```

**Response (`COMPLETED` Named Entity Recognition)**:
```json
{
  "task_id": "a429188e-6c8a-4d2b-8a8b-118f6c449c20",
  "status": "COMPLETED",
  "result": {
    "entities": [
      {
        "text": "Elon Musk",
        "label": "PERSON",
        "start": 0,
        "end": 9
      },
      {
        "text": "SpaceX",
        "label": "ORG",
        "start": 20,
        "end": 26
      },
      {
        "text": "California",
        "label": "GPE",
        "start": 30,
        "end": 40
      }
    ]
  },
  "error_message": null,
  "updated_at": "2026-10-02T09:16:05"
}
```

**Response (`FAILED`)**:
```json
{
  "task_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "FAILED",
  "result": null,
  "error_message": "Inference computation error: text exceeds supported limit",
  "updated_at": "2026-10-02T09:17:10"
}
```

---

## 12. Local Development Setup

### Prerequisites
- Python 3.10+
- MySQL 8.0+
- RabbitMQ 3.8+

### Setup Instructions
1. **Clone the repository and enter the directory**:
   ```bash
   cd "Asynchronous NLP Service"
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download NLP models**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Update DATABASE_URL and CELERY_BROKER_URL to match your local services
   ```

6. **Start FastAPI API Server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

7. **Start Celery Worker (in a separate terminal)**:
   ```bash
   celery -A app.celery_app.celery_instance worker --loglevel=info
   ```

---

## 13. Docker & Docker Compose Setup

The entire multi-service distributed system can be built and launched with a single command.

### Launch All Services
```bash
docker compose up -d --build
```

Docker Compose spins up:
- **`nlp_mysql`**: MySQL 8.0 on container port 3306 (mapped to host 3308) with health checks.
- **`nlp_rabbitmq`**: RabbitMQ with management console on ports 5672 (AMQP) and 15672 (HTTP UI).
- **`nlp_api`**: FastAPI service waiting for MySQL and RabbitMQ healthy status.
- **`nlp_worker`**: Celery worker pre-loading spaCy and Hugging Face models on startup.

### Inspect Service Status
```bash
docker compose ps
```

### View Live Logs
```bash
# All services
docker compose logs -f

# API logs
docker compose logs -f api

# Celery Worker logs
docker compose logs -f worker
```

### Stop Services
```bash
docker compose down
```

To remove all persistent data volumes:
```bash
docker compose down -v
```

---

## 14. Testing Guide

The test suite covers API validation, Celery task state transitions, and end-to-end integration flows.

### Run Unit and Task Tests
```bash
pytest tests/test_api.py tests/test_tasks.py -v
```

### Run End-to-End Integration Tests
```bash
pytest tests/test_integration.py -v
```

### Run Full Test Suite with Coverage
```bash
pytest --cov=app tests/ -v
```

---

## 15. Scaling Workers

Because Celery workers communicate over RabbitMQ's AMQP protocol, scaling the worker pool horizontally requires no code changes:

```bash
docker compose up -d --scale worker=4
```

This runs 4 independent worker containers consuming tasks from the shared RabbitMQ `nlp_tasks` queue concurrently.

---

## 16. Error Handling & Troubleshooting

| Symptom | Root Cause | Solution |
|---|---|---|
| `503 Service Unavailable` on POST | RabbitMQ broker is unreachable or rejecting connections | Verify RabbitMQ is running (`docker compose ps`) and verify `CELERY_BROKER_URL`. |
| Task stays in `PENDING` | Celery worker is not running or consuming from the wrong queue | Ensure worker is started with `celery -A app.celery_app.celery_instance worker`. |
| Task status is `FAILED` | Model inference raised an exception or invalid text encountered | Query `GET /api/nlp/status/{task_id}` to read `error_message`. |
| Port 3306 in use on host | A local MySQL instance is already running on port 3306 | Docker Compose maps host port `3308:3306` to avoid collisions. |

---

## 17. Production Considerations

1. **Dead Letter Exchanges (DLX)**: In high-scale deployments, configure a RabbitMQ Dead Letter Exchange for tasks that repeatedly fail all retries.
2. **Result Retention & TTL**: Implement periodic cleanup or MySQL partitioning for task rows older than 30 days.
3. **Queue Monitoring**: Use Celery Flower or Prometheus RabbitMQ Exporter to monitor queue length and alert on worker lag.
4. **Secrets Management**: Store database passwords and broker secrets in Docker Secrets, HashiCorp Vault, or AWS Secrets Manager.
