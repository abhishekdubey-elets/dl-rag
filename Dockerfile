# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# digitalLEARNING RAG API image.
# NOTE: installs torch + sentence-transformers → the first build is large/slow.
# For a CPU-only slimmer image, pin the CPU torch wheel index in pyproject.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.4 \
    POETRY_VIRTUALENVS_CREATE=false \
    HF_HOME=/app/models_cache

# System deps. Uncomment tesseract-ocr + poppler-utils to enable PDF OCR.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    # tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Dependency layer (cached until pyproject/lock change).
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --only main

# Application code.
COPY src ./src
COPY README.md ./
RUN poetry install --only main

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "dl_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
