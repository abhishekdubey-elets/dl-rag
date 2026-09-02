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

# Statistical NER for the knowledge graph (speaker / person extraction). The
# model is a pinned wheel so the layer is reproducible; the entity extractor
# degrades to gazetteer + honorific detection if this layer is ever dropped.
ARG SPACY_MODEL_URL=https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
RUN pip install "spacy>=3.8,<3.9" "${SPACY_MODEL_URL}" \
    && python -c "import spacy; spacy.load('en_core_web_sm'); print('spaCy model OK')"

# Application code.
COPY src ./src
COPY README.md ./
RUN poetry install --only main

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "dl_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
