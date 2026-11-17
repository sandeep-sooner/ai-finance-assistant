# One image serves both the FastAPI backend and the built React UI on a single port.

# ---------- Stage 1: build the React front end ----------
FROM node:24-slim AS frontend
WORKDIR /chat
COPY chat/package.json chat/package-lock.json ./
RUN npm ci
COPY chat/ ./
RUN npm run build


# ---------- Stage 2: Python runtime ----------
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# Runtime dependencies only (the test tools at the end of requirements.txt are skipped).
COPY requirements.txt .
RUN sed '/^# tests and evals/,$d' requirements.txt > requirements-runtime.txt \
    && pip install -r requirements-runtime.txt

COPY config.yaml .
COPY src/ src/
COPY --from=frontend /chat/dist chat/dist

# Run as a non-root user; .cache (RAG index) is created at startup.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/api/health', timeout=4)"

# Hosting platforms (Render, Railway, Fly) inject $PORT.
CMD ["sh", "-c", "uvicorn src.web_app.api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
