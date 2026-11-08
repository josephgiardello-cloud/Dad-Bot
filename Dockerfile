FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# System deps: curl (health checks), ffmpeg (audio), espeak-ng (pyttsx3 fallback TTS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    espeak-ng \
    gosu \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder

RUN /usr/local/bin/python -m pip install uv==0.13.7
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never

RUN python -m venv "$VIRTUAL_ENV"

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --extra voice --extra service --extra mcp --extra reranker

COPY launch.py ./
COPY Dad.py ./
COPY dad_streamlit.py ./
COPY dad_profile.template.json ./
COPY dadbot/ ./dadbot/
COPY dadbot_system/ ./dadbot_system/
COPY evaluation/ ./evaluation/
COPY static/ ./static/

RUN uv sync --locked --no-dev --no-editable --extra voice --extra service --extra mcp --extra reranker

FROM base AS runtime

RUN groupadd --system dadbot && useradd --system --gid dadbot --create-home dadbot

COPY --from=builder /opt/venv /opt/venv
COPY launch.py ./
COPY Dad.py ./
COPY dad_streamlit.py ./
COPY dad_profile.template.json ./
COPY dadbot/ ./dadbot/
COPY dadbot_system/ ./dadbot_system/
COPY evaluation/ ./evaluation/
COPY static/ ./static/
COPY docker/entrypoint.sh /usr/local/bin/dadbot-entrypoint

RUN sed -i 's/\r$//' /usr/local/bin/dadbot-entrypoint && \
    chmod +x /usr/local/bin/dadbot-entrypoint && \
    mkdir -p /var/lib/dadbot /var/log/dadbot/session_logs /app/static && \
    chown -R dadbot:dadbot /app /var/lib/dadbot /var/log/dadbot

ENV DADBOT_HOME=/var/lib/dadbot \
    DADBOT_LOG_ROOT=/var/log/dadbot \
    DADBOT_PROFILE_PATH=/var/lib/dadbot/dad_profile.json \
    DADBOT_MEMORY_PATH=/var/lib/dadbot/dad_memory.json \
    DADBOT_SEMANTIC_DB_PATH=/var/lib/dadbot/dad_memory_semantic.sqlite3 \
    DADBOT_SESSION_LOG_DIR=/var/log/dadbot/session_logs \
    DADBOT_RELATIONAL_LEDGER_PATH=/var/log/dadbot/session_logs/relational_ledger.jsonl \
    DADBOT_AUTO_INIT_PROFILE=true \
    DADBOT_API_HOST=0.0.0.0 \
    DADBOT_API_PORT=8010 \
    DADBOT_API_WORKERS=2 \
    DADBOT_GLOBAL_CONFLUENCE_MODE=enforce \
    DADBOT_ALLOW_LEGACY_CONFLUENCE_KEY=0 \
    DADBOT_JSON_LOGS=true \
    DADBOT_MODE=api

EXPOSE 8010 8501

USER root

ENTRYPOINT ["dadbot-entrypoint"]
CMD ["api"]