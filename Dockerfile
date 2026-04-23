FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: ffmpeg (audio), espeak-ng (pyttsx3 fallback TTS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg espeak-ng && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY Dad.py ./
COPY dad_streamlit.py ./
COPY dad_profile.template.json ./
COPY dadbot/ ./dadbot/
COPY dadbot_system/ ./dadbot_system/
COPY static/ ./static/

# Install with voice + service extras so both UI and API modes work
RUN python -m pip install --upgrade pip && \
    python -m pip install -e .[voice,service]

RUN mkdir -p /var/lib/dadbot /var/log/dadbot /app/static /app/session_logs

ENV DADBOT_PROFILE_PATH=/var/lib/dadbot/dad_profile.json \
    DADBOT_MEMORY_PATH=/var/lib/dadbot/dad_memory.json \
    DADBOT_SEMANTIC_DB_PATH=/var/lib/dadbot/dad_memory_semantic.sqlite3 \
    DADBOT_SESSION_LOG_DIR=/var/log/dadbot/session_logs \
    DADBOT_AUTO_INIT_PROFILE=true \
    DADBOT_API_HOST=0.0.0.0 \
    DADBOT_API_PORT=8010 \
    DADBOT_API_WORKERS=2 \
    DADBOT_JSON_LOGS=true \
    # Set to "streamlit" to run the UI instead of the API
    DADBOT_MODE=api

EXPOSE 8010 8501

# Entrypoint: DADBOT_MODE=api → serve API, DADBOT_MODE=streamlit → serve UI
CMD ["sh", "-c", \
    "if [ \"$DADBOT_MODE\" = \"streamlit\" ]; then \
        python -m streamlit run dad_streamlit.py \
            --server.port=8501 --server.address=0.0.0.0 \
            --browser.gatherUsageStats=false; \
    else \
        python Dad.py --serve-api; \
    fi"]