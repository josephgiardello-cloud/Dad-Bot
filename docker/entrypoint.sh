#!/bin/sh
set -eu

MODE="${1:-${DADBOT_MODE:-api}}"

mkdir -p "${DADBOT_HOME:-/var/lib/dadbot}" "${DADBOT_LOG_ROOT:-/var/log/dadbot}" "${DADBOT_SESSION_LOG_DIR:-/var/log/dadbot/session_logs}"

case "$MODE" in
  streamlit|ui)
    exec python -m streamlit run dad_streamlit.py \
      --server.port=8501 \
      --server.address=0.0.0.0 \
      --browser.gatherUsageStats=false
    ;;
  api|serve-api)
    exec python Dad.py --serve-api
    ;;
  *)
    exec "$@"
    ;;
esac