#!/bin/sh
set -eu

MODE="${1:-${DADBOT_MODE:-api}}"

mkdir -p "${DADBOT_HOME:-/var/lib/dadbot}" "${DADBOT_LOG_ROOT:-/var/log/dadbot}" "${DADBOT_SESSION_LOG_DIR:-/var/log/dadbot/session_logs}"

case "$MODE" in
  streamlit|ui)
    exec python launch.py --ui
    ;;
  api|serve-api)
    exec python launch.py --api
    ;;
  *)
    exec "$@"
    ;;
esac