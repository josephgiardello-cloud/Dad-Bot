#!/bin/sh
set -eu

MODE="${1:-${DADBOT_MODE:-api}}"

mkdir -p "${DADBOT_HOME:-/var/lib/dadbot}" "${DADBOT_LOG_ROOT:-/var/log/dadbot}" "${DADBOT_SESSION_LOG_DIR:-/var/log/dadbot/session_logs}"
chown -R dadbot:dadbot "${DADBOT_HOME:-/var/lib/dadbot}" "${DADBOT_LOG_ROOT:-/var/log/dadbot}"

case "$MODE" in
  streamlit|ui)
    exec gosu dadbot python launch.py --ui
    ;;
  api|serve-api)
    exec gosu dadbot python launch.py --api
    ;;
  *)
    exec gosu dadbot "$@"
    ;;
esac