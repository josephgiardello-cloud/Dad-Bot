from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

from dadbot_system.security import ServiceTokenManager

_API_URL = os.environ.get("DADBOT_CRASH_RECOVERY_API_URL", "http://127.0.0.1:8010").rstrip("/")
_API_PREFIX = os.environ.get("DADBOT_API_VERSION_PREFIX", "/v1").strip("/")
_TIMEOUT_SECONDS = float(os.environ.get("DADBOT_CRASH_RECOVERY_TIMEOUT_SECONDS", "120"))
_COMPOSE = ["docker", "compose", "--profile", "production"]
_REMINDER_PATH = "/var/lib/dadbot/dad_memory.json"


def _production_credentials() -> tuple[str, str, str]:
    env_path = Path(__file__).resolve().parents[2] / ".env.production"
    values: dict[str, str] = {}
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")

    secret = os.environ.get("DADBOT_API_TOKEN_SECRET") or values.get("DADBOT_API_TOKEN_SECRET")
    if not secret:
        raise RuntimeError("DADBOT_API_TOKEN_SECRET must be configured for the production recovery test")
    issuer = os.environ.get("DADBOT_API_TOKEN_ISSUER") or values.get("DADBOT_API_TOKEN_ISSUER") or "dadbot"
    tenant_id = f"recovery-{uuid4().hex[:12]}"
    token = ServiceTokenManager(secret, issuer=issuer).issue(
        subject="production-crash-recovery-test",
        tenant_id=tenant_id,
        scopes=("read", "write"),
        ttl_seconds=600,
    )
    api_secret = os.environ.get("DADBOT_SECRET_KEY") or values.get("DADBOT_SECRET_KEY") or ""
    return tenant_id, token, api_secret


def _request(
    method: str,
    path: str,
    token: str,
    payload: dict | None = None,
    *,
    api_secret: str = "",
) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if api_secret:
        headers["X-DADBOT-KEY"] = api_secret
    request = urllib.request.Request(
        f"{_API_URL}/{_API_PREFIX}/{path.lstrip('/')}",
        data=body,
        method=method,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def _compose(*args: str, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*_COMPOSE, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def _reminder_count(marker: str, tenant_id: str) -> int:
    script = (
        "import os, psycopg, sys; "
        "connection = psycopg.connect(os.environ['DADBOT_POSTGRES_DSN']); "
        "row = connection.execute("
        "\"SELECT count(*) FROM dadbot_session_state AS state "
        "CROSS JOIN LATERAL jsonb_array_elements(COALESCE(state.payload->'reminders', '[]'::jsonb)) AS reminder(value) "
        "WHERE state.session_id = %s AND position(%s in reminder.value->>'title') > 0\", "
        "(f'tenant-doc:{sys.argv[2]}:memory', sys.argv[1])).fetchone(); "
        "print(row[0]); connection.close()"
    )
    result = _compose(
        "exec",
        "-T",
        "dadbot-api",
        "python",
        "-c",
        script,
        marker,
        tenant_id,
        timeout=_TIMEOUT_SECONDS,
    )
    return int(result.stdout.strip())


def _start_reminder_watcher(
    marker: str,
    tenant_id: str,
) -> tuple[subprocess.Popen[str], queue.Queue[str | None]]:
    script = """\
import os
import psycopg
import signal
import sys
import time

marker = sys.argv[1]
tenant_id = sys.argv[2]
deadline = time.monotonic() + float(sys.argv[3])
connection = psycopg.connect(os.environ["DADBOT_POSTGRES_DSN"])
print("READY", flush=True)
while time.monotonic() < deadline:
    row = connection.execute(
        "SELECT count(*) FROM dadbot_session_state AS state "
        "CROSS JOIN LATERAL jsonb_array_elements(COALESCE(state.payload->'reminders', '[]'::jsonb)) AS reminder(value) "
        "WHERE state.session_id = %s AND position(%s in reminder.value->>'title') > 0",
        (f"tenant-doc:{tenant_id}:memory", marker),
    ).fetchone()
    count = int(row[0])
    if count:
        print(f"FOUND:{count}", flush=True)
        os.kill(1, signal.SIGKILL)
        break
    time.sleep(0.2)
else:
    print("WATCH_TIMEOUT", flush=True)
connection.close()
"""
    watcher = subprocess.Popen(
        [
            *_COMPOSE,
            "exec",
            "-T",
            "dadbot-api",
            "python",
            "-u",
            "-c",
            script,
            marker,
            tenant_id,
            str(_TIMEOUT_SECONDS),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        if watcher.stdout is not None:
            for line in watcher.stdout:
                lines.put(line.strip())
        lines.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    return watcher, lines


def _cleanup_test_tenant(tenant_id: str) -> None:
    prefix = f"tenant:{tenant_id}:%"
    document_prefix = f"tenant-doc:{tenant_id}:%"
    sql = (
        "BEGIN; "
        f"DELETE FROM dadbot_task_state WHERE payload->>'tenant_id' = '{tenant_id}'; "
        f"DELETE FROM dadbot_event_log WHERE session_id LIKE '{prefix}'; "
        f"DELETE FROM dadbot_session_state WHERE session_id LIKE '{prefix}' "
        f"OR session_id LIKE '{document_prefix}'; "
        "COMMIT;"
    )
    _compose(
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "dadbot",
        "-d",
        "dadbot",
        "-c",
        sql,
    )


def _wait_for_health() -> None:
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{_API_URL}/health", timeout=3) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise AssertionError("Production API did not become healthy after restart")


def _remove_probe_reminders(marker: str) -> None:
    script = """\
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
if not any(sys.argv[2] in str(item.get("title", "")) for item in data.get("reminders", [])):
    sys.exit(0)
probe_ids = {
    str(item.get("id") or "")
    for item in data.get("reminders", [])
    if sys.argv[2] in str(item.get("title", ""))
}
data["reminders"] = [
    item for item in data.get("reminders", [])
    if sys.argv[2] not in str(item.get("title", ""))
]
internal_state = dict(data.get("internal_state") or {})
internal_state["tool_idempotency"] = {
    key: reminder_id
    for key, reminder_id in dict(internal_state.get("tool_idempotency") or {}).items()
    if str(reminder_id) not in probe_ids
}
data["internal_state"] = internal_state
path.parent.mkdir(parents=True, exist_ok=True)
fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(data, stream, indent=2)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(name, path)
"""
    _compose(
        "exec",
        "-T",
        "dadbot-api",
        "python",
        "-c",
        script,
        _REMINDER_PATH,
        marker,
    )


@pytest.mark.integration
@pytest.mark.durability
@pytest.mark.timeout(420)
def test_sigkill_during_real_tool_turn_recovers_once():
    if os.environ.get("DADBOT_RUN_PRODUCTION_CRASH_RECOVERY") != "1":
        pytest.skip("Set DADBOT_RUN_PRODUCTION_CRASH_RECOVERY=1 to run against production Compose.")

    tenant_id, token, api_secret = _production_credentials()
    marker = f"durability-probe-{uuid4().hex}"
    session_id = f"recovery-{uuid4().hex[:12]}"
    title = f"Production persistence probe {marker}"
    created = False
    killed = False
    restarted = False
    crash_observed_at: float | None = None
    watcher: subprocess.Popen[str] | None = None

    try:
        with urllib.request.urlopen(f"{_API_URL}/health", timeout=5) as response:
            assert response.status == 200

        watcher, watcher_lines = _start_reminder_watcher(marker, tenant_id)
        try:
            watcher_status = watcher_lines.get(timeout=_TIMEOUT_SECONDS)
        except queue.Empty as exc:
            raise AssertionError("The reminder watcher did not start") from exc
        assert watcher_status == "READY", f"Reminder watcher exited before starting: {watcher_status}"

        accepted = _request(
            "POST",
            f"sessions/{session_id}/chat",
            token,
            {
                "tenant_id": tenant_id,
                "user_input": f"Remind me to verify the persistence probe {marker}",
                "requested_model": os.environ.get("DADBOT_DEFAULT_MODEL", "qwen3.5:9b"),
            },
            api_secret=api_secret,
        )
        task_id = str(accepted["task_id"])

        deadline = time.monotonic() + _TIMEOUT_SECONDS
        next_status_check = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                watcher_status = watcher_lines.get(timeout=0.1)
            except queue.Empty:
                watcher_status = ""
            if watcher_status is None:
                raise AssertionError("The reminder watcher exited before observing the tool side effect")
            if watcher_status.startswith("FOUND:"):
                created = True
                crash_observed_at = time.time()
                count = int(watcher_status.split(":", 1)[1])
                assert count == 1, f"Tool created {count} copies before the crash"
                break
            if watcher_status == "WATCH_TIMEOUT":
                break
            if time.monotonic() >= next_status_check:
                status = _request("GET", f"tasks/{task_id}", token, api_secret=api_secret).get("task") or {}
                if str(status.get("status") or "").lower() == "failed":
                    raise AssertionError(f"Production worker failed before the tool call: {status.get('error')}")
                if str(status.get("status") or "").lower() == "completed":
                    raise AssertionError("The task completed before a crash-after-side-effect window was observed")
                next_status_check = time.monotonic() + 10
        else:
            raise AssertionError("The real worker did not persist the reminder before the crash")

        if not created:
            raise AssertionError("The real worker did not persist the reminder before the crash")

        assert watcher is not None
        killed = True
        _compose("restart", "dadbot-api")
        _wait_for_health()
        restarted = True
        assert _reminder_count(marker, tenant_id) == 1, (
            "The API restart lost the reminder before the recovered task completed"
        )

        deadline = time.monotonic() + _TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            task = _request("GET", f"tasks/{task_id}", token, api_secret=api_secret).get("task") or {}
            if str(task.get("status") or "").lower() == "failed":
                raise AssertionError(f"Recovered task failed: {task.get('error')}")
            response = _request("GET", f"tasks/{task_id}", token, api_secret=api_secret).get("response")
            if response is not None:
                assert str(task.get("status") or "").lower() == "completed"
                assert crash_observed_at is not None
                completed_at = str(task.get("completed_at") or "")
                assert completed_at, "Recovered task did not persist its completion timestamp"
                completed_epoch = datetime.fromisoformat(
                    completed_at.replace("Z", "+00:00"),
                ).timestamp()
                assert completed_epoch > crash_observed_at, (
                    "The task completed before the API crash instead of recovering afterward"
                )
                assert _reminder_count(marker, tenant_id) == 1, "Recovery duplicated or lost the reminder side effect"
                return
            time.sleep(5)
        raise AssertionError("The queued task was not resumed after the production API restart")
    finally:
        try:
            if watcher is not None and watcher.poll() is None:
                watcher.terminate()
                watcher.wait(timeout=5)
        finally:
            try:
                if killed and not restarted:
                    _compose("restart", "dadbot-api")
                    _wait_for_health()
            finally:
                try:
                    _remove_probe_reminders(marker)
                finally:
                    _cleanup_test_tenant(tenant_id)


@pytest.mark.integration
@pytest.mark.durability
@pytest.mark.timeout(180)
def test_real_set_reminder_execution_is_confirmed_by_postgres():
    if os.environ.get("DADBOT_RUN_PRODUCTION_TOOL_CERTIFICATION") != "1":
        pytest.skip(
            "Set DADBOT_RUN_PRODUCTION_TOOL_CERTIFICATION=1 to run against production Compose.",
        )

    tenant_id, token, api_secret = _production_credentials()
    marker = f"tool-certification-{uuid4().hex}"
    session_id = f"certification-{uuid4().hex[:12]}"

    try:
        before_count = _reminder_count(marker, tenant_id)
        assert before_count == 0

        accepted = _request(
            "POST",
            f"sessions/{session_id}/chat",
            token,
            {
                "tenant_id": tenant_id,
                "user_input": f"Remind me to verify the real tool execution {marker}",
                "requested_model": os.environ.get("DADBOT_DEFAULT_MODEL", "qwen3.5:9b"),
            },
            api_secret=api_secret,
        )
        task_id = str(accepted["task_id"])

        deadline = time.monotonic() + _TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            result = _request("GET", f"tasks/{task_id}", token, api_secret=api_secret)
            task = result.get("task") or {}
            status = str(task.get("status") or "").lower()
            if status == "failed":
                raise AssertionError(f"Production worker failed: {task.get('error')}")
            if status == "completed":
                assert result.get("response") is not None, "Completed task has no persisted response"
                break
            time.sleep(1)
        else:
            raise AssertionError("Production tool-execution task did not complete")

        assert _reminder_count(marker, tenant_id) == before_count + 1, (
            "The real set_reminder execution did not create exactly one durable Postgres reminder",
        )
    finally:
        _remove_probe_reminders(marker)
        _cleanup_test_tenant(tenant_id)
