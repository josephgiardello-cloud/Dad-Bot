from pathlib import Path


class SystemRepair:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def create_missing_modules(self, missing: dict):
        for name, path in missing.items():
            self._write_stub(name, Path(path))

    def _write_stub(self, name: str, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        if name == "event_store":
            content = self._event_store_stub()
        elif name == "session_store":
            content = self._session_store_stub()
        elif name == "execution_ledger":
            content = self._execution_ledger_stub()
        else:
            content = self._generic_stub(name)

        path.write_text(content, encoding="utf-8")

    def _event_store_stub(self) -> str:
        return """from collections import deque


class EventStore:
    def __init__(self, maxlen=1000):
        self._events = deque(maxlen=maxlen)

    def append(self, event):
        self._events.append(event)

    def all(self):
        return list(self._events)
"""

    def _session_store_stub(self) -> str:
        return """class SessionStore:
    def __init__(self):
        self._sessions = {}

    def get(self, session_id):
        return self._sessions.get(session_id)

    def set(self, session_id, state):
        self._sessions[session_id] = state
"""

    def _execution_ledger_stub(self) -> str:
        return """class ExecutionLedger:
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)

    def read(self):
        return self.events
"""

    def _generic_stub(self, name: str) -> str:
        return f"class {name.title().replace('_', '')}:\n    pass\n"
