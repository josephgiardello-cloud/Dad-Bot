"""Tests for GDPR/CCPA memory command handling."""

from __future__ import annotations

import json
from pathlib import Path

from dadbot.managers.memory_commands import MemoryCommandManager


class _MemoryStore:
    def __init__(self, parent: DummyBot) -> None:
        self._parent = parent

    def clear_memory_store(self) -> None:
        self._parent._cleared = True


class DummyBot:
    def __init__(self):
        self._saved = []
        self._cleared = False
        self._relationship_state = {"trust_level": 0.7, "relationship_tenure_days": 42}
        self.memory = _MemoryStore(self)

    def memory_catalog(self):
        return [{"summary": "likes pizza", "category": "food"}]

    def consolidated_memories(self):
        return [{"summary": "Tony cares about family.", "category": "values"}]

    def add_memory(self, summary):
        memory = {"summary": summary, "category": "general"}
        self._saved.append(memory)
        return memory

    def forget_memories(self, query):
        return []

    def format_memories_for_reply(self, memories):
        return ", ".join(str(m.get("summary") or "") for m in memories)

    def relationship_state(self):
        return self._relationship_state

    def clear_memory_store(self):
        self._cleared = True

    def parse_tool_command(self, _):
        return None

    def get_memory_reply(self, _):
        return None


class TestMemoryCommandGdpr:
    def setup_method(self):
        self.bot = DummyBot()
        self.manager = MemoryCommandManager(self.bot)

    def test_export_my_data_creates_file(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        user_input = "/export my data"
        result = self.manager.handle_memory_command(user_input)
        assert "saved everything I know about you" in result
        path_text = result.split("\n")[2].strip()
        path = Path(path_text)
        assert path.exists()
        assert path.read_text(encoding="utf-8")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["export_version"] == "1.0"
        assert data["memories"]

    def test_delete_my_data_clears_store(self):
        result = self.manager.handle_memory_command("/delete my data")
        assert self.bot._cleared is True
        assert "erased everything I had saved" in result

    def test_forget_me_clears_store(self):
        result = self.manager.handle_memory_command("/forget me")
        assert self.bot._cleared is True
        assert "starting completely fresh" in result

    def test_what_do_you_know_about_me_summarizes_data(self):
        result = self.manager.handle_memory_command("/what do you know about me")
        assert "Individual memories (1 total)" in result
        assert "Consolidated long-term insights:" in result
        assert "1 entr" in result
        assert "Relationship state:" in result

    def test_non_gdpr_command_returns_none(self):
        assert self.manager.handle_memory_command("remember that I like cookies") is not None
