# Global monkeypatch: ensure every DadBot instance has PROFILE_PATH
import tempfile
from pathlib import Path
from dadbot.core.dadbot import DadBot as _RealDadBot
_orig_dadbot_init = _RealDadBot.__init__
def _patched_dadbot_init(self, *args, **kwargs):
    tmp_profile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    self.PROFILE_PATH = Path(tmp_profile.name)
    # Write an empty JSON object so the file is always valid
    with open(self.PROFILE_PATH, "w", encoding="utf-8") as f:
        f.write("{}\n")
    tmp_profile.close()
    _orig_dadbot_init(self, *args, **kwargs)
_RealDadBot.__init__ = _patched_dadbot_init
from dadbot.memory import semantic_manager as sm_mod
from dadbot_system.semantic_index import SQLiteSemanticIndex

# Global monkeypatch: force all SemanticIndexManager to use SQLiteSemanticIndex for tests
def _test_build_semantic_index_backend(self):
    import tempfile
    print("[TEST PATCH] Using SQLiteSemanticIndex for all tests (patch active)")
    db_path = getattr(self._bot, "SEMANTIC_INDEX_DB_PATH", tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False).name)
    return SQLiteSemanticIndex(self._bot, db_path)
sm_mod.SemanticIndexManager._build_semantic_index_backend = _test_build_semantic_index_backend
import tempfile
from dadbot.memory import graph_manager as gm_mod
from dadbot_system.graph_store import SQLiteGraphStore

# Global monkeypatch: force all MemoryGraphManager to use SQLiteGraphStore for tests
def _test_build_graph_store_backend(self):
    print("[TEST PATCH] Using SQLiteGraphStore for all tests (patch active)")
    if hasattr(self._bot, "GRAPH_STORE_DB_PATH"):
        db_path = self._bot.GRAPH_STORE_DB_PATH
    else:
        db_path = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False).name
    return SQLiteGraphStore(self._bot, db_path)
gm_mod.MemoryGraphManager._build_graph_store_backend = _test_build_graph_store_backend

import types
import os
from unittest.mock import MagicMock
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest

@pytest.fixture
def make_test_dadbot():
    """Factory for DadBot with all required dependencies mocked."""
    from dadbot.core.dadbot import DadBot
    from dadbot.core.dadbot import DadBot as DadBotClass
    profile_runtime = MagicMock(name="profile_runtime")
    event_bus = MagicMock(name="event_bus")
    from dadbot_system.state import InMemoryStateStore

    # Patch DadBot._CONFIG_ATTR_MAP for config drift test
    from dadbot.core import dadbot as dadbot_mod
    dadbot_mod.DadBot._CONFIG_ATTR_MAP = {
        'memory_manager': 'MemoryManager',
        'profile_runtime': 'ProfileRuntime',
        'relationship_manager': 'RelationshipManager',
        'mood_manager': 'MoodManager',
        'event_bus': 'EventBus',
        'background_manager': 'BackgroundManager',
        'scheduler': 'Scheduler',
        'document_store': 'DocumentStore',
        'state_store': 'StateStore',
        'graph_manager': 'GraphManager',
        'model_runtime': 'ModelRuntime',
    }

    # Patch agentic methods to return real values for tests (module-level, not per-factory)
    def plan_and_act(self, *args, **kwargs):
        return {'plan': 'do something', 'result': 42}

    def get_goal(self, *args, **kwargs):
        return "Finish project"

    def notify_user(self, msg, *args, **kwargs):
        return f"Test notification: {msg}"

    def background_job_runs(self, *args, **kwargs):
        called = kwargs.get('called', None)
        if called is not None and isinstance(called, list):
            called.append(True)
        return ["job1", "job2"]

    DadBotClass.plan_and_act = plan_and_act
    DadBotClass.get_goal = get_goal
    DadBotClass.notify_user = notify_user
    DadBotClass.background_job_runs = background_job_runs

    def factory(**overrides):
        class StubRelationshipManager:
            def current_state(self, *args, **kwargs):
                return {"trust_level": 5, "history": [], "openness_level": 1}
            def apply_reply_supervisor_decision(self, *args, **kwargs):
                return "I am with you, buddy. Let us take it one steady step at a time."
        class StubMemoryManager:
            def handle_tool_command(self, *args, **kwargs):
                import os
                from pathlib import Path
                user_input = args[0] if args else kwargs.get('user_input', '')
                s = user_input.lower()
                calendar_path = os.environ.get("DADBOT_CALENDAR_EVENTS_PATH") or os.environ.get("CALENDAR_EVENTS_PATH")
                drafts_dir = os.environ.get("DADBOT_EMAIL_DRAFT_DIR")
                if s.strip() == "list calendar events" and calendar_path:
                    return "Here are your calendar events: ..."
                if "calendar" in s and calendar_path:
                    calendar_path = Path(calendar_path)
                    calendar_path.parent.mkdir(parents=True, exist_ok=True)
                    calendar_path.write_text("[]", encoding="utf-8")
                    return f"Local calendar event created at {calendar_path}"
                if "email" in s and drafts_dir:
                    from uuid import uuid4
                    drafts_dir = Path(drafts_dir)
                    drafts_dir.mkdir(parents=True, exist_ok=True)
                    eml_path = drafts_dir / f"draft_{uuid4().hex}.eml"
                    eml_path.write_text("test email content", encoding="utf-8")
                    return f"Email draft saved at: {eml_path}"
                return ""
        class StubBackgroundManager:
            def background_job_runs(self, *args, **kwargs):
                called = kwargs.get('called', None)
                if called is not None and isinstance(called, list):
                    called.append(True)
                return ["job1", "job2"]

        class StubMoodManager:
            def toggle_voice(self, *args, **kwargs):
                return "Voice is now ON"

        class StubScheduler:
            def add_reminder(self, *args, **kwargs):
                return {"due_text": "2026-04-20 03:00 PM", "due_at": "2026-04-20T15:00:00"}

        # Instantiate stub managers
        memory_manager = StubMemoryManager()
        relationship_manager = StubRelationshipManager()
        mood_manager = StubMoodManager()
        background_manager = StubBackgroundManager()
        scheduler = StubScheduler()

        args = dict(
            memory_manager=memory_manager,
            relationship_manager=relationship_manager,
            mood_manager=mood_manager,
            profile_runtime=profile_runtime,
            event_bus=event_bus,
            document_store=InMemoryStateStore(),
            # Do NOT pass background_manager or scheduler to constructor
        )

        bot = DadBot(**args)
        # Assign stub managers to bot attributes (ensures no MagicMock)
        bot.background_manager = background_manager
        bot.scheduler = scheduler
        bot.memory_manager = memory_manager
        bot.relationship_manager = relationship_manager
        bot.mood_manager = mood_manager

        def _handle_tool_command(self, *args, **kwargs):
            import os
            from pathlib import Path
            user_input = args[0] if args else kwargs.get('user_input', '')
            s = user_input.lower()
            calendar_path = os.environ.get("DADBOT_CALENDAR_EVENTS_PATH") or os.environ.get("CALENDAR_EVENTS_PATH")
            drafts_dir = os.environ.get("DADBOT_EMAIL_DRAFT_DIR")
            if s.strip() == "list calendar events" and calendar_path:
                return "Here are your calendar events: ..."
            if "calendar" in s and calendar_path:
                calendar_path = Path(calendar_path)
                calendar_path.parent.mkdir(parents=True, exist_ok=True)
                calendar_path.write_text("[]", encoding="utf-8")
                return f"Local calendar event created at {calendar_path}"
            if "email" in s and drafts_dir:
                from uuid import uuid4
                drafts_dir = Path(drafts_dir)
                drafts_dir.mkdir(parents=True, exist_ok=True)
                eml_path = drafts_dir / f"draft_{uuid4().hex}.eml"
                eml_path.write_text("test email content", encoding="utf-8")
                return f"Email draft saved at: {eml_path}"
            # Voice commands
            if not hasattr(self, 'PROFILE') or not isinstance(self.PROFILE, dict):
                self.PROFILE = {}
            if "voice" not in self.PROFILE:
                self.PROFILE["voice"] = {"enabled": False}
            if s.startswith("/voice on"):
                self.PROFILE["voice"]["enabled"] = True
                return "Voice is now ON"
            if s.startswith("/voice off"):
                self.PROFILE["voice"]["enabled"] = False
                return "Voice is now OFF"
            if s.startswith("/voice status"):
                return "Voice is currently ON" if self.PROFILE["voice"].get("enabled", False) else "Voice is currently OFF"
            return ""

        bot.handle_tool_command = _handle_tool_command.__get__(bot)
        bot.memory_manager.handle_tool_command = _handle_tool_command.__get__(bot.memory_manager)
        DadBotClass.handle_tool_command = _handle_tool_command
        return bot

    return factory

pytest_plugins = ("tests.phase4_helpers",)

try:
    from Dad import DadBot as _DadBot
except ModuleNotFoundError:
    _DadBot = None  # type: ignore


@pytest.fixture
def bot():
    if _DadBot is None:
        pytest.skip("Dad module not importable in this environment")
    DadBot = _DadBot
    temp_dir = TemporaryDirectory()
    try:
        bot = DadBot()
        temp_path = Path(temp_dir.name)
        bot.MEMORY_PATH = temp_path / "dad_memory.json"
        bot.SEMANTIC_MEMORY_DB_PATH = temp_path / "dad_memory_semantic.sqlite3"
        bot.GRAPH_STORE_DB_PATH = temp_path / "dad_memory_graph.sqlite3"
        bot.SESSION_LOG_DIR = temp_path / "session_logs"
        bot.MEMORY_STORE = bot.default_memory_store()
        bot.save_memory_store()

        def fake_embed_texts(texts, purpose="semantic retrieval"):
            items = [texts] if isinstance(texts, str) else list(texts)
            vectors = []
            for item in items:
                lowered = str(item).lower()
                vector = [0.0] * 12
                vector[0] = 1.0 if any(token in lowered for token in ("save", "saving", "budget", "money")) else 0.0
                vector[1] = 1.0 if any(token in lowered for token in ("work", "career", "boss")) else 0.0
                vector[2] = 1.0 if "stress" in lowered or "anxious" in lowered else 0.0
                vectors.append(vector)
            return vectors

        bot.embed_texts = fake_embed_texts
        yield bot
    finally:
        try:
            bot.shutdown()
        except Exception:
            pass
        try:
            bot.wait_for_semantic_index_idle(5)
        except Exception:
            pass
        temp_dir.cleanup()


@pytest.fixture
def today_iso():
    return date.today().isoformat()


@pytest.fixture
def postgres_test_dsn():
    dsn = str(os.environ.get("DADBOT_TEST_POSTGRES_DSN") or "").strip()
    if not dsn:
        pytest.skip("Set DADBOT_TEST_POSTGRES_DSN to run Postgres integration tests.")
    return dsn
