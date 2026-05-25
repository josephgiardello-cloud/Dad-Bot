from unittest.mock import MagicMock
from dadbot.core.dadbot import DadBot
from dadbot.typing import MemoryManager, RelationshipManager, MoodManager, ProfileRuntime, EventBus

def make_test_dadbot(**kwargs):
    """Create a DadBot instance with mock managers for testing."""
    memory_manager_mock = MagicMock(spec=MemoryManager)
    memory_manager_mock.load_memory_store = lambda: None
    profile_runtime_mock = MagicMock(spec=ProfileRuntime)
    profile_runtime_mock.profile = {
        "style": "test-style",
        "family": {
            "dad": {"birthdate": "2000-01-01"},
            "carrie": {"birthdate": "2000-01-02"},
            "tony": {"birthdate": "2000-01-03"},
            "marriage": {"date": "2020-01-01"},
        },
        "education": "test-education",
        "chat_routing": {"topic_rules": [], "core_fact_ids": []},
        "facts": {},
    }
    from dadbot.tools.executors import tool_registry
    defaults = {
        "memory_manager": memory_manager_mock,
        "relationship_manager": MagicMock(spec=RelationshipManager),
        "mood_manager": MagicMock(spec=MoodManager),
        "profile_runtime": profile_runtime_mock,
        "event_bus": MagicMock(spec=EventBus),
        "metrics": MagicMock(name="metrics"),
        "smart_home": MagicMock(name="smart_home"),
        "asr": MagicMock(name="asr"),
        "tts": MagicMock(name="tts"),
        "tool_registry": tool_registry,
    }
    defaults.update(kwargs)
    class DummyDocStore:
        def __init__(self):
            self._store = {}
        def load_session_state(self, key):
            return self._store.get(key)
        def save_session_state(self, key, value):
            self._store[key] = value
    class DummyConfig:
        model_name = "test-model"
    class PatchedDadBot(DadBot):
        @property
        def metrics(self):
            return super().metrics
        @metrics.setter
        def metrics(self, value):
            self.services.metrics = value

        @property
        def smart_home(self):
            return super().smart_home
        @smart_home.setter
        def smart_home(self, value):
            self.services.smart_home = value

        @property
        def asr(self):
            return super().asr
        @asr.setter
        def asr(self, value):
            self.services.asr = value

        @property
        def tts(self):
            return super().tts
        @tts.setter
        def tts(self, value):
            self.services.tts = value
        def __init__(self, *a, **k):
            self._memory_manager = defaults["memory_manager"]
            # Patch required runtime constants before boot
            self.CONTEXT_TOKEN_BUDGET = 4096
            self.PREFERRED_EMBEDDING_MODELS = ("test-embedding-model",)
            self.STREAM_TIMEOUT_SECONDS = 30
            self.STREAM_MAX_CHARS = 4096
            self.GRAPH_REFRESH_DEBOUNCE_SECONDS = 1.0
            super().__init__(*a, **k)
            # Force test services after super().__init__ in case overwritten
            self.services = services_ns

        @property
        def tool_registry(self):
            return super().tool_registry
        @tool_registry.setter
        def tool_registry(self, value):
            # Allow runtime wiring to assign this property
            self.services.tool_registry = value

    # Forcibly remove any lingering property for tool_registry
    try:
        delattr(TestDadBot, "tool_registry")
    except Exception:
        pass
    # Print class dict for debug
    # print("TestDadBot dict after property removal:", TestDadBot.__dict__)
    from types import SimpleNamespace
    services_ns = SimpleNamespace(**defaults)
    bot = PatchedDadBot(
        memory_manager=defaults["memory_manager"],
        relationship_manager=defaults["relationship_manager"],
        mood_manager=defaults["mood_manager"],
        profile_runtime=defaults["profile_runtime"],
        event_bus=defaults["event_bus"],
        services=services_ns,
        document_store=DummyDocStore(),
    )
    bot.config = DummyConfig()
    return bot
