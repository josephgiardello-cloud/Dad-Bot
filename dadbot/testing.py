from unittest.mock import MagicMock
from dadbot.core.dadbot import DadBot
from dadbot.typing import MemoryManager, RelationshipManager, MoodManager, ProfileRuntime, EventBus

def make_test_dadbot(**kwargs):
    """Create a DadBot instance with mock managers for testing."""
    defaults = {
        "memory_manager": MagicMock(spec=MemoryManager),
        "relationship_manager": MagicMock(spec=RelationshipManager),
        "mood_manager": MagicMock(spec=MoodManager),
        "profile_runtime": MagicMock(spec=ProfileRuntime),
        "event_bus": MagicMock(spec=EventBus),
    }
    defaults.update(kwargs)
    return DadBot(**defaults)
