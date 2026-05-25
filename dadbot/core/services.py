"""
Central service wiring for DadBot.
Composes and injects all core components for extensibility and integration.
"""
from dadbot.agentic.planner import ReActPlanner
from dadbot.tools.executors import (
    ReminderExecutor, CalendarExecutor, WebSearchExecutor, ToolRegistry
)
from dadbot.core.logging_config import configure_logging
from dadbot.core.metrics import metrics
from dadbot.smart_home.mqtt_client import SmartHomeMQTTClient
from dadbot.speech.asr import ASR
from dadbot.speech.tts import TTS

def build_services(**overrides):
    """
    Compose all core services for DadBot.
    Allows dependency injection for testing and extension.
    """
    # Phase B tool registry with all executors
    tool_registry = overrides.get('tool_registry') or ToolRegistry()
    tool_registry.register('reminder', ReminderExecutor())
    tool_registry.register('calendar', CalendarExecutor())
    tool_registry.register('web_search', WebSearchExecutor())
    planner = overrides.get('planner') or ReActPlanner(tool_registry, overrides.get('llm_port', lambda prompt: "LLM output"))
    services = {
        'planner': planner,
        'tool_registry': tool_registry,
        'metrics': overrides.get('metrics', metrics),
        'smart_home': overrides.get('smart_home', SmartHomeMQTTClient('localhost')),
        'asr': overrides.get('asr', ASR()),
        'tts': overrides.get('tts', TTS()),
    }
    return services
