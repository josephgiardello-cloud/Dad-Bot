"""
Central service wiring for DadBot.
Composes and injects all core components for extensibility and integration.
"""
from dadbot.agentic.planner import ReActPlanner
from dadbot.core.metrics import metrics
from dadbot.runtime.env_loader import is_production_environment
from dadbot.speech.asr import ASR
from dadbot.speech.tts import TTS


# Kernel boundary: do not import from dadbot.tools.executors here.
# Tools must be explicitly injected until this composition path has authorized executors.
class ToolRegistry:
    def __init__(self): self._tools = {}
    def register(self, name, tool): self._tools[name] = tool
    def get(self, name): return self._tools.get(name)
    def describe_all(self): return ', '.join(self._tools.keys())

def build_services(**overrides):
    """
    Compose all core services for DadBot.
    Allows dependency injection for testing and extension.
    """
    if is_production_environment():
        missing = [
            name
            for name in ("tool_registry", "planner")
            if overrides.get(name) is None
        ]
        if missing:
            raise RuntimeError(
                "Production service composition requires explicitly injected: "
                + ", ".join(missing),
            )

    # This legacy planner registry stays empty unless callers inject real executors.
    tool_registry = overrides.get('tool_registry')
    if tool_registry is None:
        tool_registry = ToolRegistry()
    planner = overrides.get('planner')
    if planner is None:
        planner = ReActPlanner(
            tool_registry,
            overrides.get('llm_port', lambda prompt: "LLM output"),
        )
    services = {
        'planner': planner,
        'tool_registry': tool_registry,
        'metrics': overrides.get('metrics', metrics),
        'asr': overrides.get('asr', ASR()),
        'tts': overrides.get('tts', TTS()),
    }
    return services
