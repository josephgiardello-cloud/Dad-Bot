"""Model subsystem — deterministic LLM and embedding boundary.

Public API:
- ModelPort: Abstract interface for all model I/O
- OllamaModelAdapter: Concrete Ollama implementation
- ModelConfig: Immutable configuration
- ModelCallError: Raised on model call failure
"""

from dadbot.runtime.model.model_call_port import (
    DeterminismContext,
    ModelCallError,
    ModelConfig,
    ModelPort,
)
from dadbot.runtime.model.ollama_model_adapter import OllamaModelAdapter

__all__ = [
    "DeterminismContext",
    "ModelCallError",
    "ModelConfig",
    "ModelPort",
    "OllamaModelAdapter",
]
