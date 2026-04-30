"""Transitional manager imports during the Dad.py package split."""

from .conversation_persistence import ConversationPersistenceManager
from .long_term import LongTermSignalsManager
from .maintenance import MaintenanceScheduler
from .memory_commands import MemoryCommandManager
from .memory_coordination import MemoryCoordinator
from .memory_query import MemoryQueryManager
from .multimodal import MultimodalManager
from .profile_runtime import ProfileRuntimeManager
from .prompt_assembly import PromptAssemblyManager
from .reply_finalization import ReplyFinalizationManager
from .reply_generation import ReplyGenerationManager
from .reply_supervisor import ReplySupervisorManager
from .runtime_client import RuntimeClientManager
from .runtime_interface import RuntimeInterfaceManager
from .runtime_model import RuntimeModelManager
from .runtime_orchestration import RuntimeOrchestrationManager
from .runtime_storage import RuntimeStorageManager
from .safety import SafetySupportManager
from .session_summary import SessionSummaryManager
from .status_reporting import StatusReportingManager

__all__ = [
    "ConversationPersistenceManager",
    "LongTermSignalsManager",
    "MaintenanceScheduler",
    "MemoryCommandManager",
    "MemoryCoordinator",
    "MemoryQueryManager",
    "MultimodalManager",
    "ProfileRuntimeManager",
    "PromptAssemblyManager",
    "ReplyFinalizationManager",
    "ReplyGenerationManager",
    "ReplySupervisorManager",
    "RuntimeClientManager",
    "RuntimeInterfaceManager",
    "RuntimeModelManager",
    "RuntimeOrchestrationManager",
    "RuntimeStorageManager",
    "SafetySupportManager",
    "SessionSummaryManager",
    "StatusReportingManager",
]
