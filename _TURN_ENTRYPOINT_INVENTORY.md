# Turn Entrypoint Inventory

| File | Function | Status |
| --- | --- | --- |
| `dadbot/core/turn_mixin.py` | `execute_turn` | canonical production adapter |
| `dadbot/core/turn_mixin.py` | `process_user_message` | legacy facade wrapper -> canonical adapter |
| `dadbot/core/turn_mixin.py` | `process_user_message_async` | legacy facade wrapper -> canonical adapter |
| `dadbot/core/turn_mixin.py` | `process_user_message_stream` | legacy facade wrapper -> canonical adapter |
| `dadbot/core/turn_mixin.py` | `process_user_message_stream_async` | legacy facade wrapper -> canonical adapter |
| `dadbot/core/turn_mixin.py` | `handle_turn_sync` | legacy facade wrapper -> canonical adapter |
| `dadbot/core/turn_mixin.py` | `handle_turn_async` | legacy facade wrapper -> canonical adapter |
| `dadbot/core/turn_mixin.py` | `run_turn` | compatibility wrapper for typed replay/recovery contract |
| `dadbot/services/turn_service.py` | `process_user_message` | legacy service wrapper -> canonical adapter |
| `dadbot/services/turn_service.py` | `process_user_message_async` | legacy service wrapper -> canonical adapter |
| `dadbot/services/turn_service.py` | `process_user_message_stream` | legacy service wrapper -> canonical adapter |
| `dadbot/services/turn_service.py` | `process_user_message_stream_async` | legacy service wrapper -> canonical adapter |
| `dadbot/core/orchestrator.py` | `handle_turn` | internal async kernel executor |
| `dadbot/core/orchestrator.py` | `run` | internal sync bridge |
| `dadbot/core/orchestrator.py` | `run_async` | internal async bridge |
| `dadbot/core/control_plane.py` | `submit_turn` | kernel scheduling boundary |
| `dadbot/runtime_core/services.py` | `DadBotLLMService.generate_reply` | production caller migrated to canonical adapter |
| `dadbot/managers/runtime_interface.py` | `RuntimeInterfaceManager.chat_loop` | production caller migrated to canonical adapter |
| `dadbot/runtime_core/streamlit_runtime.py` | `UIRuntimeAPI.send_user_message` | production caller migrated to canonical adapter |
| `dadbot/app_runtime.py` | CLI prompt loop callsite | production caller migrated to canonical adapter |