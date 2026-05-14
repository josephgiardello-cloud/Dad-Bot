# DadBot Assistant API Contract

**Version:** 1.0  
**Status:** STABLE (locked)  
**Last Updated:** 2026-05-06

---

## Executive Summary

The **Assistant API** is the sole public interface for all external interactions with the DadBot execution kernel. This contract defines a stable, immutable five-method surface that shields users from 100+ internal modules, deterministic reducers, replay engines, execution policies, invariant gates, and graph machinery.

**Core principle:** Users see only materialized results; kernels and internal state remain hidden.

---

## Immutable Public Surface

The following five methods constitute the complete public API. **No additional methods may be added without explicit user approval.**

### 1. `chat(message: str, debug: bool = False) -> dict`

**Purpose:** Execute a user message through the full orchestration pipeline.

**Parameters:**
- `message` (str): User input to process
- `debug` (bool, optional): If `True`, includes internal kernel metadata in response; defaults to `False`

**Returns:**
```python
{
    "response": str,           # User-facing materialized result
    "memory_updates": None,    # Reserved for future use; always None in v1.0
    "tool_calls": None,        # Reserved for future use; always None in v1.0
}
```

**Debug Mode Behavior:**
- `debug=False` (default): Returns only public response dictionary
- `debug=True`: Augments response with internal state (graph checkpoint, execution trace, invariant violations, policy decisions, etc.)
- **Guarantee:** Debug metadata is never leaked in production mode; diagnostics are opt-in

**Exceptions:**
- `ValueError`: Invalid message format
- `RuntimeError`: Kernel execution failure
- `TimeoutError`: Turn execution exceeded deadline

**Example:**
```python
from dadbot import AssistantRuntime

bot = AssistantRuntime()
result = bot.chat("What's a dad joke?")
print(result["response"])  # User message response

# Debug mode: see internal state
result_debug = bot.chat("Tell me another", debug=True)
print(result_debug)  # Includes kernel internals for troubleshooting
```

---

### 2. `run_task(message: str) -> str`

**Purpose:** Submit a message for background execution; return immediately with a task identifier.

**Parameters:**
- `message` (str): User input to process asynchronously

**Returns:**
- `task_id` (str): Opaque identifier for querying task status later

**Behavior:**
- Submits turn to kernel background task queue
- Returns immediately without waiting for completion
- Task executes asynchronously; caller must poll `get_state(task_id)` to track progress

**Exceptions:**
- `ValueError`: Invalid message format
- `RuntimeError`: Background task queue full or unavailable

**Example:**
```python
task_id = bot.run_task("Analyze this dataset...")
# Do other work...
status = bot.get_state(task_id)
while status["status"] != "completed":
    print(f"Status: {status['status']}")
    status = bot.get_state(task_id)
print(f"Result: {status['result']}")
```

---

### 3. `get_state(task_id: str) -> dict`

**Purpose:** Query the status and metadata of a background task.

**Parameters:**
- `task_id` (str): Identifier returned by `run_task()`

**Returns:**
```python
{
    "status": str,              # One of: "pending", "running", "completed", "failed"
    "progress": int,            # Percentage completion (0-100)
    "result": str | None,       # Materialized result (None until completed)
    "error": str | None,        # Error message (None if no failure)
    "metadata": dict,           # Task execution metadata (timestamps, token counts, etc.)
}
```

**Status Values:**
- `"pending"`: Task queued, not yet started
- `"running"`: Task actively executing
- `"completed"`: Task finished successfully
- `"failed"`: Task execution errored

**Exceptions:**
- `KeyError`: `task_id` not found in kernel or background manager

**Example:**
```python
task_id = bot.run_task("Large job...")
status = bot.get_state(task_id)
print(f"Progress: {status['progress']}%")
if status["status"] == "completed":
    print(f"Done! Result: {status['result']}")
```

---

### 4. `reset_session() -> None`

**Purpose:** Clear all session state and reinitialize the kernel.

**Parameters:** (none)

**Returns:** `None`

**Behavior:**
- Clears session memory, execution ledger, and active tasks
- Resets runtime state to initial condition
- All pending background tasks are cancelled

**Side Effects:**
- All in-flight `task_id` references become invalid
- Memory is cleared; queries to `memory()` return empty results
- Chat history is erased

**Exceptions:**
- `RuntimeError`: Kernel reset failed (e.g., persisted state unlock issue)

**Example:**
```python
# After several interactions...
bot.reset_session()
# Kernel now in clean state; previous tasks/memory gone
```

---

### 5. `memory(query: str, limit: int = 5) -> list[dict]`

**Purpose:** Query the kernel's memory manager for relevant stored memories.

**Parameters:**
- `query` (str): Natural language search query
- `limit` (int, optional): Maximum number of results to return; defaults to 5

**Returns:**
```python
[
    {
        "id": str,           # Memory entry identifier
        "content": str,      # Memory text
        "relevance": float,  # Semantic relevance score (0.0-1.0)
        "timestamp": str,    # ISO 8601 timestamp of creation/update
        "tags": list[str],   # Metadata tags
    },
    ...
]
```

**Returns empty list if no memories match query.**

**Exceptions:**
- `ValueError`: Invalid query format
- `RuntimeError`: Memory manager unavailable

**Example:**
```python
memories = bot.memory("dad jokes told recently", limit=10)
for mem in memories:
    print(f"[{mem['relevance']:.2f}] {mem['content']}")
```

---

## Architectural Boundary

### EXPOSED (Public Contract)

- ✅ `AssistantRuntime` class (entry point)
- ✅ Five immutable methods: `chat()`, `run_task()`, `get_state()`, `reset_session()`, `memory()`
- ✅ Return dictionaries with documented keys
- ✅ Optional `debug` parameter for diagnostics

### HIDDEN (Internal Only)

- ❌ `execution_graph` (internal kernel)
- ❌ `invariant_engine`, `invariant_gate` (policy enforcement)
- ❌ `execution_policy` (control decisions)
- ❌ `control_plane` (orchestration)
- ❌ `canonical_execution_reducer` (deterministic state fold)
- ❌ `execution_replay_engine` (determinism verification)
- ❌ `execution_ledger` (event store)
- ❌ Any module prefixed `execution_`, `graph_`, `invariant_`, `kernel_`, `ledger_`, or `replay_`
- ❌ Mixin classes (`boot_mixin`, `turn_mixin`, `action_mixin`, etc.)
- ❌ All 100+ core modules (visible only in `dadbot/core/` directory)

**Rationale:** Public API is a thin facade over a complex, evolving kernel. Users don't need (and shouldn't have) access to internal machinery. This boundary prevents API fragmentation and protects against future kernel refactors.

---

## Contract Enforcement

### Method Signature Immutability

Each method's signature is locked:

```python
def chat(self, message: str, debug: bool = False) -> dict
def run_task(self, message: str) -> str
def get_state(self, task_id: str) -> dict
def reset_session(self) -> None
def memory(self, query: str, limit: int = 5) -> list[dict]
```

**Parameter additions or removals are breaking changes** and require major version bump.

### Response Shape Stability

Response dictionaries have fixed keys. New fields may be added only in reserved slots (`memory_updates`, `tool_calls` in `chat()`). Existing fields are immutable.

### Export Policy

- `AssistantRuntime` is exported from `dadbot/__init__.py` and available via `from dadbot import AssistantRuntime`
- No other internal classes, modules, or functions are exported
- Package version follows semantic versioning (MAJOR.MINOR.PATCH)

---

## Versioning

**Current Version:** 1.0

- **Major bump (2.0):** Breaking signature changes, return type changes, method removals
- **Minor bump (1.1):** New optional parameters, new reserved fields populated, new methods (requires user approval)
- **Patch bump (1.0.1):** Bug fixes, behavior corrections, internal optimizations with stable contract

---

## Usage Patterns

### Simple Chat
```python
from dadbot import AssistantRuntime

bot = AssistantRuntime()
response = bot.chat("How can I debug this?")
print(response["response"])
```

### Async Background Task
```python
task_id = bot.run_task("Run a long analysis...")

# Poll in loop
import time
while True:
    state = bot.get_state(task_id)
    if state["status"] == "completed":
        print(state["result"])
        break
    elif state["status"] == "failed":
        print(f"Error: {state['error']}")
        break
    time.sleep(1)
```

### Debugging
```python
result = bot.chat("What happened?", debug=True)
# result now contains internal kernel state for diagnostics
print(result)  # Includes trace, policy decisions, invariant checks
```

### Memory Search
```python
memories = bot.memory("previous solutions for this problem", limit=20)
for mem in memories:
    print(f"Relevance {mem['relevance']}: {mem['content']}")
```

### Session Cleanup
```python
# Clear all state
bot.reset_session()

# Kernel is now fresh
result = bot.chat("Start over")
```

---

## FAQ

**Q: Can I call kernel methods directly?**  
A: No. The kernel is internal. All interactions must flow through the five public methods.

**Q: Can I access `DadBot.process_user_message()` or other internal APIs?**  
A: No. These are internal implementation details. Use `AssistantRuntime.chat()` instead.

**Q: What if I need a new method or parameter?**  
A: File a request with the maintainers. The contract may evolve with explicit approval and version bumps.

**Q: Is `debug=True` safe for production?**  
A: No. Debug mode exposes internal kernel state and may leak sensitive information. Use only in diagnostics/development.

**Q: Can I subclass AssistantRuntime?**  
A: Not recommended. The class is sealed for compatibility. Create a wrapper instead if customization is needed.

---

## Maintenance & Evolution

This contract is locked until a major version bump. Changes are tracked in this file. Community feedback and feature requests are encouraged but require formal review before integration.

**Last validated:** DEV lane full pass (350+ tests green), no regressions.

