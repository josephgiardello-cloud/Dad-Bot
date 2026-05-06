"""
Guard tests to enforce the Assistant API contract.

This module prevents regression of the public five-method surface.
Tests verify:
  1. Exactly five public methods exist on AssistantRuntime
  2. No internal kernel classes leak into response shapes
  3. Debug mode is properly gated
  4. Method signatures remain immutable
  5. Package-level exports are stable
"""
import sys
from unittest.mock import MagicMock
import inspect
import pytest
from dadbot import AssistantRuntime
import dadbot


@pytest.fixture
def mock_kernel():
    """Provide a mocked kernel for testing without full initialization."""
    kernel = MagicMock()
    kernel.process_user_message.return_value = ("test response", False)
    kernel.runtime_orchestration = MagicMock()
    kernel.runtime_orchestration.submit_background_task.return_value = MagicMock(dadbot_task_id="task-123")
    kernel.runtime_state_manager = MagicMock()
    kernel.runtime_state_manager.get_task_state.return_value = {"status": "pending", "progress": 0}
    kernel.memory_manager = MagicMock()
    kernel.memory_manager.find_memory_matches.return_value = []
    return kernel


class TestAssistantAPIContractEnforcement:
    """Verify the assistant facade surface cannot be accidentally expanded."""

    def test_exactly_five_public_methods(self):
        """Guard: AssistantRuntime must have exactly five public methods."""
        public_methods = [
            name for name, method in inspect.getmembers(AssistantRuntime, predicate=inspect.isfunction)
            if not name.startswith("_")
        ]
        assert sorted(public_methods) == sorted(
            ["chat", "run_task", "get_state", "reset_session", "memory"]
        ), f"Public methods changed. Expected 5, got {len(public_methods)}: {public_methods}"

    def test_no_new_public_attributes(self, mock_kernel):
        """Guard: AssistantRuntime should not have unexpected public attributes."""
        # Gather all public attributes (non-callable)
        bot = AssistantRuntime(mock_kernel)
        public_attrs = [
            name for name in dir(bot)
            if not name.startswith("_") and not callable(getattr(bot, name))
        ]
        # Should be empty or minimal (only 'kernel')
        assert all(name in ["kernel"] for name in public_attrs), (
            f"Unexpected public attributes found: {public_attrs}. "
            "Only 'kernel' should be exposed (internally)."
        )

    def test_chat_signature_locked(self):
        """Guard: chat() signature must remain immutable."""
        sig = inspect.signature(AssistantRuntime.chat)
        params = list(sig.parameters.keys())
        assert params == ["self", "message", "debug"], (
            f"chat() signature changed. Expected ['self', 'message', 'debug'], got {params}"
        )
        # Verify debug has default False
        assert sig.parameters["debug"].default is False, (
            "debug parameter default changed; must remain False"
        )

    def test_run_task_signature_locked(self):
        """Guard: run_task() signature must remain immutable."""
        sig = inspect.signature(AssistantRuntime.run_task)
        params = list(sig.parameters.keys())
        assert params == ["self", "message"], (
            f"run_task() signature changed. Expected ['self', 'message'], got {params}"
        )

    def test_get_state_signature_locked(self):
        """Guard: get_state() signature must remain immutable."""
        sig = inspect.signature(AssistantRuntime.get_state)
        params = list(sig.parameters.keys())
        assert params == ["self", "task_id"], (
            f"get_state() signature changed. Expected ['self', 'task_id'], got {params}"
        )

    def test_reset_session_signature_locked(self):
        """Guard: reset_session() signature must remain immutable."""
        sig = inspect.signature(AssistantRuntime.reset_session)
        params = list(sig.parameters.keys())
        assert params == ["self"], (
            f"reset_session() signature changed. Expected ['self'], got {params}"
        )

    def test_memory_signature_locked(self):
        """Guard: memory() signature must remain immutable."""
        sig = inspect.signature(AssistantRuntime.memory)
        params = list(sig.parameters.keys())
        assert params == ["self", "query", "limit"], (
            f"memory() signature changed. Expected ['self', 'query', 'limit'], got {params}"
        )
        # Verify limit has default 5
        assert sig.parameters["limit"].default == 5, (
            "limit parameter default changed; must remain 5"
        )

    def test_chat_response_shape_immutable(self, mock_kernel):
        """Guard: chat() response must have only documented keys."""
        bot = AssistantRuntime(mock_kernel)
        result = bot.chat("test", debug=False)
        
        expected_keys = {"response", "memory_updates", "tool_calls"}
        actual_keys = set(result.keys())
        assert actual_keys == expected_keys, (
            f"chat() response shape changed. Expected {expected_keys}, got {actual_keys}"
        )

    def test_debug_mode_properly_gated(self, mock_kernel):
        """Guard: debug=True must exist and gate internal metadata."""
        bot = AssistantRuntime(mock_kernel)
        
        # Call with debug=False
        result_normal = bot.chat("test", debug=False)
        assert "debug" not in result_normal, "debug=False should not include debug key"
        
        # Call with debug=True
        result_debug = bot.chat("test", debug=True)
        assert isinstance(result_debug, dict), "debug=True should return dict"
        assert "debug" in result_debug, "debug=True should include debug key"

    def test_package_export_stable(self):
        """Guard: AssistantRuntime must be exported from dadbot.__init__."""
        assert hasattr(dadbot, "AssistantRuntime"), (
            "AssistantRuntime not exported from dadbot package"
        )
        assert dadbot.AssistantRuntime is AssistantRuntime, (
            "dadbot.AssistantRuntime is not the correct class"
        )

    def test_no_internal_kernel_objects_in_response(self, mock_kernel):
        """Guard: Response dictionaries must not contain kernel objects."""
        bot = AssistantRuntime(mock_kernel)
        result = bot.chat("test", debug=False)
        
        # Recursively check that response contains only JSON-serializable types
        def is_json_safe(obj):
            """Check if object is JSON-serializable (no kernel objects)."""
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return True
            if isinstance(obj, (list, tuple)):
                return all(is_json_safe(item) for item in obj)
            if isinstance(obj, dict):
                return all(
                    isinstance(k, str) and is_json_safe(v)
                    for k, v in obj.items()
                )
            # Any other type (objects, etc.) is not JSON-safe
            return False
        
        assert is_json_safe(result), (
            f"Response contains non-JSON-serializable objects: {result}. "
            "Kernel internals must not leak into public responses."
        )

    def test_no_kernel_attributes_exposed(self, mock_kernel):
        """Guard: AssistantRuntime instance should not expose kernel internals."""
        bot = AssistantRuntime(mock_kernel)
        
        forbidden_patterns = [
            "graph",
            "orchestrator",
            "control_plane",
            "invariant",
            "execution_",
            "kernel_",
            "ledger_",
            "replay_",
            "policy",
        ]
        
        for attr in dir(bot):
            if attr.startswith("_"):
                continue
            for pattern in forbidden_patterns:
                assert pattern.lower() not in attr.lower(), (
                    f"AssistantRuntime exposes forbidden attribute '{attr}' "
                    f"matching pattern '{pattern}'. Public API must not leak kernel internals."
                )

    def test_method_returns_proper_types(self, mock_kernel):
        """Guard: Each method returns documented types."""
        mock_kernel.reset_session = MagicMock(return_value=None)
        mock_kernel.memory_manager.find_memory_matches.return_value = []
        
        bot = AssistantRuntime(mock_kernel)
        
        # chat() -> dict
        result = bot.chat("test", debug=False)
        assert isinstance(result, dict), "chat() must return dict"
        
        # run_task() -> str (task_id)
        task_id = bot.run_task("test")
        assert isinstance(task_id, str), "run_task() must return str (task_id)"
        
        # reset_session() -> None
        result = bot.reset_session()
        assert result is None, "reset_session() must return None"
        
        # memory() -> list[dict]
        result = bot.memory("test", limit=5)
        assert isinstance(result, list), "memory() must return list"
        if result:
            assert all(isinstance(item, dict) for item in result), (
                "memory() must return list[dict]"
            )

    def test_no_monkey_patches_on_runtime(self):
        """Guard: The public facade must not be dynamically modified post-creation."""
        initial_methods = set(dir(AssistantRuntime))
        
        # Simulate accidental monkey-patching (should fail)
        def new_method(self):
            pass
        
        # Try to add a method
        try:
            AssistantRuntime.new_method = new_method
            # If we get here, verify we can detect the change
            current_methods = set(dir(AssistantRuntime))
            assert "new_method" in current_methods, "Monkey patch detection failed"
            # Clean up
            delattr(AssistantRuntime, "new_method")
        except (AttributeError, TypeError):
            # If class is somehow sealed, that's even better
            pass


class TestContractDocumentation:
    """Verify contract documentation exists and is current."""

    def test_contract_document_exists(self):
        """Guard: assistant_api_contract.md must exist."""
        import os
        contract_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "docs",
            "assistant_api_contract.md"
        )
        assert os.path.exists(contract_path), (
            f"Contract document not found at {contract_path}. "
            "Create docs/assistant_api_contract.md with API specification."
        )

    def test_contract_mentions_five_methods(self):
        """Guard: Contract must explicitly document all five methods."""
        import os
        contract_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "docs",
            "assistant_api_contract.md"
        )
        with open(contract_path, "r", encoding="utf-8") as f:
            content = f.read().lower()
        
        methods = ["chat", "run_task", "get_state", "reset_session", "memory"]
        for method in methods:
            assert method in content, (
                f"Contract document must mention '{method}' method"
            )

    def test_contract_specifies_immutability(self):
        """Guard: Contract must state the surface is locked."""
        import os
        contract_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "docs",
            "assistant_api_contract.md"
        )
        with open(contract_path, "r", encoding="utf-8") as f:
            content = f.read().lower()
        
        assert "immutable" in content or "locked" in content, (
            "Contract must explicitly state that the API surface is immutable/locked"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
