from types import SimpleNamespace

import pytest

import dadbot.app_runtime as app_runtime
from dadbot.runtime_adapter import runtime_contract_errors


class _RuntimeStub:
    @classmethod
    def initialize_profile_file(cls, profile_path=None, force=False):
        return True

    @staticmethod
    def default_planner_debug_state():
        return {}

    def clear_memory_store(self):
        return None

    def export_memory_store(self, export_path):
        return None

    def print_system_message(self, message):
        return None

    def chat_loop(self):
        return None

    def chat_loop_via_service(self, service_client, session_id=None):
        return None


def _args(**overrides):
    base = {
        "stop_streamlit": False,
        "init_profile": False,
        "serve_api": False,
        "cli": False,
        "disable_service_client": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_runtime_contract_errors_accepts_valid_runtime():
    assert runtime_contract_errors(_RuntimeStub) == []


def test_runtime_contract_errors_reports_missing_and_non_callable():
    class _BrokenRuntime:
        initialize_profile_file = 1

    errors = runtime_contract_errors(_BrokenRuntime)

    assert "attribute is not callable: initialize_profile_file" in errors
    assert "missing attribute: default_planner_debug_state" in errors


def test_runtime_contract_errors_reports_descriptor_mismatch():
    class _BadDescriptorRuntime:
        def initialize_profile_file(cls, profile_path=None, force=False):
            return True

        @classmethod
        def default_planner_debug_state(cls):
            return {}

        def clear_memory_store(self):
            return None

        def export_memory_store(self, export_path):
            return None

        def print_system_message(self, message):
            return None

        def chat_loop(self):
            return None

        def chat_loop_via_service(self, service_client, session_id=None):
            return None

    errors = runtime_contract_errors(_BadDescriptorRuntime)

    assert "attribute has wrong descriptor: initialize_profile_file (expected classmethod)" in errors
    assert "attribute has wrong descriptor: default_planner_debug_state (expected staticmethod)" in errors


def test_runtime_contract_errors_reports_required_parameter_mismatch():
    class _BadSignatureRuntime:
        @classmethod
        def initialize_profile_file(cls, profile_path=None, force=False):
            return True

        @staticmethod
        def default_planner_debug_state():
            return {}

        def __init__(self, model_name="llama3.2", *, append_signoff=True):
            return None

        def clear_memory_store(self):
            return None

        def export_memory_store(self):
            return None

        def print_system_message(self):
            return None

        def chat_loop(self):
            return None

        def chat_loop_via_service(self, session_id=None):
            return None

    errors = runtime_contract_errors(_BadSignatureRuntime)

    assert "attribute missing parameter: export_memory_store.export_path" in errors
    assert "attribute missing parameter: print_system_message.message" in errors
    assert "attribute missing parameter: chat_loop_via_service.service_client" in errors
    assert "attribute missing parameter: __init__.light_mode" in errors
    assert "attribute missing parameter: __init__.tenant_id" in errors


def test_check_dependencies_skips_when_pytest_env_set(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "x")
    args = _args()

    app_runtime.check_dependencies(args, base_script_path=tmp_path / "Dad.py", runtime_cls=None)


def test_check_dependencies_skips_non_runtime_actions(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    calls = []

    def _boom(_runtime_cls):
        calls.append("contract")
        raise AssertionError("should not be called")

    monkeypatch.setattr(app_runtime, "runtime_contract_errors", _boom)
    app_runtime.check_dependencies(_args(stop_streamlit=True), base_script_path=tmp_path / "Dad.py", runtime_cls=None)
    app_runtime.check_dependencies(_args(init_profile=True), base_script_path=tmp_path / "Dad.py", runtime_cls=None)
    assert calls == []


def test_check_dependencies_raises_for_contract_errors(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(app_runtime, "runtime_contract_errors", lambda _cls: ["missing attribute: chat_loop"])

    try:
        app_runtime.check_dependencies(_args(), base_script_path=tmp_path / "Dad.py", runtime_cls=_RuntimeStub)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Runtime class failed app contract validation" in str(exc)
        assert "missing attribute: chat_loop" in str(exc)


def test_check_dependencies_raises_when_ollama_unreachable(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(app_runtime, "runtime_contract_errors", lambda _cls: [])

    def _raise_import(_name):
        raise RuntimeError("offline")

    monkeypatch.setattr(app_runtime.importlib, "import_module", _raise_import)

    try:
        app_runtime.check_dependencies(_args(), base_script_path=tmp_path / "Dad.py", runtime_cls=_RuntimeStub)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Ollama is not reachable" in str(exc)


def test_check_dependencies_skips_ollama_for_cli_service_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(app_runtime, "runtime_contract_errors", lambda _cls: [])

    called = {"imported": False}

    def _import_module(_name):
        called["imported"] = True
        raise AssertionError("ollama import should not be attempted")

    monkeypatch.setattr(app_runtime.importlib, "import_module", _import_module)
    app_runtime.check_dependencies(
        _args(cli=True, disable_service_client=False),
        base_script_path=tmp_path / "Dad.py",
        runtime_cls=_RuntimeStub,
    )
    assert called["imported"] is False


def test_check_dependencies_raises_when_graph_config_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(app_runtime, "runtime_contract_errors", lambda _cls: [])

    class _OllamaOK:
        @staticmethod
        def ps():
            return {}

    monkeypatch.setattr(app_runtime.importlib, "import_module", lambda _name: _OllamaOK)
    monkeypatch.setenv("DADBOT_ENABLE_TURN_GRAPH", "1")
    monkeypatch.setenv("DADBOT_TURN_GRAPH_CONFIG_PATH", "missing-config.yaml")

    try:
        app_runtime.check_dependencies(_args(), base_script_path=tmp_path / "Dad.py", runtime_cls=_RuntimeStub)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Turn graph is enabled" in str(exc)
        assert "missing-config.yaml" in str(exc)


def test_check_dependencies_accepts_existing_relative_graph_config(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(app_runtime, "runtime_contract_errors", lambda _cls: [])

    class _OllamaOK:
        @staticmethod
        def ps():
            return {}

    (tmp_path / "config.yaml").write_text("graph: {}\n", encoding="utf-8")
    monkeypatch.setattr(app_runtime.importlib, "import_module", lambda _name: _OllamaOK)
    monkeypatch.setenv("DADBOT_ENABLE_TURN_GRAPH", "1")
    monkeypatch.delenv("DADBOT_TURN_GRAPH_CONFIG_PATH", raising=False)

    app_runtime.check_dependencies(_args(), base_script_path=tmp_path / "Dad.py", runtime_cls=_RuntimeStub)


def test_initialize_startup_logging_configures_logging(monkeypatch):
    class _Config:
        telemetry = object()

    captured = {}

    monkeypatch.setattr(app_runtime.ServiceConfig, "from_environment", staticmethod(lambda: _Config()))

    def _configure_logging(telemetry, force=False):
        captured["telemetry"] = telemetry
        captured["force"] = force

    monkeypatch.setattr(app_runtime, "configure_logging", _configure_logging)
    app_runtime.initialize_startup_logging(force=True)

    assert captured["telemetry"] is _Config.telemetry
    assert captured["force"] is True


def test_check_system_resources_warns_when_ram_low(monkeypatch, caplog):
    import sys
    import types
    import logging

    fake_psutil = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(available=3.2 * 1024 ** 3, total=16 * 1024 ** 3)
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    args = _args(model="gemma3:4b")
    caplog.set_level(logging.WARNING)

    app_runtime.check_system_resources(args)

    assert "Low RAM warning" in caplog.text


def test_check_system_resources_raises_when_critically_low_ram(monkeypatch):
    import sys
    import types

    fake_psutil = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(available=1.5 * 1024 ** 3, total=16 * 1024 ** 3)
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    args = _args(model="gemma3:4b")
    with pytest.raises(RuntimeError, match="Critically low RAM"):
        app_runtime.check_system_resources(args)
