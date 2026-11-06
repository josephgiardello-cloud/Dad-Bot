import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from test_streamlit_smoke import _prepare_streamlit_sandbox


def test_canonical_streamlit_boot_and_turn(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    sandbox = tmp_path / "phase1_app"
    sandbox.mkdir()
    _prepare_streamlit_sandbox(repo_root, sandbox)
    shutil.copy2(sandbox / "dad_profile.template.json", sandbox / "dad_profile.json")
    code = textwrap.dedent(
        """
        from streamlit.testing.v1 import AppTest
        from dadbot.services.agent_service import AgentService

        app = AppTest.from_file("dad_streamlit.py").run(timeout=45)
        assert not app.exception, list(app.exception)
        bot = app.session_state["bot"]
        registry = bot.services.registry
        assert bot.service_registry is registry
        orchestrator = bot._get_turn_orchestrator()
        assert orchestrator is bot.services.turn_orchestrator
        assert orchestrator.registry is registry
        assert orchestrator.graph.registry is registry
        assert registry.get("turn_graph").registry is registry
        persistence = registry.get("persistence_service")
        assert persistence is bot.services.persistence_service
        assert persistence is registry.get("persistence")
        assert persistence is registry.get("storage")
        assert persistence is orchestrator.registry.get("persistence_service")
        assert bot.model_port is not None
        agent_service = registry.get("agent_service")
        assert isinstance(agent_service, AgentService)
        assert agent_service.bot is bot
        assert bot.profile_runtime.profile
        assert bot.memory.memory_projection() is not None
        requests = []

        async def generate_async(messages, **kwargs):
            requests.append(kwargs)
            return "Start by clearing one corner of your desk."

        bot.model_port.generate_async = generate_async
        app.session_state["auto_speak_replies"] = False
        app.chat_input[0].set_value("Help me organize my desk.").run(timeout=45)
        assert not app.exception, list(app.exception)
        assert requests, "The canonical turn never invoked the model"
        assert any(request.get("purpose") == "chat" for request in requests), requests
        history = app.session_state["chat_threads"]["default"]
        assert history[-1]["role"] == "assistant", history
        assert history[-1]["text"] == "Start by clearing one corner of your desk.", history[-1]["text"]
        assert not app.session_state["message_outbox"], app.session_state["message_outbox"]
        assert bot.services.registry is registry
        assert bot.service_registry is registry
        navigation = next(radio for radio in app.radio if radio.label == "Navigation")
        navigation.set_value("Smart Home").run(timeout=45)
        assert not app.exception, list(app.exception)
        rendered_copy = "\\n".join(str(item.value) for item in app.markdown)
        assert "Home Assistant MCP controls with an explicit tool allowlist." in rendered_copy
        assert "MQTT controls" not in rendered_copy
        bot.shutdown()
        """
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("DADBOT_")}
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=sandbox,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        env=env,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
