from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys
import textwrap
import time
import webbrowser
from pathlib import Path
from socket import AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR, socket
from urllib.request import urlopen

from dadbot.runtime_adapter import runtime_contract_errors
from dadbot_system import (
	CompositeStateStore,
	DadBotOrchestrator,
	DadServiceClient,
	InMemoryEventBus,
	InMemoryStateStore,
	LocalMultiprocessBroker,
	PostgresStateStore,
	RedisStateStore,
	ServiceConfig,
	WorkerProcessManager,
	configure_logging,
	configure_tracing,
	create_api_app,
	normalize_tenant_id,
)

logger = logging.getLogger(__name__)


def env_truthy(name, default=False):
	raw_value = os.environ.get(name)
	if raw_value is None:
		return default
	return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def initialize_startup_logging(*, force=False):
	"""Initialize logging as early as possible for startup diagnostics."""
	try:
		config = ServiceConfig.from_environment()
		configure_logging(config.telemetry, force=force)
		logger.debug("Startup logging initialized.")
	except Exception as exc:
		print(f"Warning: startup logging initialization failed: {exc}", file=sys.stderr)


def _resolve_graph_config_path(base_script_path: Path):
	configured = str(os.environ.get("DADBOT_TURN_GRAPH_CONFIG_PATH") or "config.yaml").strip() or "config.yaml"
	configured_path = Path(configured)
	if configured_path.is_absolute():
		return configured_path
	return (base_script_path.parent / configured_path).resolve()


def check_dependencies(args, *, base_script_path: Path, runtime_cls):
	"""Fail fast on critical startup prerequisites before opening UI/CLI."""
	if "PYTEST_CURRENT_TEST" in os.environ:
		return

	if args.stop_streamlit or args.init_profile:
		return

	contract_issues = runtime_contract_errors(runtime_cls)
	if contract_issues:
		details = "; ".join(contract_issues)
		raise RuntimeError(f"Runtime class failed app contract validation: {details}")

	needs_local_runtime = (not args.serve_api) and ((not args.cli) or bool(args.disable_service_client))
	if needs_local_runtime:
		try:
			ollama_module = importlib.import_module("ollama")
			_ = ollama_module.ps()
		except Exception as exc:
			raise RuntimeError(
				"Ollama is not reachable. Start Ollama before launching Dad Bot local runtime."
			) from exc

	if needs_local_runtime and env_truthy("DADBOT_ENABLE_TURN_GRAPH", default=False):
		graph_config_path = _resolve_graph_config_path(base_script_path)
		if not graph_config_path.exists():
			raise RuntimeError(
				f"Turn graph is enabled, but config file was not found: {graph_config_path}"
			)

	check_system_resources(args)


# â”€â”€â”€ Model memory requirements (minimum RAM in GB) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_MODEL_MIN_RAM_GB: dict[str, float] = {
	"llama3.2:1b": 1.5,
	"llama3.2": 2.5,
	"phi4:mini": 3.0,
	"gemma3:4b": 4.5,
	"gemma3:1b": 2.0,
	"qwen2.5:3b": 3.5,
}
_DEFAULT_MIN_RAM_GB = 2.5


def check_system_resources(args) -> None:
	"""Warn or fail if system RAM is critically insufficient for the selected model.

	Uses *psutil* if available; silently skips the check when it is not installed
	so development machines without psutil still work fine.
	"""
	try:
		import psutil  # type: ignore[import]
	except ImportError:
		return

	model_name = str(getattr(args, "model", None) or "").strip().lower()
	required_gb = _MODEL_MIN_RAM_GB.get(model_name, _DEFAULT_MIN_RAM_GB)

	mem = psutil.virtual_memory()
	available_gb = mem.available / (1024 ** 3)
	total_gb = mem.total / (1024 ** 3)

	logger = logging.getLogger(__name__)

	if available_gb < required_gb * 0.5:
		raise RuntimeError(
			f"Critically low RAM: {available_gb:.1f} GB available, "
			f"but the model needs at least {required_gb:.1f} GB. "
			"Close other applications or choose a smaller model (--model llama3.2:1b)."
		)

	if available_gb < required_gb:
		logger.warning(
			"Low RAM warning: %.1f GB available, model may need up to %.1f GB. "
			"Performance may be degraded. Consider --model llama3.2:1b for lighter footprint.",
			available_gb,
			required_gb,
		)
	elif total_gb >= 8 and available_gb < 2.0:
		logger.warning(
			"System has %.1f GB total RAM but only %.1f GB free. "
			"Close other apps for best performance.",
			total_gb,
			available_gb,
		)


def parse_args(argv=None):
	parser = argparse.ArgumentParser(add_help=True)
	parser.add_argument("--cli", action="store_true", help="Run the terminal chat interface.")
	parser.add_argument("--serve-api", action="store_true", help="Run the multiprocess API service instead of Streamlit.")
	parser.add_argument("--stop-streamlit", action="store_true", help="Stop Dad Bot Streamlit processes for this workspace and free localhost:8501.")
	parser.add_argument("--service-url", metavar="URL", default="", help="Dad Bot API base URL for CLI/client execution.")
	parser.add_argument("--disable-service-client", action="store_true", help="Force CLI to use the in-process DadBot runtime instead of the API service.")
	parser.add_argument("--clear-memory", action="store_true", help="Clear Dad's saved memory store and exit.")
	parser.add_argument("--export-memory", metavar="PATH", help="Export Dad's saved memory store to a JSON file and exit.")
	parser.add_argument("--init-profile", action="store_true", help="Create a starter dad_profile.json and exit if one does not already exist.")
	parser.add_argument("--model", metavar="NAME", help="Override the default Ollama model for this run.")
	parser.add_argument("--worker-count", metavar="N", type=int, default=None, help="Number of queue workers when running --serve-api.")
	parser.add_argument("--api-host", metavar="HOST", default=None, help="Bind host for --serve-api.")
	parser.add_argument("--api-port", metavar="PORT", type=int, default=None, help="Bind port for --serve-api.")
	parser.add_argument("--redis-url", metavar="URL", default=None, help="Redis URL for fast session/task state.")
	parser.add_argument("--postgres-dsn", metavar="DSN", default=None, help="Postgres DSN for durable state.")
	parser.add_argument("--tenant-id", metavar="ID", default="", help="Tenant/customer identifier used to isolate persisted state.")
	parser.add_argument("--otel", action="store_true", help="Enable OpenTelemetry if the SDK is installed.")
	parser.add_argument("--no-signoff", action="store_true", help="Disable Dad's default reply signoff for this run.")
	parser.add_argument("--light", action="store_true", help="Use a lighter runtime that skips mood detection and extra review passes.")
	return parser.parse_args(argv)


def minimal_streamlit_stub_source():
	return textwrap.dedent(
		"""
		import streamlit as st

		from Dad import DadBot

		st.set_page_config(page_title="Dad Bot", page_icon="ðŸ§”", layout="centered")

		bot = DadBot()

		if "messages" not in st.session_state:
			st.session_state.messages = [
				{"role": "assistant", "content": bot.opening_message("Hey buddy, I'm right here.")}
			]

		st.title("Dad Bot")
		st.caption("Auto-generated minimal Streamlit chat because dad_streamlit.py was missing.")

		for message in st.session_state.messages:
			with st.chat_message(message["role"]):
				st.markdown(message["content"])

		if prompt := st.chat_input("Talk to Dad..."):
			st.session_state.messages.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)
			with st.chat_message("assistant"):
				reply, _should_end = bot.process_user_message(prompt)
				reply = reply or bot.reply_finalization.append_signoff("I'm here, buddy.")
				st.markdown(reply)
			st.session_state.messages.append({"role": "assistant", "content": reply})
			bot.persist_conversation_async()
			st.rerun()
		"""
	).strip() + "\n"


def ensure_streamlit_app_file(streamlit_app_path):
	if streamlit_app_path.exists():
		return False

	streamlit_app_path.write_text(minimal_streamlit_stub_source(), encoding="utf-8")
	return True


def launch_streamlit_app(*, append_signoff=True, light_mode=False, script_path: str | Path | None = None):
	def required_streamlit_port():
		configured_port = os.environ.get("DADBOT_STREAMLIT_PORT", "8501")
		try:
			required_port = int(configured_port)
		except (TypeError, ValueError) as exc:
			raise RuntimeError(
				f"DADBOT_STREAMLIT_PORT must be a valid integer port. Got: {configured_port!r}"
			) from exc

		if required_port != 8501:
			raise RuntimeError(
				f"Dad Bot is pinned to port 8501. Update or unset DADBOT_STREAMLIT_PORT (current value: {configured_port!r})."
			)

		with socket(AF_INET, SOCK_STREAM) as candidate_socket:
			candidate_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
			try:
				candidate_socket.bind(("127.0.0.1", required_port))
			except OSError as exc:
				raise RuntimeError(
					"Dad Bot requires localhost:8501, but that port is already in use. "
					"Stop the existing process using 8501 and try again."
				) from exc

		return required_port

	def wait_for_streamlit(url, process, timeout_seconds=15):
		deadline = time.monotonic() + timeout_seconds
		while time.monotonic() < deadline:
			try:
				with urlopen(url, timeout=1):
					return True
			except Exception:
				if process.poll() is not None:
					return False
				time.sleep(0.25)
		return False

	base_path = Path(script_path) if script_path is not None else Path.cwd() / "Dad.py"
	streamlit_app_path = base_path.with_name("dad_streamlit.py")
	stub_created = ensure_streamlit_app_file(streamlit_app_path)
	if stub_created:
		print(f"dad_streamlit.py was missing, so a minimal stub was created at {streamlit_app_path}.")
	chosen_port = required_streamlit_port()
	local_url = f"http://localhost:{chosen_port}"
	command_env = os.environ.copy()
	if append_signoff:
		command_env.pop("DADBOT_NO_SIGNOFF", None)
	else:
		command_env["DADBOT_NO_SIGNOFF"] = "1"
	if light_mode:
		command_env["DADBOT_LIGHT_MODE"] = "1"
	else:
		command_env.pop("DADBOT_LIGHT_MODE", None)
	command = [
		sys.executable,
		"-m",
		"streamlit",
		"run",
		str(streamlit_app_path),
		"--browser.gatherUsageStats",
		"false",
		"--server.headless",
		"true",
		"--server.port",
		str(chosen_port),
	]
	process = subprocess.Popen(command, cwd=str(streamlit_app_path.parent), env=command_env)
	try:
		if wait_for_streamlit(local_url, process):
			try:
				webbrowser.open(local_url)
			except Exception:
				logger.warning("Could not open browser automatically for %s", local_url, exc_info=True)
		return process.wait()
	except KeyboardInterrupt:
		if process.poll() is None:
			process.terminate()
			try:
				return process.wait(timeout=5)
			except subprocess.TimeoutExpired:
				process.kill()
				return process.wait()
		return process.returncode or 0


def stop_streamlit_app(script_path: str | Path | None = None):
	workspace = str((Path(script_path) if script_path is not None else Path.cwd() / "Dad.py").resolve().parent)

	def port_8501_in_use():
		with socket(AF_INET, SOCK_STREAM) as candidate_socket:
			candidate_socket.settimeout(0.5)
			return candidate_socket.connect_ex(("127.0.0.1", 8501)) == 0

	if os.name == "nt":
		query = (
			f"$workspace = {workspace!r}; "
			"Get-CimInstance Win32_Process | Where-Object { "
			"$cmd = $_.CommandLine; "
			"$_.Name -ieq \"python.exe\" -and $cmd -and $cmd -like \"*$workspace*\" -and ($cmd -like \"*streamlit run*\" -or $cmd -like \"*dad_streamlit.py*\") "
			"} | Select-Object ProcessId, Name, CommandLine | ConvertTo-Json -Compress"
		)
		result = subprocess.run(
			["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
			capture_output=True,
			text=True,
			check=False,
		)

		if result.returncode != 0:
			stderr = (result.stderr or "").strip()
			if stderr:
				print(stderr, file=sys.stderr)
			return result.returncode or 1

		stdout = (result.stdout or "").strip()
		if not stdout:
			targets = []
		else:
			try:
				parsed = json.loads(stdout)
			except json.JSONDecodeError as exc:
				print(f"Could not parse stop-streamlit process list: {exc}", file=sys.stderr)
				return 1
			targets = parsed if isinstance(parsed, list) else [parsed]

		if not targets:
			print("STOP_RESULT: no matching Dad Bot Streamlit processes were running.")
			if port_8501_in_use():
				print("PORT_8501_OCCUPIED_BY_NON_DADBOT_PROCESS", file=sys.stderr)
				return 1
			print("PORT_8501_FREE")
			return 0

		stop_failed = False
		for target in sorted(targets, key=lambda item: int(item.get("ProcessId", 0)), reverse=True):
			process_id = int(target.get("ProcessId", 0))
			name = target.get("Name") or "python.exe"
			kill_result = subprocess.run(
				["taskkill", "/PID", str(process_id), "/F", "/T"],
				capture_output=True,
				text=True,
				check=False,
			)
			if kill_result.returncode == 0:
				print(f"STOPPED PID={process_id} NAME={name}")
			else:
				stop_failed = True
				error_text = (kill_result.stderr or kill_result.stdout or "").strip()
				print(f"STOP_FAILED PID={process_id} NAME={name} ERROR={error_text}", file=sys.stderr)

		if stop_failed:
			return 1

		if port_8501_in_use():
			print("PORT_8501_STILL_OCCUPIED", file=sys.stderr)
			return 1

		print("PORT_8501_FREE")
		return 0

	try:
		result = subprocess.run(
			["lsof", "-ti", "tcp:8501"],
			capture_output=True,
			text=True,
			check=False,
		)
	except FileNotFoundError:
		print("--stop-streamlit is currently supported on Windows or systems with lsof installed.", file=sys.stderr)
		return 1

	process_ids = []
	for line in (result.stdout or "").splitlines():
		line = line.strip()
		if not line:
			continue
		try:
			process_ids.append(int(line))
		except ValueError:
			continue

	if not process_ids:
		print("STOP_RESULT: no matching Dad Bot Streamlit processes were running.")
		if port_8501_in_use():
			print("PORT_8501_OCCUPIED_BY_NON_DADBOT_PROCESS", file=sys.stderr)
			return 1
		print("PORT_8501_FREE")
		return 0

	stop_failed = False
	for process_id in sorted(set(process_ids), reverse=True):
		try:
			os.kill(process_id, 15)
			print(f"STOPPED PID={process_id}")
		except OSError as exc:
			stop_failed = True
			print(f"STOP_FAILED PID={process_id} ERROR={exc}", file=sys.stderr)

	if stop_failed:
		return 1

	if port_8501_in_use():
		print("PORT_8501_STILL_OCCUPIED", file=sys.stderr)
		return 1

	print("PORT_8501_FREE")
	return 0


def build_service_state_store(config):
	fast_store = None
	durable_store = None

	if config.persistence.redis_url:
		fast_store = RedisStateStore(config.persistence.redis_url)

	if config.persistence.postgres_dsn:
		durable_store = PostgresStateStore(
			config.persistence.postgres_dsn,
			session_table=config.persistence.session_table,
			task_table=config.persistence.task_table,
			event_table=config.persistence.event_table,
		)

	if fast_store is None and durable_store is None:
		return InMemoryStateStore()

	return CompositeStateStore(fast_store=fast_store, durable_store=durable_store)


def build_customer_document_store(config):
	fast_store = None
	durable_store = None

	if config.persistence.redis_url:
		fast_store = RedisStateStore(config.persistence.redis_url)

	if config.persistence.postgres_dsn:
		durable_store = PostgresStateStore(
			config.persistence.postgres_dsn,
			session_table=config.persistence.session_table,
			task_table=config.persistence.task_table,
			event_table=config.persistence.event_table,
		)

	if durable_store is None:
		return None
	if fast_store is None:
		return durable_store
	return CompositeStateStore(fast_store=fast_store, durable_store=durable_store)


def launch_api_service(args, *, dadbot_cls=None):
	try:
		uvicorn = importlib.import_module("uvicorn")
	except ImportError as exc:
		print("FastAPI service mode requires uvicorn and fastapi to be installed.")
		raise SystemExit(1) from exc

	resolved_dadbot_cls = dadbot_cls
	if resolved_dadbot_cls is None:
		main_module = sys.modules.get("__main__")
		resolved_dadbot_cls = getattr(main_module, "DadBot", None)
		if resolved_dadbot_cls is None:
			from Dad import DadBot as ImportedDadBot

			resolved_dadbot_cls = ImportedDadBot

	config = ServiceConfig.from_environment()
	config.default_model = args.model or config.default_model
	if args.api_host:
		config.api.host = args.api_host
	if args.api_port is not None:
		config.api.port = args.api_port
	if args.worker_count is not None:
		config.workers.worker_count = max(1, int(args.worker_count))
	if args.redis_url is not None:
		config.persistence.redis_url = str(args.redis_url or "").strip()
	if args.postgres_dsn is not None:
		config.persistence.postgres_dsn = str(args.postgres_dsn or "").strip()
	if args.otel:
		config.telemetry.otel_enabled = True

	configure_logging(config.telemetry, force=True)
	configure_tracing(config.telemetry)

	broker = LocalMultiprocessBroker(max_queue_size=config.queue.max_queue_size)
	event_bus = InMemoryEventBus()
	state_store = build_service_state_store(config)
	orchestrator = DadBotOrchestrator(
		broker,
		state_store=state_store,
		event_bus=event_bus,
		planner_debug_factory=resolved_dadbot_cls.default_planner_debug_state,
	)
	worker_manager = WorkerProcessManager(broker, config)
	worker_manager.start()
	app = create_api_app(orchestrator, worker_manager=worker_manager, config=config)
	try:
		uvicorn.run(app, host=config.api.host, port=config.api.port, log_level=config.telemetry.log_level.lower())
	finally:
		worker_manager.shutdown()
	return 0


def main(argv=None, *, dadbot_cls=None, service_client_cls=DadServiceClient, script_path: str | Path | None = None):
	args = parse_args(sys.argv[1:] if argv is None else argv)
	initialize_startup_logging(force=False)
	resolved_dadbot_cls = dadbot_cls
	if resolved_dadbot_cls is None:
		main_module = sys.modules.get("__main__")
		resolved_dadbot_cls = getattr(main_module, "DadBot", None)
		if resolved_dadbot_cls is None:
			from Dad import DadBot as ImportedDadBot

			resolved_dadbot_cls = ImportedDadBot

	base_script_path = Path(script_path) if script_path is not None else Path.cwd() / "Dad.py"
	check_dependencies(args, base_script_path=base_script_path, runtime_cls=resolved_dadbot_cls)

	if args.no_signoff:
		os.environ["DADBOT_NO_SIGNOFF"] = "1"
	if args.light:
		os.environ["DADBOT_LIGHT_MODE"] = "1"
	if args.tenant_id:
		os.environ["DADBOT_TENANT_ID"] = normalize_tenant_id(args.tenant_id)

	if args.init_profile:
		created = resolved_dadbot_cls.initialize_profile_file()
		if created:
			print("Starter dad_profile.json created.")
		else:
			print("dad_profile.json already exists.")
		return 0

	if args.stop_streamlit:
		return stop_streamlit_app(script_path=base_script_path)

	if args.serve_api:
		return launch_api_service(args, dadbot_cls=resolved_dadbot_cls)

	bot = resolved_dadbot_cls(
		model_name=args.model or "llama3.2",
		append_signoff=not args.no_signoff,
		light_mode=args.light,
		tenant_id=args.tenant_id,
	)

	if args.clear_memory:
		bot.memory.clear_memory_store()
		bot.print_system_message("Dad's saved memory has been cleared.")
		return 0
	if args.export_memory:
		bot.memory.export_memory_store(args.export_memory)
		bot.print_system_message(f"Dad's saved memory was exported to {args.export_memory}.")
		return 0
	if args.cli:
		if args.disable_service_client:
			bot.chat_loop()
		else:
			client = service_client_cls()
			if args.service_url:
				client.config.base_url = args.service_url
			if args.tenant_id:
				client.config.tenant_id = normalize_tenant_id(args.tenant_id)
			bot.chat_loop_via_service(client)
		return 0

	return launch_streamlit_app(
		append_signoff=not args.no_signoff,
		light_mode=args.light,
		script_path=base_script_path,
	)


__all__ = [
	"build_customer_document_store",
	"build_service_state_store",
	"ensure_streamlit_app_file",
	"launch_api_service",
	"launch_streamlit_app",
	"main",
	"minimal_streamlit_stub_source",
	"parse_args",
	"stop_streamlit_app",
]
