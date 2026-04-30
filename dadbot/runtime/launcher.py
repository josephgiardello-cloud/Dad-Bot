from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from socket import AF_INET, SO_REUSEADDR, SOCK_STREAM, SOL_SOCKET, socket
from urllib.request import urlopen

from dadbot.core.execution_boundary import enforce_execution_role
from dadbot_system import (
    DadBotOrchestrator,
    DadServiceClient,
    InMemoryEventBus,
    LocalMultiprocessBroker,
    ServiceConfig,
    WorkerProcessManager,
    configure_logging,
    configure_tracing,
    create_api_app,
    normalize_tenant_id,
)

logger = logging.getLogger(__name__)
EXECUTION_ROLE = "disabled"


def resolve_runtime_class(dadbot_cls=None):
    resolved_dadbot_cls = dadbot_cls
    if resolved_dadbot_cls is None:
        main_module = sys.modules.get("__main__")
        resolved_dadbot_cls = getattr(main_module, "DadBot", None)
        if resolved_dadbot_cls is None:
            from Dad import DadBot as ImportedDadBot

            resolved_dadbot_cls = ImportedDadBot
    return resolved_dadbot_cls


def launch_streamlit_app(
    *,
    append_signoff=True,
    light_mode=False,
    script_path: str | Path | None = None,
    ensure_streamlit_app_file,
):
    def required_streamlit_port():
        configured_port = os.environ.get("DADBOT_STREAMLIT_PORT", "8501")
        try:
            required_port = int(configured_port)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"DADBOT_STREAMLIT_PORT must be a valid integer port. Got: {configured_port!r}",
            ) from exc

        if required_port != 8501:
            raise RuntimeError(
                f"Dad Bot is pinned to port 8501. Update or unset DADBOT_STREAMLIT_PORT (current value: {configured_port!r}).",
            )

        with socket(AF_INET, SOCK_STREAM) as candidate_socket:
            candidate_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
            try:
                candidate_socket.bind(("127.0.0.1", required_port))
            except OSError as exc:
                raise RuntimeError(
                    "Dad Bot requires localhost:8501, but that port is already in use. "
                    "Stop the existing process using 8501 and try again.",
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
    workspace_path = base_path.resolve().parent
    streamlit_app_path = workspace_path / "dadbot" / "ui" / "entrypoint.py"
    stub_created = ensure_streamlit_app_file(streamlit_app_path)
    if stub_created:
        print(
            f"The packaged Streamlit entrypoint was missing, so a minimal stub was created at {streamlit_app_path}.",
        )
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
    process = subprocess.Popen(command, cwd=str(workspace_path), env=command_env)
    try:
        if wait_for_streamlit(local_url, process):
            try:
                webbrowser.open(local_url)
            except Exception:
                logger.warning(
                    "Could not open browser automatically for %s",
                    local_url,
                    exc_info=True,
                )
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
    workspace = str(
        (Path(script_path) if script_path is not None else Path.cwd() / "Dad.py").resolve().parent,
    )

    def port_8501_in_use():
        with socket(AF_INET, SOCK_STREAM) as candidate_socket:
            candidate_socket.settimeout(0.5)
            return candidate_socket.connect_ex(("127.0.0.1", 8501)) == 0

    if os.name == "nt":
        query = (
            f"$workspace = {workspace!r}; "
            "Get-CimInstance Win32_Process | Where-Object { "
            "$cmd = $_.CommandLine; "
            '$_.Name -ieq "python.exe" -and $cmd -and $cmd -like "*$workspace*" -and ($cmd -like "*streamlit run*" -or $cmd -like "*dadbot*ui*entrypoint.py*") '
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
                print(
                    f"Could not parse stop-streamlit process list: {exc}",
                    file=sys.stderr,
                )
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
        for target in sorted(
            targets,
            key=lambda item: int(item.get("ProcessId", 0)),
            reverse=True,
        ):
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
                print(
                    f"STOP_FAILED PID={process_id} NAME={name} ERROR={error_text}",
                    file=sys.stderr,
                )

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
        print(
            "--stop-streamlit is currently supported on Windows or systems with lsof installed.",
            file=sys.stderr,
        )
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


def launch_api_service(args, *, dadbot_cls=None, build_service_state_store):
    enforce_execution_role(module="runtime.launcher", role=EXECUTION_ROLE)
    try:
        uvicorn = importlib.import_module("uvicorn")
    except ImportError as exc:
        print("FastAPI service mode requires uvicorn and fastapi to be installed.")
        raise SystemExit(1) from exc

    resolved_dadbot_cls = resolve_runtime_class(dadbot_cls)

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
        uvicorn.run(
            app,
            host=config.api.host,
            port=config.api.port,
            log_level=config.telemetry.log_level.lower(),
        )
    finally:
        worker_manager.shutdown()
    return 0


def dispatch_runtime_mode(
    args,
    *,
    dadbot_cls,
    service_client_cls=DadServiceClient,
    script_path: str | Path | None = None,
    ensure_streamlit_app_file,
    build_service_state_store,
):
    enforce_execution_role(module="runtime.launcher", role=EXECUTION_ROLE)
    base_script_path = Path(script_path) if script_path is not None else Path.cwd() / "Dad.py"

    if args.no_signoff:
        os.environ["DADBOT_NO_SIGNOFF"] = "1"
    if args.light:
        os.environ["DADBOT_LIGHT_MODE"] = "1"
    if args.tenant_id:
        os.environ["DADBOT_TENANT_ID"] = normalize_tenant_id(args.tenant_id)

    if args.init_profile:
        created = dadbot_cls.initialize_profile_file()
        if created:
            print("Starter dad_profile.json created.")
        else:
            print("dad_profile.json already exists.")
        return 0

    if args.stop_streamlit:
        return stop_streamlit_app(script_path=base_script_path)

    if args.serve_api:
        return launch_api_service(
            args,
            dadbot_cls=dadbot_cls,
            build_service_state_store=build_service_state_store,
        )

    bot = dadbot_cls(
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
        bot.print_system_message(
            f"Dad's saved memory was exported to {args.export_memory}.",
        )
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
        ensure_streamlit_app_file=ensure_streamlit_app_file,
    )
