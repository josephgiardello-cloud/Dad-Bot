from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def test_registry_import_time_budget() -> None:
    proc = _run_python(
        "import importlib, json, time;"
        "t0=time.perf_counter();"
        "importlib.import_module('dadbot.registry');"
        "print(json.dumps({'elapsed_s': time.perf_counter()-t0}))"
    )
    payload = json.loads(proc.stdout.strip())
    assert float(payload["elapsed_s"]) < 1.0


def test_observability_import_does_not_pull_heavy_runtime_modules() -> None:
    proc = _run_python(
        "import importlib, json, sys;"
        "before=set(sys.modules);"
        "importlib.import_module('dadbot.core.observability');"
        "after=set(sys.modules)-before;"
        "heavy=sorted(m for m in after if m in {'dadbot.registry','dadbot.core.orchestrator','dadbot.managers.runtime_client','litellm'});"
        "print(json.dumps({'heavy_modules': heavy}))"
    )
    payload = json.loads(proc.stdout.strip())
    assert payload["heavy_modules"] == []


def test_dadbot_import_time_budget() -> None:
    proc = _run_python(
        "import importlib, json, time;"
        "t0=time.perf_counter();"
        "importlib.import_module('dadbot.core.dadbot');"
        "print(json.dumps({'elapsed_s': time.perf_counter()-t0}))"
    )
    payload = json.loads(proc.stdout.strip())
    assert float(payload["elapsed_s"]) < 0.75


def test_dadbot_import_does_not_pull_forbidden_stack() -> None:
    proc = _run_python(
        "import importlib, json, sys;"
        "before=set(sys.modules);"
        "importlib.import_module('dadbot.core.dadbot');"
        "after=set(sys.modules)-before;"
        "forbidden=('litellm','dadbot.core.orchestrator','dadbot.managers.runtime_client');"
        "loaded=sorted(m for m in after if any(token in m for token in forbidden));"
        "print(json.dumps({'loaded': loaded}))"
    )
    payload = json.loads(proc.stdout.strip())
    assert payload["loaded"] == []