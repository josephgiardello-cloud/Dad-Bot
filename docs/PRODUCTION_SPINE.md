# Dad-Bot Runtime Spine

This file records the smallest stable runtime contract we want people to trust.

## Canonical Path

The happy path should stay short and readable:

1. `launch.py` is the public entrypoint.
2. `dadbot.app_runtime.main` decides whether we are running UI, CLI, or API.
3. The turn path enters the orchestrator, then the control plane, then the kernel boundary.
4. Scheduler, persistence, and telemetry are implementation details, not public architecture.

## What Matters

- Keep the default conversation path simple.
- Treat advanced graph, confluence, and replay machinery as optional safeguards.
- Keep legacy wrapper scripts as compatibility shims only.
- Keep `dad_streamlit.py` as the legacy Streamlit surface until the package UI wrapper fully replaces it.

## Complete-Run Contract

The runtime still requires stable execution identity, terminal lifecycle state, terminal unified execution status, a valid terminal turn state, and a non-empty final response.

## Source of Truth

If another note conflicts with this file, this file wins.
