# Dad-Bot Production Spine

This document is the authoritative runtime contract for shipped behavior.

## 1) Canonical Runtime Entry Path

A production turn must follow this path:

1. Orchestrator entry: dadbot.core.orchestrator.Orchestrator.handle_turn
2. Control-plane ingress: dadbot.core.control_plane.ExecutionControlPlane.submit_turn
3. Kernel boundary: dadbot.core.kernel_gateway.KernelGateway.submit_turn
4. Control-plane kernel path: dadbot.core.control_plane.ExecutionControlPlane._submit_turn_kernel
5. Scheduler execution: dadbot.core.control_plane.Scheduler.drain_once
6. Finalization: dadbot.core.control_plane.ExecutionControlPlane._finalize_submit_success

Public process entry wrappers route into dadbot.app_runtime.main:

- launch.py
- Dad.py (compatibility wrapper)
- api_entrypoint.py

## 2) Required Runtime Modules

These are production-critical runtime layers:

- dadbot.core
- dadbot.memory
- dadbot.runtime_adapter
- dadbot.registry
- dadbot.app_runtime

These are tooling-only (must not become runtime dependencies):

- tools
- tests
- archive
- ci
- dadbot.utils.architecture
- dadbot.utils.benchmark
- dadbot.utils.models
- dadbot.utils.oracle
- dadbot.utils.report
- dadbot.utils.signal_bus
- dadbot.utils.truth_binding

## 3) Complete-Run Contract

A run is complete only when all are true:

1. Stable execution identity exists (trace_id and job_id).
2. Lifecycle projection state is terminal (completed or failed).
3. Unified execution_result status is terminal.
4. terminal_turn_state is present and valid.
5. Final response payload is non-empty.

The runtime now enforces this in ExecutionControlPlane._assert_complete_run_contract.

## 4) Experimental Surface

Anything not required above is non-authoritative unless explicitly promoted.

Promotion requires:

1. Runtime integration through canonical path.
2. Invariant coverage in tests.
3. Documentation update in this file.

## 5) Source of Truth Rule

If a statement in older notes conflicts with this file, this file wins.
