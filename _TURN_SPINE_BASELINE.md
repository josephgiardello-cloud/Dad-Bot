# Turn Spine Baseline

## Phase 0 Snapshot

- Timestamp basis: current workspace session baseline before turn-spine rerouting.
- Git HEAD: `a6da6eb5decae94705b24c1d53e2871301baf458`
- Worktree state: dirty
- Baseline cert command: `c:/Users/josep/OneDrive/Desktop/Dad-Bot/.venv/Scripts/python.exe -m pytest -q`
- Baseline cert result: failed / timed out in `tests/system_validation/test_drift_monitor.py::test_boundary_gate_is_reproducible`

## Notes

- This is not a clean committed rollback point; it is a dirty worktree anchor on top of the recorded HEAD commit.
- Earlier session notes referred to a green cert lane, but the fresh Phase 0 snapshot above is the authoritative pre-refactor baseline for this workspace state.