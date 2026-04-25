from __future__ import annotations


class LedgerEnforcementError(Exception):
    pass


class LedgerEnforcer:
    REQUIRED_FIELDS = {
        "type",
        "session_id",
        "trace_id",
        "timestamp",
        "kernel_step_id",
    }

    def validate(self, event: dict):
        missing = self.REQUIRED_FIELDS - set(event.keys())
        if missing:
            raise LedgerEnforcementError(f"Missing fields: {missing}")

        if not event.get("kernel_step_id"):
            raise LedgerEnforcementError("Kernel lineage missing")

        return True
