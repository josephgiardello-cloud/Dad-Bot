from system_manifest import REQUIRED_CORE_MODULES
from system_repair import SystemRepair
from system_validator import SystemValidator


def ensure_system(base_path: str):
    validator = SystemValidator(base_path)
    repair = SystemRepair(base_path)

    missing = validator.check_missing_modules(REQUIRED_CORE_MODULES)

    if missing:
        print(f"[SYSTEM] Missing modules detected: {list(missing.keys())}")
        repair.create_missing_modules(missing)
        print("[SYSTEM] Auto-repair complete.")

    return validator.is_system_complete(REQUIRED_CORE_MODULES)
