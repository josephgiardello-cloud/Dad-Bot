from pathlib import Path


class SystemValidator:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def check_missing_modules(self, manifest: dict[str, str]) -> dict:
        missing = {}
        for name, rel_path in manifest.items():
            full_path = self.base_path / rel_path
            if not full_path.exists():
                missing[name] = str(full_path)
        return missing

    def is_system_complete(self, manifest: dict[str, str]) -> bool:
        return len(self.check_missing_modules(manifest)) == 0
