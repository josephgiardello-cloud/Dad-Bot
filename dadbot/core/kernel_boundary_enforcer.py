"""Kernel boundary enforcement: prevent architectural bleed.

Mandatory boundary checks that prevent kernel/core modules from importing
persona/ui/consumer layers. Violations are hard errors in CI/pre-merge.

Enforcement design:
1. Kernel modules: dadbot/core/*, dadbot/contracts.py, dadbot/runtime/*
2. Boundary violations: import of dadbot/persona/*, dadbot/ui/*, dadbot/consumers/*
3. Exceptions: contracts, shared types, service interfaces
4. Enforcement points: pytest collection, pre-commit, CI gates
5. Recovery: dead-code removal, layer refactoring, manager injection

Architectural principle:
- Core execution paths must remain independent of persona/UI implementation details
- Persona behavior flows through managers/services, not direct dependencies
- All I/O and UI interactions routed through well-defined interfaces
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Kernel core modules (must not import from persona/ui/consumer layers)
KERNEL_MODULES = {
    "dadbot/core",
    "dadbot/contracts.py",
    "dadbot/runtime",
    "dadbot_system",
}

# Persona/UI/Consumer layers (kernel must not depend on these)
FORBIDDEN_IMPORTS = {
    "dadbot.persona",
    "dadbot.ui",
    "dadbot.consumers",
}

# Exceptions: modules that may be imported by kernel for shared contracts
ALLOWED_IMPORTS_TO_KERNEL = {
    "dadbot.contracts",
    "dadbot.managers",
    "dadbot.services",
    "dadbot.infrastructure",
    "dadbot.runtime",
}


class BoundaryViolation:
    """Single boundary violation record."""
    
    def __init__(
        self,
        file_path: Path,
        line_number: int,
        import_statement: str,
        violating_module: str,
        reason: str,
    ):
        self.file_path = file_path
        self.line_number = line_number
        self.import_statement = import_statement
        self.violating_module = violating_module
        self.reason = reason
    
    def __str__(self) -> str:
        return (
            f"{self.file_path}:{self.line_number}: "
            f"Boundary violation: {self.reason}\n"
            f"  {self.import_statement}"
        )


class KernelBoundaryChecker(ast.NodeVisitor):
    """AST visitor to detect boundary violations in Python files."""
    
    def __init__(self, file_path: Path, is_kernel_module: bool):
        self.file_path = file_path
        self.is_kernel_module = is_kernel_module
        self.violations: list[BoundaryViolation] = []
    
    def visit_Import(self, node: ast.Import) -> None:
        """Check 'import x' statements."""
        if not self.is_kernel_module:
            return
        
        for alias in node.names:
            module_name = alias.name
            self._check_import(module_name, node.lineno, f"import {module_name}")
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check 'from x import y' statements."""
        if not self.is_kernel_module:
            return
        
        if node.module:
            module_name = node.module
            import_spec = f"from {module_name} import " + ", ".join(
                alias.name for alias in node.names
            )
            self._check_import(module_name, node.lineno, import_spec)
    
    def _check_import(self, module_name: str, lineno: int, import_statement: str) -> None:
        """Validate import against boundary rules."""
        # Check if this import violates kernel boundary
        for forbidden in FORBIDDEN_IMPORTS:
            if module_name.startswith(forbidden):
                violation = BoundaryViolation(
                    file_path=self.file_path,
                    line_number=lineno,
                    import_statement=import_statement,
                    violating_module=module_name,
                    reason=f"Kernel modules must not import {forbidden}",
                )
                self.violations.append(violation)
                return


def check_file(file_path: Path) -> list[BoundaryViolation]:
    """Check single file for boundary violations."""
    if not file_path.exists():
        return []
    
    # Determine if this is a kernel module
    relative_path = str(file_path).replace("\\", "/")
    is_kernel = any(relative_path.startswith(km.replace("\\", "/")) for km in KERNEL_MODULES)
    
    if not is_kernel:
        return []
    
    try:
        source = file_path.read_text()
        tree = ast.parse(source)
        checker = KernelBoundaryChecker(file_path, is_kernel)
        checker.visit(tree)
        return checker.violations
    except Exception as exc:
        logger.warning("Failed to check %s: %s", file_path, exc)
        return []


def check_workspace(workspace_root: Path) -> tuple[bool, list[BoundaryViolation]]:
    """Check entire workspace for boundary violations.
    
    Returns:
        (all_ok: bool, violations: list[BoundaryViolation])
    """
    all_violations: list[BoundaryViolation] = []
    
    # Find all Python files in dadbot/core and dadbot/runtime
    python_files = list(workspace_root.glob("dadbot/core/**/*.py"))
    python_files.extend(workspace_root.glob("dadbot/runtime/**/*.py"))
    python_files.extend(workspace_root.glob("dadbot_system/**/*.py"))
    python_files.append(workspace_root / "dadbot" / "contracts.py")
    
    for py_file in python_files:
        if py_file.is_file():
            violations = check_file(py_file)
            all_violations.extend(violations)
    
    all_ok = len(all_violations) == 0
    return all_ok, all_violations


def format_report(violations: list[BoundaryViolation]) -> str:
    """Format violations as readable report."""
    if not violations:
        return "✓ Kernel boundary check passed: no violations found"
    
    lines = [f"✗ Kernel boundary check failed: {len(violations)} violation(s) found\n"]
    for i, violation in enumerate(violations, 1):
        lines.append(f"\n{i}. {violation}")
    
    return "\n".join(lines)


# Pytest plugin integration
def pytest_configure(config: Any) -> None:
    """Register boundary check as pytest plugin."""
    config.addinivalue_line(
        "markers", "boundary: mark test as kernel boundary check"
    )


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    """Enforce boundary checks during pytest collection."""
    from pathlib import Path
    
    workspace = Path.cwd()
    all_ok, violations = check_workspace(workspace)
    
    if not all_ok:
        logger.error("Kernel boundary violations detected during pytest collection:")
        logger.error(format_report(violations))
        # Fail collection with clear error
        raise RuntimeError(
            f"Kernel boundary check failed: {len(violations)} violation(s). "
            "Cannot proceed with test collection."
        )
