"""Priority 4: Kernel Boundary Hardening tests."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from dadbot.core.kernel_boundary_enforcer import (
    KernelBoundaryChecker,
    BoundaryViolation,
    check_file,
    check_workspace,
    format_report,
)


@pytest.mark.unit
class TestBoundaryViolation:
    """Test BoundaryViolation record."""

    def test_violation_creation(self):
        """Verify violation record can be created."""
        v = BoundaryViolation(
            file_path=Path("test.py"),
            line_number=10,
            import_statement="from dadbot.ui import widget",
            violating_module="dadbot.ui",
            reason="Kernel modules must not import UI",
        )
        
        assert v.file_path == Path("test.py")
        assert v.line_number == 10
        assert "UI" in v.reason

    def test_violation_str(self):
        """Verify violation formats correctly."""
        v = BoundaryViolation(
            file_path=Path("kernel.py"),
            line_number=5,
            import_statement="import dadbot.ui",
            violating_module="dadbot.ui",
            reason="Test violation",
        )
        
        output = str(v)
        assert "kernel.py:5" in output
        assert "import dadbot.ui" in output


@pytest.mark.unit
class TestKernelBoundaryChecker:
    """Test AST-based boundary checker."""

    def test_checker_initialization(self):
        """Verify checker can be initialized."""
        checker = KernelBoundaryChecker(
            file_path=Path("test.py"),
            is_kernel_module=True,
        )
        
        assert checker.file_path == Path("test.py")
        assert checker.is_kernel_module is True
        assert len(checker.violations) == 0

    def test_detects_ui_import_violation(self):
        """Verify violation detection for UI imports."""
        import ast
        
        code = "from dadbot.ui import streamlit_app"
        tree = ast.parse(code)
        checker = KernelBoundaryChecker(Path("kernel.py"), is_kernel_module=True)
        checker.visit(tree)
        
        assert len(checker.violations) == 1
        assert "ui" in checker.violations[0].violating_module.lower()

    def test_detects_persona_import_violation(self):
        """Verify violation detection for persona imports."""
        import ast
        
        code = "from dadbot.persona import PersonaLayer"
        tree = ast.parse(code)
        checker = KernelBoundaryChecker(Path("kernel.py"), is_kernel_module=True)
        checker.visit(tree)
        
        assert len(checker.violations) == 1
        assert "persona" in checker.violations[0].violating_module.lower()

    def test_detects_consumer_import_violation(self):
        """Verify violation detection for consumer imports."""
        import ast
        
        code = "import dadbot.consumers.slack"
        tree = ast.parse(code)
        checker = KernelBoundaryChecker(Path("kernel.py"), is_kernel_module=True)
        checker.visit(tree)
        
        assert len(checker.violations) == 1
        assert "consumers" in checker.violations[0].violating_module.lower()

    def test_allows_contract_imports(self):
        """Verify that contract imports are allowed."""
        import ast
        
        code = "from dadbot.contracts import FinalizedTurnResult"
        tree = ast.parse(code)
        checker = KernelBoundaryChecker(Path("kernel.py"), is_kernel_module=True)
        checker.visit(tree)
        
        assert len(checker.violations) == 0

    def test_non_kernel_modules_not_checked(self):
        """Verify non-kernel modules are not checked."""
        import ast
        
        code = "from dadbot.ui import widget"
        tree = ast.parse(code)
        checker = KernelBoundaryChecker(Path("ui.py"), is_kernel_module=False)
        checker.visit(tree)
        
        assert len(checker.violations) == 0


@pytest.mark.unit
class TestFileChecking:
    """Test file-level boundary checking."""

    def test_check_file_valid(self):
        """Verify valid file passes check."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("from dadbot.contracts import FinalizedTurnResult\n")
            f.write("import logging\n")
            temp_path = Path(f.name)
        
        try:
            # Rename to simulate kernel module path
            kernel_path = Path(temp_path.parent) / "kernel_test.py"
            temp_path.rename(kernel_path)
            
            violations = check_file(kernel_path)
            assert len(violations) == 0
        finally:
            if kernel_path.exists():
                kernel_path.unlink()

    def test_check_file_nonexistent(self):
        """Verify nonexistent file returns empty list."""
        violations = check_file(Path("/nonexistent/file.py"))
        assert len(violations) == 0


@pytest.mark.unit
class TestFormatReport:
    """Test report formatting."""

    def test_format_report_no_violations(self):
        """Verify report for clean check."""
        report = format_report([])
        
        assert "✓" in report
        assert "no violations" in report.lower()

    def test_format_report_with_violations(self):
        """Verify report with violations."""
        violations = [
            BoundaryViolation(
                file_path=Path("core.py"),
                line_number=10,
                import_statement="import dadbot.ui",
                violating_module="dadbot.ui",
                reason="Test",
            ),
            BoundaryViolation(
                file_path=Path("orchestrator.py"),
                line_number=20,
                import_statement="from dadbot.persona import X",
                violating_module="dadbot.persona",
                reason="Test",
            ),
        ]
        
        report = format_report(violations)
        
        assert "✗" in report
        assert "2 violation" in report.lower()
        assert "core.py:10" in report
        assert "orchestrator.py:20" in report


@pytest.mark.integration
@pytest.mark.parametrize("forbidden_module", [
    "dadbot.ui",
    "dadbot.persona",
    "dadbot.consumers",
])
def test_boundary_enforcement_rejects_imports(forbidden_module):
    """Test that boundary enforcer rejects forbidden imports from kernel."""
    import ast
    
    code = f"from {forbidden_module} import something"
    tree = ast.parse(code)
    checker = KernelBoundaryChecker(Path("core/test.py"), is_kernel_module=True)
    checker.visit(tree)
    
    assert len(checker.violations) > 0, f"Should reject {forbidden_module}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
