"""
AST-based code analyzer for Annabeth's self-improvement system.
Scans server/ Python files for bare except clauses and other
improvement opportunities. Only bare-except fixes are auto-applied
(LOW risk). All others are logged/notified only.
"""

import ast
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional


class ImprovementType(Enum):
    PERFORMANCE    = "performance"
    READABILITY    = "readability"
    ERROR_HANDLING = "error_handling"
    TYPE_SAFETY    = "type_safety"
    REFACTORING    = "refactoring"


@dataclass
class ImprovementOpportunity:
    file_path: Path
    line_number: int
    improvement_type: ImprovementType
    description: str
    current_code: str
    suggested_code: Optional[str] = None
    confidence: float = 0.5


class CodeAnalyzer:
    """
    Analyzes Python source files for improvement opportunities using AST.
    Only scans files inside `src_path`; skips __pycache__.
    """

    def __init__(self, src_path: Path = Path("server")):
        self.logger    = logging.getLogger(__name__)
        self.src_path  = src_path
        self.opportunities: List[ImprovementOpportunity] = []

    def analyze_all(self) -> List[ImprovementOpportunity]:
        """Scan all .py files under src_path and return opportunities."""
        self.opportunities = []
        for py_file in self.src_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            self._analyze_file(py_file)
        return self.opportunities

    # ── Private helpers ────────────────────────────────────────────────────

    def _analyze_file(self, file_path: Path) -> None:
        try:
            content = file_path.read_text(encoding="utf-8")
            tree    = ast.parse(content)
            self._check_bare_excepts(tree, file_path)
            self._check_missing_type_hints(tree, file_path)
            self._check_complex_functions(tree, file_path)
            self._check_unused_imports(tree, file_path)
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            self.logger.warning(f"Error analyzing {file_path}: {e}")

    def _check_bare_excepts(self, tree: ast.AST, file_path: Path) -> None:
        """bare except: → except Exception:  (HIGH confidence, LOW risk)"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                self.opportunities.append(ImprovementOpportunity(
                    file_path=file_path,
                    line_number=node.lineno,
                    improvement_type=ImprovementType.ERROR_HANDLING,
                    description="Bare except clause — catches SystemExit/KeyboardInterrupt",
                    current_code="except:",
                    suggested_code="except Exception:",
                    confidence=0.9,
                ))

    def _check_missing_type_hints(self, tree: ast.AST, file_path: Path) -> None:
        """Functions lacking type annotations (MEDIUM risk — log only)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.returns is None and node.name != "__init__":
                    args_missing = any(
                        arg.annotation is None
                        for arg in node.args.args
                        if arg.arg != "self"
                    )
                    if args_missing:
                        self.opportunities.append(ImprovementOpportunity(
                            file_path=file_path,
                            line_number=node.lineno,
                            improvement_type=ImprovementType.TYPE_SAFETY,
                            description=f"Function '{node.name}' missing type hints",
                            current_code=f"def {node.name}(...)",
                            confidence=0.7,
                        ))

    def _check_complex_functions(self, tree: ast.AST, file_path: Path) -> None:
        """Functions with >10 control-flow structures (MEDIUM risk — log only)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                count = sum(
                    1 for n in ast.walk(node)
                    if isinstance(n, (ast.If, ast.For, ast.While, ast.With, ast.Try))
                )
                if count > 10:
                    self.opportunities.append(ImprovementOpportunity(
                        file_path=file_path,
                        line_number=node.lineno,
                        improvement_type=ImprovementType.REFACTORING,
                        description=(
                            f"Function '{node.name}' may be too complex "
                            f"({count} control structures)"
                        ),
                        current_code=f"def {node.name}(...)",
                        confidence=0.6,
                    ))

    def _check_unused_imports(self, tree: ast.AST, file_path: Path) -> None:
        """Potentially unused top-level imports (LOW confidence — log only)."""
        imports:    dict = {}
        used_names: set  = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name] = node
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports[alias.asname or alias.name] = node
            elif isinstance(node, ast.Name):
                used_names.add(node.id)

        for name, node in imports.items():
            if name not in used_names and not name.startswith("_"):
                self.opportunities.append(ImprovementOpportunity(
                    file_path=file_path,
                    line_number=node.lineno,
                    improvement_type=ImprovementType.READABILITY,
                    description=f"Potentially unused import: {name}",
                    current_code=f"import {name}",
                    confidence=0.5,
                ))
