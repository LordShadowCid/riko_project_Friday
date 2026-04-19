"""
AST-based code analyzer for the self-improvement system.
Identifies improvement opportunities without modifying any code.

Runs only when self_improvement.enabled: true in character_config.yaml.
All proposals are written to disk for human review — nothing is auto-applied
unless self_improvement.auto_apply_risk is explicitly raised.
"""

import ast
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Categories of improvement
CAT_LONG_FUNCTION = "long_function"
CAT_BARE_EXCEPT = "bare_except"
CAT_MAGIC_NUMBER = "magic_number"
CAT_TODO_COMMENT = "todo_comment"
CAT_MISSING_TYPE_HINT = "missing_type_hint"
CAT_EMPTY_EXCEPT = "empty_except"

# Thresholds
_MAX_FUNCTION_LINES = 60
_MAX_MAGIC_NUMBER_ABS = 1000  # numbers larger than this are never flagged


@dataclass
class ImprovementOpportunity:
    """A single potential improvement found in the codebase."""
    file: str            # workspace-relative path
    line: int            # 1-based line number
    category: str        # one of the CAT_* constants
    description: str     # human-readable problem description
    suggestion: str      # concrete suggestion for how to fix it
    severity: str = "low"    # low / medium / high


class CodeAnalyzer(ast.NodeVisitor):
    """
    Walk a Python AST and collect improvement opportunities.

    Usage:
        analyzer = CodeAnalyzer("server/main_chat.py")
        opportunities = analyzer.analyze()
    """

    def __init__(self, filepath: str, workspace_root: Optional[str] = None):
        self.filepath = filepath
        self.workspace_root = workspace_root or ""
        self._rel_path = (
            os.path.relpath(filepath, workspace_root)
            if workspace_root
            else filepath
        )
        self._source_lines: List[str] = []
        self._opportunities: List[ImprovementOpportunity] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> List[ImprovementOpportunity]:
        """Parse and walk the file, return all found opportunities."""
        self._opportunities.clear()
        self._source_lines.clear()

        try:
            source = Path(self.filepath).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("[CodeAnalyzer] Cannot read %s: %s", self.filepath, exc)
            return []

        self._source_lines = source.splitlines()
        self._scan_comments()

        try:
            tree = ast.parse(source, filename=self.filepath)
        except SyntaxError as exc:
            logger.debug("[CodeAnalyzer] Syntax error in %s: %s", self.filepath, exc)
            return []

        self.visit(tree)
        return list(self._opportunities)

    # ------------------------------------------------------------------
    # Comment scanning (no AST needed)
    # ------------------------------------------------------------------

    def _scan_comments(self):
        for lineno, line in enumerate(self._source_lines, start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                comment = stripped[1:].strip().upper()
                if comment.startswith("TODO") or comment.startswith("FIXME") or comment.startswith("HACK"):
                    self._add(
                        lineno,
                        CAT_TODO_COMMENT,
                        f"Unresolved marker at line {lineno}: {line.strip()!r}",
                        "Resolve or track this item in the project's issue tracker.",
                        severity="low",
                    )

    # ------------------------------------------------------------------
    # AST visitors
    # ------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._check_function(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # same checks

    def _check_function(self, node):
        # Long function
        end_line = getattr(node, "end_lineno", None)
        if end_line is not None:
            length = end_line - node.lineno
            if length > _MAX_FUNCTION_LINES:
                self._add(
                    node.lineno,
                    CAT_LONG_FUNCTION,
                    f"Function '{node.name}' is {length} lines long (threshold: {_MAX_FUNCTION_LINES}).",
                    f"Consider splitting '{node.name}' into smaller, focused helpers.",
                    severity="medium",
                )

        # Missing return type hint (skip __dunder__ methods and __init__)
        if (
            node.returns is None
            and not node.name.startswith("__")
            and not node.name.startswith("_")
        ):
            self._add(
                node.lineno,
                CAT_MISSING_TYPE_HINT,
                f"Public function '{node.name}' has no return type annotation.",
                f"Add a return type annotation, e.g. `def {node.name}(...) -> None:`.",
                severity="low",
            )

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        # Bare except (catches everything including KeyboardInterrupt)
        if node.type is None:
            self._add(
                node.lineno,
                CAT_BARE_EXCEPT,
                "Bare `except:` clause catches all exceptions including SystemExit.",
                "Replace with `except Exception:` or a specific exception type.",
                severity="medium",
            )

        # Empty except block (pass / ... only)
        body_types = {type(stmt) for stmt in node.body}
        if body_types <= {ast.Pass, ast.Expr}:
            # ast.Expr alone could be an ellipsis literal
            self._add(
                node.lineno,
                CAT_EMPTY_EXCEPT,
                "Exception handler body is effectively empty (pass / ...).",
                "At minimum log the exception: `logger.warning(..., exc_info=True)`.",
                severity="medium",
            )

        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        # Flag magic numbers (not in assignments named *_THRESHOLD, *_MAX, etc.)
        if isinstance(node.n if hasattr(node, "n") else None, (int, float)):
            val = node.n  # type: ignore[attr-defined]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if abs(val) > 1 and abs(val) <= _MAX_MAGIC_NUMBER_ABS and val not in (2, 10, 100):
                    # Only flag if the parent is a BinOp or Compare (not a default kwarg)
                    # We can't easily get the parent in NodeVisitor, so skip for now
                    pass  # Magic number detection is intentionally conservative
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _add(
        self,
        line: int,
        category: str,
        description: str,
        suggestion: str,
        severity: str = "low",
    ):
        self._opportunities.append(
            ImprovementOpportunity(
                file=self._rel_path,
                line=line,
                category=category,
                description=description,
                suggestion=suggestion,
                severity=severity,
            )
        )


# ---------------------------------------------------------------------------
# Convenience: scan multiple files
# ---------------------------------------------------------------------------

def scan_directory(
    root: str,
    include_patterns: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    max_files: int = 50,
) -> List[ImprovementOpportunity]:
    """
    Walk *root* and analyze Python files, returning all opportunities.

    Args:
        root: Workspace root to scan.
        include_patterns: Glob-like path substrings to include (default: server/).
        exclude_dirs: Directory names to skip (default: common junk dirs).
        max_files: Safety cap to avoid scanning huge trees.
    """
    include_patterns = include_patterns or ["server/"]
    exclude_dirs = exclude_dirs or [
        ".venv", "__pycache__", ".git", "node_modules", "build", "dist",
    ]

    results: List[ImprovementOpportunity] = []
    count = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
        if not any(rel_dir.startswith(p.rstrip("/")) for p in include_patterns):
            continue

        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            if count >= max_files:
                logger.debug("[CodeAnalyzer] Reached max_files=%d limit", max_files)
                return results

            filepath = os.path.join(dirpath, filename)
            analyzer = CodeAnalyzer(filepath, workspace_root=root)
            results.extend(analyzer.analyze())
            count += 1

    return results
