"""
Improvement generator for Annabeth's self-improvement system.
Takes ImprovementOpportunity objects from the analyzer and
produces GeneratedImprovement objects with modified code.

Risk levels:
  LOW    — bare except fixes (auto-applied)
  MEDIUM — type hints, refactoring (notify only, never auto)
  HIGH   — performance, readability (never touched)
  BLOCKED — code that fails AST validation after modification
"""

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .analyzer import ImprovementOpportunity, ImprovementType


@dataclass
class GeneratedImprovement:
    opportunity: ImprovementOpportunity
    original_code: str
    modified_code: str
    validation_result: Dict[str, Any]
    risk_level: str   # LOW | MEDIUM | HIGH | BLOCKED


class ImprovementGenerator:
    """
    Converts ImprovementOpportunity → GeneratedImprovement.
    Only `ERROR_HANDLING` (bare except) produces LOW-risk auto-applicable fixes.
    All other types return MEDIUM risk and are never auto-applied.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_improvement(
        self, opportunity: ImprovementOpportunity
    ) -> Optional[GeneratedImprovement]:
        if opportunity.improvement_type == ImprovementType.ERROR_HANDLING:
            return self._fix_bare_except(opportunity)
        # All other types: produce a MEDIUM-risk notification-only improvement
        return self._notify_only(opportunity)

    # ── Fixers ────────────────────────────────────────────────────────────────

    def _fix_bare_except(
        self, opportunity: ImprovementOpportunity
    ) -> Optional[GeneratedImprovement]:
        """Replace `except:` with `except Exception:` on the identified line."""
        try:
            content  = opportunity.file_path.read_text(encoding="utf-8")
            lines    = content.split("\n")
            idx      = opportunity.line_number - 1
            if idx < 0 or idx >= len(lines):
                return None
            original_line = lines[idx]
            if "except:" not in original_line:
                return None   # Already fixed or false positive
            lines[idx]    = original_line.replace("except:", "except Exception:", 1)
            modified      = "\n".join(lines)
            validation    = self._validate(modified, opportunity.file_path)
            return GeneratedImprovement(
                opportunity=opportunity,
                original_code=content,
                modified_code=modified,
                validation_result=validation,
                risk_level="LOW" if validation["valid"] else "BLOCKED",
            )
        except Exception as e:
            self.logger.error(f"Error generating bare-except fix: {e}")
            return None

    def _notify_only(
        self, opportunity: ImprovementOpportunity
    ) -> Optional[GeneratedImprovement]:
        """Return a MEDIUM-risk improvement that is never auto-applied."""
        try:
            content = opportunity.file_path.read_text(encoding="utf-8")
        except Exception:
            return None
        return GeneratedImprovement(
            opportunity=opportunity,
            original_code=content,
            modified_code=content,   # No change
            validation_result={"valid": True, "errors": []},
            risk_level="MEDIUM",
        )

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(code: str, file_path: Path) -> Dict[str, Any]:
        try:
            ast.parse(code)
            return {"valid": True, "errors": []}
        except SyntaxError as e:
            return {"valid": False, "errors": [str(e)]}
