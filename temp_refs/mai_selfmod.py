"""
SOURCE: https://github.com/MystiaTech/Mai/tree/main/src/selfmod/
REPO: Mai (MystiaTech)
FILES: analyzer.py, scheduler.py, generator.py (combined for Annabeth reference)
PURPOSE: Pattern reference for an AST-based self-improvement system.
         Annabeth adaptation: server/process/self_improvement/
"""

# ============================================================
# FILE: analyzer.py
# ============================================================
import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ImprovementType(Enum):
    """Types of code improvements."""
    PERFORMANCE = "performance"
    READABILITY = "readability"
    ERROR_HANDLING = "error_handling"
    TYPE_SAFETY = "type_safety"
    REFACTORING = "refactoring"


@dataclass
class ImprovementOpportunity:
    """Identified improvement opportunity."""
    file_path: Path
    line_number: int
    improvement_type: ImprovementType
    description: str
    current_code: str
    suggested_code: Optional[str] = None
    confidence: float = 0.5


class CodeAnalyzer:
    """
    Analyzes codebase for improvement opportunities using AST.
    
    Annabeth hook: analyze server/ directory Python files.
    Run periodically (e.g., weekly) or after a session ends.
    """
    
    def __init__(self, src_path: Path = Path("src")):
        self.logger = logging.getLogger(__name__)
        self.src_path = src_path
        self.opportunities: List[ImprovementOpportunity] = []

    def analyze_all(self) -> List[ImprovementOpportunity]:
        """Analyze all Python files in src directory."""
        self.opportunities = []
        for py_file in self.src_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            self._analyze_file(py_file)
        return self.opportunities

    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            self._check_bare_excepts(tree, file_path, content)
            self._check_missing_type_hints(tree, file_path, content)
            self._check_complex_functions(tree, file_path, content)
            self._check_unused_imports(tree, file_path, content)
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in {file_path}: {e}")

    def _check_bare_excepts(self, tree: ast.AST, file_path: Path, content: str) -> None:
        """Check for bare except: clauses (catches everything including SystemExit)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    self.opportunities.append(ImprovementOpportunity(
                        file_path=file_path,
                        line_number=node.lineno,
                        improvement_type=ImprovementType.ERROR_HANDLING,
                        description="Bare except clause — should catch specific exceptions",
                        current_code="except:",
                        suggested_code="except Exception:",
                        confidence=0.9
                    ))

    def _check_missing_type_hints(self, tree: ast.AST, file_path: Path, content: str) -> None:
        """Check for functions missing type hints."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.returns is None and node.name != "__init__":
                    args_missing_hints = any(
                        arg.annotation is None
                        for arg in node.args.args
                        if arg.arg != "self"
                    )
                    if args_missing_hints:
                        self.opportunities.append(ImprovementOpportunity(
                            file_path=file_path,
                            line_number=node.lineno,
                            improvement_type=ImprovementType.TYPE_SAFETY,
                            description=f"Function '{node.name}' missing type hints",
                            current_code=f"def {node.name}(...)",
                            confidence=0.7
                        ))

    def _check_complex_functions(self, tree: ast.AST, file_path: Path, content: str) -> None:
        """Check for overly complex functions (too many control structures)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                statements = len([n for n in ast.walk(node) if isinstance(n, (
                    ast.If, ast.For, ast.While, ast.With, ast.Try
                ))])
                if statements > 10:
                    self.opportunities.append(ImprovementOpportunity(
                        file_path=file_path,
                        line_number=node.lineno,
                        improvement_type=ImprovementType.REFACTORING,
                        description=f"Function '{node.name}' may be too complex ({statements} control structures)",
                        current_code=f"def {node.name}(...)",
                        confidence=0.6
                    ))

    def _check_unused_imports(self, tree: ast.AST, file_path: Path, content: str) -> None:
        """Check for potentially unused imports."""
        imports = {}
        used_names = set()
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
                    confidence=0.5
                ))


# ============================================================
# FILE: generator.py  
# ============================================================

@dataclass
class GeneratedImprovement:
    """A generated code improvement."""
    opportunity: ImprovementOpportunity
    original_code: str
    modified_code: str
    validation_result: Dict[str, Any]
    risk_level: str  # LOW, MEDIUM, HIGH, BLOCKED


class ImprovementGenerator:
    """
    Generates code improvements from analysis opportunities.
    Creates Python code changes and validates them with AST checking.
    
    Annabeth: Only auto-apply LOW risk improvements (bare except fixes).
              MEDIUM risk (type hints) require confirmation.
              HIGH/BLOCKED are never auto-applied.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_improvement(self, opportunity: ImprovementOpportunity) -> Optional[GeneratedImprovement]:
        """Generate improvement for an opportunity."""
        if opportunity.improvement_type == ImprovementType.ERROR_HANDLING:
            return self._fix_bare_except(opportunity)
        elif opportunity.improvement_type == ImprovementType.TYPE_SAFETY:
            return self._add_type_hints(opportunity)
        elif opportunity.improvement_type == ImprovementType.PERFORMANCE:
            return self._optimize_performance(opportunity)
        elif opportunity.improvement_type == ImprovementType.READABILITY:
            return self._improve_readability(opportunity)
        return None

    def _fix_bare_except(self, opportunity: ImprovementOpportunity) -> Optional[GeneratedImprovement]:
        """Fix bare except: → except Exception:  (LOW risk)"""
        try:
            content = opportunity.file_path.read_text(encoding='utf-8')
            lines = content.split("\n")
            line_idx = opportunity.line_number - 1
            original_line = lines[line_idx]
            modified_line = original_line.replace("except:", "except Exception:")
            lines[line_idx] = modified_line
            modified_code = "\n".join(lines)
            validation = self._validate_code(modified_code, opportunity.file_path)
            return GeneratedImprovement(
                opportunity=opportunity,
                original_code=content,
                modified_code=modified_code,
                validation_result=validation,
                risk_level="LOW" if validation["valid"] else "BLOCKED"
            )
        except Exception as e:
            self.logger.error(f"Error generating fix: {e}")
            return None

    def _add_type_hints(self, opportunity: ImprovementOpportunity) -> Optional[GeneratedImprovement]:
        """Add basic type hints — MEDIUM risk (AI-generated, needs review)."""
        # Simplified implementation  
        return None  # Requires LLM assistance for Annabeth

    def _optimize_performance(self, opportunity: ImprovementOpportunity) -> Optional[GeneratedImprovement]:
        """Optimize performance patterns."""
        return None  # Context-dependent, skip for Annabeth

    def _improve_readability(self, opportunity: ImprovementOpportunity) -> Optional[GeneratedImprovement]:
        """Improve readability (f-strings, etc.)."""
        return None  # Too context-dependent

    def _validate_code(self, code: str, file_path: Path) -> Dict[str, Any]:
        """Validate generated code parses correctly with AST."""
        try:
            ast.parse(code)
            return {"valid": True, "errors": []}
        except SyntaxError as e:
            return {"valid": False, "errors": [str(e)]}


# ============================================================
# FILE: scheduler.py
# ============================================================

import asyncio
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class SchedulerConfig:
    """Configuration for improvement scheduler."""
    analysis_interval_hours: float = 24.0
    auto_apply_low_risk: bool = True
    require_approval_for_medium: bool = True
    max_daily_improvements: int = 5
    # Annabeth: analyze only server/ directory
    src_path: Path = Path("server")


class ImprovementScheduler:
    """
    Schedules periodic code analysis and improvement application.
    
    Annabeth integration:
    - Start in main_chat.py alongside the aiohttp server
    - on_improvement_ready → send notification to developer via WebSocket broadcast
    - Auto-apply only LOW risk (bare except fixes)
    - Log all applied improvements to grillo_activity_log table
    
    SAFETY: Never applies improvements during active conversation.
    Uses a conversation_active flag to block improvements when user is talking.
    """
    
    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        on_improvement_ready: Optional[Callable[[GeneratedImprovement], None]] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.config = config or SchedulerConfig()
        self.on_improvement_ready = on_improvement_ready
        
        self.analyzer = CodeAnalyzer(src_path=self.config.src_path)
        self.generator = ImprovementGenerator()
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._improvements_today = 0
        self._last_reset = datetime.now(timezone.utc)
        
        # Annabeth: gate so improvements don't fire during active conversation
        self.conversation_active = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop — runs analysis every analysis_interval_hours."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                if (now - self._last_reset) > timedelta(days=1):
                    self._improvements_today = 0
                    self._last_reset = now
                
                if (self._improvements_today < self.config.max_daily_improvements
                        and not self.conversation_active):
                    await self._run_analysis()
                
                await asyncio.sleep(self.config.analysis_interval_hours * 3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(3600)  # Wait an hour on error

    async def _run_analysis(self) -> None:
        """Run code analysis and process improvements."""
        self.logger.info("Running code analysis...")
        opportunities = self.analyzer.analyze_all()
        
        for opp in opportunities:
            if opp.confidence < 0.7:
                continue
            improvement = self.generator.generate_improvement(opp)
            if not improvement:
                continue
            
            if improvement.risk_level == "LOW" and self.config.auto_apply_low_risk:
                await self._apply_improvement(improvement)
            elif self.on_improvement_ready:
                # Notify developer (Annabeth: WebSocket broadcast or log)
                self.on_improvement_ready(improvement)
            
            self._improvements_today += 1
            if self._improvements_today >= self.config.max_daily_improvements:
                break

    async def _apply_improvement(self, improvement: GeneratedImprovement) -> bool:
        """Apply an improvement — writes modified code to file."""
        try:
            file_path = improvement.opportunity.file_path
            # SAFETY: make backup before writing
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            backup_path.write_text(improvement.original_code, encoding='utf-8')
            file_path.write_text(improvement.modified_code, encoding='utf-8')
            self.logger.info(f"Auto-applied improvement to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to apply improvement: {e}")
            return False

    async def run_manual_analysis(self) -> list:
        """Run analysis manually and return opportunities."""
        return self.analyzer.analyze_all()
