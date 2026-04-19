"""
Improvement scheduler for Annabeth's self-improvement system.
Runs periodic AST analysis (default: weekly) and auto-applies
LOW-risk fixes (bare except) only when no conversation is active.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Optional

from .analyzer import CodeAnalyzer
from .generator import GeneratedImprovement, ImprovementGenerator


@dataclass
class SchedulerConfig:
    """Runtime configuration for ImprovementScheduler."""
    # Weekly by default — don't spam the dev
    analysis_interval_hours: float = 168.0
    auto_apply_low_risk: bool      = True
    require_approval_for_medium: bool = True
    max_daily_improvements: int    = 3
    # Only scan the server/ directory
    src_path: Path                 = field(default_factory=lambda: Path("server"))


class ImprovementScheduler:
    """
    Schedules periodic code analysis and safe auto-application of LOW-risk fixes.

    Safety invariants:
      - Never applies improvements when `conversation_active` is True.
      - Always writes a `.bak` backup before modifying any source file.
      - Only bare-except → except-Exception fixes are auto-applied.
      - Everything else is passed to `on_improvement_ready` for human review.
    """

    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        on_improvement_ready: Optional[Callable[[GeneratedImprovement], None]] = None,
    ):
        self.logger   = logging.getLogger(__name__)
        self.config   = config or SchedulerConfig()
        self.on_improvement_ready = on_improvement_ready

        self.analyzer  = CodeAnalyzer(src_path=self.config.src_path)
        self.generator = ImprovementGenerator()

        self._running  = False
        self._task: Optional[asyncio.Task] = None

        self._improvements_today = 0
        self._last_reset         = datetime.now(timezone.utc)

        # Set True during active user conversation to block any file writes.
        self.conversation_active = False

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background scheduler loop (must be called from async context)."""
        if self._running:
            return
        self._running = True
        self._task    = asyncio.create_task(self._scheduler_loop())
        self.logger.info("[SelfImprovement] Scheduler started.")

    def stop(self) -> None:
        """Stop the scheduler cleanly."""
        self._running = False
        if self._task:
            self._task.cancel()

    async def run_manual_analysis(self) -> list:
        """Run a one-off analysis and return all opportunities (does not apply anything)."""
        return self.analyzer.analyze_all()

    # ── Private implementation ─────────────────────────────────────────────

    async def _scheduler_loop(self) -> None:
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                if (now - self._last_reset) > timedelta(days=1):
                    self._improvements_today = 0
                    self._last_reset         = now

                if (
                    not self.conversation_active
                    and self._improvements_today < self.config.max_daily_improvements
                ):
                    await self._run_analysis()

                await asyncio.sleep(self.config.analysis_interval_hours * 3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"[SelfImprovement] Scheduler error: {e}")
                await asyncio.sleep(3600)   # Back off 1 hour on unexpected error

    async def _run_analysis(self) -> None:
        self.logger.info("[SelfImprovement] Running code analysis…")
        opportunities = self.analyzer.analyze_all()
        self.logger.info(f"[SelfImprovement] Found {len(opportunities)} opportunities.")

        for opp in opportunities:
            if self.conversation_active:
                break   # Never interrupt a conversation
            if opp.confidence < 0.7:
                continue

            improvement = self.generator.generate_improvement(opp)
            if not improvement:
                continue

            if improvement.risk_level == "LOW" and self.config.auto_apply_low_risk:
                applied = await self._apply_improvement(improvement)
                if applied:
                    self._improvements_today += 1
                    if self.on_improvement_ready:
                        self.on_improvement_ready(improvement)
            else:
                # MEDIUM / HIGH — notify without touching files
                if self.on_improvement_ready:
                    self.on_improvement_ready(improvement)

            if self._improvements_today >= self.config.max_daily_improvements:
                break

    async def _apply_improvement(self, improvement: GeneratedImprovement) -> bool:
        """Write modified code to disk with a .bak backup. Returns True on success."""
        if not improvement.validation_result.get("valid"):
            self.logger.warning(
                f"[SelfImprovement] Skipping invalid improvement for "
                f"{improvement.opportunity.file_path}"
            )
            return False
        if self.conversation_active:
            return False

        try:
            file_path   = improvement.opportunity.file_path
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            backup_path.write_text(improvement.original_code, encoding="utf-8")
            file_path.write_text(improvement.modified_code,   encoding="utf-8")
            self.logger.info(
                f"[SelfImprovement] Auto-applied: {improvement.opportunity.description} "
                f"in {file_path}:{improvement.opportunity.line_number}"
            )
            return True
        except Exception as e:
            self.logger.error(f"[SelfImprovement] Failed to apply improvement: {e}")
            return False
