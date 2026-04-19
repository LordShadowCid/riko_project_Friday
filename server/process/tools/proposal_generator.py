"""
Proposal generator for the self-improvement system.
Converts raw ImprovementOpportunity objects into persisted, reviewable proposals.

Proposals are saved to C:\\annabeth_data\\proposals.json.
Nothing is auto-applied unless self_improvement.auto_apply_risk is explicitly raised
in character_config.yaml (default: none).
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from server.process.tools.code_analyzer import ImprovementOpportunity, scan_directory

logger = logging.getLogger(__name__)

_PROPOSALS_PATH = Path(r"C:\annabeth_data\proposals.json")
_RATE_LIMIT_SECONDS = 3600  # at most once per hour

_generator_lock = threading.Lock()
_last_run_time: float = 0.0


@dataclass
class Proposal:
    """A persisted code improvement proposal."""
    id: str               # unique identifier (timestamp-based)
    created_at: str       # ISO-8601 UTC timestamp
    file: str
    line: int
    category: str
    description: str
    suggestion: str
    severity: str
    status: str = "pending"   # pending / accepted / rejected / applied


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_proposals() -> List[Proposal]:
    if not _PROPOSALS_PATH.exists():
        return []
    try:
        raw = json.loads(_PROPOSALS_PATH.read_text(encoding="utf-8"))
        return [Proposal(**p) for p in raw]
    except Exception as exc:
        logger.warning("[Proposals] Could not load proposals: %s", exc)
        return []


def _save_proposals(proposals: List[Proposal]):
    _PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        _PROPOSALS_PATH.write_text(
            json.dumps([asdict(p) for p in proposals], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.error("[Proposals] Could not save proposals: %s", exc)


def _opportunity_to_proposal(opp: ImprovementOpportunity) -> Proposal:
    ts = datetime.now(timezone.utc)
    return Proposal(
        id=f"{ts.strftime('%Y%m%d%H%M%S%f')}_{opp.file.replace('/', '_')}_{opp.line}",
        created_at=ts.isoformat(),
        file=opp.file,
        line=opp.line,
        category=opp.category,
        description=opp.description,
        suggestion=opp.suggestion,
        severity=opp.severity,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ProposalGenerator:
    """
    Scans the codebase for improvement opportunities and persists new proposals.

    De-duplicates against existing pending proposals so repeated runs don't
    create noise.
    """

    def __init__(self, workspace_root: str, max_pending: int = 100):
        self.workspace_root = workspace_root
        self.max_pending = max_pending

    def run(self) -> int:
        """
        Scan the workspace and save any new proposals.  Returns the count added.
        Rate-limited to at most once per _RATE_LIMIT_SECONDS.
        """
        global _last_run_time

        with _generator_lock:
            if time.monotonic() - _last_run_time < _RATE_LIMIT_SECONDS:
                logger.debug("[Proposals] Rate-limited, skipping scan")
                return 0

            _last_run_time = time.monotonic()

        try:
            opportunities = scan_directory(self.workspace_root)
        except Exception as exc:
            logger.warning("[Proposals] Scan error: %s", exc)
            return 0

        if not opportunities:
            return 0

        with _generator_lock:
            existing = _load_proposals()

            # Build a set of (file, line, category) for fast de-dup
            seen = {
                (p.file, p.line, p.category)
                for p in existing
                if p.status == "pending"
            }

            # Cap total pending proposals
            pending_count = sum(1 for p in existing if p.status == "pending")
            added = 0

            for opp in opportunities:
                key = (opp.file, opp.line, opp.category)
                if key in seen:
                    continue
                if pending_count >= self.max_pending:
                    logger.debug(
                        "[Proposals] Max pending proposals (%d) reached", self.max_pending
                    )
                    break

                proposal = _opportunity_to_proposal(opp)
                existing.append(proposal)
                seen.add(key)
                pending_count += 1
                added += 1

            if added:
                _save_proposals(existing)
                logger.info("[Proposals] Added %d new proposal(s)", added)

        return added

    def get_pending(self) -> List[Proposal]:
        """Return all pending proposals (read-only view)."""
        with _generator_lock:
            return [p for p in _load_proposals() if p.status == "pending"]

    def mark_rejected(self, proposal_id: str):
        """Mark a proposal as rejected so it is not re-surfaced."""
        with _generator_lock:
            proposals = _load_proposals()
            for p in proposals:
                if p.id == proposal_id:
                    p.status = "rejected"
                    break
            _save_proposals(proposals)

    def summary_for_llm(self, max_items: int = 5) -> str:
        """Return a short human-readable summary of the top pending proposals."""
        pending = self.get_pending()
        if not pending:
            return "No code improvement proposals pending."

        # Prioritise high severity
        ordered = sorted(
            pending,
            key=lambda p: {"high": 0, "medium": 1, "low": 2}.get(p.severity, 3),
        )

        lines = [f"Top {min(max_items, len(ordered))} code improvement proposals:"]
        for p in ordered[:max_items]:
            lines.append(
                f"  [{p.severity.upper()}] {p.file}:{p.line} — {p.description}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_generator_instance: Optional[ProposalGenerator] = None


def get_proposal_generator(workspace_root: Optional[str] = None) -> Optional[ProposalGenerator]:
    """
    Return the module-level ProposalGenerator singleton.
    Returns None if self_improvement is disabled in config.
    """
    global _generator_instance

    if _generator_instance is not None:
        return _generator_instance

    try:
        from server.annabeth_config import load_config, resolve_repo_path
        config = load_config()
        si_cfg = config.get("self_improvement", {})
        if not si_cfg.get("enabled", False):
            return None

        root = workspace_root or resolve_repo_path(".")
        _generator_instance = ProposalGenerator(workspace_root=str(root))
    except Exception as exc:
        logger.warning("[Proposals] Could not initialise generator: %s", exc)
        return None

    return _generator_instance
