"""
Self-modification: Annabeth adjusts her own personality based on self-eval trends.

Runs periodically (every N turns) and checks:
- avg_helpfulness < 3.5 -> make responses more detailed (increase verbosity)
- avg_appropriate_length < 3.5 -> adjust verbosity (likely too long -> decrease)
- avg_in_character < 3.5 -> increase snarkiness back up
- too many interruptions -> decrease verbosity (responses too long)

Adjustments are stored in C:\\annabeth_data\\personality.json and read as a
prompt overlay by _get_personality_overlay() in llm_scr.py.
"""
import json
import os
import time

PERSONALITY_FILE = os.path.join(
    os.environ.get("ANNABETH_DATA", r"C:\annabeth_data"), "personality.json"
)
DEFAULTS = {"verbosity": 3, "snarkiness": 4, "formality": 2}
# Don't check more than once every 10 minutes
_last_check: float = 0.0
CHECK_INTERVAL_SEC = 600
# Minimum evals needed before adjusting
MIN_EVALS = 5

# Thresholds for self-modification decisions
_INTERRUPT_RATE_HIGH = 0.30   # interrupt_rate above this → verbosity too high
_SCORE_LOW_THRESHOLD = 3.5    # avg score below this → trait needs adjustment


def _load_personality() -> dict:
    if os.path.exists(PERSONALITY_FILE):
        with open(PERSONALITY_FILE, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    return dict(DEFAULTS)


def _save_personality(p: dict):
    os.makedirs(os.path.dirname(PERSONALITY_FILE), exist_ok=True)
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(PERSONALITY_FILE), suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(p, indent=2))
        os.replace(tmp_path, PERSONALITY_FILE)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def self_modify_check():
    """Check self-eval trends and adjust personality if needed.
    
    Call this after each response. It rate-limits itself internally.
    """
    global _last_check
    now = time.time()
    if now - _last_check < CHECK_INTERVAL_SEC:
        return
    _last_check = now

    try:
        from server.process.memory.feedback import get_recent_feedback_summary
        summary = get_recent_feedback_summary(hours=6)
    except Exception as e:
        print(f"[SelfMod] Feedback fetch failed: {e}")
        return

    if summary.get("eval_count", 0) < MIN_EVALS:
        return  # Not enough data yet

    p = _load_personality()
    changed = False

    # Too many interruptions -> responses probably too long
    interruption_rate = summary["interruptions"] / max(summary["total_turns"], 1)
    if interruption_rate > _INTERRUPT_RATE_HIGH and p.get("verbosity", 3) > 1:
        p["verbosity"] = max(1, p.get("verbosity", 3) - 1)
        changed = True
        print(f"[SelfMod] High interruption rate ({interruption_rate:.0%}) -> verbosity decreased to {p['verbosity']}")

    # Low helpfulness -> try being more detailed
    if summary["avg_helpfulness"] < _SCORE_LOW_THRESHOLD and p.get("verbosity", 3) < 5:
        p["verbosity"] = min(5, p.get("verbosity", 3) + 1)
        changed = True
        print(f"[SelfMod] Low helpfulness ({summary['avg_helpfulness']}) -> verbosity increased to {p['verbosity']}")

    # Low appropriate_length -> adjust verbosity (usually means too long)
    # Skip if helpfulness is ALSO low — the two adjustments would cancel out,
    # and helpfulness matters more for user satisfaction.
    elif summary["avg_appropriate_length"] < _SCORE_LOW_THRESHOLD and p.get("verbosity", 3) > 1:
        p["verbosity"] = max(1, p.get("verbosity", 3) - 1)
        changed = True
        print(f"[SelfMod] Low length score ({summary['avg_appropriate_length']}) -> verbosity decreased to {p['verbosity']}")

    # Low in-character -> boost snarkiness back up
    if summary["avg_in_character"] < _SCORE_LOW_THRESHOLD and p.get("snarkiness", 4) < 5:
        p["snarkiness"] = min(5, p.get("snarkiness", 4) + 1)
        changed = True
        print(f"[SelfMod] Low in-character ({summary['avg_in_character']}) -> snarkiness increased to {p['snarkiness']}")

    if changed:
        _save_personality(p)
        print(f"[SelfMod] Personality updated: {p}")
    else:
        print(f"[SelfMod] Scores OK — no adjustments (h={summary['avg_helpfulness']}, c={summary['avg_in_character']}, l={summary['avg_appropriate_length']})")

    # Phase 8 — code improvement proposals (no-op when disabled)
    try:
        from server.process.tools.proposal_generator import get_proposal_generator
        gen = get_proposal_generator()
        if gen is not None:
            added = gen.run()
            if added:
                print(f"[SelfMod] {added} new code improvement proposal(s) written to proposals.json")
    except Exception as e:
        print(f"[SelfMod] Proposal scan failed (non-critical): {e}")
