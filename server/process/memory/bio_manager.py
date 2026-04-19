"""
Speaker Bio Manager

Maintains a per-speaker profile: real name, relationship, timezone,
known facts, and last-seen timestamp in SQLite.

Usage:
  from server.process.memory.bio_manager import get_bio, add_fact, update_bio

Inspired by: XargonWan/Synthetic_Heart/develop/plugins/bio_manager.py
"""
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

DB_PATH = Path(r"C:\annabeth_data\self_eval\feedback.db")

_db_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        # Note: tables already created by _ensure_table() at module load
    return _conn


def _ensure_table() -> None:
    """Ensure speaker_bio table exists (also created by feedback._init_tables)."""
    with _db_lock:
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS speaker_bio (
                speaker_id   TEXT PRIMARY KEY,
                real_name    TEXT DEFAULT '',
                relationship TEXT DEFAULT 'user',
                timezone     TEXT DEFAULT '',
                known_facts  TEXT DEFAULT '[]',
                last_seen    REAL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_bio_speaker ON speaker_bio(speaker_id);
        """)
        conn.commit()


_ensure_table()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_speaker(speaker_id: str) -> None:
    """Create a bio row for this speaker if one doesn't exist yet."""
    with _db_lock:
        conn = _get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO speaker_bio (speaker_id, last_seen) VALUES (?, ?)",
            (speaker_id, time.time()),
        )
        conn.commit()


def update_last_seen(speaker_id: str) -> None:
    """Touch the last_seen timestamp for this speaker."""
    ensure_speaker(speaker_id)
    with _db_lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE speaker_bio SET last_seen = ? WHERE speaker_id = ?",
            (time.time(), speaker_id),
        )
        conn.commit()


def update_bio(speaker_id: str, field: str, value: str) -> None:
    """Update a single text field (real_name, relationship, timezone)."""
    allowed = {"real_name", "relationship", "timezone"}
    if field not in allowed:
        return
    ensure_speaker(speaker_id)
    with _db_lock:
        conn = _get_conn()
        conn.execute(
            f"UPDATE speaker_bio SET {field} = ? WHERE speaker_id = ?",  # noqa: S608 — field is whitelisted
            (value, speaker_id),
        )
        conn.commit()


def add_fact(speaker_id: str, fact: str) -> None:
    """Append a free-text fact to the speaker's known_facts JSON list."""
    ensure_speaker(speaker_id)
    with _db_lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT known_facts FROM speaker_bio WHERE speaker_id = ?",
            (speaker_id,),
        ).fetchone()
        facts: list = json.loads(row[0]) if row else []
        if fact not in facts:
            facts.append(fact)
        # Keep only the most recent 50 facts to avoid unbounded growth
        facts = facts[-50:]
        conn.execute(
            "UPDATE speaker_bio SET known_facts = ? WHERE speaker_id = ?",
            (json.dumps(facts), speaker_id),
        )
        conn.commit()


def get_bio(speaker_id: str) -> str:
    """
    Return a short context string for LLM injection.

    Example:
        "Speaker: Dad | relationship: primary_user | facts: likes coffee, has a daughter Riley"
    """
    with _db_lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT real_name, relationship, timezone, known_facts, last_seen "
            "FROM speaker_bio WHERE speaker_id = ?",
            (speaker_id,),
        ).fetchone()

    if not row:
        return ""

    real_name, relationship, timezone, known_facts_str, last_seen = row
    parts: list[str] = []

    display = real_name or speaker_id
    parts.append(f"Speaker: {display}")

    if relationship:
        parts.append(f"relationship: {relationship}")

    if timezone:
        parts.append(f"timezone: {timezone}")

    try:
        facts: list[str] = json.loads(known_facts_str or "[]")
    except Exception:
        facts = []

    if facts:
        parts.append("facts: " + "; ".join(facts[-5:]))  # Show most recent 5

    if last_seen:
        delta = int(time.time() - last_seen)
        if delta < 3600:
            parts.append(f"last seen: {delta // 60}m ago")
        elif delta < 86400:
            parts.append(f"last seen: {delta // 3600}h ago")

    return " | ".join(parts)


def get_raw_bio(speaker_id: str) -> Optional[Dict[str, Any]]:
    """Return the raw bio dict for programmatic use."""
    with _db_lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT real_name, relationship, timezone, known_facts, last_seen "
            "FROM speaker_bio WHERE speaker_id = ?",
            (speaker_id,),
        ).fetchone()
    if not row:
        return None
    real_name, relationship, timezone, known_facts_str, last_seen = row
    return {
        "speaker_id": speaker_id,
        "real_name": real_name,
        "relationship": relationship,
        "timezone": timezone,
        "known_facts": json.loads(known_facts_str or "[]"),
        "last_seen": last_seen,
    }
