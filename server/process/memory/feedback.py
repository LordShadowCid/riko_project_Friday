"""
Implicit feedback tracker — monitors conversation signals to learn user preferences.

Tracks:
- Interruptions (negative signal — response was too long or off-topic)
- Conversation length (positive — user engaged longer)
- Response timing (how long user waited before responding)

Stores metrics in SQLite on C: NVMe for fast random I/O.
"""
import sqlite3
import time
import threading
from pathlib import Path
from typing import Optional

DB_PATH = Path(r"C:\annabeth_data\self_eval\feedback.db")

_db_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    """Get or create the SQLite connection (thread-safe)."""
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent performance
        _init_tables(_conn)
    return _conn


def _init_tables(conn: sqlite3.Connection):
    """Create feedback tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            speaker TEXT DEFAULT 'Unknown',
            event_type TEXT NOT NULL,
            value REAL DEFAULT 0,
            details TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS self_eval (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            speaker TEXT DEFAULT 'Unknown',
            user_input TEXT,
            response TEXT,
            helpfulness INTEGER DEFAULT 0,
            in_character INTEGER DEFAULT 0,
            appropriate_length INTEGER DEFAULT 0,
            notes TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(event_type);
        CREATE INDEX IF NOT EXISTS idx_feedback_time ON feedback(timestamp);
        CREATE INDEX IF NOT EXISTS idx_self_eval_time ON self_eval(timestamp);
    """)


def log_feedback(event_type: str, value: float = 0,
                 speaker: str = "Unknown", details: str = ""):
    """Log an implicit feedback event.
    
    Event types:
    - 'interruption': User interrupted Annabeth (value=1)
    - 'turn_complete': A full turn completed (value=response_time_secs)
    - 'session_length': Conversation session ended (value=num_turns)
    """
    with _db_lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO feedback (timestamp, speaker, event_type, value, details) "
            "VALUES (?, ?, ?, ?, ?)",
            (time.time(), speaker, event_type, value, details),
        )
        conn.commit()


def log_self_eval(user_input: str, response: str,
                  helpfulness: int, in_character: int,
                  appropriate_length: int, speaker: str = "Unknown",
                  notes: str = ""):
    """Log a self-evaluation score for a response (1-5 scale each)."""
    with _db_lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO self_eval "
            "(timestamp, speaker, user_input, response, helpfulness, "
            "in_character, appropriate_length, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), speaker, user_input, response,
             helpfulness, in_character, appropriate_length, notes),
        )
        conn.commit()


def get_recent_feedback_summary(hours: float = 24) -> dict:
    """Get a summary of recent feedback for self-improvement."""
    with _db_lock:
        conn = _get_conn()
        cutoff = time.time() - (hours * 3600)

        interruptions = conn.execute(
            "SELECT COUNT(*) FROM feedback "
            "WHERE event_type='interruption' AND timestamp > ?",
            (cutoff,),
        ).fetchone()[0]

        turns = conn.execute(
            "SELECT COUNT(*), AVG(value) FROM feedback "
            "WHERE event_type='turn_complete' AND timestamp > ?",
            (cutoff,),
        ).fetchone()

        eval_avg = conn.execute(
            "SELECT AVG(helpfulness), AVG(in_character), "
            "AVG(appropriate_length), COUNT(*) FROM self_eval "
            "WHERE timestamp > ?",
            (cutoff,),
        ).fetchone()

        return {
            "interruptions": interruptions,
            "total_turns": turns[0] or 0,
            "avg_response_time": round(turns[1] or 0, 2),
            "avg_helpfulness": round(eval_avg[0] or 0, 1),
            "avg_in_character": round(eval_avg[1] or 0, 1),
            "avg_appropriate_length": round(eval_avg[2] or 0, 1),
            "eval_count": eval_avg[3] or 0,
        }
