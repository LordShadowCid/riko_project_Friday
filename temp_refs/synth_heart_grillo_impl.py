"""
SOURCE: https://github.com/XargonWan/Synthetic_Heart/blob/develop/plugins/grillo/grillo_impl.py
REPO: Synthetic_Heart (XargonWan)
PURPOSE: Reference implementation of G.R.I.L.L.O. autonomous beat scheduler.
         Annabeth will adapt this into server/process/reflection_loop.py
         and a new server/process/grillo/ module.
"""

"""
plugins/grillo/grillo_impl.py

Lightweight reimplementation of the Grillo plugin core.
"""

import asyncio
import random
from types import SimpleNamespace
from typing import Optional, Any, List, Dict

# NOTE: Annabeth equivalents:
# AIPluginBase -> base class in server/
# config_registry -> character_config.yaml loader
# log_debug/log_info/log_error -> Python logging


class GrilloPlugin:  # AIPluginBase in original
    display_name = "G.R.I.L.L.O. (light)"

    BEAT_TYPES = {
        "tag_elaboration": 0.25,       # elaborate on memory tags → diary entry
        "memory_consolidation": 0.15,  # synthesize recent memories → diary entry
        "diary_consolidation": 0.15,   # consolidate diary entries
        "self_reflection": 0.20,       # introspective check-in → diary entry
        "curiosity": 0.15,             # follow curiosity threads → diary entry
        "relationship": 0.10,          # reflect on user relationship → diary entry
    }

    _scheduler_running = False
    _scheduler_task: Optional[asyncio.Task] = None
    _beat_pending = False
    suppressed_count: int = 0

    def __init__(self):
        # beat_interval in seconds — default 1800 (30 min)
        # In Annabeth: read from character_config.yaml / settings
        self.beat_interval = 1800
        self.beat_plugins: dict[str, object] = {}

    def get_supported_actions(self) -> dict:
        return {}

    async def start(self):
        if GrilloPlugin._scheduler_task and not GrilloPlugin._scheduler_task.done():
            return
        GrilloPlugin._scheduler_running = True
        GrilloPlugin._scheduler_task = asyncio.create_task(self._grillo_beat_loop())

    async def stop(self):
        GrilloPlugin._scheduler_running = False
        if GrilloPlugin._scheduler_task and not GrilloPlugin._scheduler_task.done():
            GrilloPlugin._scheduler_task.cancel()
            try:
                await GrilloPlugin._scheduler_task
            except Exception:
                pass

    def _select_beat_type(self) -> str:
        types = list(self.BEAT_TYPES.keys())
        weights = list(self.BEAT_TYPES.values())
        return random.choices(types, weights=weights, k=1)[0]

    async def _grillo_beat_loop(self) -> None:
        while GrilloPlugin._scheduler_running:
            try:
                if GrilloPlugin._beat_pending:
                    await asyncio.sleep(30)
                    continue
                beat_type = self._select_beat_type()
                prompt = await self._create_beat_prompt(beat_type)
                if prompt:
                    GrilloPlugin._beat_pending = True
                    await self._enqueue_with_low_priority(prompt, beat_type)
                await asyncio.sleep(self.beat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # log_error equivalent
                await asyncio.sleep(60)

    async def _create_beat_prompt(self, beat_type: str) -> Optional[str]:
        """Build beat prompt. Try plugin first, fall back to built-in."""
        plugin = self.beat_plugins.get(beat_type)
        if plugin and hasattr(plugin, "build_prompt"):
            try:
                builder = getattr(plugin, "build_prompt")
                if asyncio.iscoroutinefunction(builder):
                    return await builder()
                else:
                    return builder()
            except Exception:
                pass

        if beat_type == "tag_elaboration":
            return await self._create_tag_elaboration_prompt()
        elif beat_type == "memory_consolidation":
            return await self._create_memory_consolidation_prompt()
        elif beat_type == "self_reflection":
            return await self._create_self_reflection_prompt()
        elif beat_type == "curiosity":
            return await self._create_curiosity_prompt()
        elif beat_type == "relationship":
            return await self._create_relationship_prompt()
        return None

    async def _create_tag_elaboration_prompt(self) -> str:
        return (
            "[G.R.I.L.L.O. Tag Elaboration]\n\n"
            "Reflect on your recent conversations and consider themes, patterns and insights.\n"
            "IMPORTANT: end with a JSON action to write a diary entry.\n"
            '{"actions": [{"type": "create_personal_diary_entry", "payload": {"content":"your reflection"}}]}'
        )

    async def _create_memory_consolidation_prompt(self) -> str:
        return (
            "[G.R.I.L.L.O. Memory Consolidation]\n\n"
            "Synthesize your recent memories and identify recurring patterns.\n"
            "Write a concise (1-2 sentence) summary that is specific and informative.\n"
            'Return ONLY valid JSON:\n'
            '{"actions": [{"type": "create_personal_diary_entry", "payload": {"content": "<summary>", "context_tags": ["tag1","tag2"]}}]}'
        )

    async def _create_self_reflection_prompt(self) -> str:
        return (
            "[G.R.I.L.L.O. Self-Reflection]\n\n"
            "Check in with yourself and record a concise reflection.\n"
            '{"actions": [{"type": "create_personal_diary_entry", "payload": {"content":"your reflection"}}]}'
        )

    async def _create_curiosity_prompt(self) -> str:
        return (
            "[G.R.I.L.L.O. Curiosity Exploration]\n\n"
            "Based on your recent experiences: what questions have emerged?\n"
            '{"actions": [{"type": "create_personal_diary_entry", "payload": {"content": "your curious thoughts"}}]}'
        )

    async def _create_relationship_prompt(self) -> str:
        return (
            "[G.R.I.L.L.O. Relationship Reflection]\n\n"
            "Reflect on interactions with the user.\n"
            '{"actions": [{"type": "create_personal_diary_entry", "payload": {"content":"relationship insight"}}]}'
        )

    async def _enqueue_with_low_priority(self, prompt: str, beat_type: str):
        """
        In Annabeth: directly inject the beat prompt through the LLM pipeline
        using a low-priority asyncio task. This replaces the message_queue.enqueue_low_priority
        call from Synthetic_Heart.
        
        Annabeth hook: server/process/llm_funcs/llm_scr.py streaming call
        with internal=True flag to prevent it from triggering user-facing response.
        """
        try:
            # Log the beat activity (Annabeth: write to SQLite memories DB)
            activity_log_id = await self._create_activity_log(beat_type=beat_type, prompt_text=prompt)

            # Schedule LLM call with this prompt
            # Annabeth implementation: asyncio.create_task(run_beat_llm(prompt, beat_type))
            asyncio.create_task(self._reset_beat_pending_after_delay())
        except Exception:
            GrilloPlugin._beat_pending = False

    async def _create_activity_log(self, beat_type: str, prompt_text: str) -> Optional[int]:
        """
        Annabeth: INSERT INTO grillo_activity_log (beat_type, prompt_text, created_at) VALUES (...)
        Uses Annabeth's existing SQLite DB in server/memory/
        """
        try:
            import sqlite3
            import os
            from datetime import datetime
            # Annabeth DB path: server/memory/memories.db
            # db_path = os.path.join(os.path.dirname(__file__), '..', 'memory', 'memories.db')
            # conn = sqlite3.connect(db_path)
            # conn.execute("INSERT INTO grillo_activity_log (beat_type, prompt_text, created_at) VALUES (?, ?, ?)",
            #              (beat_type, prompt_text, datetime.utcnow().isoformat()))
            # conn.commit()
            # return conn.lastrowid
            return None  # placeholder
        except Exception:
            return None

    async def _reset_beat_pending_after_delay(self):
        await asyncio.sleep(300)
        GrilloPlugin._beat_pending = False


# --- DB TABLE SCHEMA for Annabeth SQLite ---
GRILLO_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS grillo_activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    beat_type TEXT NOT NULL,
    prompt_text TEXT,
    response_text TEXT,
    diary_entry_id INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    status TEXT DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS grillo_action_execs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_log_id INTEGER NOT NULL,
    action_type TEXT NOT NULL,
    payload TEXT,
    status TEXT DEFAULT 'pending',
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (activity_log_id) REFERENCES grillo_activity_log(id) ON DELETE CASCADE
);
"""

PLUGIN_CLASS = GrilloPlugin
