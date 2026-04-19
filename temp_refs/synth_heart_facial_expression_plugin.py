"""
SOURCE: https://github.com/XargonWan/Synthetic_Heart/blob/develop/plugins/facial_expression_plugin.py
REPO: Synthetic_Heart (XargonWan)
PURPOSE: Reference for parsing [em_NAME:INTENSITY] LLM output tags and
         driving Unity VRM blendshape expressions via timed WebSocket events.
         Annabeth adaptation: server/process/facial_expression_handler.py
         Unity adaptation: unity/Scripts/FacialExpressionReceiver.cs
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FacialExpressionEvent:
    """A single expression tag parsed from LLM output."""
    position: int    # character index in the original text
    name: Optional[str]    # expression name (e.g. "smile"), None = reset
    intensity: float  # 0.0 - 1.0


@dataclass 
class _TimelineEvent:
    delay: float
    name: Optional[str]
    intensity: float


def parse_facial_expressions(text: str):
    """
    Parse [em_NAME:INTENSITY] tags from text.
    Returns (clean_text, list[FacialExpressionEvent])
    
    Examples:
      "Ciao! [em_grin:0.9] Come va?" → ("Ciao! Come va?", [FacialExpressionEvent(6, "grin", 0.9)])
      "[em] reset to base state" → bare [em] means clear/reset
    
    Annabeth: import re; regex = r'\[em(?:_([a-zA-Z_]+))?(?::([0-9.]+))?\]'
    """
    import re
    pattern = re.compile(r'\[em(?:_([a-zA-Z_]+))?(?::([0-9.]+))?\]')
    events = []
    clean_parts = []
    last_end = 0
    clean_pos = 0
    
    for m in pattern.finditer(text):
        clean_parts.append(text[last_end:m.start()])
        name = m.group(1)  # None for bare [em]
        intensity = float(m.group(2)) if m.group(2) else 1.0
        events.append(FacialExpressionEvent(
            position=clean_pos + len(text[last_end:m.start()]),
            name=name,
            intensity=intensity
        ))
        clean_pos += len(text[last_end:m.start()])
        last_end = m.end()
    
    clean_parts.append(text[last_end:])
    clean_text = "".join(clean_parts)
    return clean_text, events


class FacialExpressionPlugin:
    """
    Plugin that handles LLM facial expression tags.
    
    Flow:
    1. LLM outputs text with [em_smile:0.8] tags
    2. process_message_text() strips tags, schedules expression timeline
    3. _play_expression_timeline() pushes WebSocket events at timed delays
    4. Unity AnnabethExpressionReceiver.cs applies VRM blendshape weights
    
    Prompt injection (get_supported_actions):
      "You can embed facial expression tags in your message text: [em_NAME:INTENSITY]
       Available: smile, grin, sad, blush, surprised, angry, thinking
       INTENSITY: float 0.0-1.0.
       Each expression persists until the next [em_...] tag or end of audio.
       Use [em] (bare) to reset to base emotional state mid-sentence.
       Do NOT add a reset tag at the end — it happens automatically.
       These tags are invisible to users."
    
    Annabeth VRM blendshape names (common):
      smile, happy, angry, sad, surprised, blush, wink, thinking, concentrating
    """

    AVAILABLE_EXPRESSIONS = [
        "smile", "grin", "sad", "blush", "surprised", "angry",
        "thinking", "happy", "wink", "concentrating"
    ]

    def get_prompt_injection(self) -> str:
        """Inject expression instructions into system prompt."""
        expr_names = ", ".join(self.AVAILABLE_EXPRESSIONS)
        return (
            f"You can embed facial expression tags in your message text: [em_NAME:INTENSITY]\n"
            f"Available: {expr_names}\n"
            f"INTENSITY: float 0.0-1.0.\n"
            f"Each expression persists until the next [em_...] tag or end of audio.\n"
            f"At the end of audio, face automatically returns to base emotional state.\n"
            f"Use [em] (bare) to return to base emotional state mid-sentence.\n"
            f"Do NOT add a reset tag at the end — it happens automatically.\n"
            f"These tags are invisible to users. Example: \"Ciao! [em_grin:0.9] Come va?\""
        )

    async def process_message_text(
        self,
        text: str,
        session_id: str,
        audio_duration_s: Optional[float] = None,
        websocket_sender=None,  # Annabeth: avatar_server.py broadcast function
    ) -> str:
        """
        Parse text for tags and schedule expression timeline.
        Returns cleaned text (tags stripped).
        
        audio_duration_s: if provided, syncs expressions to TTS audio length.
        websocket_sender: async callable that sends JSON to Unity WebSocket clients.
        """
        clean, events = parse_facial_expressions(text)
        if events and websocket_sender:
            total_chars = len(clean)
            chars_per_sec = 12  # configurable: how fast TTS speaks
            asyncio.create_task(
                self._play_expression_timeline(
                    events, total_chars, session_id,
                    chars_per_sec, audio_duration_s, websocket_sender
                )
            )
        return clean

    async def _play_expression_timeline(
        self,
        events: List[FacialExpressionEvent],
        total_chars: int,
        session_id: str,
        chars_per_sec: float,
        audio_duration_s: Optional[float],
        websocket_sender,
    ) -> None:
        """
        Drive Unity via WebSocket through a sequence of expression events.
        
        WebSocket message format (to Unity):
        {
          "type": "facial_expression",
          "expression": "smile",    // or null for reset
          "intensity": 0.8,
          "targets": {"mouthSmile": 0.8, "cheekPuff": 0.3}  // optional VRM exact targets
        }
        """
        import json
        
        total_duration = (
            audio_duration_s
            if audio_duration_s and audio_duration_s > 0
            else (total_chars / chars_per_sec if chars_per_sec > 0 else 1.0)
        )
        
        timeline: List[_TimelineEvent] = []
        for ev in events:
            frac = ev.position / total_chars if total_chars > 0 else 0.0
            delay = frac * total_duration
            timeline.append(_TimelineEvent(delay, ev.name, ev.intensity))
        
        start = asyncio.get_event_loop().time()
        for item in timeline:
            now = asyncio.get_event_loop().time()
            sleep_for = item.delay - (now - start)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            
            # Send to Unity via WebSocket
            msg = {
                "type": "facial_expression",
                "expression": item.name,  # None = clear/reset
                "intensity": item.intensity,
            }
            try:
                await websocket_sender(json.dumps(msg))
            except Exception:
                pass
        
        # After all events: hold until end of audio, then reset
        elapsed = asyncio.get_event_loop().time() - start
        remaining = total_duration - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)
        
        # Send reset
        reset_msg = {"type": "facial_expression", "expression": None, "intensity": 0}
        try:
            await websocket_sender(json.dumps(reset_msg))
        except Exception:
            pass


# ==============================================================
# UNITY C# RECEIVER STUB (for reference)
# ==============================================================
# Create: unity/Scripts/AnnabethExpressionReceiver.cs
# 
# Receives {"type":"facial_expression","expression":"smile","intensity":0.8}
# Maps expression name to VRM blendshape keys:
#   smile → "happy" VRM preset or custom shape
#   grin  → "happy" at higher intensity
#   sad   → "sad" VRM preset
#   etc.
#
# Uses Vrm10RuntimeExpression.SetWeight(key, weight) or
# VRMBlendShapeProxy.ImmediatelySetValue(BlendShapeKey, value)
