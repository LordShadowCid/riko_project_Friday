"""
Self-evaluation loop — Annabeth rates her own responses in the background.

After each response, the LLM scores itself on:
- Helpfulness (1-5)
- In-character-ness (1-5)
- Appropriate length (1-5)

Results are stored in SQLite via feedback.py for trend analysis.
"""
import json
import threading
import requests
from typing import Optional

from server.annabeth_config import load_config
from server.utils import get_ollama_settings


def _quick_llm(prompt: str) -> Optional[str]:
    settings = get_ollama_settings()
    payload = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": "You are a quality evaluator. "
             "Respond ONLY with the requested JSON. No extra text."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "keep_alive": settings["keep_alive"],
        "options": {"num_ctx": 512, "temperature": 0.1},
    }
    try:
        r = requests.post(
            f"{settings['host']}/api/chat", json=payload, timeout=15
        )
        r.raise_for_status()
        return (r.json().get("message") or {}).get("content", "")
    except Exception:
        return None


def self_evaluate(user_input: str, response: str, speaker: str = "Unknown"):
    """Run self-evaluation in a background thread."""
    def _do_eval():
        from server.process.memory.feedback import log_self_eval

        prompt = (
            f"Rate this voice assistant response on a scale of 1-5:\n\n"
            f"User said: \"{user_input}\"\n"
            f"Assistant replied: \"{response}\"\n\n"
            f"Score these three criteria (1=poor, 5=excellent):\n"
            f"1. helpfulness: Was the response useful and on-topic?\n"
            f"2. in_character: Did it sound like a snarky anime girl assistant?\n"
            f"3. appropriate_length: Was it the right length for a voice response?\n\n"
            f"Return ONLY JSON: "
            f'{{\"helpfulness\": N, \"in_character\": N, \"appropriate_length\": N}}'
        )
        result = _quick_llm(prompt)
        if not result:
            return

        try:
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            scores = json.loads(cleaned)
            h = int(scores.get("helpfulness", 3))
            c = int(scores.get("in_character", 3))
            l = int(scores.get("appropriate_length", 3))
            # Clamp to 1-5
            h, c, l = max(1, min(5, h)), max(1, min(5, c)), max(1, min(5, l))
            log_self_eval(
                user_input=user_input, response=response,
                helpfulness=h, in_character=c, appropriate_length=l,
                speaker=speaker,
            )
            print(f"[SelfEval] h={h} c={c} l={l}")
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    thread = threading.Thread(target=_do_eval, daemon=True, name="self-eval")
    thread.start()
