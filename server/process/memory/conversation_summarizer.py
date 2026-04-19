"""
Conversation summarizer — extracts facts and summaries after each chat exchange.

Uses the local Ollama LLM to analyze the conversation and extract:
- Key facts about the user (stored in 'facts' collection)
- A brief conversation summary (stored in 'conversations' collection)

Runs asynchronously in a background thread to avoid blocking the chat loop.
"""
import json
import threading
import requests
from typing import Optional

from server.annabeth_config import load_config
from server.utils import get_ollama_settings


def _llm_extract(prompt: str) -> Optional[str]:
    """Run a quick, non-streaming LLM call for extraction tasks."""
    settings = get_ollama_settings()
    payload = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": "You extract information from conversations. "
             "Respond ONLY with the requested JSON. No explanations."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "keep_alive": settings["keep_alive"],
        "options": {"num_ctx": 1024, "temperature": 0.1},
    }
    try:
        r = requests.post(
            f"{settings['host']}/api/chat", json=payload, timeout=30
        )
        r.raise_for_status()
        return (r.json().get("message") or {}).get("content", "")
    except Exception as e:
        print(f"[Memory] LLM extraction failed: {e}")
        return None


def extract_and_store(user_text: str, assistant_text: str,
                      speaker: str = "Unknown") -> None:
    """
    Extract facts and summary from a conversation exchange, store in memory.
    Runs in a background thread to not block the chat loop.
    """
    def _do_extract():
        from server.process.memory.memory_store import get_memory_store
        store = get_memory_store()

        # 1) Extract facts about the user
        # NOTE: We intentionally do NOT pass assistant_text to this prompt.
        # Including it causes the LLM to store facts about the assistant's
        # behavior, and those leak into RAG injections causing repetition.
        fact_prompt = (
            f"The user (speaker ID tag: \"{speaker}\" — this is NOT their real name, "
            f"just an internal label) said: \"{user_text}\"\n\n"
            "Extract any personal facts about the user from what they said.\n"
            "IMPORTANT: The speaker tag (e.g. 'Dad', 'Riley') is just an ID label, "
            "NOT the user's name. Do NOT record it as a fact.\n"
            "Only record facts the USER reveals about THEMSELVES.\n"
            "Examples: their hobbies, preferences, job, family details, "
            "things they like/dislike, things they mentioned doing.\n"
            "Return a JSON array of fact strings. "
            "If no facts, return an empty array [].\n"
            "Example: [\"User likes sci-fi movies\", \"User has a daughter named Riley\"]"
        )
        fact_response = _llm_extract(fact_prompt)
        if fact_response:
            try:
                # Try to parse JSON from the response
                # Handle markdown code blocks
                cleaned = fact_response.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
                facts = json.loads(cleaned)
                if isinstance(facts, list):
                    for fact in facts:
                        if isinstance(fact, str) and len(fact) > 5:
                            store.add_fact(fact, subject=speaker, speaker=speaker)
                            print(f"[Memory] Learned: {fact}")
                            # Also populate structured bio table (Phase 5 — Synthetic_Heart)
                            if speaker and speaker not in {"Unknown", None}:
                                try:
                                    from server.process.memory.bio_manager import (
                                        ensure_speaker, add_fact as bio_add_fact,
                                    )
                                    ensure_speaker(speaker)
                                    bio_add_fact(speaker, fact)
                                except Exception as _be:
                                    pass  # non-fatal
            except (json.JSONDecodeError, ValueError):
                pass

        # 2) Store conversation record — raw user text only, NO LLM summarization.
        # Previously we used the LLM to generate a summary, but the 8B model
        # would include assistant response text despite instructions not to,
        # poisoning the RAG context and causing the LLM to parrot old responses.
        # Raw user text is sufficient for ChromaDB's embedding-based recall.
        if user_text and len(user_text.strip()) > 5:
            store.add_conversation(f"User said: {user_text.strip()}", speaker=speaker)
            print(f"[Memory] Stored user message for recall")

        # 3) Self-compress if conversations collection is getting large
        try:
            store.compress_if_needed(threshold=500)
        except Exception as _ce:
            print(f"[Memory] Compression skipped (non-fatal): {_ce}")

    thread = threading.Thread(target=_do_extract, daemon=True,
                              name="memory-extract")
    thread.start()
