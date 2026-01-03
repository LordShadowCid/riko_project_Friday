"""LLM wrapper with local JSON chat history.

Backends:
- OpenAI (Responses API) when OPENAI_API_KEY is set.
- Ollama (local) when OPENAI_API_KEY is not set.

Supports streaming for faster response times.
Includes RAM-based response caching for faster repeated queries.
"""

import json
import os
import re
import hashlib
from collections import OrderedDict
from typing import Generator, Callable

from openai import OpenAI
import requests

from server.annabeth_config import load_config


char_config = load_config()

# ============ Response Cache ============
# RAM-based cache for recent responses to avoid regenerating similar queries
# Uses LRU eviction to keep memory bounded

class ResponseCache:
    """LRU cache for LLM responses, stored in RAM."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, user_input: str, context_hash: str = "") -> str:
        """Create a cache key from user input and context."""
        # Normalize input: lowercase, strip whitespace
        normalized = user_input.lower().strip()
        key_str = f"{context_hash}:{normalized}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, user_input: str, context_hash: str = "") -> str | None:
        """Get cached response if available."""
        key = self._make_key(user_input, context_hash)
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None
    
    def put(self, user_input: str, response: str, context_hash: str = ""):
        """Store a response in cache."""
        key = self._make_key(user_input, context_hash)
        
        # If key exists, move to end
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = response
            return
        
        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        
        self._cache[key] = response
    
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
        }

# Global cache instance
_cache_cfg = char_config.get("response_cache", {}) or {}
_cache_enabled = _cache_cfg.get("enabled", True)
_cache_max_size = _cache_cfg.get("max_entries", 100)
_response_cache = ResponseCache(max_size=_cache_max_size) if _cache_enabled else None

def get_response_cache() -> ResponseCache | None:
    """Get the global response cache instance."""
    return _response_cache


def _get_context_hash(messages: list) -> str:
    """Create a hash of recent conversation context for cache keying."""
    # Use last 2 exchanges (4 messages) for context
    recent = messages[-4:] if len(messages) > 4 else messages
    context_str = ""
    for m in recent:
        if isinstance(m, dict):
            role = m.get("role", "")
            content = _content_to_text(m.get("content", ""))
            context_str += f"{role}:{content[:100]}"  # Truncate long messages
    return hashlib.md5(context_str.encode()).hexdigest()[:8]

# ============ End Response Cache ============


def _resolve_openai_api_key() -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY") or char_config.get("OPENAI_API_KEY")
    if not api_key:
        return None
    api_key = str(api_key).strip()
    if api_key in {"sk-YOURAPIKEY", "YOUR_API_KEY"}:
        return None
    return api_key


def _get_openai_client() -> OpenAI:
    api_key = _resolve_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "OpenAI API key is not set. Set OPENAI_API_KEY as an environment variable or set OPENAI_API_KEY in character_config.yaml."
        )
    return OpenAI(api_key=api_key)


def _get_ollama_settings() -> dict:
    """Get Ollama configuration as a dict."""
    ollama_cfg = char_config.get("ollama", {}) or {}
    host = str(ollama_cfg.get("host") or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").strip()
    if host.startswith("0.0.0.0"):
        host = "http://127.0.0.1:11434"
    if not host.startswith("http://") and not host.startswith("https://"):
        host = "http://" + host

    # Reasonable default; user can override in config.
    model = str(ollama_cfg.get("model") or "llama3.1:8b").strip()
    keep_alive = ollama_cfg.get("keep_alive", 3600)  # Keep model loaded for 1 hour
    stream = ollama_cfg.get("stream", True)  # Enable streaming by default
    num_ctx = ollama_cfg.get("num_ctx", 2048)  # Smaller context = faster generation
    
    return {
        "host": host.rstrip("/"),
        "model": model,
        "keep_alive": keep_alive,
        "stream": stream,
        "num_ctx": num_ctx,
    }


def _content_to_text(content) -> str:
    """Normalize OpenAI-style content blocks into a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if "text" in item and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                    continue
                if item.get("type") in {"input_text", "output_text"} and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
    return str(content)


def _messages_to_role_content(messages) -> list[dict]:
    """Convert stored history into Ollama chat message format."""
    out: list[dict] = []
    if not isinstance(messages, list):
        return out
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        text = _content_to_text(m.get("content"))
        out.append({"role": role, "content": text})
    return out


def _messages_to_prompt(messages) -> str:
    """Flatten messages into a single text prompt for /api/generate fallback."""
    lines: list[str] = []
    if not isinstance(messages, list):
        return ""
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        text = _content_to_text(m.get("content"))
        if not text:
            continue
        if role == "system":
            lines.append(f"System: {text}")
        elif role == "user":
            lines.append(f"User: {text}")
        else:
            lines.append(f"Assistant: {text}")

    lines.append("Assistant:")
    return "\n".join(lines)

# Constants
HISTORY_FILE = char_config.get('history_file', 'chat_history.json')
MODEL = char_config.get('model', 'gpt-4.1-mini')
MAX_HISTORY_TURNS = int(char_config.get('max_history_turns', 20))
SYSTEM_PROMPT =  [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": char_config['presets']['default']['system_prompt']  
                }
            ]
        }
    ]

# Load/save chat history
def _trim_history(messages):
    """Keep the system prompt and the last N user/assistant turns."""
    if not isinstance(messages, list):
        return SYSTEM_PROMPT

    system = SYSTEM_PROMPT
    rest = [m for m in messages if isinstance(m, dict) and m.get('role') != 'system']

    max_messages = max(2, MAX_HISTORY_TURNS * 2)
    trimmed = rest[-max_messages:]
    return system + trimmed


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return _trim_history(json.load(f))
    return SYSTEM_PROMPT

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(_trim_history(history), f, indent=2)



def get_annabeth_response(messages):
    api_key = _resolve_openai_api_key()
    if api_key:
        client = _get_openai_client()

        # Call OpenAI with system prompt + history
        response = client.responses.create(
            model=MODEL,
            input=messages,
            temperature=1,
            top_p=1,
            max_output_tokens=2048,
            stream=False,
            text={
                "format": {
                    "type": "text"
                }
            },
        )

        return response

    # Ollama fallback (non-streaming)
    settings = _get_ollama_settings()
    chat_payload = {
        "model": settings["model"],
        "messages": _messages_to_role_content(messages),
        "stream": False,
        "keep_alive": settings["keep_alive"],
        "options": {
            "num_ctx": settings["num_ctx"],
        },
    }

    # Normalize into an OpenAI-like shape the rest of the file expects.
    class _OllamaResp:
        def __init__(self, text: str):
            self.output_text = text

    try:
        r = requests.post(f"{settings['host']}/api/chat", json=chat_payload, timeout=60)
        if r.status_code == 404:
            raise requests.HTTPError("/api/chat not supported", response=r)
        r.raise_for_status()
        data = r.json()
        text = (((data or {}).get("message") or {}).get("content"))
        return _OllamaResp(str(text or ""))
    except requests.HTTPError:
        prompt = _messages_to_prompt(messages)
        gen_payload = {
            "model": settings["model"],
            "prompt": prompt,
            "stream": False,
            "keep_alive": settings["keep_alive"],
        }
        r2 = requests.post(f"{settings['host']}/api/generate", json=gen_payload, timeout=120)
        r2.raise_for_status()
        data2 = r2.json()
        return _OllamaResp(str((data2 or {}).get("response") or ""))


def stream_ollama_response(messages) -> Generator[str, None, str]:
    """
    Stream response from Ollama, yielding complete sentences as they arrive.
    Returns the full response text at the end.
    
    Yields sentences as they complete (ending with .!? or newline).
    """
    settings = _get_ollama_settings()
    
    chat_payload = {
        "model": settings["model"],
        "messages": _messages_to_role_content(messages),
        "stream": True,
        "keep_alive": settings["keep_alive"],
        "options": {
            "num_ctx": settings["num_ctx"],
        },
    }
    
    full_response = ""
    buffer = ""
    # Pattern to split on sentence endings
    sentence_pattern = re.compile(r'([.!?]+[\s\n]+|[\n]+)')
    
    try:
        with requests.post(
            f"{settings['host']}/api/chat", 
            json=chat_payload, 
            stream=True,
            timeout=120,
        ) as r:
            r.raise_for_status()
            
            for line in r.iter_lines():
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                content = (data.get("message") or {}).get("content", "")
                if content:
                    buffer += content
                    full_response += content
                    
                    # Check if we have complete sentences to yield
                    parts = sentence_pattern.split(buffer)
                    
                    # If we have at least one complete sentence
                    if len(parts) > 1:
                        # Combine sentence parts (text + punctuation pairs)
                        i = 0
                        while i < len(parts) - 1:
                            if i + 1 < len(parts):
                                sentence = parts[i] + parts[i + 1]
                                sentence = sentence.strip()
                                if sentence:
                                    yield sentence
                                i += 2
                            else:
                                break
                        
                        # Keep the incomplete part in buffer
                        buffer = parts[-1] if parts else ""
                
                # Check if done
                if data.get("done"):
                    break
        
        # Yield any remaining text
        if buffer.strip():
            yield buffer.strip()
            
    except Exception as e:
        print(f"[LLM] Streaming error: {e}")
        if buffer.strip():
            yield buffer.strip()
    
    return full_response


def llm_response(user_input):

    messages = load_history()

    # Append user message to memory
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_text", "text": user_input}
        ]
    })


    annabeth_response = get_annabeth_response(messages)


    # just append assistant message to regular response. 
    messages.append({
    "role": "assistant",
    "content": [
        {"type": "output_text", "text": annabeth_response.output_text}
    ]
    })

    save_history(messages)
    return annabeth_response.output_text


def llm_response_streaming(user_input, on_sentence: Callable[[str], None] = None, speaker_name: str = None) -> str:
    """
    Get LLM response with streaming. Calls on_sentence callback for each sentence.
    Returns the full response text.
    
    Uses RAM cache for faster repeated queries when enabled.
    
    Args:
        user_input: User's message
        on_sentence: Callback function that receives each sentence as it's ready
        speaker_name: Name of the speaker (for multi-user support)
        
    Returns:
        Full response text
    """
    messages = load_history()
    
    # Format the message with speaker info if available
    if speaker_name and speaker_name != "Unknown":
        formatted_input = f"[{speaker_name}]: {user_input}"
    else:
        formatted_input = user_input
    
    # Check response cache first
    cache = get_response_cache()
    if cache:
        context_hash = _get_context_hash(messages)
        cached_response = cache.get(formatted_input, context_hash)
        if cached_response:
            print(f"[LLM] Cache hit! ({cache.stats()['hit_rate']} overall)")
            # Still need to add to history and call callback
            messages.append({
                "role": "user",
                "content": [{"type": "input_text", "text": formatted_input}]
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "output_text", "text": cached_response}]
            })
            save_history(messages)
            if on_sentence:
                on_sentence(cached_response)
            return cached_response

    # Append user message to memory
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_text", "text": formatted_input}
        ]
    })

    # Check if we should use streaming
    settings = _get_ollama_settings()
    api_key = _resolve_openai_api_key()
    
    if api_key or not settings.get("stream", True):
        # Use non-streaming for OpenAI or if streaming disabled
        response = get_annabeth_response(messages)
        full_text = response.output_text
        if on_sentence:
            on_sentence(full_text)
    else:
        # Stream from Ollama
        full_text = ""
        for sentence in stream_ollama_response(messages):
            full_text += " " + sentence if full_text else sentence
            if on_sentence:
                on_sentence(sentence)
        full_text = full_text.strip()

    # Save to history
    messages.append({
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": full_text}
        ]
    })

    save_history(messages)
    
    # Store in cache for future use
    if cache and full_text:
        context_hash = _get_context_hash(messages[:-2])  # Context before this exchange
        cache.put(formatted_input, full_text, context_hash)
    
    return full_text


if __name__ == "__main__":
    print('running main')