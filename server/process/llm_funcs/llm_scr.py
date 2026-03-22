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
from server.utils import get_ollama_settings as _get_ollama_settings


char_config = load_config()


# ============ TTS Sentence Chunking ============
# Break long sentences at natural pause points to reduce TTS latency spikes

def chunk_long_sentence(sentence: str, max_chars: int = 250) -> list[str]:
    """
    Break long sentences at natural pause points (commas, semicolons, colons).
    This reduces TTS latency spikes on long sentences.
    
    Args:
        sentence: The sentence to potentially chunk
        max_chars: Maximum characters before trying to split (default 150)
    
    Returns:
        List of sentence chunks (may be just [sentence] if short enough)
    """
    if len(sentence) <= max_chars:
        return [sentence]
    
    chunks = []
    # Split on natural pause points: comma, semicolon, colon, dash
    pause_pattern = re.compile(r'([,;:\-–—]\s+)')
    parts = pause_pattern.split(sentence)
    
    current = ""
    for i, part in enumerate(parts):
        # If adding this part exceeds max and we have content, save current chunk
        if len(current) + len(part) > max_chars and current.strip():
            chunks.append(current.strip())
            current = part
        else:
            current += part
    
    # Don't forget the last chunk
    if current.strip():
        chunks.append(current.strip())
    
    # If we couldn't split well, just return original
    if not chunks:
        return [sentence]
    
    return chunks

# ============ End TTS Sentence Chunking ============


# ============ Repetition Guard ============
# Detect when the LLM falls into a loop producing identical or near-identical
# responses and force a regeneration with higher temperature to break out.

# In-memory deque of recent assistant response prefixes.  Survives across turns
# even when the previous turn's daemon thread hasn't flushed chat_history.json yet
# (race condition after user interrupts playback).
from collections import deque as _deque
_recent_responses_mem: _deque[str] = _deque(maxlen=4)

# Rotating fallback pool — avoids single-string poisoning if any sneak into history
_FALLBACK_POOL = [
    "Hmm, what was that? I think I zoned out for a second. Could you say that again?",
    "Wait, sorry, I totally spaced out! What did you say?",
    "Ah, my brain just glitched for a sec. One more time?",
    "Oops, I missed that. Say it again for me?",
    "Hold on, I blanked out there. Can you repeat that?",
]
_FALLBACK_SET_LOWER = {f.strip().lower() for f in _FALLBACK_POOL}

def _is_fallback_like(text: str) -> bool:
    """Check if text is similar to any fallback response."""
    from difflib import SequenceMatcher
    norm = text.strip().lower()
    for fb in _FALLBACK_SET_LOWER:
        if SequenceMatcher(None, norm, fb).ratio() >= 0.75:
            return True
    return False

def _pick_fallback() -> str:
    """Pick a random fallback response from the pool."""
    import random
    return random.choice(_FALLBACK_POOL)

def _get_recent_assistant_texts(messages: list, n: int = 4) -> list[str]:
    """Return the text of the last N assistant responses from history."""
    texts = []
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            texts.append(_content_to_text(m.get("content", "")).strip().lower())
            if len(texts) >= n:
                break
    return texts


def _is_repetition(candidate: str, recent_texts: list[str], threshold: float = 0.85) -> bool:
    """Check if candidate is a near-duplicate of any recent response.
    
    Uses BOTH full-text and prefix-based comparison.  The prefix check
    catches outputs that start identically but diverge into gibberish.
    """
    if not candidate or not recent_texts:
        return False
    from difflib import SequenceMatcher
    norm = candidate.strip().lower()
    prefix = norm[:120]  # First ~120 chars catch "same opener" pattern
    for prev in recent_texts:
        if not prev:
            continue
        # Full text similarity
        if SequenceMatcher(None, norm, prev).ratio() >= threshold:
            return True
        # Prefix similarity — catches same-start-different-gibberish-tail
        if len(prefix) > 40 and len(prev) > 40:
            prev_prefix = prev[:120]
            if SequenceMatcher(None, prefix, prev_prefix).ratio() >= 0.85:
                return True
    return False


def _is_gibberish(text: str, threshold: float = 0.05) -> bool:
    """Detect garbled/gibberish text caused by excessive temperature.
    
    Checks multiple signals:
    - Non-ASCII character ratio (accented, symbols, control chars)
    - HTML entities that shouldn't appear in spoken text
    - Smooshed-word runs (very long tokens with no spaces)
    - Alternating-case gibberish (LeTs KeEp ThIe StUfF)
    """
    if not text:
        return False
    # --- Non-ASCII ratio ---
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ratio = non_ascii / len(text) if text else 0
    if ratio > threshold:
        print(f"[LLM] Gibberish detected: {non_ascii}/{len(text)} non-ASCII chars ({ratio:.1%})")
        return True
    # --- HTML entities (should never appear in spoken text) ---
    import re as _re
    html_entities = _re.findall(r'&(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+|[a-z]+);', text)
    if len(html_entities) >= 3:
        print(f"[LLM] Gibberish detected: {len(html_entities)} HTML entities in response")
        return True
    # --- Smooshed-word runs (tokens >30 chars with no spaces) ---
    long_tokens = [w for w in text.split() if len(w) > 30]
    if len(long_tokens) >= 2:
        print(f"[LLM] Gibberish detected: {len(long_tokens)} smooshed-word tokens")
        return True
    # --- Alternating case gibberish (e.g. "LeTs KeEp ThIe StUfF") ---
    words = text.split()
    if len(words) >= 6:
        mixed_case_count = sum(
            1 for w in words
            if len(w) > 2 and any(c.isupper() for c in w[1:]) and any(c.islower() for c in w[1:])
        )
        if mixed_case_count / len(words) > 0.4:
            print(f"[LLM] Gibberish detected: {mixed_case_count}/{len(words)} mixed-case words")
            return True
    # --- Slash/hyphen-fragmented words (e.g. "th/at si/nk i/n wh/iLl") ---
    if len(words) >= 4:
        frag_count = sum(1 for w in words if _re.search(r'[a-zA-Z][/\\][a-zA-Z]', w))
        if frag_count >= 3:
            print(f"[LLM] Gibberish detected: {frag_count} slash-fragmented words")
            return True
    return False


def _dedup_history(messages: list) -> list:
    """Remove duplicate/near-identical consecutive assistant responses from history.
    
    When the LLM sees repeated assistant responses in its context, it generates
    more of the same.  This breaks that echo chamber by keeping only the latest
    instance of any run of duplicates.
    """
    from difflib import SequenceMatcher
    if not messages:
        return messages
    
    result = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            result.append(msg)
            i += 1
            continue
        
        # Look ahead: skip this assistant msg if the NEXT assistant msg is very similar
        curr_text = _content_to_text(msg.get("content", "")).strip().lower()
        
        # Also skip if content is gibberish
        raw_text = _content_to_text(msg.get("content", ""))
        if _is_gibberish(raw_text):
            # Skip this message AND its preceding user message if we just added one
            if result and isinstance(result[-1], dict) and result[-1].get("role") == "user":
                result.pop()  # Remove the orphaned user message too
            i += 1
            continue
        
        # Check if a later assistant message is near-identical
        skip = False
        for j in range(i + 1, min(i + 3, len(messages))):
            later = messages[j]
            if isinstance(later, dict) and later.get("role") == "assistant":
                later_text = _content_to_text(later.get("content", "")).strip().lower()
                if len(curr_text) > 40 and len(later_text) > 40:
                    prefix_ratio = SequenceMatcher(None, curr_text[:120], later_text[:120]).ratio()
                    if prefix_ratio >= 0.80:
                        # Skip this one — the later duplicate will be kept (or also pruned)
                        skip = True
                        # Also remove the preceding user message to keep pairs aligned
                        if result and isinstance(result[-1], dict) and result[-1].get("role") == "user":
                            result.pop()
                        break
        
        if not skip:
            result.append(msg)
        i += 1
    
    return result


def _strip_accumulated_prefix(new_text: str, prev_assistant_texts: list) -> str:
    """Strip accumulated prefix when the model echoes previous responses.

    Detects the pattern where the LLM generates
    "old response A + old response B + new content" and returns only the
    genuinely new content.  This prevents snowballing history where each
    turn grows longer.
    """
    from difflib import SequenceMatcher

    if not new_text or not prev_assistant_texts:
        return new_text

    new_norm = new_text.strip()
    if len(new_norm) < 60:
        return new_text

    # Find the longest previous response whose text appears as a prefix
    # of the new response.  Check longest candidates first so we strip
    # as much accumulated echo as possible.
    best_end = 0
    for prev in sorted(prev_assistant_texts, key=len, reverse=True):
        prev_norm = prev.strip()
        if len(prev_norm) < 40:
            continue

        check_len = min(len(prev_norm), len(new_norm))
        sm = SequenceMatcher(
            None,
            new_norm[:check_len].lower(),
            prev_norm[:check_len].lower(),
        )
        ratio = sm.ratio()

        if ratio >= 0.75 and check_len > best_end:
            best_end = check_len

    if best_end >= 60:
        remainder = new_text[best_end:].strip()
        # Clean orphaned leading punctuation / connectors
        remainder = remainder.lstrip(' .,;:!?-\u2013\u2014')
        remainder = remainder.strip()

        if len(remainder) >= 20:
            print(f"[LLM] Stripped {best_end}-char accumulated prefix from response")
            return remainder
        else:
            print(f"[LLM] Would strip prefix but only {len(remainder)} chars remain \u2014 keeping original")

    return new_text

# ============ End Repetition Guard ============


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

# ============ Tool Calling (Phase 5) ============
# The abliterated model doesn't support Ollama's native tool API, so we use
# keyword-based routing: detect when tools are needed from the user's input,
# run them, and inject the results as context before the LLM generates.

def _tools_enabled() -> bool:
    """Check if tool calling is enabled in config."""
    return (char_config.get("tools", {}) or {}).get("enabled", False)


def _match_and_run_tools(user_input: str, speaker_name: str = "Unknown") -> str | None:
    """Match user input to tools, execute them, and return context to inject.

    Returns a string of tool results to inject as a system message,
    or None if no tools matched.
    """
    import re as _re
    text = user_input.lower().strip()
    results = []

    from server.process.tools.tool_defs import (
        execute_get_current_time,
        execute_remember_fact,
        execute_recall_memory,
        execute_web_search,
        execute_adjust_personality,
    )

    # --- Time / Date ---
    time_patterns = [
        r"\bwhat time\b", r"\bwhat.s the time\b", r"\btell me the time\b",
        r"\bwhat day\b", r"\bwhat.s the day\b", r"\bwhat.s the date\b",
        r"\bwhat date\b", r"\bwhat month\b", r"\bwhat year\b",
        r"\bcurrent time\b", r"\bcurrent date\b",
    ]
    if any(_re.search(p, text) for p in time_patterns):
        result = execute_get_current_time()
        results.append(f"[Current time] {result}")
        print(f"[Tools] Matched: get_current_time -> {result}")

    # --- Remember fact ---
    remember_patterns = [
        r"\bremember (that|this)\b", r"\bdon.t forget\b",
        r"\bplease remember\b", r"\bkeep in mind\b",
        r"\bnote that\b", r"\bsave (that|this)\b",
    ]
    if any(_re.search(p, text) for p in remember_patterns):
        # Extract what to remember (text after the trigger phrase)
        fact = user_input.strip()  # Store the full user message as the fact
        result = execute_remember_fact(fact=fact, subject=speaker_name)
        results.append(f"[Memory stored] {result}")
        print(f"[Tools] Matched: remember_fact -> {result}")

    # --- Recall memory ---
    recall_patterns = [
        r"\bdo you remember\b", r"\bdid (i|we) (tell|mention|say)\b",
        r"\bwhat did (i|we) (say|tell|talk) about\b",
        r"\brecall\b", r"\bwhat do you know about\b",
    ]
    if any(_re.search(p, text) for p in recall_patterns):
        result = execute_recall_memory(query=user_input)
        if "No relevant memories" not in result:
            results.append(f"[Retrieved memories]\n{result}")
            print(f"[Tools] Matched: recall_memory -> found results")
        else:
            results.append("[Retrieved memories] Nothing found in memory.")
            print(f"[Tools] Matched: recall_memory -> no results")

    # --- Personality adjustment ---
    personality_map = {
        r"\bbe more (verbose|talkative|detailed)\b": ("verbosity", "increase"),
        r"\bbe less (verbose|talkative|detailed)\b": ("verbosity", "decrease"),
        r"\bshorter (responses?|answers?|replies?)\b": ("verbosity", "decrease"),
        r"\blonger (responses?|answers?|replies?)\b": ("verbosity", "increase"),
        r"\bbe more snarky\b": ("snarkiness", "increase"),
        r"\bbe less snarky\b": ("snarkiness", "decrease"),
        r"\bbe (more )?(polite|nice|kind)\b": ("snarkiness", "decrease"),
        r"\bbe more formal\b": ("formality", "increase"),
        r"\bbe (more )?casual\b": ("formality", "decrease"),
    }
    for pattern, (param, direction) in personality_map.items():
        if _re.search(pattern, text):
            result = execute_adjust_personality(parameter=param, direction=direction)
            results.append(f"[Personality adjusted] {result}")
            print(f"[Tools] Matched: adjust_personality({param}, {direction}) -> {result}")
            break

    # --- Web search (broader trigger, only if nothing else matched) ---
    if not results:
        search_patterns = [
            r"\bsearch (for|about)\b", r"\blook up\b", r"\bgoogle\b",
            r"\bfind (out|me|info)\b", r"\bwhat is a?\b", r"\bwho is\b",
            r"\bwhat are\b", r"\bhow (do|does|to|can)\b",
        ]
        if any(_re.search(p, text) for p in search_patterns):
            # Only trigger web search for genuine questions, not conversational
            if len(text) > 15 and "?" in user_input:
                result = execute_web_search(query=user_input)
                if "failed" not in result.lower():
                    results.append(f"[Web search]\n{result}")
                    print(f"[Tools] Matched: web_search -> got results")

    if not results:
        return None

    return "\n".join(results)


def _get_personality_overlay() -> str:
    """Read personality adjustments from personality.json and return a prompt snippet."""
    pf = os.path.join(os.environ.get("ANNABETH_DATA", r"C:\annabeth_data"), "personality.json")
    if not os.path.exists(pf):
        return ""
    try:
        with open(pf, "r", encoding="utf-8") as f:
            p = json.loads(f.read())
        parts = []
        v = p.get("verbosity", 3)
        if v <= 2:
            parts.append("Keep responses very short (1 sentence).")
        elif v >= 4:
            parts.append("Feel free to give longer, more detailed responses (3-5 sentences).")
        s = p.get("snarkiness", 4)
        if s <= 2:
            parts.append("Be more polite and less snarky than usual.")
        elif s >= 4:
            parts.append("Be extra snarky, teasing, and playful.")
        fm = p.get("formality", 2)
        if fm >= 4:
            parts.append("Use proper grammar and a more formal tone.")
        elif fm <= 2:
            parts.append("Use casual, relaxed language with slang.")
        return " ".join(parts)
    except Exception:
        return ""

# ============ End Tool Calling ============


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
def _sanitize_response(text: str, max_len: int = 600) -> str:
    """Clean an assistant response before saving to history.

    Strips HTML entities, asterisk actions, excess punctuation, and
    truncates overly long responses so they don't fill the context window
    with noise.
    """
    if not text:
        return text
    import re as _re
    # Decode HTML entities -> plain text
    import html as _html
    text = _html.unescape(text)
    # Remove any remaining raw HTML entity patterns
    text = _re.sub(r'&[a-z]+;', ' ', text)
    text = _re.sub(r'&#x?[0-9a-fA-F]+;', ' ', text)
    # Remove asterisk actions *like this*
    text = _re.sub(r'\*[^*]+\*', '', text)
    # Collapse excessive punctuation (!!!!! -> !)
    text = _re.sub(r'([!?.]){3,}', r'\1', text)
    # Collapse whitespace
    text = _re.sub(r'\s+', ' ', text).strip()
    # Truncate
    if len(text) > max_len:
        text = text[:max_len].rsplit(' ', 1)[0] + '...'
    return text


def _trim_history(messages):
    """Keep the system prompt and the last N user/assistant turns.

    Also sanitises assistant responses (strip HTML entities, truncate)
    and drops gibberish entries so they don't poison future context.
    """
    if not isinstance(messages, list):
        return SYSTEM_PROMPT

    system = SYSTEM_PROMPT
    rest = [m for m in messages if isinstance(m, dict) and m.get('role') != 'system']

    max_messages = max(2, MAX_HISTORY_TURNS * 2)
    trimmed = rest[-max_messages:]

    # Sanitise assistant responses
    cleaned = []
    for m in trimmed:
        if m.get('role') == 'assistant':
            raw = _content_to_text(m.get('content', ''))
            sanitised = _sanitize_response(raw)
            if _is_gibberish(sanitised):
                # Drop this assistant msg AND its preceding user msg
                if cleaned and cleaned[-1].get('role') == 'user':
                    cleaned.pop()
                continue
            m = dict(m)
            m['content'] = [{'type': 'output_text', 'text': sanitised}]
        cleaned.append(m)

    return system + cleaned


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8-sig") as f:
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
    settings = _get_ollama_settings(char_config)
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


def stream_ollama_response(messages, temp_boost: float = 0.0) -> Generator[str, None, str]:
    """
    Stream response from Ollama, yielding complete sentences as they arrive.
    Returns the full response text at the end.
    
    Args:
        messages: Chat messages to send to Ollama
        temp_boost: Additional temperature to add (for breaking repetition loops)
    
    Yields sentences as they complete (ending with .!? or newline).
    """
    settings = _get_ollama_settings(char_config)
    
    options = {
        "num_ctx": settings["num_ctx"],
        # Token-level repeat suppression: penalise tokens that appeared
        # in the last 256 tokens of the prompt (covers recent assistant
        # responses in the chat history).  Default is 1.1 / 64 which is
        # too mild for an 8B model that loops easily.
        # NOTE: 1.5 caused garbled text — 1.35 balances repetition vs coherence.
        "repeat_penalty": 1.35,
        "repeat_last_n": 512,
        "num_predict": 300,  # Cap response length to reduce slowness & rambling
    }
    if temp_boost > 0:
        options["temperature"] = 0.7 + temp_boost  # Default ~0.7, boost from there
    
    chat_payload = {
        "model": settings["model"],
        "messages": _messages_to_role_content(messages),
        "stream": True,
        "keep_alive": settings["keep_alive"],
        "options": options,
    }
    
    full_response = ""
    buffer = ""
    # Pattern to split on sentence endings
    # Primary: sentence endings. Secondary: mid-sentence breaks for long buffers.
    sentence_pattern = re.compile(r'([.!?]+[\s\n]+|[\n]+)')
    mid_break_pattern = re.compile(r'([,;:]\s+|—\s*|\s-\s)')
    EAGER_FLUSH_LEN = 80  # Flush at mid-sentence break if buffer exceeds this
    
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
                                    # Chunk long sentences for smoother TTS
                                    for chunk in chunk_long_sentence(sentence):
                                        yield chunk
                                i += 2
                            else:
                                break
                        
                        # Keep the incomplete part in buffer
                        buffer = parts[-1] if parts else ""
                    elif len(buffer) > EAGER_FLUSH_LEN:
                        # Eager flush: split at comma/semicolon for faster first audio
                        mid_parts = mid_break_pattern.split(buffer)
                        if len(mid_parts) > 1:
                            # Yield everything up to and including the break
                            flushed = ""
                            for j in range(0, len(mid_parts) - 1, 2):
                                piece = mid_parts[j]
                                sep = mid_parts[j + 1] if j + 1 < len(mid_parts) else ""
                                flushed += piece + sep
                            flushed = flushed.strip()
                            if flushed:
                                for chunk in chunk_long_sentence(flushed):
                                    yield chunk
                            buffer = mid_parts[-1] if mid_parts else ""
                
                # Check if done
                if data.get("done"):
                    break
        
        # Yield any remaining text (also chunk if long)
        if buffer.strip():
            for chunk in chunk_long_sentence(buffer.strip()):
                yield chunk
            
    except Exception as e:
        print(f"[LLM] Streaming error: {e}")
        if buffer.strip():
            for chunk in chunk_long_sentence(buffer.strip()):
                yield chunk
    
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
            # Check that the cached response isn't a repeat of recent responses
            mem_responses = [t for t in _recent_responses_mem]
            if _is_repetition(cached_response, mem_responses, threshold=0.90):
                print(f"[LLM] Cache hit REJECTED (would repeat recent response)")
                cached_response = None  # Fall through to live generation
            else:
                print(f"[LLM] Cache hit! ({cache.stats()['hit_rate']} overall)")
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

    # --- RAG: Inject relevant long-term memories (ephemeral, not saved to history) ---
    def _build_llm_messages(base_messages: list, inject_memories: bool = True) -> list:
        """Build ephemeral LLM messages, optionally with memory injection."""
        llm_msgs = list(base_messages)
        if not inject_memories:
            return llm_msgs
        try:
            from server.process.memory.memory_store import get_memory_store
            store = get_memory_store()
            memories = store.recall_all(user_input, n_results=3)
            if memories:
                memory_lines = [m["text"] for m in memories if m.get("text")]
                if memory_lines:
                    memory_context = (
                        "You remember the following from past conversations:\n- "
                        + "\n- ".join(memory_lines[:2])
                        + "\nUse these memories naturally if relevant. "
                        "Don't force them into the conversation."
                    )
                    llm_msgs.insert(-1, {
                        "role": "system",
                        "content": [{"type": "input_text", "text": memory_context}]
                    })
                    print(f"[Memory] Injected {len(memory_lines)} memories")
        except Exception as e:
            print(f"[Memory] Recall failed (non-fatal): {e}")
        return llm_msgs

    llm_messages = _build_llm_messages(messages, inject_memories=True)

    # --- Personality overlay (from self-modification) ---
    personality_snippet = _get_personality_overlay()
    if personality_snippet:
        llm_messages.insert(1, {
            "role": "system",
            "content": [{"type": "input_text", "text": personality_snippet}]
        })

    # --- Tool calling (Phase 5) ---
    tool_was_used = False
    if _tools_enabled():
        tool_context = _match_and_run_tools(user_input, speaker_name=speaker_name or "Unknown")
        if tool_context:
            tool_was_used = True
            # Inject tool results as an ephemeral system message right before
            # the user's latest message so the LLM can incorporate them naturally
            llm_messages.insert(-1, {
                "role": "system",
                "content": [{"type": "input_text", "text":
                    f"Tool results (use naturally in your response, don't mention tools):\n{tool_context}"}]
            })
            print(f"[Tools] Injected tool context into LLM messages")

    # Collect recent assistant responses for post-generation repetition check.
    # Merge file history with in-memory deque for completeness.
    history_responses = [
        t for t in _get_recent_assistant_texts(messages, n=4)
        if not _is_fallback_like(t)
    ]
    mem_responses = [t for t in _recent_responses_mem if not _is_fallback_like(t)]
    _seen_resp = set()
    recent_responses: list[str] = []
    for _r in list(mem_responses) + history_responses:
        _key = _r[:60]
        if _key not in _seen_resp:
            _seen_resp.add(_key)
            recent_responses.append(_r)

    # Check if we should use streaming
    settings = _get_ollama_settings(char_config)
    api_key = _resolve_openai_api_key()
    
    max_retries = 2
    temp_boost = 0.0
    is_repeat = False

    # --- Pre-generation: always deduplicate history ---
    # Removes near-identical assistant responses so the LLM doesn't see an
    # echo chamber that causes it to accumulate old responses in its output.
    llm_messages = _dedup_history(llm_messages)

    # Mild temperature nudge when recent responses share the same opening.
    if len(recent_responses) >= 2:
        from difflib import SequenceMatcher as _SM
        if _SM(None, recent_responses[0][:120], recent_responses[1][:120]).ratio() >= 0.80:
            temp_boost = 0.15
            print(f"[LLM] Similar recent openings — temp_boost={temp_boost}")

    for attempt in range(max_retries + 1):
        if api_key or not settings.get("stream", True):
            # Non-streaming path (OpenAI or streaming disabled)
            response = get_annabeth_response(llm_messages)
            full_text = response.output_text
            is_repeat = _is_repetition(full_text, recent_responses, threshold=0.90)
            is_fallback_clone = _is_fallback_like(full_text)
            is_gibberish = _is_gibberish(full_text)
            if is_repeat or is_fallback_clone or is_gibberish:
                reason = "gibberish" if is_gibberish else ("fallback-clone" if is_fallback_clone else "repetition")
                if attempt < max_retries:
                    temp_boost += 0.10
                    llm_messages = _build_llm_messages(messages, inject_memories=False)
                    llm_messages = _dedup_history(llm_messages)
                    print(f"[LLM] {reason} detected (attempt {attempt+1}), retrying, temp_boost={temp_boost}")
                    continue
                else:
                    print(f"[LLM] {reason} persisted after retries — using fallback")
                    full_text = _pick_fallback()
                    if on_sentence:
                        on_sentence(full_text)
                    break
            if on_sentence and full_text:
                on_sentence(full_text)
            break
        else:
            # STREAMING PATH with early-abort repeat detection.
            # Buffer sentences until we accumulate enough text (~40 chars)
            # to reliably compare against recent responses. Only then
            # release the buffered sentences to TTS. This catches short
            # first sentences like "Um, okay." that would slip past a
            # per-sentence check.
            full_text = ""
            early_abort = False
            prefix_checked = False
            buffered_sentences = []  # Hold back until prefix check passes
            _EARLY_CHECK_LEN = 40   # Chars needed before we can check

            for sentence in stream_ollama_response(llm_messages, temp_boost=temp_boost):
                if not full_text:
                    full_text = sentence
                else:
                    full_text += " " + sentence

                # --- Early repeat detection: buffer until enough text ---
                if not prefix_checked:
                    buffered_sentences.append(sentence)
                    prefix_so_far = full_text.strip().lower()

                    if len(prefix_so_far) >= _EARLY_CHECK_LEN and recent_responses:
                        from difflib import SequenceMatcher as _SM2
                        for prev in recent_responses:
                            if len(prev) < _EARLY_CHECK_LEN:
                                continue
                            check_len = min(len(prefix_so_far), len(prev))
                            ratio = _SM2(None, prefix_so_far[:check_len], prev[:check_len]).ratio()
                            if ratio >= 0.75:
                                early_abort = True
                                print(f"[LLM] Early repeat detected after {len(buffered_sentences)} sentence(s), {len(prefix_so_far)} chars (ratio={ratio:.2f}) -- aborting")
                                break

                        if early_abort:
                            break

                        # Passed! Flush all buffered sentences to TTS
                        prefix_checked = True
                        if on_sentence:
                            for buf_s in buffered_sentences:
                                on_sentence(buf_s)
                        buffered_sentences = []
                        continue
                    else:
                        # Not enough text yet — keep buffering
                        continue

                # Subsequent sentences stream normally
                if on_sentence:
                    on_sentence(sentence)

            # If we never reached the check threshold (very short response),
            # flush whatever we buffered
            if not prefix_checked and not early_abort and buffered_sentences:
                if on_sentence:
                    for buf_s in buffered_sentences:
                        on_sentence(buf_s)

            full_text = full_text.strip()

            if early_abort:
                if attempt < max_retries:
                    temp_boost += 0.25
                    llm_messages = _build_llm_messages(messages, inject_memories=False)
                    llm_messages = _dedup_history(llm_messages)
                    print(f"[LLM] Retrying with temp_boost={temp_boost} (attempt {attempt+1})")
                    continue
                else:
                    print(f"[LLM] Repeat persisted after retries -- using fallback")
                    full_text = _pick_fallback()
                    is_repeat = False  # fallback is fine to save
                    if on_sentence:
                        on_sentence(full_text)
                    break

            # Post-generation checks (for cases that slip past early detection).
            is_gibberish_flag = _is_gibberish(full_text)
            is_repeat = _is_repetition(full_text, recent_responses, threshold=0.90)

            if is_gibberish_flag:
                print(f"[LLM] WARNING: Gibberish streamed to user -- will be sanitized in history")
            if is_repeat:
                print(f"[LLM] WARNING: Repeat streamed to user -- NOT saving to history")

            break

    was_fallback = _is_fallback_like(full_text)

    # Record in in-memory deque for cross-turn tracking
    if full_text and not was_fallback:
        _recent_responses_mem.append(full_text.strip().lower())

    # If a repeat was detected on the streaming path, do NOT save the
    # assistant response to history.  Saving it would reinforce the loop.
    # We still save the user message so context isn't lost.
    if is_repeat and not was_fallback:
        # Keep the user message that was already appended to `messages`
        # but skip appending the assistant echo.
        save_history(messages)
        print(f"[LLM] Skipped saving repeated assistant response to history")
    else:
        # Normal save path
        saved_text = _sanitize_response(full_text) if not was_fallback else full_text

        # Strip accumulated prefix: if the model echoed previous responses
        # at the start of this one, save only the genuinely new content.
        if not was_fallback:
            prev_texts = _get_recent_assistant_texts(messages, n=4)
            saved_text = _strip_accumulated_prefix(saved_text, prev_texts)
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": saved_text}
            ]
        })

        save_history(messages)
    
    # Store in cache (skip fallback responses and tool-assisted responses)
    if cache and full_text and not was_fallback and not tool_was_used:
        context_hash = _get_context_hash(messages[:-2])
        cache.put(formatted_input, full_text, context_hash)
    
    # Background: Extract facts + store conversation summary (skip fallbacks)
    if not was_fallback:
        try:
            from server.process.memory.conversation_summarizer import extract_and_store
            _speaker = speaker_name or "Unknown"
            extract_and_store(formatted_input, full_text, speaker=_speaker)
        except Exception as e:
            print(f"[Memory] Background extraction failed (non-fatal): {e}")

        try:
            from server.process.memory.self_eval import self_evaluate
            self_evaluate(formatted_input, full_text, speaker=speaker_name or "Unknown")
        except Exception as e:
            print(f"[SelfEval] Launch failed (non-fatal): {e}")

        # Self-modification check (rate-limited internally)
        try:
            from server.process.tools.self_modify import self_modify_check
            self_modify_check()
        except Exception as e:
            print(f"[SelfMod] Check failed (non-fatal): {e}")

    return full_text


if __name__ == "__main__":
    print('running main')