"""LLM wrapper with local JSON chat history.

Backends:
- OpenAI (Responses API) when OPENAI_API_KEY is set.
- Ollama (local) when OPENAI_API_KEY is not set.
"""

import json
import os

from openai import OpenAI
import requests

from server.riko_config import load_config


char_config = load_config()


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


def _get_ollama_settings() -> tuple[str, str]:
    ollama_cfg = char_config.get("ollama", {}) or {}
    host = str(ollama_cfg.get("host") or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").strip()
    if host.startswith("0.0.0.0"):
        host = "http://127.0.0.1:11434"
    if not host.startswith("http://") and not host.startswith("https://"):
        host = "http://" + host

    # Reasonable default; user can override in config.
    model = str(ollama_cfg.get("model") or "llama3.1:8b").strip()
    return host.rstrip("/"), model


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



def get_riko_response_no_tool(messages):
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

    # Ollama fallback
    host, ollama_model = _get_ollama_settings()
    chat_payload = {
        "model": ollama_model,
        "messages": _messages_to_role_content(messages),
        "stream": False,
    }

    # Normalize into an OpenAI-like shape the rest of the file expects.
    class _OllamaResp:
        def __init__(self, text: str):
            self.output_text = text

    try:
        r = requests.post(f"{host}/api/chat", json=chat_payload, timeout=60)
        if r.status_code == 404:
            raise requests.HTTPError("/api/chat not supported", response=r)
        r.raise_for_status()
        data = r.json()
        text = (((data or {}).get("message") or {}).get("content"))
        return _OllamaResp(str(text or ""))
    except requests.HTTPError:
        prompt = _messages_to_prompt(messages)
        gen_payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        r2 = requests.post(f"{host}/api/generate", json=gen_payload, timeout=120)
        r2.raise_for_status()
        data2 = r2.json()
        return _OllamaResp(str((data2 or {}).get("response") or ""))


def llm_response(user_input):

    messages = load_history()

    # Append user message to memory
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_text", "text": user_input}
        ]
    })


    riko_test_response = get_riko_response_no_tool(messages)


    # just append assistant message to regular response. 
    messages.append({
    "role": "assistant",
    "content": [
        {"type": "output_text", "text": riko_test_response.output_text}
    ]
    })

    save_history(messages)
    return riko_test_response.output_text


if __name__ == "__main__":
    print('running main')