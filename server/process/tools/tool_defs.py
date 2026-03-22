"""
Tool definitions for Annabeth — Phase 5.

Each tool is defined as:
- An Ollama-compatible JSON schema (for the LLM to call)
- A Python executor function (runs when the LLM invokes it)

Tools available:
- get_current_time: Returns the current date/time
- remember_fact: Stores a fact in long-term memory
- recall_memory: Searches long-term memory for relevant info
- get_self_eval_summary: Gets recent self-evaluation scores
- web_search: Searches the web for information (DuckDuckGo)
- adjust_personality: Self-modification — tweaks personality parameters
"""
import datetime
import json
import re
from typing import Any


# ============ Ollama Tool Schemas ============
# These are sent to Ollama in the `tools` parameter.
# Llama 3.1 uses these to decide when/how to call a tool.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date, time, and day of week. Use when the user asks what time or day it is.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Store something important about the user in long-term memory so you can recall it later. Use this when the user tells you something personal worth remembering (preferences, facts about themselves, requests to remember something).",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to remember, written as a statement. Example: 'User's favorite color is blue'",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Who or what this fact is about. Example: 'Dad', 'Riley'",
                    },
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Search your long-term memory for information about past conversations or stored facts. Use when the user asks 'do you remember' or references something from a past conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory. Example: 'favorite food', 'what we talked about yesterday'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use when the user asks about something you don't know, recent events, weather, news, or facts you're unsure about.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Keep it concise and specific.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_personality",
            "description": "Adjust your own personality parameters based on feedback. Only use this if the user explicitly asks you to change how you respond (be more/less snarky, longer/shorter responses, etc).",
            "parameters": {
                "type": "object",
                "properties": {
                    "parameter": {
                        "type": "string",
                        "enum": ["verbosity", "snarkiness", "formality"],
                        "description": "Which personality aspect to adjust",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["increase", "decrease"],
                        "description": "Whether to increase or decrease this trait",
                    },
                },
                "required": ["parameter", "direction"],
            },
        },
    },
]

# Quick lookup by name
TOOL_MAP = {t["function"]["name"]: t for t in TOOL_SCHEMAS}


# ============ Tool Executors ============

def execute_get_current_time(**_kwargs) -> str:
    now = datetime.datetime.now()
    return now.strftime("It's %A, %B %d, %Y at %I:%M %p.")


def execute_remember_fact(fact: str, subject: str = "Unknown", **_kwargs) -> str:
    try:
        from server.process.memory.memory_store import get_memory_store
        store = get_memory_store()
        store.add_fact(fact, subject=subject, speaker=subject)
        return f"Remembered: {fact}"
    except Exception as e:
        return f"Failed to store memory: {e}"


def execute_recall_memory(query: str, **_kwargs) -> str:
    try:
        from server.process.memory.memory_store import get_memory_store
        store = get_memory_store()
        results = store.recall_all(query, n_results=3)
        if not results:
            return "No relevant memories found."
        lines = []
        for r in results:
            text = r.get("text", "")
            dist = r.get("distance", 1.0)
            if dist < 1.5:  # Only include reasonably relevant results
                lines.append(f"- {text}")
        return "\n".join(lines) if lines else "No relevant memories found."
    except Exception as e:
        return f"Memory search failed: {e}"


def execute_web_search(query: str, **_kwargs) -> str:
    """Search the web using DuckDuckGo Lite (no API key needed)."""
    import urllib.request
    import urllib.parse
    try:
        url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query})
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Extract text snippets from DuckDuckGo Lite results
        # Results are in <td> tags with class "result-snippet"
        snippets = re.findall(
            r'<td\s+class="result-snippet"[^>]*>(.*?)</td>',
            html, re.DOTALL | re.IGNORECASE
        )
        if not snippets:
            # Fallback: try <a class="result-link"> text
            snippets = re.findall(
                r'<a[^>]+class="result-link"[^>]*>(.*?)</a>',
                html, re.DOTALL | re.IGNORECASE
            )

        # Clean HTML tags from snippets
        clean = []
        for s in snippets[:3]:
            text = re.sub(r'<[^>]+>', '', s).strip()
            if text:
                clean.append(text)

        if clean:
            return "Web search results:\n" + "\n".join(f"- {s}" for s in clean)
        return f"No results found for '{query}'."
    except Exception as e:
        return f"Web search failed: {e}"


def execute_adjust_personality(parameter: str, direction: str, **_kwargs) -> str:
    """Adjust personality parameters stored in a JSON file."""
    import os
    personality_file = os.path.join(
        os.environ.get("ANNABETH_DATA", r"C:\annabeth_data"),
        "personality.json"
    )
    defaults = {"verbosity": 3, "snarkiness": 4, "formality": 2}
    try:
        if os.path.exists(personality_file):
            with open(personality_file, "r") as f:
                personality = json.loads(f.read())
        else:
            personality = dict(defaults)

        current = personality.get(parameter, defaults.get(parameter, 3))
        if direction == "increase":
            new_val = min(5, current + 1)
        else:
            new_val = max(1, current - 1)

        personality[parameter] = new_val
        os.makedirs(os.path.dirname(personality_file), exist_ok=True)
        with open(personality_file, "w") as f:
            f.write(json.dumps(personality, indent=2))

        return f"Adjusted {parameter}: {current} -> {new_val} (scale 1-5)"
    except Exception as e:
        return f"Failed to adjust personality: {e}"


# ============ Executor Router ============

_EXECUTORS = {
    "get_current_time": execute_get_current_time,
    "remember_fact": execute_remember_fact,
    "recall_memory": execute_recall_memory,
    "web_search": execute_web_search,
    "adjust_personality": execute_adjust_personality,
}


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name and return result as a string.
    
    Args:
        name: Tool function name
        arguments: Dict of arguments parsed from the LLM's tool call
        
    Returns:
        String result to feed back to the LLM
    """
    executor = _EXECUTORS.get(name)
    if not executor:
        return f"Unknown tool: {name}"
    try:
        result = executor(**arguments)
        print(f"[Tool] {name}({arguments}) -> {result[:100]}")
        return str(result)
    except Exception as e:
        print(f"[Tool] {name} failed: {e}")
        return f"Tool error: {e}"
