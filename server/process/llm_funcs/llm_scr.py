"""LLM wrapper (OpenAI Responses API) with local JSON chat history."""

import json
import os
from openai import OpenAI

from riko_config import load_config

char_config = load_config()
client = OpenAI(api_key=char_config.get('OPENAI_API_KEY'))

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
        with open(HISTORY_FILE, "r") as f:
            return _trim_history(json.load(f))
    return SYSTEM_PROMPT

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(_trim_history(history), f, indent=2)



def get_riko_response_no_tool(messages):

    # Call OpenAI with system prompt + history
    response = client.responses.create(
        model=MODEL,
        input= messages,
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