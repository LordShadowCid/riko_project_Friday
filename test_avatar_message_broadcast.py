"""Focused regression test for avatar outbound WebSocket messages.

Validates that the avatar server broadcasts the key runtime messages that
Unity and the web clients depend on after a connection is established.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import web


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


pass_count = 0
fail_count = 0


def PASS(name, detail=""):
    global pass_count
    pass_count += 1
    print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))


def FAIL(name, detail=""):
    global fail_count
    fail_count += 1
    print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))


async def receive_json(ws: aiohttp.ClientWebSocketResponse, timeout: float = 5.0) -> dict:
    msg = await ws.receive(timeout=timeout)
    if msg.type != aiohttp.WSMsgType.TEXT:
        raise RuntimeError(f"Expected TEXT message, got {msg.type}")
    return json.loads(msg.data)


async def main() -> int:
    print("#" * 60)
    print("#  ANNABETH AVATAR MESSAGE BROADCAST TEST")
    print(f"#  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 60)

    from client import avatar_server
    from shared.config import MessageType
    from shared.state import reset_state

    reset_state()

    app = avatar_server.create_app(ROOT)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    if site._server is None or not site._server.sockets:
        FAIL("WebSocket test server startup", "no bound sockets")
        await runner.cleanup()
        return 1

    port = site._server.sockets[0].getsockname()[1]
    PASS("WebSocket test server startup", f"port={port}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                # Initial state snapshot messages.
                await receive_json(ws)
                await receive_json(ws)

                await avatar_server.speak_start("Hello there")
                start_msg = await receive_json(ws)
                if start_msg.get("type") == MessageType.SPEAK_START.value and start_msg.get("text") == "Hello there":
                    PASS("speak_start broadcast")
                else:
                    FAIL("speak_start broadcast", str(start_msg))

                await avatar_server.speak_end()
                end_msg = await receive_json(ws)
                if end_msg.get("type") == MessageType.SPEAK_END.value:
                    PASS("speak_end broadcast")
                else:
                    FAIL("speak_end broadcast", str(end_msg))

                await avatar_server.set_emotion("happy")
                emotion_msg = await receive_json(ws)
                if emotion_msg.get("type") == MessageType.EMOTION.value and emotion_msg.get("emotion") == "happy":
                    PASS("emotion broadcast")
                else:
                    FAIL("emotion broadcast", str(emotion_msg))

                await avatar_server.send_debug_status("[MIC] Listening...", "user text", "response text")
                debug_msg = await receive_json(ws)
                if (
                    debug_msg.get("type") == MessageType.DEBUG_STATUS.value
                    and debug_msg.get("status") == "[MIC] Listening..."
                    and debug_msg.get("user_text") == "user text"
                    and debug_msg.get("response_text") == "response text"
                ):
                    PASS("debug_status broadcast")
                else:
                    FAIL("debug_status broadcast", str(debug_msg))

                word_timings = [
                    {"word": "Hello", "start": 0.0, "end": 0.2},
                    {"word": "world", "start": 0.2, "end": 0.45},
                ]
                await avatar_server.send_read_highlight("Hello world", word_timings, sentence_index=3)
                highlight_msg = await receive_json(ws)
                if (
                    highlight_msg.get("type") == MessageType.READ_HIGHLIGHT.value
                    and highlight_msg.get("sentence") == "Hello world"
                    and highlight_msg.get("sentence_index") == 3
                    and highlight_msg.get("word_timings") == word_timings
                ):
                    PASS("read_highlight broadcast")
                else:
                    FAIL("read_highlight broadcast", str(highlight_msg))

                await avatar_server.send_read_clear()
                clear_msg = await receive_json(ws)
                if clear_msg.get("type") == MessageType.READ_CLEAR.value:
                    PASS("read_clear broadcast")
                else:
                    FAIL("read_clear broadcast", str(clear_msg))
    finally:
        await runner.cleanup()

    total = pass_count + fail_count
    print("\n" + "=" * 60)
    print(f"  SUMMARY: {pass_count} PASS / {fail_count} FAIL  (total {total})")
    print("=" * 60)
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))