"""Focused regression test for avatar WebSocket state sync.

Validates that a newly connected client receives the current mode/silence
snapshot and that subsequent mode/silence changes are broadcast back out.
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
    print("#  ANNABETH AVATAR STATE SYNC TEST")
    print(f"#  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 60)

    from client.avatar_server import create_app
    from shared.config import CompanionMode, MessageType
    from shared.state import get_companion_state, reset_state

    reset_state()
    state = get_companion_state()
    state.mode = CompanionMode.DANCE_FULL
    state.silenced = True

    app = create_app(ROOT)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    sockets = getattr(site, "_server", None)
    if sockets is None or not site._server.sockets:
        FAIL("WebSocket test server startup", "no bound sockets")
        await runner.cleanup()
        return 1

    port = site._server.sockets[0].getsockname()[1]
    PASS("WebSocket test server startup", f"port={port}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                first = await receive_json(ws)
                second = await receive_json(ws)

                if first.get("type") == MessageType.MODE_CHANGE.value and first.get("mode") == CompanionMode.DANCE_FULL.value:
                    PASS("Initial mode snapshot")
                else:
                    FAIL("Initial mode snapshot", str(first))

                if second.get("type") == MessageType.SET_SILENCE.value and second.get("silenced") is True:
                    PASS("Initial silence snapshot")
                else:
                    FAIL("Initial silence snapshot", str(second))

                await ws.send_json({
                    "type": MessageType.MODE_CHANGE.value,
                    "mode": CompanionMode.ACTIVE.value,
                })
                mode_update = await receive_json(ws)
                if mode_update.get("type") == MessageType.MODE_CHANGE.value and mode_update.get("mode") == CompanionMode.ACTIVE.value:
                    PASS("Mode change broadcast")
                else:
                    FAIL("Mode change broadcast", str(mode_update))

                await ws.send_json({
                    "type": MessageType.SET_SILENCE.value,
                    "silenced": False,
                })
                silence_update = await receive_json(ws)
                if silence_update.get("type") == MessageType.SET_SILENCE.value and silence_update.get("silenced") is False:
                    PASS("Silence state broadcast")
                else:
                    FAIL("Silence state broadcast", str(silence_update))
    finally:
        await runner.cleanup()

    total = pass_count + fail_count
    print("\n" + "=" * 60)
    print(f"  SUMMARY: {pass_count} PASS / {fail_count} FAIL  (total {total})")
    print("=" * 60)
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))