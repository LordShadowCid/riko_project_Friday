"""
Avatar WebSocket Server
Serves the VRM model and sends animation commands to the web frontend
Includes system audio analysis for beat-reactive dancing
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import Set, Optional, Dict, Any
from aiohttp import web
import aiohttp

# Add parent directory to path for shared imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared import (
    CompanionMode,
    MessageType,
    get_config,
    get_companion_state,
)

# Type aliases
WebSocketClient = web.WebSocketResponse

# Connected WebSocket clients
clients: Set[WebSocketClient] = set()

# Audio analyzer reference
_audio_analyzer = None
_audio_broadcast_task: Optional["asyncio.Task[None]"] = None

_config = get_config()


def _get_state():
    """Return the live shared companion state instance."""
    return get_companion_state()


def get_current_mode() -> CompanionMode:
    """Get the current companion mode."""
    return _get_state().mode


def is_chat_silenced() -> bool:
    """Check if chat is silenced (S key toggle)."""
    return _get_state().silenced


def set_chat_silenced(silenced: bool) -> None:
    """Set chat silence state."""
    _get_state().silenced = silenced


def toggle_chat_silence() -> bool:
    """Toggle chat silence on/off."""
    return _get_state().toggle_silence()


def is_listening_paused() -> bool:
    """Check if listening should be paused (silenced OR not in active mode)."""
    return _get_state().is_listening_paused()


async def _send_state_snapshot(ws: WebSocketClient) -> None:
    """Send the current shared companion state to a newly connected client."""
    state = _get_state()
    await ws.send_str(json.dumps({
        "type": MessageType.MODE_CHANGE.value,
        "mode": state.mode.value,
    }))
    await ws.send_str(json.dumps({
        "type": MessageType.SET_SILENCE.value,
        "silenced": state.silenced,
    }))


async def _broadcast_mode_change() -> None:
    state = _get_state()
    await broadcast({
        "type": MessageType.MODE_CHANGE.value,
        "mode": state.mode.value,
    })


async def _broadcast_silence_state() -> None:
    state = _get_state()
    await broadcast({
        "type": MessageType.SET_SILENCE.value,
        "silenced": state.silenced,
    })

async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """Handle WebSocket connections from the avatar frontend"""
    ws = web.WebSocketResponse(heartbeat=20.0)  # Ping every 20s to keep connection alive
    await ws.prepare(request)
    
    clients.add(ws)
    print(f"[Avatar] Client connected. Total: {len(clients)}")
    await _send_state_snapshot(ws)
    
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                # Handle incoming messages from frontend
                data = json.loads(msg.data)
                msg_type = data.get('type', '')
                
                if msg_type == MessageType.MODE_CHANGE.value:
                    mode_str = data.get('mode', 'active')
                    try:
                        _get_state().mode = CompanionMode(mode_str)
                        await _broadcast_mode_change()
                    except ValueError:
                        print(f"[Avatar] Unknown mode: {mode_str}")
                        
                elif msg_type == MessageType.TOGGLE_SILENCE.value:
                    toggle_chat_silence()
                    await _broadcast_silence_state()
                    
                elif msg_type == MessageType.SET_SILENCE.value:
                    set_chat_silenced(data.get('silenced', False))
                    await _broadcast_silence_state()
                
                elif msg_type == MessageType.READ_PAUSE.value:
                    # Q key pressed - pause read-aloud immediately
                    try:
                        from shared import get_read_aloud_manager
                        from server.process.asr_func.asr_vad import get_interrupt_flag
                        read_aloud = get_read_aloud_manager()
                        if read_aloud.state.is_reading:
                            # Set interrupt flag to stop audio playback immediately
                            get_interrupt_flag().set()
                            read_aloud.request_pause()
                            print("[Avatar] Read-aloud pause requested + audio interrupted (Q key)")
                        else:
                            print("[Avatar] Not currently reading")
                    except Exception as e:
                        print(f"[Avatar] Read pause error: {e}")
                
                elif msg_type == MessageType.READ_RESUME.value:
                    # R key pressed - resume read-aloud
                    try:
                        from shared import get_read_aloud_manager
                        read_aloud = get_read_aloud_manager()
                        if read_aloud.state.is_paused:
                            read_aloud.resume()
                            print("[Avatar] Read-aloud resumed (R key)")
                        else:
                            print("[Avatar] Not currently paused")
                    except Exception as e:
                        print(f"[Avatar] Read resume error: {e}")

                elif msg_type == 'audio_config':
                    # Feature #24: Update audio analyzer config from Unity
                    if _audio_analyzer:
                        _audio_analyzer.update_config(
                            sound_threshold=data.get('sound_threshold'),
                            filter_apps=data.get('filter_apps')
                        )

                elif msg_type == MessageType.SWITCH_VOICE.value:
                    # Switch RVC voice model at runtime
                    voice_name = data.get('voice', '').strip()
                    if voice_name:
                        try:
                            from server.process.tts_func.rvc_convert import switch_voice
                            result = switch_voice(voice_name)
                            await ws.send_json({
                                "type": "voice_changed",
                                "ok": result["ok"],
                                "voice": result["voice"],
                                "error": result.get("error"),
                            })
                            if result["ok"]:
                                print(f"[Avatar] Voice switched to: {result['voice']}")
                            else:
                                print(f"[Avatar] Voice switch failed: {result.get('error')}")
                        except Exception as e:
                            await ws.send_json({
                                "type": "voice_changed",
                                "ok": False,
                                "voice": "",
                                "error": str(e),
                            })
                            print(f"[Avatar] Voice switch error: {e}")

                elif msg_type == MessageType.LIST_VOICES.value:
                    # Return available RVC voice models
                    try:
                        from server.process.tts_func.rvc_convert import list_voices
                        voices = list_voices()
                        await ws.send_json({
                            "type": "voice_list",
                            "voices": voices,
                        })
                    except Exception as e:
                        await ws.send_json({
                            "type": "voice_list",
                            "voices": [],
                            "error": str(e),
                        })

                elif msg_type == 'selected_text':
                    # Browser extension: user highlighted text in browser
                    text = data.get('text', '').strip()
                    if text:
                        _get_state().browser_selected_text = text

                elif msg_type == MessageType.SHUTDOWN.value:
                    # Unity is closing — signal the backend to exit
                    print("[Avatar] Shutdown requested by Unity frontend")
                    _get_state().shutdown_requested = True
                    
                else:
                    print(f"[Avatar] Received: {data}")
                    
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"[Avatar] WebSocket error: {ws.exception()}")
    finally:
        clients.discard(ws)
        print(f"[Avatar] Client disconnected. Total: {len(clients)}")
    
    return ws


async def broadcast(message: Dict[str, Any]) -> None:
    """Send a message to all connected clients"""
    if not clients:
        return
    
    data = json.dumps(message)
    await asyncio.gather(*[
        client.send_str(data) 
        for client in clients 
        if not client.closed
    ], return_exceptions=True)


async def speak_start(text: Optional[str] = None) -> None:
    """Notify frontend that TTS is starting"""
    await broadcast({
        "type": MessageType.SPEAK_START.value,
        "text": text
    })


async def speak_end() -> None:
    """Notify frontend that TTS has finished"""
    await broadcast({
        "type": MessageType.SPEAK_END.value
    })


async def set_emotion(emotion: str) -> None:
    """Set avatar emotion (happy, sad, angry, surprised)"""
    await broadcast({
        "type": MessageType.EMOTION.value,
        "emotion": emotion
    })


async def send_audio_data(data: Dict[str, Any]) -> None:
    """Send audio analysis data to all clients"""
    await broadcast({
        "type": MessageType.AUDIO_ANALYSIS.value,
        **data
    })


async def send_read_highlight(sentence: str, word_timings: list, sentence_index: int = 0) -> None:
    """Send read-aloud highlight data to all clients (browser extension)."""
    await broadcast({
        "type": MessageType.READ_HIGHLIGHT.value,
        "sentence": sentence,
        "sentence_index": sentence_index,
        "word_timings": word_timings,
    })


async def send_read_clear() -> None:
    """Clear read-aloud highlights."""
    await broadcast({
        "type": MessageType.READ_CLEAR.value,
    })


async def send_debug_status(status: str, user_text: str = "", response_text: str = "") -> None:
    """Send debug status info to all clients for the on-screen overlay."""
    await broadcast({
        "type": MessageType.DEBUG_STATUS.value,
        "status": status,
        "user_text": user_text,
        "response_text": response_text,
    })


async def _audio_broadcast_loop() -> None:
    """Continuously broadcast audio analysis data"""
    global _audio_analyzer
    
    # Use configured frame rate
    frame_delay = 1.0 / _config.audio.audio_broadcast_fps
    
    while True:
        if _audio_analyzer and clients:
            analysis = _audio_analyzer.get_analysis()
            await send_audio_data(analysis)
        await asyncio.sleep(frame_delay)


def start_audio_analyzer() -> bool:
    """Start the system audio analyzer"""
    global _audio_analyzer, _audio_broadcast_task
    
    try:
        # Import from the same directory
        client_dir = _config.paths.client_dir
        if str(client_dir) not in sys.path:
            sys.path.insert(0, str(client_dir))
        
        from audio_analyzer import SystemAudioAnalyzer

        # Read preferred loopback device from character_config.yaml (if set)
        preferred_device = ""
        try:
            from server.annabeth_config import load_config as _load_cfg
            _cfg = _load_cfg()
            preferred_device = (
                (_cfg.get("audio_capture") or _cfg.get("audio") or {})
                .get("loopback_device_name", "")
            ).strip()
        except Exception:
            pass

        _audio_analyzer = SystemAudioAnalyzer(preferred_device_name=preferred_device)
        if _audio_analyzer.start():
            print("[Avatar] System audio analyzer started")
            return True
        else:
            print("[Avatar] Failed to start audio analyzer (no loopback device)")
            _audio_analyzer = None  # Don't broadcast zeros if capture never started
            return False
    except ImportError as e:
        print(f"[Avatar] Audio analyzer not available: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[Avatar] Error starting audio analyzer: {e}")
        import traceback
        traceback.print_exc()
        return False


def stop_audio_analyzer() -> None:
    """Stop the audio analyzer"""
    global _audio_analyzer
    if _audio_analyzer:
        _audio_analyzer.stop()
        _audio_analyzer = None


async def index_handler(_request: web.Request) -> web.FileResponse:
    """Serve the main HTML page"""
    html_path = _config.paths.client_dir / "index.html"
    return web.FileResponse(html_path)


async def companion_handler(_request: web.Request) -> web.FileResponse:
    """Serve the desktop companion HTML page (transparent, minimal UI)"""
    html_path = _config.paths.client_dir / "companion.html"
    return web.FileResponse(html_path)


def create_app(repo_root: Optional[Path] = None) -> web.Application:
    """Create the aiohttp application"""
    if repo_root is None:
        repo_root = _config.paths.project_root
    
    app = web.Application()
    
    # Routes
    app.router.add_get('/', index_handler)
    app.router.add_get('/companion', companion_handler)
    app.router.add_get('/ws', websocket_handler)
    
    # Static file routes for models
    models_path = _config.paths.models_dir
    if models_path.exists():
        app.router.add_static('/models', models_path, show_index=True)
    
    # Static file routes for VRMA animations
    animations_path = _config.paths.animations_dir
    if animations_path.exists():
        app.router.add_static('/animations', animations_path, show_index=True)
    
    # Static files for client assets
    client_path = _config.paths.client_dir
    if client_path.exists():
        app.router.add_static('/client', client_path, show_index=True)
    
    return app


async def run_server(host: Optional[str] = None, port: Optional[int] = None) -> web.AppRunner:
    """Run the avatar server"""
    if host is None:
        host = _config.server.avatar_host
    if port is None:
        port = _config.server.avatar_port
        
    app = create_app()
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, host, port)
    await site.start()
    
    vrm_path = _config.paths.vrm_model_path
    print(f"[Avatar] Server running at http://localhost:{port}")
    print(f"[Avatar] Open http://localhost:{port} in your browser")
    print(f"[Avatar] VRM path: {vrm_path}")
    
    return runner


# Global reference for the server
_server_runner: Optional[web.AppRunner] = None
_server_task: Optional["asyncio.Task[None]"] = None


async def _emotion_broadcast_loop() -> None:
    """Broadcast the dominant emotion to all clients every 5 seconds."""
    while True:
        if clients:
            try:
                from server.process.memory.emotion_state import get_dominant_emotion
                dominant = get_dominant_emotion()
                await broadcast({
                    "type": MessageType.EMOTION.value,
                    "emotion": dominant,
                })
            except Exception:
                pass
        await asyncio.sleep(5.0)


async def start_avatar_server() -> web.AppRunner:
    """Start the avatar server (call from main_chat.py)"""
    global _server_runner, _audio_broadcast_task
    _server_runner = await run_server()
    
    # Start audio analyzer
    start_audio_analyzer()
    
    # Start audio broadcast loop
    _audio_broadcast_task = asyncio.create_task(_audio_broadcast_loop())

    # Start emotion decay loop (runs in a background daemon thread)
    try:
        from server.process.memory.emotion_state import start_decay_loop
        start_decay_loop(interval_seconds=60)
    except Exception as e:
        print(f"[Avatar] Emotion decay loop start failed (non-fatal): {e}")

    # Start periodic emotion broadcast (every 5 s)
    asyncio.create_task(_emotion_broadcast_loop())

    return _server_runner


def get_avatar_api() -> Dict[str, Any]:
    """Get the avatar control functions"""
    return {
        'speak_start': speak_start,
        'speak_end': speak_end,
        'set_emotion': set_emotion,
        'broadcast': broadcast,
        'send_audio_data': send_audio_data,
        'send_read_highlight': send_read_highlight,
        'send_read_clear': send_read_clear,
        'send_debug_status': send_debug_status,
        'start_audio_analyzer': start_audio_analyzer,
        'stop_audio_analyzer': stop_audio_analyzer,
        'get_current_mode': get_current_mode,
        'is_listening_paused': is_listening_paused,
        'is_chat_silenced': is_chat_silenced,
        'toggle_chat_silence': toggle_chat_silence,
        'set_chat_silenced': set_chat_silenced,
        # Add new state accessors
        'get_state': _get_state,
        'get_config': lambda: _config,
    }


if __name__ == "__main__":
    # Standalone mode for testing
    async def main():
        runner = await run_server()
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n[Avatar] Shutting down...")
        finally:
            await runner.cleanup()
    
    asyncio.run(main())
