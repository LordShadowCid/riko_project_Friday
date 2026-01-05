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

# Get shared state instance
_state = get_companion_state()
_config = get_config()


def get_current_mode() -> CompanionMode:
    """Get the current companion mode."""
    return _state.mode


def is_chat_silenced() -> bool:
    """Check if chat is silenced (S key toggle)."""
    return _state.silenced


def set_chat_silenced(silenced: bool) -> None:
    """Set chat silence state."""
    _state.silenced = silenced


def toggle_chat_silence() -> bool:
    """Toggle chat silence on/off."""
    return _state.toggle_silence()


def is_listening_paused() -> bool:
    """Check if listening should be paused (silenced OR not in active mode)."""
    return _state.is_listening_paused()

async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """Handle WebSocket connections from the avatar frontend"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    clients.add(ws)
    print(f"[Avatar] Client connected. Total: {len(clients)}")
    
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                # Handle incoming messages from frontend
                data = json.loads(msg.data)
                msg_type = data.get('type', '')
                
                if msg_type == MessageType.MODE_CHANGE.value:
                    mode_str = data.get('mode', 'active')
                    try:
                        _state.mode = CompanionMode(mode_str)
                    except ValueError:
                        print(f"[Avatar] Unknown mode: {mode_str}")
                        
                elif msg_type == MessageType.TOGGLE_SILENCE.value:
                    toggle_chat_silence()
                    
                elif msg_type == MessageType.SET_SILENCE.value:
                    set_chat_silenced(data.get('silenced', False))
                    
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
        
        _audio_analyzer = SystemAudioAnalyzer()
        if _audio_analyzer.start():
            print("[Avatar] System audio analyzer started")
            return True
        else:
            print("[Avatar] Failed to start audio analyzer (no loopback device)")
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


async def start_avatar_server() -> web.AppRunner:
    """Start the avatar server (call from main_chat.py)"""
    global _server_runner, _audio_broadcast_task
    _server_runner = await run_server()
    
    # Start audio analyzer
    start_audio_analyzer()
    
    # Start audio broadcast loop
    _audio_broadcast_task = asyncio.create_task(_audio_broadcast_loop())
    
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
        'start_audio_analyzer': start_audio_analyzer,
        'stop_audio_analyzer': stop_audio_analyzer,
        'get_current_mode': get_current_mode,
        'is_listening_paused': is_listening_paused,
        'is_chat_silenced': is_chat_silenced,
        'toggle_chat_silence': toggle_chat_silence,
        'set_chat_silenced': set_chat_silenced,
        # Add new state accessors
        'get_state': lambda: _state,
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
