"""
Avatar WebSocket Server
Serves the VRM model and sends animation commands to the web frontend
Includes system audio analysis for beat-reactive dancing
"""
import asyncio
import json
import os
from pathlib import Path
from aiohttp import web
import aiohttp

# Connected WebSocket clients
clients = set()

# Audio analyzer reference
_audio_analyzer = None
_audio_broadcast_task = None

# Current companion mode (active, idle, dance_beat, dance_full)
_current_mode = "active"

# Chat silence state (separate from mode - can dance while silenced)
_chat_silenced = False

def get_current_mode():
    """Get the current companion mode."""
    return _current_mode

def is_chat_silenced():
    """Check if chat is silenced (S key toggle)."""
    return _chat_silenced

def set_chat_silenced(silenced: bool):
    """Set chat silence state."""
    global _chat_silenced
    _chat_silenced = silenced
    print(f"[Avatar] Chat silenced: {silenced}")

def toggle_chat_silence():
    """Toggle chat silence on/off."""
    global _chat_silenced
    _chat_silenced = not _chat_silenced
    print(f"[Avatar] Chat silenced: {_chat_silenced}")
    return _chat_silenced

def is_listening_paused():
    """Check if listening should be paused (silenced OR not in active mode)."""
    return _chat_silenced or _current_mode != "active"

async def websocket_handler(request):
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
                if data.get('type') == 'mode_change':
                    global _current_mode
                    _current_mode = data.get('mode', 'active')
                    print(f"[Avatar] Mode changed to: {_current_mode}")
                elif data.get('type') == 'toggle_silence':
                    toggle_chat_silence()
                elif data.get('type') == 'set_silence':
                    set_chat_silenced(data.get('silenced', False))
                else:
                    print(f"[Avatar] Received: {data}")
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"[Avatar] WebSocket error: {ws.exception()}")
    finally:
        clients.discard(ws)
        print(f"[Avatar] Client disconnected. Total: {len(clients)}")
    
    return ws


async def broadcast(message: dict):
    """Send a message to all connected clients"""
    if not clients:
        return
    
    data = json.dumps(message)
    await asyncio.gather(*[
        client.send_str(data) 
        for client in clients 
        if not client.closed
    ], return_exceptions=True)


async def speak_start(text: str = None):
    """Notify frontend that TTS is starting"""
    await broadcast({
        "type": "speak_start",
        "text": text
    })


async def speak_end():
    """Notify frontend that TTS has finished"""
    await broadcast({
        "type": "speak_end"
    })


async def set_emotion(emotion: str):
    """Set avatar emotion (happy, sad, angry, surprised)"""
    await broadcast({
        "type": "emotion",
        "emotion": emotion
    })


async def send_audio_data(data: dict):
    """Send audio analysis data to all clients"""
    await broadcast({
        "type": "audio_analysis",
        **data
    })


async def _audio_broadcast_loop():
    """Continuously broadcast audio analysis data"""
    global _audio_analyzer
    
    while True:
        if _audio_analyzer and clients:
            analysis = _audio_analyzer.get_analysis()
            await send_audio_data(analysis)
        await asyncio.sleep(0.033)  # ~30 FPS


def start_audio_analyzer():
    """Start the system audio analyzer"""
    global _audio_analyzer, _audio_broadcast_task
    
    try:
        # Import from the same directory
        import sys
        from pathlib import Path
        client_dir = Path(__file__).parent
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


def stop_audio_analyzer():
    """Stop the audio analyzer"""
    global _audio_analyzer
    if _audio_analyzer:
        _audio_analyzer.stop()
        _audio_analyzer = None


async def index_handler(request):
    """Serve the main HTML page"""
    html_path = Path(__file__).parent / "index.html"
    return web.FileResponse(html_path)


async def companion_handler(request):
    """Serve the desktop companion HTML page (transparent, minimal UI)"""
    html_path = Path(__file__).parent / "companion.html"
    return web.FileResponse(html_path)


def create_app(repo_root: Path = None):
    """Create the aiohttp application"""
    if repo_root is None:
        repo_root = Path(__file__).parent.parent
    
    app = web.Application()
    
    # Routes
    app.router.add_get('/', index_handler)
    app.router.add_get('/companion', companion_handler)
    app.router.add_get('/ws', websocket_handler)
    
    # Static file routes for models
    models_path = repo_root / "models"
    if models_path.exists():
        app.router.add_static('/models', models_path, show_index=True)
    
    # Static file routes for VRMA animations
    animations_path = repo_root / "animations"
    if animations_path.exists():
        app.router.add_static('/animations', animations_path, show_index=True)
    
    # Static files for client assets
    client_path = repo_root / "client"
    if client_path.exists():
        app.router.add_static('/client', client_path, show_index=True)
    
    return app


async def run_server(host='0.0.0.0', port=8765):
    """Run the avatar server"""
    repo_root = Path(__file__).parent.parent
    app = create_app(repo_root)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, host, port)
    await site.start()
    
    print(f"[Avatar] Server running at http://localhost:{port}")
    print(f"[Avatar] Open http://localhost:{port} in your browser")
    print(f"[Avatar] VRM path: {repo_root / 'models' / 'vrm' / 'claire_avatar.vrm'}")
    
    return runner


# Global reference for the server
_server_runner = None
_server_task = None


async def start_avatar_server():
    """Start the avatar server (call from main_chat.py)"""
    global _server_runner, _audio_broadcast_task
    _server_runner = await run_server()
    
    # Start audio analyzer
    start_audio_analyzer()
    
    # Start audio broadcast loop
    _audio_broadcast_task = asyncio.create_task(_audio_broadcast_loop())
    
    return _server_runner


def get_avatar_api():
    """Get the avatar control functions"""
    return {
        'speak_start': speak_start,
        'speak_end': speak_end,
        'set_emotion': set_emotion,
        'broadcast': broadcast,
        'send_audio_data': send_audio_data,
        'start_audio_analyzer': start_audio_analyzer,
        'stop_audio_analyzer': stop_audio_analyzer,
        'get_current_mode': get_current_mode,
        'is_listening_paused': is_listening_paused,
        'is_chat_silenced': is_chat_silenced,
        'toggle_chat_silence': toggle_chat_silence,
        'set_chat_silenced': set_chat_silenced
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
