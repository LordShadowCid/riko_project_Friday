"""
Avatar WebSocket Server
Serves the VRM model and sends animation commands to the web frontend
"""
import asyncio
import json
import os
from pathlib import Path
from aiohttp import web
import aiohttp

# Connected WebSocket clients
clients = set()

async def websocket_handler(request):
    """Handle WebSocket connections from the avatar frontend"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    clients.add(ws)
    print(f"[Avatar] Client connected. Total: {len(clients)}")
    
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                # Handle incoming messages from frontend if needed
                data = json.loads(msg.data)
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


async def index_handler(request):
    """Serve the main HTML page"""
    html_path = Path(__file__).parent / "index.html"
    return web.FileResponse(html_path)


def create_app(repo_root: Path = None):
    """Create the aiohttp application"""
    if repo_root is None:
        repo_root = Path(__file__).parent.parent
    
    app = web.Application()
    
    # Routes
    app.router.add_get('/', index_handler)
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
    global _server_runner
    _server_runner = await run_server()
    return _server_runner


def get_avatar_api():
    """Get the avatar control functions"""
    return {
        'speak_start': speak_start,
        'speak_end': speak_end,
        'set_emotion': set_emotion,
        'broadcast': broadcast
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
