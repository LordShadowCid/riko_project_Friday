"""
Desktop Companion - Transparent, always-on-top VRM avatar window
Uses pywebview to wrap the existing Three.js avatar as a desktop overlay
"""
import asyncio
import contextlib
import threading
import webview
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client.avatar_server import create_app
from aiohttp import web


class DesktopCompanion:
    """Desktop companion window manager"""
    
    def __init__(self, width=400, height=600, x=None, y=None):
        self.width = width
        self.height = height
        self.x = x  # None = right side of screen
        self.y = y  # None = bottom of screen
        self.window = None
        self.server_runner = None
        self.loop = None
        self.server_thread = None
        self.port = 8766  # Use different port to avoid conflicts
        self.server_ready = threading.Event()
        self.server_error = None
        
    def _run_server_loop(self):
        """Run the aiohttp server in a background thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        async def start_server():
            repo_root = Path(__file__).parent.parent
            app = create_app(repo_root)
            
            runner = web.AppRunner(app)
            await runner.setup()
            
            # Try multiple ports if the first one is busy
            for port in [self.port, 8767, 8768, 8769, 8770]:
                try:
                    site = web.TCPSite(runner, '127.0.0.1', port, reuse_address=True)
                    await site.start()
                    self.port = port  # Update to actual port
                    print(f"[Desktop Companion] Server running on http://127.0.0.1:{port}")
                    self.server_runner = runner
                    self.server_ready.set()
                    break
                except OSError:
                    print(f"[Desktop Companion] Port {port} busy, trying next...")
                    continue
            else:
                raise RuntimeError("Could not find an available desktop companion port")
            
            # Keep the server running
            while True:
                await asyncio.sleep(1)
        
        try:
            self.loop.run_until_complete(start_server())
        except Exception as e:
            self.server_error = e
            self.server_ready.set()
            print(f"[Desktop Companion] Server error: {e}")
    
    def start(self):
        """Start the desktop companion"""
        # Start the avatar server in a background thread
        self.server_thread = threading.Thread(target=self._run_server_loop, daemon=True)
        self.server_thread.start()

        if not self.server_ready.wait(timeout=5):
            raise RuntimeError("Desktop companion server timed out during startup")

        if self.server_error is not None:
            raise RuntimeError(f"Desktop companion server failed to start: {self.server_error}") from self.server_error
        
        # Calculate position (bottom-right by default)
        if self.x is None or self.y is None:
            try:
                # Try to get screen size
                import ctypes
                user32 = ctypes.windll.user32
                screen_width = user32.GetSystemMetrics(0)
                screen_height = user32.GetSystemMetrics(1)
                
                if self.x is None:
                    self.x = screen_width - self.width - 50
                if self.y is None:
                    self.y = screen_height - self.height - 100
            except Exception:
                # Fallback position
                self.x = self.x or 1000
                self.y = self.y or 400
        
        # Create the transparent window - simplified, no js_api to avoid recursion
        self.window = webview.create_window(
            title='Annabeth',
            url=f'http://127.0.0.1:{self.port}/companion',
            width=self.width,
            height=self.height,
            x=self.x,
            y=self.y,
            frameless=True,
            on_top=True,
            transparent=True,
            resizable=False,
        )
        
        print(f"[Desktop Companion] Window created at ({self.x}, {self.y})")
        print("[Desktop Companion] Ready!")
        
        # Start the webview - try without specifying gui to let it pick best option
        try:
            webview.start(debug=False)
        finally:
            self.stop()
    
    def stop(self):
        """Stop the desktop companion"""
        if self.window:
            with contextlib.suppress(Exception):
                self.window.destroy()
            self.window = None

        if self.loop and self.server_runner:
            cleanup_future = asyncio.run_coroutine_threadsafe(self.server_runner.cleanup(), self.loop)
            with contextlib.suppress(Exception):
                cleanup_future.result(timeout=5)
            self.server_runner = None

        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop = None

        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)
        self.server_thread = None


def main():
    """Run the desktop companion"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Annabeth Desktop Companion')
    parser.add_argument('--width', type=int, default=400, help='Window width')
    parser.add_argument('--height', type=int, default=600, help='Window height')
    parser.add_argument('--x', type=int, default=None, help='X position (default: right side)')
    parser.add_argument('--y', type=int, default=None, help='Y position (default: bottom)')
    
    args = parser.parse_args()
    
    companion = DesktopCompanion(
        width=args.width,
        height=args.height,
        x=args.x,
        y=args.y
    )
    
    try:
        companion.start()
    except KeyboardInterrupt:
        print("\n[Desktop Companion] Shutting down...")
        companion.stop()


if __name__ == '__main__':
    main()
