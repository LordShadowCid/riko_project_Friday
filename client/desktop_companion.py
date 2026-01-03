"""
Desktop Companion - Transparent, always-on-top VRM avatar window
Uses pywebview to wrap the existing Three.js avatar as a desktop overlay
"""
import asyncio
import threading
import webview
from pathlib import Path
import sys
import time

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
                    break
                except OSError:
                    print(f"[Desktop Companion] Port {port} busy, trying next...")
                    continue
            else:
                print("[Desktop Companion] Could not find an available port!")
                return
            
            # Keep the server running
            while True:
                await asyncio.sleep(1)
        
        try:
            self.loop.run_until_complete(start_server())
        except Exception as e:
            print(f"[Desktop Companion] Server error: {e}")
    
    def start(self):
        """Start the desktop companion"""
        # Start the avatar server in a background thread
        self.server_thread = threading.Thread(target=self._run_server_loop, daemon=True)
        self.server_thread.start()
        
        # Give server time to start and find a port
        time.sleep(2)
        
        # Check if server started
        if self.server_runner is None:
            print("[Desktop Companion] Server failed to start!")
            return
        
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
            except:
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
        webview.start(debug=False)
    
    def stop(self):
        """Stop the desktop companion"""
        if self.window:
            self.window.destroy()
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)


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
