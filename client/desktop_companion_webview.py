"""
Desktop Companion using PyQt6 WebEngine.
Embeds the three-vrm HTML viewer in a transparent, always-on-top window.
Includes a local HTTP server for serving files.
Supports global hotkeys for mode switching.
"""
import sys
import os
import threading
import http.server
import socketserver
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow, QSystemTrayIcon, QMenu
from PyQt6.QtCore import Qt, QUrl, QPoint, QTimer, QObject, pyqtSlot, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QAction, QKeySequence, QShortcut
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PyQt6.QtWebChannel import QWebChannel

# Try to import keyboard for global hotkeys
try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    print("Note: 'keyboard' module not installed. Global hotkeys disabled.")
    print("Install with: pip install keyboard")

# Try to import audio analyzer
try:
    from audio_analyzer import SystemAudioAnalyzer
    HAS_AUDIO_ANALYZER = True
except ImportError:
    HAS_AUDIO_ANALYZER = False
    print("Note: Audio analyzer not available. Dance modes won't react to music.")


# Companion modes
class CompanionMode:
    ACTIVE = "active"           # Normal mode - listening and responding
    IDLE = "idle"               # Idle mode - just vibing, not listening
    DANCE_BEAT = "dance_beat"   # Beat-reactive dance (procedural)
    DANCE_FULL = "dance_full"   # Full-body dance animation (VRMA)


# Simple HTTP server for serving local files
class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that doesn't log to console."""
    
    def log_message(self, format, *args):
        pass  # Suppress logging
    
    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()


def start_http_server(directory: str, port: int = 8765):
    """Start HTTP server in background thread."""
    os.chdir(directory)
    handler = QuietHTTPHandler
    
    # Allow port reuse to avoid "Address already in use" errors
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"HTTP server running at http://localhost:{port}")
        httpd.serve_forever()

class TransparentWebEnginePage(QWebEnginePage):
    """Custom page with transparent background."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Enable local file access for loading VRM models
        self.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        self.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True
        )
    
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        """Capture JavaScript console messages."""
        # Show all JS console messages for debugging
        print(f"js: {message}")


class PyBridge(QObject):
    """Bridge object exposed to JavaScript for window control."""
    
    # Signal to notify JS of mode changes
    modeChanged = pyqtSignal(str)
    
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.current_mode = CompanionMode.ACTIVE
        self.chat_silenced = False  # S key toggles this
    
    @pyqtSlot(int, int)
    def moveWindow(self, deltaX, deltaY):
        """Move the window by delta amount."""
        pos = self.window.pos()
        self.window.move(pos.x() + deltaX, pos.y() + deltaY)
    
    @pyqtSlot(result=str)
    def getMode(self):
        """Get current companion mode."""
        return self.current_mode
    
    @pyqtSlot(str)
    def setMode(self, mode):
        """Set companion mode."""
        self.current_mode = mode
        print(f"Mode changed to: {mode}")
    
    def cycleMode(self):
        """Cycle through modes."""
        modes = [CompanionMode.ACTIVE, CompanionMode.IDLE, CompanionMode.DANCE_BEAT, CompanionMode.DANCE_FULL]
        current_idx = modes.index(self.current_mode) if self.current_mode in modes else 0
        next_idx = (current_idx + 1) % len(modes)
        self.current_mode = modes[next_idx]
        # Notify JS of mode change (safely)
        self._safe_set_mode(self.current_mode)
        print(f"Mode cycled to: {self.current_mode}")
        return self.current_mode
    
    def _safe_set_mode(self, mode):
        """Safely call setCompanionMode, waiting if needed."""
        js_code = f"""
            if (typeof window.setCompanionMode === 'function') {{
                window.setCompanionMode('{mode}');
            }} else {{
                setTimeout(function() {{
                    if (typeof window.setCompanionMode === 'function') {{
                        window.setCompanionMode('{mode}');
                    }}
                }}, 500);
            }}
        """
        self.window.web_view.page().runJavaScript(js_code)


class DesktopCompanionWindow(QMainWindow):
    """Transparent frameless window with embedded web VRM viewer."""
    
    def __init__(self):
        super().__init__()
        
        # Audio analyzer for dance modes
        self.audio_analyzer = None
        self.audio_timer = None
        
        # Window flags for transparent, frameless, always-on-top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool  # Hide from taskbar
        )
        
        # Enable transparency
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Set window properties
        self.setWindowTitle("Annabeth - Desktop Companion")
        self.setGeometry(100, 100, 500, 600)
        
        # Create web view
        self.web_view = QWebEngineView()
        self.web_page = TransparentWebEnginePage(self.web_view)
        self.web_view.setPage(self.web_page)
        
        # Make web view background transparent
        self.web_view.setStyleSheet("background: transparent;")
        self.web_page.setBackgroundColor(QColor(0, 0, 0, 0))
        
        # Set up QWebChannel for Python-JavaScript communication
        self.channel = QWebChannel()
        self.bridge = PyBridge(self)
        self.channel.registerObject("pyBridge", self.bridge)
        self.web_page.setWebChannel(self.channel)
        
        # Set as central widget
        self.setCentralWidget(self.web_view)
        
        # Start audio analyzer for dance modes
        self._start_audio_analyzer()
    
    def _start_audio_analyzer(self):
        """Start the system audio analyzer for beat-reactive dance."""
        if not HAS_AUDIO_ANALYZER:
            print("[Companion] Audio analyzer not available")
            return
        
        try:
            self.audio_analyzer = SystemAudioAnalyzer()
            if self.audio_analyzer.start():
                print("[Companion] Audio analyzer started - dance modes will react to music!")
                
                # Set up timer to send audio data to JS (30 FPS)
                self.audio_timer = QTimer(self)
                self.audio_timer.timeout.connect(self._send_audio_data)
                self.audio_timer.start(33)  # ~30 FPS
            else:
                print("[Companion] Failed to start audio analyzer (no loopback device)")
        except Exception as e:
            print(f"[Companion] Error starting audio analyzer: {e}")
    
    def _send_audio_data(self):
        """Send audio analysis data to JavaScript."""
        if not self.audio_analyzer:
            return
        
        # Get current audio analysis
        data = self.audio_analyzer.get_analysis()
        
        # Debug: Log audio data periodically
        if not hasattr(self, '_audio_log_count'):
            self._audio_log_count = 0
        self._audio_log_count += 1
        if self._audio_log_count % 100 == 1:  # Log every ~3 seconds
            print(f"[Audio] bass={data['bass']:.2f} mid={data['mid']:.2f} energy={data['energy']:.2f} beat={data['beat']}")
        
        # Send audio data to JS (always send so dance works even without music)
        js_code = f"""
            if (typeof window.handleAudioAnalysis === 'function') {{
                window.handleAudioAnalysis({{
                    bass: {data['bass']},
                    mid: {data['mid']},
                    high: {data['high']},
                    energy: {data['energy']},
                    beat: {'true' if data['beat'] else 'false'},
                    beatIntensity: {data['beatIntensity']}
                }});
            }}
        """
        self.web_view.page().runJavaScript(js_code)
    
    def closeEvent(self, event):
        """Clean up on close."""
        if self.audio_timer:
            self.audio_timer.stop()
        if self.audio_analyzer:
            self.audio_analyzer.stop()
        super().closeEvent(event)
    
    def mousePressEvent(self, event):
        """Handle mouse press for window dragging."""
        if event.button() == Qt.MouseButton.RightButton:
            self._drag_position = event.globalPosition().toPoint()
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for window dragging."""
        if self._drag_position is not None:
            delta = event.globalPosition().toPoint() - self._drag_position
            self.move(self.pos() + delta)
            self._drag_position = event.globalPosition().toPoint()
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.RightButton:
            self._drag_position = None
        super().mouseReleaseEvent(event)
    
    def keyPressEvent(self, event):
        """Handle key presses."""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_F5:
            # Reload the page
            self.web_view.reload()
            print("Reloading page...")
        elif event.key() == Qt.Key.Key_Space:
            # Cycle modes with spacebar when window focused
            self.bridge.cycleMode()
        elif event.key() == Qt.Key.Key_D:
            # D key = toggle dance mode
            self._toggle_dance()
        elif event.key() == Qt.Key.Key_S:
            # S key = toggle silence (pause chat listening/responding)
            print("[S KEY] Toggling silence...")
            self._toggle_silence()
        elif event.key() == Qt.Key.Key_1:
            # 1 = Active mode
            self._set_mode(CompanionMode.ACTIVE)
        elif event.key() == Qt.Key.Key_2:
            # 2 = Idle mode  
            self._set_mode(CompanionMode.IDLE)
        elif event.key() == Qt.Key.Key_3:
            # 3 = Dance Beat mode
            self._set_mode(CompanionMode.DANCE_BEAT)
        elif event.key() == Qt.Key.Key_4:
            # 4 = Dance Full mode
            self._set_mode(CompanionMode.DANCE_FULL)
    
    def _toggle_dance(self):
        """Toggle between dance modes."""
        if self.bridge.current_mode == CompanionMode.DANCE_BEAT:
            self.bridge.current_mode = CompanionMode.DANCE_FULL
        elif self.bridge.current_mode == CompanionMode.DANCE_FULL:
            self.bridge.current_mode = CompanionMode.ACTIVE
        else:
            self.bridge.current_mode = CompanionMode.DANCE_BEAT
        self._notify_mode()
    
    def _toggle_silence(self):
        """Toggle chat silence (S key) - pauses listening/responding."""
        # Toggle local state first for immediate UI feedback
        self.bridge.chat_silenced = not getattr(self.bridge, 'chat_silenced', False)
        status = "🔇 SILENCED" if self.bridge.chat_silenced else "🔊 LISTENING"
        print(f"Chat: {status}")
        
        # Send toggle_silence message to server via WebSocket
        silenced_state = "true" if self.bridge.chat_silenced else "false"
        js_code = f"""
            (function() {{
                if (window.ws && window.ws.readyState === WebSocket.OPEN) {{
                    window.ws.send(JSON.stringify({{ type: 'toggle_silence' }}));
                    console.log('✅ Sent toggle_silence to server - silenced: {silenced_state}');
                    return 'sent';
                }} else {{
                    console.log('❌ WebSocket not connected (state: ' + (window.ws ? window.ws.readyState : 'null') + ')');
                    return 'not_connected';
                }}
            }})();
        """
        self.web_view.page().runJavaScript(js_code, lambda result: print(f"[Silence] WebSocket: {result}"))
    
    def _toggle_active(self):
        """Toggle between active and idle."""
        if self.bridge.current_mode == CompanionMode.ACTIVE:
            self.bridge.current_mode = CompanionMode.IDLE
        else:
            self.bridge.current_mode = CompanionMode.ACTIVE
        self._notify_mode()
    
    def _set_mode(self, mode):
        """Set a specific mode."""
        self.bridge.current_mode = mode
        self._notify_mode()
    
    def _notify_mode(self):
        """Notify JavaScript of mode change."""
        mode = self.bridge.current_mode
        # Use a safer approach that checks if function exists
        js_code = f"""
            if (typeof window.setCompanionMode === 'function') {{
                window.setCompanionMode('{mode}');
            }} else {{
                console.log('Waiting for setCompanionMode...');
                setTimeout(function() {{
                    if (typeof window.setCompanionMode === 'function') {{
                        window.setCompanionMode('{mode}');
                    }}
                }}, 500);
            }}
        """
        self.web_view.page().runJavaScript(js_code)
        print(f"Mode: {mode}")


class HotkeyManager(QObject):
    """Manages global hotkeys using the keyboard library."""
    
    # Signal emitted when a hotkey is pressed (thread-safe)
    hotkeyPressed = pyqtSignal(str)
    
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.hotkeys = []
        
        # Connect signal to slot for thread-safe execution
        self.hotkeyPressed.connect(self._handle_hotkey)
        
        if HAS_KEYBOARD:
            self._register_hotkeys()
    
    def _register_hotkeys(self):
        """Register global hotkeys."""
        # Ctrl+Shift+A: Toggle Active/Idle mode
        keyboard.add_hotkey('ctrl+shift+a', lambda: self.hotkeyPressed.emit('toggle_active'))
        self.hotkeys.append('ctrl+shift+a')
        
        # Ctrl+Shift+D: Toggle Dance mode (cycles between beat and full)
        keyboard.add_hotkey('ctrl+shift+d', lambda: self.hotkeyPressed.emit('toggle_dance'))
        self.hotkeys.append('ctrl+shift+d')
        
        # Ctrl+Shift+M: Cycle through all modes
        keyboard.add_hotkey('ctrl+shift+m', lambda: self.hotkeyPressed.emit('cycle_modes'))
        self.hotkeys.append('ctrl+shift+m')
        
        print("Global hotkeys registered:")
        print("  Ctrl+Shift+A: Toggle Active/Idle")
        print("  Ctrl+Shift+D: Toggle Dance modes")
        print("  Ctrl+Shift+M: Cycle all modes")
    
    def _handle_hotkey(self, action):
        """Handle hotkey action on the main thread (slot)."""
        if action == 'toggle_active':
            self._toggle_active()
        elif action == 'toggle_dance':
            self._toggle_dance()
        elif action == 'cycle_modes':
            self._cycle_modes()
    
    def _toggle_active(self):
        """Toggle between active and idle."""
        if self.bridge.current_mode == CompanionMode.ACTIVE:
            self.bridge.current_mode = CompanionMode.IDLE
        else:
            self.bridge.current_mode = CompanionMode.ACTIVE
        self._notify_mode_change()
    
    def _toggle_dance(self):
        """Toggle between dance modes."""
        if self.bridge.current_mode == CompanionMode.DANCE_BEAT:
            self.bridge.current_mode = CompanionMode.DANCE_FULL
        elif self.bridge.current_mode == CompanionMode.DANCE_FULL:
            self.bridge.current_mode = CompanionMode.IDLE
        else:
            self.bridge.current_mode = CompanionMode.DANCE_BEAT
        self._notify_mode_change()
    
    def _cycle_modes(self):
        """Cycle through all modes."""
        self.bridge.cycleMode()
    
    def _notify_mode_change(self):
        """Notify JavaScript of mode change."""
        mode = self.bridge.current_mode
        self.bridge._safe_set_mode(mode)
        print(f"Mode: {mode}")
    
    def cleanup(self):
        """Remove all hotkeys."""
        if HAS_KEYBOARD:
            for hotkey in self.hotkeys:
                try:
                    keyboard.remove_hotkey(hotkey)
                except:
                    pass


def main():
    # High DPI support
    if hasattr(Qt, 'HighDpiScaleFactorRoundingPolicy'):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    
    app = QApplication(sys.argv)
    
    # Start HTTP server for serving files
    project_root = Path(__file__).parent.parent.absolute()
    server_port = 8766  # Use different port to avoid conflicts
    
    server_thread = threading.Thread(
        target=start_http_server,
        args=(str(project_root), server_port),
        daemon=True
    )
    server_thread.start()
    
    # Give server time to start
    import time
    time.sleep(0.3)
    
    # Create window that loads from HTTP server
    window = DesktopCompanionWindow()
    
    # Load the companion.html from HTTP server
    url = f"http://localhost:{server_port}/client/companion.html"
    print(f"Loading: {url}")
    window.web_view.load(QUrl(url))
    
    window.show()
    
    # Setup global hotkey manager
    hotkey_manager = HotkeyManager(window.bridge)
    
    print("=" * 50)
    print("Desktop Companion (WebEngine)")
    print("=" * 50)
    print("Controls (when window focused):")
    print("  S: Toggle chat silence (mute listening)")
    print("  D: Toggle dance modes")
    print("  1-4: Quick mode select")
    print("  Space: Cycle all modes")
    print("  F5: Reload page")
    print("  ESC: Close")
    print("-" * 50)
    print("Right-click + drag: Move window")
    print("=" * 50)
    
    result = app.exec()
    hotkey_manager.cleanup()
    sys.exit(result)


if __name__ == "__main__":
    main()