"""
Desktop Companion using PyQt6 WebEngine.
Embeds the three-vrm HTML viewer in a transparent, always-on-top window.
Includes a local HTTP server for serving files.
"""
import sys
import os
import threading
import http.server
import socketserver
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt, QUrl, QPoint, QTimer, QObject, pyqtSlot
from PyQt6.QtGui import QColor
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PyQt6.QtWebChannel import QWebChannel


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


class PyBridge(QObject):
    """Bridge object exposed to JavaScript for window control."""
    
    def __init__(self, window):
        super().__init__()
        self.window = window
    
    @pyqtSlot(int, int)
    def moveWindow(self, deltaX, deltaY):
        """Move the window by delta amount."""
        pos = self.window.pos()
        self.window.move(pos.x() + deltaX, pos.y() + deltaY)


class DesktopCompanionWindow(QMainWindow):
    """Transparent frameless window with embedded web VRM viewer."""
    
    def __init__(self):
        super().__init__()
        
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
    
    print("=" * 50)
    print("Desktop Companion (WebEngine)")
    print("=" * 50)
    print("Controls:")
    print("  Right-click + drag: Move window")
    print("  F5: Reload page")
    print("  ESC: Close")
    print("=" * 50)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()