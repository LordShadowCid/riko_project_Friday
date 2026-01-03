"""
Test script for PyQt6 transparent window with OpenGL.
This tests if WA_TranslucentBackground works on Windows.
"""
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
import math

class TransparentGLWidget(QOpenGLWidget):
    """OpenGL widget with transparent background."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        
        # Timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)  # ~60 FPS
        
    def initializeGL(self):
        """Set up OpenGL context."""
        glClearColor(0.0, 0.0, 0.0, 0.0)  # Transparent clear color
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
    def resizeGL(self, w, h):
        """Handle resize."""
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h if h > 0 else 1
        glOrtho(-aspect, aspect, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        
    def paintGL(self):
        """Render a spinning triangle."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        glRotatef(self.angle, 0, 0, 1)
        
        # Draw a colorful triangle
        glBegin(GL_TRIANGLES)
        glColor4f(1.0, 0.0, 0.0, 0.9)  # Red with some transparency
        glVertex2f(0.0, 0.6)
        glColor4f(0.0, 1.0, 0.0, 0.9)  # Green
        glVertex2f(-0.5, -0.3)
        glColor4f(0.0, 0.0, 1.0, 0.9)  # Blue
        glVertex2f(0.5, -0.3)
        glEnd()
        
        # Draw a character silhouette (simple)
        glColor4f(0.8, 0.6, 0.8, 1.0)  # Light purple
        self.draw_circle(0, 0.2, 0.15)  # Head
        
        # Body
        glColor4f(0.6, 0.4, 0.7, 1.0)
        glBegin(GL_POLYGON)
        glVertex2f(-0.12, 0.05)
        glVertex2f(0.12, 0.05)
        glVertex2f(0.15, -0.5)
        glVertex2f(-0.15, -0.5)
        glEnd()
        
    def draw_circle(self, cx, cy, r, segments=32):
        """Draw a filled circle."""
        glBegin(GL_POLYGON)
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            glVertex2f(cx + x, cy + y)
        glEnd()
        
    def animate(self):
        """Update animation."""
        self.angle = (self.angle + 1) % 360
        self.update()


class TransparentWindow(QMainWindow):
    """Main window with transparency enabled."""
    
    def __init__(self):
        super().__init__()
        
        # Window flags for transparency
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        
        # Enable transparency
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Set size and position
        self.setGeometry(100, 100, 400, 500)
        
        # Central widget
        central = QWidget()
        central.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCentralWidget(central)
        
        # Layout
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add OpenGL widget
        self.gl_widget = TransparentGLWidget()
        layout.addWidget(self.gl_widget)
        
        # Status label
        self.status_label = QLabel("🎭 Transparency Test - Drag to move, ESC to close")
        self.status_label.setStyleSheet("""
            QLabel {
                color: white;
                background: rgba(0, 0, 0, 0.5);
                padding: 5px;
                border-radius: 5px;
                font-size: 11px;
            }
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # For dragging
        self.drag_position = None
        
    def mousePressEvent(self, event):
        """Start drag."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            
    def mouseMoveEvent(self, event):
        """Handle drag."""
        if event.buttons() == Qt.MouseButton.LeftButton and self.drag_position:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
            
    def keyPressEvent(self, event):
        """Handle key presses."""
        if event.key() == Qt.Key.Key_Escape:
            self.close()


def main():
    app = QApplication(sys.argv)
    
    # Enable OpenGL
    from PyQt6.QtGui import QSurfaceFormat
    fmt = QSurfaceFormat()
    fmt.setAlphaBufferSize(8)  # Request alpha channel
    fmt.setSamples(4)  # Antialiasing
    QSurfaceFormat.setDefaultFormat(fmt)
    
    window = TransparentWindow()
    window.show()
    
    print("=" * 50)
    print("PyQt6 Transparent OpenGL Test")
    print("=" * 50)
    print("If you see a spinning triangle and character")
    print("with a TRANSPARENT background (you can see your")
    print("desktop through), then transparency is WORKING!")
    print("")
    print("If background is BLACK or WHITE, transparency")
    print("is NOT working with OpenGL on this system.")
    print("")
    print("Press ESC to close, drag to move.")
    print("=" * 50)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
