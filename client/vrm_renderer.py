"""
VRM Renderer using PyQt6 and ModernGL.
Renders VRM avatars with transparency support.
"""
import sys
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import moderngl

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QSurfaceFormat

from vrm_loader import VRMLoader, VRMData, VRMMesh


# Vertex shader - handles model/view/projection and basic lighting
VERTEX_SHADER = """
#version 330

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

in vec3 in_position;
in vec3 in_normal;

out vec3 v_normal;

void main() {
    vec4 world_pos = model * vec4(in_position, 1.0);
    gl_Position = projection * view * world_pos;
    v_normal = in_normal;  // Pass through to prevent optimization
}
"""

# Fragment shader - simple solid color for debugging
FRAGMENT_SHADER = """
#version 330

uniform vec4 base_color;

in vec3 v_normal;

out vec4 frag_color;

void main() {
    // Use normal to affect color slightly (prevents optimization)
    vec3 n = normalize(v_normal);
    float shade = 0.5 + 0.5 * n.z;  // Basic lighting based on normal Z
    frag_color = vec4(base_color.rgb * shade, base_color.a);
}
"""


class VRMRenderMesh:
    """Renderable mesh data for OpenGL."""
    
    def __init__(self, ctx: moderngl.Context, mesh: VRMMesh):
        self.name = mesh.name
        self.base_color = mesh.base_color
        
        # Ensure we have normals
        if mesh.normals is None:
            # Generate flat normals
            normals = np.zeros_like(mesh.positions)
            normals[:, 2] = 1.0  # Default forward-facing
        else:
            normals = mesh.normals
        
        # Create vertex buffer (positions + normals interleaved)
        vertex_data = np.hstack([
            mesh.positions.astype(np.float32),
            normals.astype(np.float32)
        ])
        
        self.vbo = ctx.buffer(vertex_data.tobytes())
        
        # Create index buffer if we have indices
        self.ibo = None
        self.vertex_count = len(mesh.positions)
        self.index_count = 0
        
        if mesh.indices is not None:
            indices = mesh.indices.astype(np.uint32)
            self.ibo = ctx.buffer(indices.tobytes())
            self.index_count = len(indices)
        
        self.vao = None
    
    def create_vao(self, ctx: moderngl.Context, program: moderngl.Program):
        """Create vertex array object."""
        if self.ibo:
            self.vao = ctx.vertex_array(
                program,
                [(self.vbo, '3f 3f', 'in_position', 'in_normal')],
                self.ibo
            )
        else:
            self.vao = ctx.vertex_array(
                program,
                [(self.vbo, '3f 3f', 'in_position', 'in_normal')]
            )
    
    def render(self, program: moderngl.Program):
        """Render this mesh."""
        if self.vao is None:
            return
            
        # Set material color
        program['base_color'].value = self.base_color
        
        if self.ibo:
            self.vao.render(moderngl.TRIANGLES)
        else:
            self.vao.render(moderngl.TRIANGLES, vertices=self.vertex_count)


class VRMRenderer(QOpenGLWidget):
    """OpenGL widget for rendering VRM models with transparency."""
    
    def __init__(self, vrm_path: str, parent=None):
        super().__init__(parent)
        self.vrm_path = vrm_path
        self.vrm_data: Optional[VRMData] = None
        self.render_meshes: List[VRMRenderMesh] = []
        
        self.ctx: Optional[moderngl.Context] = None
        self.program: Optional[moderngl.Program] = None
        
        # Camera/view settings - VRM is ~1.6m tall, face at Y~1.6
        self.camera_distance = 3.0
        self.camera_height = 1.4  # Look at chest/face level
        self.rotation = 180.0  # Start facing front (VRM faces -Z)
        self.auto_rotate = False
        
        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)  # ~60 FPS
        
        # Dragging state
        self.last_mouse_pos = None
        
    def initializeGL(self):
        """Initialize OpenGL context and load VRM."""
        # Create moderngl context from Qt's existing context
        # standalone=False tells moderngl to use the current OpenGL context
        self.ctx = moderngl.create_context(require=330)
        
        print(f"OpenGL: {self.ctx.info['GL_VERSION']}")
        print(f"Renderer: {self.ctx.info['GL_RENDERER']}")
        
        # Enable transparency
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Enable depth testing
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        # DISABLE backface culling for debugging (faces might be wound wrong)
        # self.ctx.enable(moderngl.CULL_FACE)
        
        # Create shader program
        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )
        
        # Load VRM
        self._load_vrm()
        
    def _load_vrm(self):
        """Load VRM model and create render meshes."""
        try:
            loader = VRMLoader(self.vrm_path)
            self.vrm_data = loader.load()
            
            # Create render meshes
            for mesh in self.vrm_data.meshes:
                try:
                    render_mesh = VRMRenderMesh(self.ctx, mesh)
                    render_mesh.create_vao(self.ctx, self.program)
                    self.render_meshes.append(render_mesh)
                except Exception as e:
                    print(f"Failed to create render mesh {mesh.name}: {e}")
                    
            print(f"Created {len(self.render_meshes)} render meshes")
            
        except Exception as e:
            print(f"Failed to load VRM: {e}")
            import traceback
            traceback.print_exc()
    
    def resizeGL(self, w, h):
        """Handle resize."""
        self.ctx.viewport = (0, 0, w, h)
        
    def paintGL(self):
        """Render the VRM model."""
        # DEBUG: Clear with visible dark blue background
        self.ctx.clear(0.1, 0.1, 0.3, 1.0)
        
        if not self.render_meshes:
            return
        
        # Create matrices
        width = self.width()
        height = self.height()
        aspect = width / height if height > 0 else 1.0
        
        # Projection matrix (perspective)
        fov = 30.0
        near = 0.1
        far = 100.0
        projection = self._perspective_matrix(fov, aspect, near, far)
        
        # View matrix (camera looking at model)
        import math
        cam_x = math.sin(math.radians(self.rotation)) * self.camera_distance
        cam_z = math.cos(math.radians(self.rotation)) * self.camera_distance
        cam_y = self.camera_height
        
        # Look at the face area (Y~1.5)
        target_y = 1.5
        
        view = self._look_at_matrix(
            eye=(cam_x, cam_y, cam_z),
            target=(0, target_y, 0),
            up=(0, 1, 0)
        )
        
        # Debug output (first frame only)
        if not hasattr(self, '_debug_printed'):
            print(f"Camera: eye=({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}), target=(0, {target_y:.2f}, 0)")
            self._debug_printed = True
        
        # Model matrix (identity for now, model is at origin)
        model = np.eye(4, dtype=np.float32)
        
        # Set uniforms
        self.program['projection'].write(projection.tobytes())
        self.program['view'].write(view.tobytes())
        self.program['model'].write(model.tobytes())
        
        # Render all meshes
        for mesh in self.render_meshes:
            mesh.render(self.program)
    
    def _perspective_matrix(self, fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Create perspective projection matrix (column-major for OpenGL)."""
        import math
        f = 1.0 / math.tan(math.radians(fov) / 2.0)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[3, 2] = -1.0
        m[2, 3] = (2.0 * far * near) / (near - far)
        return m
    
    def _look_at_matrix(self, eye: Tuple, target: Tuple, up: Tuple) -> np.ndarray:
        """Create look-at view matrix."""
        eye = np.array(eye, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)
        
        f = target - eye
        f = f / np.linalg.norm(f)
        
        s = np.cross(f, up)
        s_len = np.linalg.norm(s)
        if s_len > 0:
            s = s / s_len
        else:
            s = np.array([1, 0, 0], dtype=np.float32)
        
        u = np.cross(s, f)
        
        # Build view matrix (column-major for OpenGL)
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = s[0]
        m[1, 0] = s[1]
        m[2, 0] = s[2]
        m[0, 1] = u[0]
        m[1, 1] = u[1]
        m[2, 1] = u[2]
        m[0, 2] = -f[0]
        m[1, 2] = -f[1]
        m[2, 2] = -f[2]
        m[3, 0] = -np.dot(s, eye)
        m[3, 1] = -np.dot(u, eye)
        m[3, 2] = np.dot(f, eye)
        
        return m
    
    def update_animation(self):
        """Update animation state."""
        if self.auto_rotate:
            self.rotation = (self.rotation + 0.5) % 360
        self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse press for rotation."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        """Handle mouse drag for rotation."""
        if self.last_mouse_pos is not None:
            dx = event.pos().x() - self.last_mouse_pos.x()
            self.rotation += dx * 0.5
            self.last_mouse_pos = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self.last_mouse_pos = None
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y() / 120
        self.camera_distance = max(0.5, min(10.0, self.camera_distance - delta * 0.2))
        self.update()


class TransparentVRMWindow(QMainWindow):
    """Transparent window for VRM avatar display."""
    
    def __init__(self, vrm_path: str):
        super().__init__()
        
        # Window flags - DEBUG: removed Tool flag so it shows in taskbar
        # Also removed frameless for debugging
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
        )
        
        # DEBUG: Disable transparency for now to see if window appears
        # self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Set window title so we can find it
        self.setWindowTitle("VRM Renderer - Annabeth")
        
        # Set size and position
        self.setGeometry(100, 100, 400, 500)
        
        # Central widget
        central = QWidget()
        # DEBUG: disabled transparency
        # central.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCentralWidget(central)
        
        # Layout
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add VRM renderer
        self.vrm_renderer = VRMRenderer(vrm_path)
        layout.addWidget(self.vrm_renderer)
        
        # For window dragging
        self.drag_position = None
        
    def mousePressEvent(self, event):
        """Start window drag with right-click."""
        if event.button() == Qt.MouseButton.RightButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        """Handle window drag."""
        if event.buttons() == Qt.MouseButton.RightButton and self.drag_position:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
        else:
            super().mouseMoveEvent(event)
            
    def keyPressEvent(self, event):
        """Handle key presses."""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_R:
            self.vrm_renderer.auto_rotate = not self.vrm_renderer.auto_rotate


def main():
    app = QApplication(sys.argv)
    
    # Set up OpenGL format for transparency
    fmt = QSurfaceFormat()
    fmt.setAlphaBufferSize(8)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    
    # Find VRM file
    vrm_path = "models/vrm/claire_avatar.vrm"
    if len(sys.argv) > 1:
        vrm_path = sys.argv[1]
    
    if not Path(vrm_path).exists():
        print(f"VRM file not found: {vrm_path}")
        return 1
    
    window = TransparentVRMWindow(vrm_path)
    window.show()
    
    print("=" * 50)
    print("VRM Renderer")
    print("=" * 50)
    print("Controls:")
    print("  Left-click + drag: Rotate model")
    print("  Right-click + drag: Move window")
    print("  Mouse wheel: Zoom in/out")
    print("  R: Toggle auto-rotation")
    print("  ESC: Close")
    print("=" * 50)
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
