"""
VRM Renderer using PyQt6 and PyOpenGL.
Renders VRM avatars with transparency support.
"""
import sys
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from ctypes import c_void_p

from OpenGL.GL import *
from OpenGL.GLU import *

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QSurfaceFormat

from vrm_loader import VRMLoader, VRMData, VRMMesh


# Vertex shader with skeletal animation support
VERTEX_SHADER = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// Bone matrices (max 200 bones for VRM models)
uniform mat4 bone_matrices[200];
uniform int use_skinning;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in ivec4 in_joints;   // Bone indices
layout(location = 4) in vec4 in_weights;   // Bone weights

out vec3 v_normal;
out vec3 v_position;
out vec2 v_uv;

void main() {
    vec4 pos = vec4(in_position, 1.0);
    vec3 norm = in_normal;
    
    // Apply skeletal animation if enabled
    if (use_skinning == 1 && (in_weights.x + in_weights.y + in_weights.z + in_weights.w) > 0.0) {
        mat4 skin_matrix = 
            bone_matrices[in_joints.x] * in_weights.x +
            bone_matrices[in_joints.y] * in_weights.y +
            bone_matrices[in_joints.z] * in_weights.z +
            bone_matrices[in_joints.w] * in_weights.w;
        
        pos = skin_matrix * pos;
        norm = mat3(skin_matrix) * norm;
    }
    
    vec4 world_pos = model * pos;
    gl_Position = projection * view * world_pos;
    v_normal = mat3(model) * norm;
    v_position = world_pos.xyz;
    v_uv = in_uv;
}
"""

# Fragment shader
FRAGMENT_SHADER = """
#version 330 core

uniform vec4 base_color;
uniform vec3 light_dir;
uniform sampler2D tex_diffuse;
uniform int use_texture;

in vec3 v_normal;
in vec3 v_position;
in vec2 v_uv;

out vec4 frag_color;

void main() {
    vec3 n = normalize(v_normal);
    vec3 l = normalize(light_dir);
    
    // Simple diffuse lighting
    float ndotl = max(dot(n, l), 0.0);
    float shade = 0.4 + 0.6 * ndotl;  // Ambient + diffuse
    
    vec4 tex_color;
    if (use_texture == 1) {
        tex_color = texture(tex_diffuse, v_uv);
        // Discard fully transparent pixels
        if (tex_color.a < 0.1) discard;
    } else {
        tex_color = base_color;
    }
    
    vec3 color = tex_color.rgb * shade;
    frag_color = vec4(color, tex_color.a);
}
"""


def compile_shader(source: str, shader_type: int) -> int:
    """Compile a shader from source."""
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    
    # Check for errors
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compilation error: {error}")
    
    return shader


def create_program(vertex_src: str, fragment_src: str) -> int:
    """Create a shader program."""
    vertex_shader = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_src, GL_FRAGMENT_SHADER)
    
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    
    # Check for errors
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program linking error: {error}")
    
    # Clean up shaders (they're now part of the program)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    
    return program


class VRMRenderMesh:
    """Renderable mesh data for OpenGL."""
    
    def __init__(self, mesh: VRMMesh):
        self.name = mesh.name
        self.base_color = mesh.base_color
        self.texture_index = mesh.texture_index
        self.vertex_count = len(mesh.positions)
        self.index_count = 0
        
        # Store data for later GPU upload
        self.positions = mesh.positions.astype(np.float32)
        
        if mesh.normals is not None:
            self.normals = mesh.normals.astype(np.float32)
        else:
            # Generate default normals
            self.normals = np.zeros_like(self.positions)
            self.normals[:, 2] = 1.0
        
        # UVs for texturing
        if mesh.uvs is not None:
            self.uvs = mesh.uvs.astype(np.float32)
        else:
            self.uvs = np.zeros((len(self.positions), 2), dtype=np.float32)
        
        # Joint indices and weights for skinning
        if mesh.joint_indices is not None:
            self.joint_indices = mesh.joint_indices.astype(np.int32)
        else:
            self.joint_indices = np.zeros((len(self.positions), 4), dtype=np.int32)
            
        if mesh.joint_weights is not None:
            self.joint_weights = mesh.joint_weights.astype(np.float32)
        else:
            self.joint_weights = np.zeros((len(self.positions), 4), dtype=np.float32)
        
        self.has_skinning = mesh.joint_indices is not None and mesh.joint_weights is not None
        
        # Debug: check joint index range
        if self.has_skinning:
            max_joint = np.max(self.joint_indices)
            min_joint = np.min(self.joint_indices)
            print(f"    {mesh.name}: joint indices range [{min_joint}, {max_joint}]")
        
        self.indices = None
        if mesh.indices is not None:
            self.indices = mesh.indices.astype(np.uint32)
            self.index_count = len(self.indices)
        
        # OpenGL handles (created later)
        self.vao = None
        self.vbo_positions = None
        self.vbo_normals = None
        self.vbo_uvs = None
        self.vbo_joints = None
        self.vbo_weights = None
        self.ebo = None
        self.gl_texture = None  # OpenGL texture handle
        
    def create_gl_objects(self, textures: List = None):
        """Create OpenGL objects. Must be called with valid GL context."""
        # Create VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Position VBO (location 0)
        self.vbo_positions = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_positions)
        glBufferData(GL_ARRAY_BUFFER, self.positions.nbytes, self.positions, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        # Normal VBO (location 1)
        self.vbo_normals = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
        glBufferData(GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
        
        # UV VBO (location 2)
        self.vbo_uvs = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_uvs)
        glBufferData(GL_ARRAY_BUFFER, self.uvs.nbytes, self.uvs, GL_STATIC_DRAW)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)
        
        # Joint indices VBO (location 3) - integer attribute
        self.vbo_joints = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_joints)
        glBufferData(GL_ARRAY_BUFFER, self.joint_indices.nbytes, self.joint_indices, GL_STATIC_DRAW)
        glVertexAttribIPointer(3, 4, GL_INT, 0, None)  # Note: IPointer for integers
        glEnableVertexAttribArray(3)
        
        # Joint weights VBO (location 4)
        self.vbo_weights = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_weights)
        glBufferData(GL_ARRAY_BUFFER, self.joint_weights.nbytes, self.joint_weights, GL_STATIC_DRAW)
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(4)
        
        # Index buffer
        if self.indices is not None:
            self.ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
        
        # Create texture if available
        if textures and self.texture_index is not None and self.texture_index < len(textures):
            tex_data = textures[self.texture_index]
            if tex_data is not None:
                self.gl_texture = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, self.gl_texture)
                
                # Set texture parameters
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                
                # Upload texture data (RGBA)
                height, width = tex_data.shape[:2]
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, 
                            GL_RGBA, GL_UNSIGNED_BYTE, tex_data)
                glGenerateMipmap(GL_TEXTURE_2D)
                
                glBindTexture(GL_TEXTURE_2D, 0)
        
    def render(self, program: int):
        """Render this mesh."""
        if self.vao is None:
            return
        
        # Set texture or base color
        use_tex_loc = glGetUniformLocation(program, "use_texture")
        if self.gl_texture:
            glUniform1i(use_tex_loc, 1)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.gl_texture)
            tex_loc = glGetUniformLocation(program, "tex_diffuse")
            glUniform1i(tex_loc, 0)
        else:
            glUniform1i(use_tex_loc, 0)
            loc = glGetUniformLocation(program, "base_color")
            glUniform4f(loc, *self.base_color)
        
        glBindVertexArray(self.vao)
        
        if self.ebo is not None:
            glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
            
        glBindVertexArray(0)


class VRMRenderer(QOpenGLWidget):
    """OpenGL widget for rendering VRM models with transparency."""
    
    def __init__(self, vrm_path: str, parent=None):
        super().__init__(parent)
        self.vrm_path = vrm_path
        self.vrm_data: Optional[VRMData] = None
        self.render_meshes: List[VRMRenderMesh] = []
        
        self.program = None
        
        # Camera settings - VRM is ~1.6m tall, face at Y~1.6
        self.camera_distance = 3.0
        self.camera_height = 1.5
        self.rotation = 180.0  # Start facing the front of the model
        self.auto_rotate = False
        
        # Pose/skeletal animation
        self.bone_matrices = None  # Will be 128x4x4 array (final skinning matrices)
        self.bone_local_rotations = None  # Local rotation overrides for poses
        self.inverse_bind_matrices = None  # From VRM
        self.bone_parents = None  # Parent indices for hierarchy
        self.bone_rest_transforms = None  # Rest pose local transforms
        self.node_to_joint_map = {}  # Maps node indices to joint indices
        self.use_skinning = False
        self.current_pose = "idle"  # Current pose name
        
        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)  # ~60 FPS
        
        # Dragging state
        self.last_mouse_pos = None
        
    def initializeGL(self):
        """Initialize OpenGL context and load VRM."""
        print(f"OpenGL Version: {glGetString(GL_VERSION).decode()}")
        print(f"Renderer: {glGetString(GL_RENDERER).decode()}")
        
        # Set clear color - transparent for desktop companion
        glClearColor(0.0, 0.0, 0.0, 0.0)
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Create shader program
        try:
            self.program = create_program(VERTEX_SHADER, FRAGMENT_SHADER)
            print("Shader program created successfully")
        except Exception as e:
            print(f"Failed to create shader program: {e}")
            return
        
        # Load VRM
        self._load_vrm()
        
    def _create_debug_triangle(self):
        """Create a simple triangle to test rendering at VRM model location."""
        # A larger triangle at Y=1.5 (where VRM face is)
        positions = np.array([
            [0.0, 2.0, 0.0],    # Top
            [-0.5, 1.0, 0.0],   # Bottom left
            [0.5, 1.0, 0.0],    # Bottom right
        ], dtype=np.float32)
        
        normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        
        self.debug_vao = glGenVertexArrays(1)
        glBindVertexArray(self.debug_vao)
        
        vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        vbo_norm = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_norm)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
        print("Debug triangle created at Y=1.0-2.0")
        
    def _load_vrm(self):
        """Load VRM model and create render meshes."""
        try:
            loader = VRMLoader(self.vrm_path)
            self.vrm_data = loader.load()
            
            # Get textures
            textures = self.vrm_data.textures
            
            # Create render meshes
            for mesh in self.vrm_data.meshes:
                try:
                    render_mesh = VRMRenderMesh(mesh)
                    render_mesh.create_gl_objects(textures)
                    self.render_meshes.append(render_mesh)
                except Exception as e:
                    print(f"Failed to create render mesh {mesh.name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Count textured and skinned meshes
            textured = sum(1 for m in self.render_meshes if m.gl_texture)
            skinned = sum(1 for m in self.render_meshes if m.has_skinning)
            print(f"Created {len(self.render_meshes)} render meshes ({textured} with textures, {skinned} with skinning)")
            
            # Initialize bone matrices
            self._init_bone_matrices()
            
            # Set initial pose
            self.set_pose("idle")
            
        except Exception as e:
            print(f"Failed to load VRM: {e}")
            import traceback
            traceback.print_exc()
    
    def _init_bone_matrices(self):
        """Initialize bone matrices array from VRM skeleton data."""
        num_bones = len(self.vrm_data.bones)
        
        # Create arrays for 200 bones max
        self.bone_matrices = np.zeros((200, 4, 4), dtype=np.float32)
        self.bone_local_rotations = np.zeros((200, 4, 4), dtype=np.float32)
        self.inverse_bind_matrices = np.zeros((200, 4, 4), dtype=np.float32)
        self.bone_rest_transforms = np.zeros((200, 4, 4), dtype=np.float32)
        self.bone_parents = np.full(200, -1, dtype=np.int32)
        
        # Initialize all to identity
        for i in range(200):
            self.bone_matrices[i] = np.eye(4, dtype=np.float32)
            self.bone_local_rotations[i] = np.eye(4, dtype=np.float32)
            self.inverse_bind_matrices[i] = np.eye(4, dtype=np.float32)
            self.bone_rest_transforms[i] = np.eye(4, dtype=np.float32)
        
        # Copy bone data from VRM
        for i, bone in enumerate(self.vrm_data.bones):
            if i >= 200:
                break
            self.inverse_bind_matrices[i] = bone.inverse_bind_matrix.copy()
            self.bone_rest_transforms[i] = bone.local_matrix.copy()
            self.bone_parents[i] = bone.parent_index
        
        # Build name to joint index mapping
        self.bone_name_to_joint = {}
        for i, bone in enumerate(self.vrm_data.bones):
            if i >= 200:
                break
            if bone.name:
                # Store lowercase name for matching
                self.bone_name_to_joint[bone.name.lower()] = i
        
        # VRM bone naming convention mapping
        # VRM humanoid names like "leftUpperArm" map to skeleton names like "J_Bip_L_UpperArm"
        self.humanoid_to_joint = {}
        
        for hbone_name, node_idx in self.vrm_data.humanoid_bones.items():
            # Parse humanoid name to find side and bone type
            side = ""
            bone_part = hbone_name
            if hbone_name.startswith("left"):
                side = "_l_"
                bone_part = hbone_name[4:]  # Remove "left"
            elif hbone_name.startswith("right"):
                side = "_r_"
                bone_part = hbone_name[5:]  # Remove "right"
            else:
                side = "_c_"  # Center bone
            
            # Search for matching bone in skeleton
            for i, bone in enumerate(self.vrm_data.bones):
                if i >= 200 or not bone.name:
                    continue
                bname_lower = bone.name.lower()
                # Check if bone contains the side and part name
                if side.lower() in bname_lower and bone_part.lower() in bname_lower:
                    self.humanoid_to_joint[hbone_name] = i
                    break
        
        # Debug: print humanoid bone mapping
        print(f"\nHumanoid bone mapping found ({len(self.humanoid_to_joint)} bones):")
        for hbone_name in sorted(self.humanoid_to_joint.keys())[:10]:
            idx = self.humanoid_to_joint[hbone_name]
            print(f"  {hbone_name}: joint={idx}, bone={self.vrm_data.bones[idx].name}")
        if len(self.humanoid_to_joint) > 10:
            print(f"  ... and {len(self.humanoid_to_joint) - 10} more")
        
        # Disable skinning - model has 274 bones but GPU can only handle ~200 in uniforms
        # The model looks correct in rest pose without skinning
        self.use_skinning = False
        print(f"\nSkinning DISABLED (model has {num_bones} bones, exceeds GPU uniform limit)")
    
    def set_pose(self, pose_name: str):
        """Set a predefined pose."""
        self.current_pose = pose_name
        
        if not self.vrm_data or not self.use_skinning:
            return
        
        # Reset all local rotations to identity
        for i in range(200):
            self.bone_local_rotations[i] = np.eye(4, dtype=np.float32)
        
        if pose_name == "idle":
            # Relaxed idle pose - arms slightly down from T-pose
            self._set_bone_rotation("leftUpperArm", 0, 0, 25)   # Arms down (positive Z = down for left)
            self._set_bone_rotation("rightUpperArm", 0, 0, -25)  # Arms down (negative Z = down for right)
            self._set_bone_rotation("leftLowerArm", 5, 0, 0)   # Slight bend forward
            self._set_bone_rotation("rightLowerArm", 5, 0, 0)
            
        elif pose_name == "wave":
            # Waving pose - right arm up and bent
            self._set_bone_rotation("rightUpperArm", 0, 0, 150)  # Arm way up
            self._set_bone_rotation("rightLowerArm", 0, -90, 0)   # Forearm bent inward
            self._set_bone_rotation("leftUpperArm", 0, 0, 25)    # Left arm relaxed
            
        elif pose_name == "thinking":
            # Hand on chin thinking pose  
            self._set_bone_rotation("rightUpperArm", 60, 0, -30)
            self._set_bone_rotation("rightLowerArm", 0, -120, 0)
            self._set_bone_rotation("head", 0, -10, -5)  # Slight head tilt
            self._set_bone_rotation("leftUpperArm", 0, 0, 25)
            
        elif pose_name == "happy":
            # Arms slightly raised, happy pose
            self._set_bone_rotation("leftUpperArm", 0, 0, 35)
            self._set_bone_rotation("rightUpperArm", 0, 0, -35)
            self._set_bone_rotation("leftLowerArm", -20, 0, 0)
            self._set_bone_rotation("rightLowerArm", -20, 0, 0)
            self._set_bone_rotation("head", -5, 0, 0)  # Look up slightly
        
        elif pose_name == "tpose":
            # T-pose - disable skinning entirely for clean rest pose
            for i in range(200):
                self.bone_matrices[i] = np.eye(4, dtype=np.float32)
            self.update()
            return  # Skip _compute_bone_matrices
        
        # Recalculate bone matrices with hierarchy
        self._compute_bone_matrices()
        self.update()
    
    def _set_bone_rotation(self, bone_name: str, rx: float, ry: float, rz: float):
        """Set local rotation for a humanoid bone (degrees)."""
        # Use the pre-computed humanoid to joint mapping
        if not hasattr(self, 'humanoid_to_joint') or bone_name not in self.humanoid_to_joint:
            return
            
        joint_idx = self.humanoid_to_joint[bone_name]
        if joint_idx >= 200:
            return
        
        # Create rotation matrix from euler angles (XYZ order)
        import math
        rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
        
        # Individual rotation matrices
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        
        # Rotation X
        rot_x = np.array([
            [1, 0, 0, 0],
            [0, cx, -sx, 0],
            [0, sx, cx, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Rotation Y
        rot_y = np.array([
            [cy, 0, sy, 0],
            [0, 1, 0, 0],
            [-sy, 0, cy, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Rotation Z
        rot_z = np.array([
            [cz, -sz, 0, 0],
            [sz, cz, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Combined: R = Rz * Ry * Rx
        self.bone_local_rotations[joint_idx] = rot_z @ rot_y @ rot_x
    
    def _compute_bone_matrices(self):
        """Compute final bone matrices for skinning.
        
        For GPU skinning, the formula is:
        skinned_vertex = sum(weight[i] * bone_matrix[i] * vertex)
        
        Where bone_matrix[i] should transform from bind pose to current pose.
        When no animation is applied, bone_matrix should be identity.
        """
        num_bones = min(200, len(self.vrm_data.bones))
        
        # For simple pose changes, we only need to apply local rotations
        # The inverse bind matrices handle the rest pose transformation
        
        for i in range(num_bones):
            # Start with identity
            self.bone_matrices[i] = np.eye(4, dtype=np.float32)
        
        # Apply rotations to the bones that have been set
        # This is a simplified approach - just apply the rotation directly
        for i in range(num_bones):
            if not np.allclose(self.bone_local_rotations[i], np.eye(4)):
                # This bone has a rotation applied
                # For now, just use the rotation as the bone matrix
                # This works because we're only rotating, not translating
                self.bone_matrices[i] = self.bone_local_rotations[i]
    
    def resizeGL(self, w, h):
        """Handle resize."""
        glViewport(0, 0, w, h)
        
    def paintGL(self):
        """Render the VRM model."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        if self.program is None:
            return
        
        glUseProgram(self.program)
        
        # Create matrices
        width = self.width()
        height = self.height()
        aspect = width / height if height > 0 else 1.0
        
        # Projection matrix
        projection = self._perspective_matrix(30.0, aspect, 0.1, 100.0)
        
        # View matrix - camera looking at model
        import math
        cam_x = math.sin(math.radians(self.rotation)) * self.camera_distance
        cam_z = math.cos(math.radians(self.rotation)) * self.camera_distance
        cam_y = self.camera_height
        
        target_y = 1.5  # Look at face level
        
        view = self._look_at_matrix(
            eye=(cam_x, cam_y, cam_z),
            target=(0, target_y, 0),
            up=(0, 1, 0)
        )
        
        # Model matrix (identity)
        model = np.eye(4, dtype=np.float32)
        
        # Set uniforms
        loc = glGetUniformLocation(self.program, "projection")
        glUniformMatrix4fv(loc, 1, GL_FALSE, projection)
        
        loc = glGetUniformLocation(self.program, "view")
        glUniformMatrix4fv(loc, 1, GL_FALSE, view)
        
        loc = glGetUniformLocation(self.program, "model")
        glUniformMatrix4fv(loc, 1, GL_FALSE, model)
        
        # Light from the front (camera direction) - slightly above and to the right
        loc = glGetUniformLocation(self.program, "light_dir")
        glUniform3f(loc, 0.3, 0.5, -1.0)  # Negative Z = toward the model's front
        
        # Upload bone matrices for skinning
        loc = glGetUniformLocation(self.program, "use_skinning")
        glUniform1i(loc, 1 if self.use_skinning else 0)
        
        if self.use_skinning and self.bone_matrices is not None:
            for i in range(min(200, len(self.bone_matrices))):
                loc = glGetUniformLocation(self.program, f"bone_matrices[{i}]")
                if loc >= 0:
                    glUniformMatrix4fv(loc, 1, GL_FALSE, self.bone_matrices[i])
        
        # Render all VRM meshes
        for mesh in self.render_meshes:
            mesh.render(self.program)
            
        glUseProgram(0)
    
    def _perspective_matrix(self, fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Create perspective projection matrix (column-major for OpenGL)."""
        import math
        f = 1.0 / math.tan(math.radians(fov) / 2.0)
        m = np.zeros((4, 4), dtype=np.float32)
        # Column-major layout for OpenGL
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[3, 2] = (2.0 * far * near) / (near - far)
        m[2, 3] = -1.0
        return m
    
    def _look_at_matrix(self, eye: Tuple, target: Tuple, up: Tuple) -> np.ndarray:
        """Create look-at view matrix (column-major for OpenGL)."""
        eye = np.array(eye, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)
        
        f = target - eye
        f = f / np.linalg.norm(f)
        
        s = np.cross(f, up)
        s_norm = np.linalg.norm(s)
        if s_norm > 0:
            s = s / s_norm
        else:
            s = np.array([1, 0, 0], dtype=np.float32)
        
        u = np.cross(s, f)
        
        # Column-major layout for OpenGL
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
            self.rotation = (self.rotation + 1) % 360
        self.update()
        
    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.pos()
        elif event.button() == Qt.MouseButton.RightButton:
            # Store position for window dragging
            self._drag_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        """Handle mouse drag for rotation or window movement."""
        if event.buttons() == Qt.MouseButton.LeftButton and self.last_mouse_pos:
            # Rotate model
            delta = event.pos() - self.last_mouse_pos
            self.rotation += delta.x() * 0.5
            self.last_mouse_pos = event.pos()
            self.update()
        elif event.buttons() == Qt.MouseButton.RightButton and hasattr(self, '_drag_pos'):
            # Move the parent window
            window = self.window()
            if window:
                new_pos = event.globalPosition().toPoint()
                delta = new_pos - self._drag_pos
                window.move(window.pos() + delta)
                self._drag_pos = new_pos
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.RightButton:
            if hasattr(self, '_drag_pos'):
                del self._drag_pos
        super().mouseReleaseEvent(event)
        
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y()
        self.camera_distance -= delta * 0.002
        self.camera_distance = max(0.5, min(10.0, self.camera_distance))
        self.update()


class TransparentVRMWindow(QMainWindow):
    """Main window with transparent VRM renderer."""
    
    def __init__(self, vrm_path: str):
        super().__init__()
        
        # Frameless transparent window
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("VRM Renderer - Annabeth")
        
        # Set size and position
        self.setGeometry(100, 100, 500, 600)
        
        # Central widget
        central = QWidget()
        central.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCentralWidget(central)
        
        # Layout
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add VRM renderer
        self.vrm_renderer = VRMRenderer(vrm_path)
        layout.addWidget(self.vrm_renderer)
        
        # For window dragging
        self.drag_position = None
            
    def keyPressEvent(self, event):
        """Handle key presses."""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_R:
            self.vrm_renderer.auto_rotate = not self.vrm_renderer.auto_rotate
            print(f"Auto-rotate: {self.vrm_renderer.auto_rotate}")
        elif event.key() == Qt.Key.Key_1:
            self.vrm_renderer.set_pose("idle")
            print("Pose: idle")
        elif event.key() == Qt.Key.Key_2:
            self.vrm_renderer.set_pose("wave")
            print("Pose: wave")
        elif event.key() == Qt.Key.Key_3:
            self.vrm_renderer.set_pose("thinking")
            print("Pose: thinking")
        elif event.key() == Qt.Key.Key_4:
            self.vrm_renderer.set_pose("happy")
            print("Pose: happy")
        elif event.key() == Qt.Key.Key_T:
            # T-pose (reset)
            self.vrm_renderer.set_pose("tpose")
            print("Pose: T-pose (default)")
        # S key removed - skinning disabled for this model (274 bones exceeds GPU limit)


def main():
    app = QApplication(sys.argv)
    
    # Set up OpenGL format with alpha for transparency
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
    print("VRM Renderer (PyOpenGL)")
    print("=" * 50)
    print("Controls:")
    print("  Left-click + drag: Rotate view")
    print("  Mouse wheel: Zoom in/out")
    print("  R: Toggle auto-rotation")
    print("  1: Idle pose")
    print("  2: Wave pose")
    print("  3: Thinking pose")
    print("  4: Happy pose")
    print("  T: T-pose (default)")
    print("  ESC: Close")
    print("=" * 50)
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
