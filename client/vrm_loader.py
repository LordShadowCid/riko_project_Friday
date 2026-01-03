"""
VRM File Loader using pygltflib.
Parses VRM (glTF 2.0 based) files and extracts mesh data for rendering.
"""
import json
import struct
import numpy as np
from pathlib import Path
from pygltflib import GLTF2
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any


@dataclass
class VRMMesh:
    """Represents a single mesh primitive for rendering."""
    name: str
    positions: np.ndarray  # Nx3 float32
    normals: Optional[np.ndarray] = None  # Nx3 float32
    uvs: Optional[np.ndarray] = None  # Nx2 float32
    indices: Optional[np.ndarray] = None  # Mx3 uint32 (triangles)
    joint_indices: Optional[np.ndarray] = None  # Nx4 uint16 (bone indices)
    joint_weights: Optional[np.ndarray] = None  # Nx4 float32 (bone weights)
    morph_targets: List[np.ndarray] = field(default_factory=list)  # List of Nx3 position deltas
    base_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    texture_index: Optional[int] = None


@dataclass 
class VRMBone:
    """Represents a bone in the skeleton."""
    name: str
    index: int
    parent_index: int
    local_matrix: np.ndarray  # 4x4 transform
    inverse_bind_matrix: np.ndarray  # 4x4 inverse bind pose
    children: List[int] = field(default_factory=list)


@dataclass
class VRMExpression:
    """Represents a VRM expression (blend shape group)."""
    name: str
    preset: str  # e.g., "happy", "angry", "blink"
    morph_target_binds: List[Tuple[int, int, float]] = field(default_factory=list)  # (mesh_idx, target_idx, weight)


@dataclass
class VRMData:
    """Complete VRM model data."""
    meshes: List[VRMMesh] = field(default_factory=list)
    bones: List[VRMBone] = field(default_factory=list)
    expressions: List[VRMExpression] = field(default_factory=list)
    textures: List[np.ndarray] = field(default_factory=list)  # RGBA images
    vrm_meta: Dict[str, Any] = field(default_factory=dict)
    humanoid_bones: Dict[str, int] = field(default_factory=dict)  # VRM humanoid bone mapping


class VRMLoader:
    """Loads VRM files and extracts rendering data."""
    
    def __init__(self, vrm_path: str):
        self.path = Path(vrm_path)
        self.gltf: Optional[GLTF2] = None
        self.binary_data: bytes = b""
        
    def load(self) -> VRMData:
        """Load and parse the VRM file."""
        print(f"Loading VRM: {self.path}")
        
        # Load glTF/GLB - VRM files are GLB (binary) format
        # Use load_binary for .vrm files
        if self.path.suffix.lower() in ['.vrm', '.glb']:
            self.gltf = GLTF2().load_binary(str(self.path))
        else:
            self.gltf = GLTF2().load(str(self.path))
        
        # Get binary buffer data
        if self.gltf.buffers and len(self.gltf.buffers) > 0:
            buffer = self.gltf.buffers[0]
            if buffer.uri is None:
                # GLB format - data is embedded
                self.binary_data = self.gltf.binary_blob() or b""
            else:
                # Separate .bin file
                bin_path = self.path.parent / buffer.uri
                with open(bin_path, 'rb') as f:
                    self.binary_data = f.read()
        
        vrm_data = VRMData()
        
        # Parse VRM extensions
        self._parse_vrm_extensions(vrm_data)
        
        # Parse meshes
        self._parse_meshes(vrm_data)
        
        # Parse skeleton
        self._parse_skeleton(vrm_data)
        
        # Parse textures (basic info only for now)
        self._parse_textures(vrm_data)
        
        print(f"Loaded: {len(vrm_data.meshes)} meshes, {len(vrm_data.bones)} bones, {len(vrm_data.expressions)} expressions")
        
        return vrm_data
    
    def _parse_vrm_extensions(self, vrm_data: VRMData):
        """Parse VRM-specific extensions."""
        if not self.gltf.extensions:
            return
            
        # Try VRM 1.0 first, then VRM 0.x
        vrm_ext = None
        if hasattr(self.gltf.extensions, 'VRMC_vrm'):
            vrm_ext = self.gltf.extensions.get('VRMC_vrm', {})
            vrm_data.vrm_meta['version'] = '1.0'
        elif hasattr(self.gltf.extensions, 'VRM'):
            vrm_ext = self.gltf.extensions.get('VRM', {})
            vrm_data.vrm_meta['version'] = '0.x'
        
        # Try to access as dict
        if isinstance(self.gltf.extensions, dict):
            vrm_ext = self.gltf.extensions.get('VRMC_vrm') or self.gltf.extensions.get('VRM')
            
        if vrm_ext:
            vrm_data.vrm_meta['extension'] = vrm_ext
            print(f"Found VRM extension: version {vrm_data.vrm_meta.get('version', 'unknown')}")
            
            # Parse expressions/blend shapes
            self._parse_expressions(vrm_ext, vrm_data)
            
            # Parse humanoid bone mapping
            self._parse_humanoid(vrm_ext, vrm_data)
    
    def _parse_expressions(self, vrm_ext: dict, vrm_data: VRMData):
        """Parse VRM expressions (blend shape groups)."""
        # VRM 1.0 format
        if 'expressions' in vrm_ext:
            expr_data = vrm_ext['expressions']
            if 'preset' in expr_data:
                for name, expr_info in expr_data['preset'].items():
                    expr = VRMExpression(name=name, preset=name)
                    if 'morphTargetBinds' in expr_info:
                        for bind in expr_info['morphTargetBinds']:
                            expr.morph_target_binds.append((
                                bind.get('node', 0),
                                bind.get('index', 0),
                                bind.get('weight', 1.0)
                            ))
                    vrm_data.expressions.append(expr)
                    
        # VRM 0.x format
        elif 'blendShapeMaster' in vrm_ext:
            blend_groups = vrm_ext['blendShapeMaster'].get('blendShapeGroups', [])
            for group in blend_groups:
                expr = VRMExpression(
                    name=group.get('name', 'unknown'),
                    preset=group.get('presetName', 'unknown')
                )
                for bind in group.get('binds', []):
                    expr.morph_target_binds.append((
                        bind.get('mesh', 0),
                        bind.get('index', 0),
                        bind.get('weight', 100.0) / 100.0  # VRM 0.x uses 0-100
                    ))
                vrm_data.expressions.append(expr)
    
    def _parse_humanoid(self, vrm_ext: dict, vrm_data: VRMData):
        """Parse humanoid bone mapping."""
        if 'humanoid' not in vrm_ext:
            return
            
        humanoid = vrm_ext['humanoid']
        human_bones = humanoid.get('humanBones', [])
        
        # VRM 1.0 format (dict)
        if isinstance(human_bones, dict):
            for bone_name, bone_info in human_bones.items():
                if isinstance(bone_info, dict):
                    vrm_data.humanoid_bones[bone_name] = bone_info.get('node', -1)
                else:
                    vrm_data.humanoid_bones[bone_name] = bone_info
                
        # VRM 0.x format (list)
        elif isinstance(human_bones, list):
            for bone_info in human_bones:
                if isinstance(bone_info, dict):
                    bone_name = bone_info.get('bone', 'unknown')
                    vrm_data.humanoid_bones[bone_name] = bone_info.get('node', -1)
    
    def _parse_meshes(self, vrm_data: VRMData):
        """Parse mesh primitives."""
        if not self.gltf.meshes:
            return
            
        for mesh_idx, mesh in enumerate(self.gltf.meshes):
            for prim_idx, primitive in enumerate(mesh.primitives):
                mesh_name = f"{mesh.name or f'mesh_{mesh_idx}'}_{prim_idx}"
                
                # Get vertex positions (required)
                if hasattr(primitive.attributes, 'POSITION') and primitive.attributes.POSITION is not None:
                    positions = self._get_accessor_data(primitive.attributes.POSITION)
                else:
                    continue  # Skip primitives without positions
                
                vrm_mesh = VRMMesh(
                    name=mesh_name,
                    positions=positions
                )
                
                # Get normals
                if hasattr(primitive.attributes, 'NORMAL') and primitive.attributes.NORMAL is not None:
                    vrm_mesh.normals = self._get_accessor_data(primitive.attributes.NORMAL)
                
                # Get UVs
                if hasattr(primitive.attributes, 'TEXCOORD_0') and primitive.attributes.TEXCOORD_0 is not None:
                    vrm_mesh.uvs = self._get_accessor_data(primitive.attributes.TEXCOORD_0)
                
                # Get indices
                if primitive.indices is not None:
                    vrm_mesh.indices = self._get_accessor_data(primitive.indices)
                
                # Get joint indices for skinning
                if hasattr(primitive.attributes, 'JOINTS_0') and primitive.attributes.JOINTS_0 is not None:
                    vrm_mesh.joint_indices = self._get_accessor_data(primitive.attributes.JOINTS_0)
                
                # Get joint weights for skinning
                if hasattr(primitive.attributes, 'WEIGHTS_0') and primitive.attributes.WEIGHTS_0 is not None:
                    vrm_mesh.joint_weights = self._get_accessor_data(primitive.attributes.WEIGHTS_0)
                
                # Get morph targets
                if primitive.targets:
                    for target in primitive.targets:
                        if hasattr(target, 'POSITION') and target.POSITION is not None:
                            target_positions = self._get_accessor_data(target.POSITION)
                            vrm_mesh.morph_targets.append(target_positions)
                
                # Get material color
                if primitive.material is not None and self.gltf.materials:
                    material = self.gltf.materials[primitive.material]
                    if material.pbrMetallicRoughness and material.pbrMetallicRoughness.baseColorFactor:
                        vrm_mesh.base_color = tuple(material.pbrMetallicRoughness.baseColorFactor)
                    if material.pbrMetallicRoughness and material.pbrMetallicRoughness.baseColorTexture:
                        vrm_mesh.texture_index = material.pbrMetallicRoughness.baseColorTexture.index
                
                vrm_data.meshes.append(vrm_mesh)
                print(f"  Mesh '{vrm_mesh.name}': {len(vrm_mesh.positions)} vertices, {len(vrm_mesh.morph_targets)} morph targets")
    
    def _parse_skeleton(self, vrm_data: VRMData):
        """Parse skeleton/bones."""
        if not self.gltf.skins or len(self.gltf.skins) == 0:
            return
            
        skin = self.gltf.skins[0]  # Usually VRM has one skin
        
        # Get inverse bind matrices
        inverse_bind_matrices = None
        if skin.inverseBindMatrices is not None:
            ibm_data = self._get_accessor_data(skin.inverseBindMatrices)
            inverse_bind_matrices = ibm_data.reshape(-1, 4, 4)
        
        for i, joint_idx in enumerate(skin.joints):
            node = self.gltf.nodes[joint_idx]
            
            # Get local transform
            local_matrix = np.eye(4, dtype=np.float32)
            if node.matrix:
                local_matrix = np.array(node.matrix, dtype=np.float32).reshape(4, 4)
            else:
                # Build from TRS
                if node.translation:
                    local_matrix[3, :3] = node.translation
                # TODO: Add rotation and scale
            
            # Get inverse bind matrix
            ibm = np.eye(4, dtype=np.float32)
            if inverse_bind_matrices is not None and i < len(inverse_bind_matrices):
                ibm = inverse_bind_matrices[i]
            
            # Find parent
            parent_idx = -1
            for j, potential_parent_joint in enumerate(skin.joints):
                parent_node = self.gltf.nodes[potential_parent_joint]
                if parent_node.children and joint_idx in parent_node.children:
                    parent_idx = j
                    break
            
            bone = VRMBone(
                name=node.name or f"bone_{i}",
                index=i,
                parent_index=parent_idx,
                local_matrix=local_matrix,
                inverse_bind_matrix=ibm
            )
            vrm_data.bones.append(bone)
    
    def _parse_textures(self, vrm_data: VRMData):
        """Parse and extract texture images."""
        if not self.gltf.textures or not self.gltf.images:
            return
            
        print(f"  Loading {len(self.gltf.textures)} textures...")
        
        for tex_idx, texture in enumerate(self.gltf.textures):
            if texture.source is None:
                vrm_data.textures.append(None)
                continue
                
            image = self.gltf.images[texture.source]
            
            # Get image data from buffer view
            if image.bufferView is not None:
                bv = self.gltf.bufferViews[image.bufferView]
                offset = bv.byteOffset or 0
                length = bv.byteLength
                image_data = self.binary_data[offset:offset + length]
                
                # Decode image using PIL
                try:
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(image_data))
                    img = img.convert('RGBA')
                    # Convert to numpy array
                    tex_array = np.array(img, dtype=np.uint8)
                    vrm_data.textures.append(tex_array)
                except Exception as e:
                    print(f"    Failed to load texture {tex_idx}: {e}")
                    vrm_data.textures.append(None)
            else:
                vrm_data.textures.append(None)
                
        print(f"  Loaded {len([t for t in vrm_data.textures if t is not None])} textures successfully")
    
    def _get_accessor_data(self, accessor_idx: int) -> np.ndarray:
        """Extract data from a glTF accessor."""
        accessor = self.gltf.accessors[accessor_idx]
        buffer_view = self.gltf.bufferViews[accessor.bufferView]
        
        # Calculate offset
        offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
        
        # Determine dtype and shape
        component_type_map = {
            5120: np.int8,
            5121: np.uint8,
            5122: np.int16,
            5123: np.uint16,
            5125: np.uint32,
            5126: np.float32,
        }
        dtype = component_type_map.get(accessor.componentType, np.float32)
        
        type_count_map = {
            'SCALAR': 1,
            'VEC2': 2,
            'VEC3': 3,
            'VEC4': 4,
            'MAT4': 16,
        }
        component_count = type_count_map.get(accessor.type, 1)
        
        # Calculate byte length
        byte_length = accessor.count * component_count * np.dtype(dtype).itemsize
        
        # Extract data
        raw_data = self.binary_data[offset:offset + byte_length]
        data = np.frombuffer(raw_data, dtype=dtype)
        
        # Reshape if needed
        if component_count > 1:
            data = data.reshape(-1, component_count)
        
        return data


def test_load_vrm(vrm_path: str):
    """Test loading a VRM file."""
    loader = VRMLoader(vrm_path)
    vrm_data = loader.load()
    
    print("\n=== VRM Summary ===")
    print(f"Meshes: {len(vrm_data.meshes)}")
    print(f"Bones: {len(vrm_data.bones)}")
    print(f"Expressions: {len(vrm_data.expressions)}")
    print(f"Humanoid bones: {list(vrm_data.humanoid_bones.keys())[:10]}...")
    
    if vrm_data.expressions:
        print("\nExpressions:")
        for expr in vrm_data.expressions[:10]:
            print(f"  - {expr.name} ({expr.preset}): {len(expr.morph_target_binds)} binds")
    
    # Calculate total vertices
    total_verts = sum(len(m.positions) for m in vrm_data.meshes)
    total_tris = sum(len(m.indices) // 3 if m.indices is not None else 0 for m in vrm_data.meshes)
    print(f"\nTotal vertices: {total_verts:,}")
    print(f"Total triangles: {total_tris:,}")
    
    return vrm_data


if __name__ == "__main__":
    import sys
    
    # Default to the project's VRM file
    vrm_path = "models/vrm/claire_avatar.vrm"
    if len(sys.argv) > 1:
        vrm_path = sys.argv[1]
    
    test_load_vrm(vrm_path)
