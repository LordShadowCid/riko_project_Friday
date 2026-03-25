using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace Annabeth.Dance
{
    /// <summary>
    /// Feature #19: Parses and plays VMD (Vocaloid Motion Data) files.
    /// Reads bone keyframes and morph keyframes from VMD binary format,
    /// maps MMD bone names to Unity HumanBodyBones, and applies animations.
    /// </summary>
    public class VmdPlayer : MonoBehaviour
    {
        // ── VMD Data Structures ─────────────────────────────────────

        public struct BoneKeyframe
        {
            public string BoneName;
            public uint Frame;
            public Vector3 Position;
            public Quaternion Rotation;
        }

        public struct MorphKeyframe
        {
            public string MorphName;
            public uint Frame;
            public float Weight;
        }

        public class VmdData
        {
            public string ModelName;
            public List<BoneKeyframe> BoneFrames = new();
            public List<MorphKeyframe> MorphFrames = new();
        }

        // ── MMD Bone → Unity HumanBodyBones mapping ─────────────────

        private static readonly Dictionary<string, HumanBodyBones> BoneMap = new()
        {
            // Core
            { "センター", HumanBodyBones.Hips },
            { "上半身", HumanBodyBones.Spine },
            { "上半身2", HumanBodyBones.Chest },
            { "首", HumanBodyBones.Neck },
            { "頭", HumanBodyBones.Head },
            // Left arm
            { "左肩", HumanBodyBones.LeftShoulder },
            { "左腕", HumanBodyBones.LeftUpperArm },
            { "左ひじ", HumanBodyBones.LeftLowerArm },
            { "左手首", HumanBodyBones.LeftHand },
            // Right arm
            { "右肩", HumanBodyBones.RightShoulder },
            { "右腕", HumanBodyBones.RightUpperArm },
            { "右ひじ", HumanBodyBones.RightLowerArm },
            { "右手首", HumanBodyBones.RightHand },
            // Left leg
            { "左足", HumanBodyBones.LeftUpperLeg },
            { "左ひざ", HumanBodyBones.LeftLowerLeg },
            { "左足首", HumanBodyBones.LeftFoot },
            { "左つま先", HumanBodyBones.LeftToes },
            // Right leg
            { "右足", HumanBodyBones.RightUpperLeg },
            { "右ひざ", HumanBodyBones.RightLowerLeg },
            { "右足首", HumanBodyBones.RightFoot },
            { "右つま先", HumanBodyBones.RightToes },
        };

        // ── Playback state ──────────────────────────────────────────

        private VmdData _data;
        private Animator _animator;
        private readonly Dictionary<HumanBodyBones, Transform> _boneTransforms = new();
        private readonly Dictionary<string, int> _morphIndices = new();
        private SkinnedMeshRenderer _faceMesh;
        private float _playbackTime;
        private float _totalDuration;
        private bool _isPlaying;
        private bool _loop = true;
        private const float FrameRate = 30f; // VMD standard frame rate

        public bool IsPlaying => _isPlaying;
        public float Progress => _totalDuration > 0 ? _playbackTime / _totalDuration : 0f;

        public event Action OnFinished;

        public void Initialize(Animator animator)
        {
            _animator = animator;
            CacheBoneTransforms();
            CacheFaceMesh();
        }

        private void CacheBoneTransforms()
        {
            _boneTransforms.Clear();
            if (_animator == null) return;
            foreach (var kvp in BoneMap)
            {
                var t = _animator.GetBoneTransform(kvp.Value);
                if (t != null)
                    _boneTransforms[kvp.Value] = t;
            }
        }

        private void CacheFaceMesh()
        {
            _morphIndices.Clear();
            _faceMesh = null;
            if (_animator == null) return;

            // Search for SkinnedMeshRenderer with blendshapes (typically "Face" or "Body")
            foreach (var smr in _animator.GetComponentsInChildren<SkinnedMeshRenderer>())
            {
                if (smr.sharedMesh != null && smr.sharedMesh.blendShapeCount > 0)
                {
                    _faceMesh = smr;
                    for (int i = 0; i < smr.sharedMesh.blendShapeCount; i++)
                        _morphIndices[smr.sharedMesh.GetBlendShapeName(i)] = i;
                    break;
                }
            }
        }

        // ── VMD Binary Parser ───────────────────────────────────────

        /// <summary>
        /// Parse a VMD file from disk.
        /// </summary>
        public static VmdData Parse(string filePath)
        {
            if (!File.Exists(filePath))
            {
                Debug.LogError($"[VMD] File not found: {filePath}");
                return null;
            }

            using var stream = File.OpenRead(filePath);
            using var reader = new BinaryReader(stream);

            var data = new VmdData();

            // Header: 30 bytes magic
            byte[] magic = reader.ReadBytes(30);
            string header = Encoding.ASCII.GetString(magic).TrimEnd('\0');
            if (!header.StartsWith("Vocaloid Motion Data"))
            {
                Debug.LogError($"[VMD] Invalid header: {header}");
                return null;
            }

            // Model name: 20 bytes (Shift_JIS)
            byte[] nameBytes = reader.ReadBytes(20);
            data.ModelName = Encoding.GetEncoding(932).GetString(nameBytes).TrimEnd('\0');

            // Bone keyframes
            uint boneCount = reader.ReadUInt32();
            for (uint i = 0; i < boneCount; i++)
            {
                byte[] boneNameBytes = reader.ReadBytes(15);
                string boneName = Encoding.GetEncoding(932).GetString(boneNameBytes).TrimEnd('\0');
                uint frame = reader.ReadUInt32();

                // Position (x, y, z) — MMD is left-handed, Unity is left-handed too but axes differ
                float px = reader.ReadSingle();
                float py = reader.ReadSingle();
                float pz = reader.ReadSingle();

                // Rotation (quaternion x, y, z, w)
                float rx = reader.ReadSingle();
                float ry = reader.ReadSingle();
                float rz = reader.ReadSingle();
                float rw = reader.ReadSingle();

                // Interpolation data: 64 bytes (skip for now)
                reader.ReadBytes(64);

                // Convert MMD → Unity coordinate system
                // MMD: left-handed (X-right, Y-up, Z-forward)
                // Unity: left-handed (X-right, Y-up, Z-forward) but mirrored Z for motions
                data.BoneFrames.Add(new BoneKeyframe
                {
                    BoneName = boneName,
                    Frame = frame,
                    Position = new Vector3(-px, py, -pz) * 0.08f, // Scale down from MMD units
                    Rotation = new Quaternion(-rx, ry, -rz, rw)    // Mirror X and Z axes
                });
            }

            // Morph keyframes
            if (stream.Position < stream.Length - 4)
            {
                uint morphCount = reader.ReadUInt32();
                for (uint i = 0; i < morphCount; i++)
                {
                    byte[] morphNameBytes = reader.ReadBytes(15);
                    string morphName = Encoding.GetEncoding(932).GetString(morphNameBytes).TrimEnd('\0');
                    uint frame = reader.ReadUInt32();
                    float weight = reader.ReadSingle();

                    data.MorphFrames.Add(new MorphKeyframe
                    {
                        MorphName = morphName,
                        Frame = frame,
                        Weight = weight
                    });
                }
            }

            Debug.Log($"[VMD] Parsed: {data.ModelName}, {data.BoneFrames.Count} bone frames, {data.MorphFrames.Count} morph frames");
            return data;
        }

        // ── Playback ────────────────────────────────────────────────

        public bool LoadAndPlay(string filePath, bool loop = true)
        {
            _data = Parse(filePath);
            if (_data == null) return false;

            _loop = loop;
            _playbackTime = 0f;

            // Calculate total duration
            uint maxFrame = 0;
            foreach (var bf in _data.BoneFrames)
                if (bf.Frame > maxFrame) maxFrame = bf.Frame;
            foreach (var mf in _data.MorphFrames)
                if (mf.Frame > maxFrame) maxFrame = mf.Frame;
            _totalDuration = maxFrame / FrameRate;

            _isPlaying = true;
            return true;
        }

        public void Stop()
        {
            _isPlaying = false;
            _playbackTime = 0f;
        }

        public void Pause() => _isPlaying = false;
        public void Resume() => _isPlaying = true;

        private void LateUpdate()
        {
            if (!_isPlaying || _data == null || _animator == null) return;

            _playbackTime += Time.deltaTime;
            if (_playbackTime >= _totalDuration)
            {
                if (_loop)
                    _playbackTime %= _totalDuration;
                else
                {
                    _isPlaying = false;
                    OnFinished?.Invoke();
                    return;
                }
            }

            float currentFrame = _playbackTime * FrameRate;
            ApplyBoneFrames(currentFrame);
            ApplyMorphFrames(currentFrame);
        }

        private void ApplyBoneFrames(float currentFrame)
        {
            // Group frames by bone name and interpolate
            // Simple approach: find nearest before/after frames per bone
            foreach (var kvp in BoneMap)
            {
                if (!_boneTransforms.TryGetValue(kvp.Value, out var boneTransform))
                    continue;

                BoneKeyframe? prev = null;
                BoneKeyframe? next = null;

                foreach (var bf in _data.BoneFrames)
                {
                    if (bf.BoneName != kvp.Key) continue;
                    if (bf.Frame <= currentFrame)
                    {
                        if (!prev.HasValue || bf.Frame > prev.Value.Frame)
                            prev = bf;
                    }
                    if (bf.Frame >= currentFrame)
                    {
                        if (!next.HasValue || bf.Frame < next.Value.Frame)
                            next = bf;
                    }
                }

                if (!prev.HasValue && !next.HasValue) continue;

                Quaternion targetRot;
                if (prev.HasValue && next.HasValue && next.Value.Frame != prev.Value.Frame)
                {
                    float t = (currentFrame - prev.Value.Frame) / (next.Value.Frame - prev.Value.Frame);
                    targetRot = Quaternion.Slerp(prev.Value.Rotation, next.Value.Rotation, t);
                }
                else if (prev.HasValue)
                    targetRot = prev.Value.Rotation;
                else
                    targetRot = next.Value.Rotation;

                boneTransform.localRotation = targetRot;
            }
        }

        private void ApplyMorphFrames(float currentFrame)
        {
            if (_faceMesh == null) return;

            // Collect unique morph names involved
            var morphNames = new HashSet<string>();
            foreach (var mf in _data.MorphFrames)
                morphNames.Add(mf.MorphName);

            foreach (string morphName in morphNames)
            {
                if (!_morphIndices.TryGetValue(morphName, out int blendIndex))
                    continue;

                MorphKeyframe? prev = null;
                MorphKeyframe? next = null;

                foreach (var mf in _data.MorphFrames)
                {
                    if (mf.MorphName != morphName) continue;
                    if (mf.Frame <= currentFrame)
                    {
                        if (!prev.HasValue || mf.Frame > prev.Value.Frame)
                            prev = mf;
                    }
                    if (mf.Frame >= currentFrame)
                    {
                        if (!next.HasValue || mf.Frame < next.Value.Frame)
                            next = mf;
                    }
                }

                if (!prev.HasValue && !next.HasValue) continue;

                float weight;
                if (prev.HasValue && next.HasValue && next.Value.Frame != prev.Value.Frame)
                {
                    float t = (currentFrame - prev.Value.Frame) / (next.Value.Frame - prev.Value.Frame);
                    weight = Mathf.Lerp(prev.Value.Weight, next.Value.Weight, t);
                }
                else if (prev.HasValue)
                    weight = prev.Value.Weight;
                else
                    weight = next.Value.Weight;

                _faceMesh.SetBlendShapeWeight(blendIndex, weight * 100f);
            }
        }
    }
}
