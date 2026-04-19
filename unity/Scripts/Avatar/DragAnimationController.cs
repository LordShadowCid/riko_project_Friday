using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UniVRM10;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Applies 2nd-order spring physics sway to hips and arms when the window moves.
    /// Also still drives VRM10 spring bone joints for hair/clothes.
    /// Based on Mate Engine's AvatarSwayController.cs spring formula:
    ///   acceleration = ω²(target - x) - 2ζωv
    /// where ω = springFrequency * 2π, ζ = dampingRatio.
    /// Provides natural overshoot + settle on drag/release.
    /// </summary>
    public class DragAnimationController : MonoBehaviour
    {
#if UNITY_STANDALONE_WIN
        [Header("Spring Physics")]
        [SerializeField] private float springFrequency = 2.6f;  // Hz
        [SerializeField] private float dampingRatio = 0.35f;     // Under-damped for bounce
        [SerializeField] private float intensity = 1.0f;         // Global multiplier

        [Header("Sway Limits")]
        [SerializeField] private float maxLeanZ = 25f;   // Left/right lean degrees
        [SerializeField] private float maxLeanX = 12f;    // Forward/back lean degrees
        [SerializeField] private float armSwingScale = 0.5f;

        [Header("Spring Bone (hair/clothes)")]
        [SerializeField] private float springBoneImpact = 0.05f;

        // Spring state — 2D (Z = left/right lean, X = forward/back lean)
        private float _springPosZ, _springVelZ;  // Lateral sway
        private float _springPosX, _springVelX;  // Depth sway

        // Window tracking
        private Vector2Int _prevWindowPos;
        private Vector2 _windowVelocity;
        private IntPtr _hwnd;

        // VRM references
        private Vrm10Instance _vrm;
        private Animator _animator;
        private readonly List<VRM10SpringBoneJoint> _joints = new();

        // Bone references
        private Transform _hips;
        private Transform _leftUpperArm, _rightUpperArm;
        private Quaternion _origHips, _origLUA, _origRUA;

        private bool _enabled = true;

        [StructLayout(LayoutKind.Sequential)]
        private struct RECT { public int left, top, right, bottom; }

        [DllImport("user32.dll")]
        private static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

        [DllImport("user32.dll")]
        private static extern IntPtr GetActiveWindow();

        private void Start()
        {
            _hwnd = GetActiveWindow();
            _prevWindowPos = GetWindowPosition();
        }

        public void Initialize(Vrm10Instance vrm)
        {
            _vrm = vrm;
            _animator = vrm?.GetComponent<Animator>();

            // Cache spring bone joints for hair/clothes
            _joints.Clear();
            _joints.AddRange(vrm.GetComponentsInChildren<VRM10SpringBoneJoint>(true));

            // Cache humanoid bones for body sway
            if (_animator != null)
            {
                _hips = _animator.GetBoneTransform(HumanBodyBones.Hips);
                _leftUpperArm = _animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
                _rightUpperArm = _animator.GetBoneTransform(HumanBodyBones.RightUpperArm);
            }

            if (_hips) _origHips = _hips.localRotation;
            if (_leftUpperArm) _origLUA = _leftUpperArm.localRotation;
            if (_rightUpperArm) _origRUA = _rightUpperArm.localRotation;

            Debug.Log($"[DragAnimation] Initialized: {_joints.Count} spring joints, sway on hips/arms");
        }

        private void LateUpdate()
        {
            if (_hwnd == IntPtr.Zero || _vrm == null) return;

            // Track window velocity
            var pos = GetWindowPosition();
            var delta = pos - _prevWindowPos;
            _prevWindowPos = pos;

            // Smooth window velocity (pixels/frame → normalized)
            float rawVelX = -delta.x * 0.01f; // Horizontal: negative because lean opposite to movement
            float rawVelY = delta.y * 0.005f;  // Vertical: forward lean when moving down
            _windowVelocity = Vector2.Lerp(_windowVelocity, new Vector2(rawVelX, rawVelY), 0.3f);

            // Apply spring bone forces for hair/clothes only while moving
            if (_joints.Count > 0)
            {
                if (delta != Vector2Int.zero)
                {
                    Vector3 force = new Vector3(-delta.x, delta.y, 0).normalized * springBoneImpact;
                    foreach (var joint in _joints)
                    {
                        if (joint == null) continue;
                        joint.m_gravityDir = force.normalized;
                        joint.m_gravityPower = force.magnitude;
                        _vrm.Runtime?.SpringBone?.SetJointLevel(joint.transform, joint.Blittable);
                    }
                }
                else
                {
                    // Reset spring bone gravity when not moving
                    foreach (var joint in _joints)
                    {
                        if (joint == null) continue;
                        joint.m_gravityPower = 0f;
                        _vrm.Runtime?.SpringBone?.SetJointLevel(joint.transform, joint.Blittable);
                    }
                }
            }

            if (!_enabled || intensity <= 0f) return;

            // 2nd-order spring: a = ω²(target - x) - 2ζωv
            float omega = springFrequency * 2f * Mathf.PI;
            float dt = Time.deltaTime;

            // Target position driven by window velocity
            float targetZ = Mathf.Clamp(_windowVelocity.x * maxLeanZ * intensity, -maxLeanZ, maxLeanZ);
            float targetX = Mathf.Clamp(_windowVelocity.y * maxLeanX * intensity, -maxLeanX, maxLeanX);

            // Spring Z (lateral lean)
            float accelZ = omega * omega * (targetZ - _springPosZ) - 2f * dampingRatio * omega * _springVelZ;
            _springVelZ += accelZ * dt;
            _springPosZ += _springVelZ * dt;
            _springPosZ = Mathf.Clamp(_springPosZ, -maxLeanZ, maxLeanZ);

            // Spring X (depth lean)
            float accelX = omega * omega * (targetX - _springPosX) - 2f * dampingRatio * omega * _springVelX;
            _springVelX += accelX * dt;
            _springPosX += _springVelX * dt;
            _springPosX = Mathf.Clamp(_springPosX, -maxLeanX, maxLeanX);

            // Skip if negligible
            if (Mathf.Abs(_springPosZ) < 0.01f && Mathf.Abs(_springPosX) < 0.01f) return;

            // Apply hip sway from original base rotation (NOT additive)
            if (_hips)
            {
                Quaternion sway = Quaternion.Euler(_springPosX, 0f, _springPosZ);
                _hips.localRotation = _origHips * sway;
            }

            // Apply arm swing from original base rotation
            float armZ = _springPosZ * armSwingScale;
            if (_leftUpperArm)
            {
                Quaternion swing = Quaternion.Euler(0f, 0f, armZ);
                _leftUpperArm.localRotation = _origLUA * swing;
            }
            if (_rightUpperArm)
            {
                Quaternion swing = Quaternion.Euler(0f, 0f, -armZ);
                _rightUpperArm.localRotation = _origRUA * swing;
            }
        }

        private Vector2Int GetWindowPosition()
        {
            GetWindowRect(_hwnd, out RECT rect);
            return new Vector2Int(rect.left, rect.top);
        }

        // ── Settings API ────────────────────────────────────────
        public void SetSwayEnabled(bool enabled) => _enabled = enabled;
        public void SetIntensity(float val) => intensity = val;
        public void SetSpringFrequency(float freq) => springFrequency = freq;
        public void SetDampingRatio(float ratio) => dampingRatio = ratio;
#else
        public void Initialize(Vrm10Instance vrm)
        {
            Debug.Log("[DragAnimation] Editor mode — spring physics disabled.");
        }
        public void SetSwayEnabled(bool enabled) { }
        public void SetIntensity(float val) { }
        public void SetSpringFrequency(float freq) { }
        public void SetDampingRatio(float ratio) { }
#endif
    }
}
