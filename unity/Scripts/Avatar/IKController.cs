using System;
using UnityEngine;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Inverse Kinematics controller based on Mate Engine's IKFix + HandHolder patterns.
    /// Provides:
    ///   - Foot grounding for sitting poses (legs dangle/rest on surface)
    ///   - Hand placement during drag (arms reach up)
    ///   - Bone caching with SetAnimator() rebind on VRM swap
    /// </summary>
    [DefaultExecutionOrder(1000)] // Run LateUpdate after animations and other scripts
    public class IKController : MonoBehaviour
    {
        /// <summary>Fired when IK enabled state changes.</summary>
        public event Action<bool> OnIKEnabledChanged;

        // ── Cached bones (Mate Engine pattern: cache once, rebind on model swap) ──
        private Transform _hips;
        private Transform _leftUpperLeg;
        private Transform _rightUpperLeg;
        private Transform _leftLowerLeg;
        private Transform _rightLowerLeg;
        private Transform _leftFoot;
        private Transform _rightFoot;
        private Transform _leftHand;
        private Transform _rightHand;
        private Transform _leftShoulder;
        private Transform _rightShoulder;
        private Transform _chest;
        private Transform _head;

        private Animator _animator;
        private bool _initialized;

        // ── Settings ────────────────────────────────────────────
        [SerializeField] private bool _enabled = true;
        [SerializeField] private float _footGroundOffset = 0.02f;
        [SerializeField] private float _sittingLegAngle = 80f;
        [SerializeField] private float _sittingKneeAngle = 90f;
        [SerializeField] private float _ikBlendSpeed = 6f;

        // ── Runtime state ───────────────────────────────────────
        private bool _isSitting;
        private bool _isDragging;
        private float _sittingBlend;    // 0 = no sitting IK, 1 = full sitting pose
        private float _draggingBlend;   // 0 = no drag IK, 1 = full drag pose
        private Vector3 _dragHandTarget; // Screen-space cursor → world target for hands

        // ── Rig completeness checks ─────────────────────────────
        private bool _hasCompleteLegRig;
        private bool _hasCompleteArmRig;

        // ── Saved rest rotations (before IK) ────────────────────
        private Quaternion _restLeftUpperLeg;
        private Quaternion _restRightUpperLeg;
        private Quaternion _restLeftLowerLeg;
        private Quaternion _restRightLowerLeg;
        private Quaternion _restLeftFoot;
        private Quaternion _restRightFoot;
        private Quaternion _restLeftHand;
        private Quaternion _restRightHand;
        private Quaternion _restLeftShoulder;
        private Quaternion _restRightShoulder;

        // ── Public API ──────────────────────────────────────────

        /// <summary>Call after VRM loads to cache bone references (Mate Engine SetAnimator pattern).</summary>
        public void Initialize(UniVRM10.Vrm10Instance vrm)
        {
            _animator = vrm.GetComponentInChildren<Animator>();
            if (_animator == null)
            {
                Debug.LogWarning("[IKController] VRM has no Animator, IK disabled");
                _initialized = false;
                return;
            }
            CacheBones();
            _sittingBlend = 0f;
            _draggingBlend = 0f;
            _hasCompleteLegRig = _leftUpperLeg != null && _rightUpperLeg != null &&
                                  _leftLowerLeg != null && _rightLowerLeg != null &&
                                  _leftFoot != null && _rightFoot != null;
            _hasCompleteArmRig = _leftHand != null && _rightHand != null &&
                                  _leftShoulder != null && _rightShoulder != null;
            _initialized = true;
            Debug.Log($"[IKController] Initialized, legs={_hasCompleteLegRig}, arms={_hasCompleteArmRig}");
        }

        public void SetEnabled(bool enabled)
        {
            _enabled = enabled;
            OnIKEnabledChanged?.Invoke(enabled);
        }

        public bool IsEnabled => _enabled;

        /// <summary>Notify that the avatar is now sitting on a window.</summary>
        public void SetSitting(bool sitting)
        {
            _isSitting = sitting;
        }

        /// <summary>Notify that the avatar is being dragged.</summary>
        public void SetDragging(bool dragging)
        {
            _isDragging = dragging;
        }

        // ── Bone caching (Mate Engine pattern) ──────────────────

        private void CacheBones()
        {
            if (_animator == null) return;

            _hips = _animator.GetBoneTransform(HumanBodyBones.Hips);
            _leftUpperLeg = _animator.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
            _rightUpperLeg = _animator.GetBoneTransform(HumanBodyBones.RightUpperLeg);
            _leftLowerLeg = _animator.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
            _rightLowerLeg = _animator.GetBoneTransform(HumanBodyBones.RightLowerLeg);
            _leftFoot = _animator.GetBoneTransform(HumanBodyBones.LeftFoot);
            _rightFoot = _animator.GetBoneTransform(HumanBodyBones.RightFoot);
            _leftHand = _animator.GetBoneTransform(HumanBodyBones.LeftHand);
            _rightHand = _animator.GetBoneTransform(HumanBodyBones.RightHand);
            _leftShoulder = _animator.GetBoneTransform(HumanBodyBones.LeftShoulder);
            _rightShoulder = _animator.GetBoneTransform(HumanBodyBones.RightShoulder);
            _chest = _animator.GetBoneTransform(HumanBodyBones.Chest);
            _head = _animator.GetBoneTransform(HumanBodyBones.Head);
        }

        // ── LateUpdate: Apply IK after animations ──────────────

        private void LateUpdate()
        {
            if (!_initialized || !_enabled) return;

            float dt = Time.deltaTime;

            // Blend sitting IK
            float sittingTarget = _isSitting ? 1f : 0f;
            _sittingBlend = Mathf.MoveTowards(_sittingBlend, sittingTarget, _ikBlendSpeed * dt);

            // Blend dragging IK
            float draggingTarget = _isDragging ? 1f : 0f;
            _draggingBlend = Mathf.MoveTowards(_draggingBlend, draggingTarget, _ikBlendSpeed * dt);

            // Snapshot animation rest rotations BEFORE applying IK
            SnapshotRestRotations();

            if (_sittingBlend > 0.001f)
                ApplySittingIK(_sittingBlend);

            if (_draggingBlend > 0.001f)
                ApplyDragHandIK(_draggingBlend);
        }

        private void SnapshotRestRotations()
        {
            if (_leftUpperLeg != null)  _restLeftUpperLeg  = _leftUpperLeg.localRotation;
            if (_rightUpperLeg != null) _restRightUpperLeg = _rightUpperLeg.localRotation;
            if (_leftLowerLeg != null)  _restLeftLowerLeg  = _leftLowerLeg.localRotation;
            if (_rightLowerLeg != null) _restRightLowerLeg = _rightLowerLeg.localRotation;
            if (_leftFoot != null)      _restLeftFoot      = _leftFoot.localRotation;
            if (_rightFoot != null)     _restRightFoot     = _rightFoot.localRotation;
            if (_leftHand != null)      _restLeftHand      = _leftHand.localRotation;
            if (_rightHand != null)     _restRightHand     = _rightHand.localRotation;
            if (_leftShoulder != null)  _restLeftShoulder  = _leftShoulder.localRotation;
            if (_rightShoulder != null) _restRightShoulder = _rightShoulder.localRotation;
        }

        // ── Sitting IK: Legs dangle/rest on surface ─────────────

        private void ApplySittingIK(float blend)
        {
            if (!_hasCompleteLegRig) return;

            // Rotate upper legs forward (thighs horizontal for sitting)
            if (_leftUpperLeg != null)
            {
                _leftUpperLeg.localRotation = _restLeftUpperLeg *
                    Quaternion.AngleAxis(_sittingLegAngle * blend, Vector3.right);
            }

            if (_rightUpperLeg != null)
            {
                _rightUpperLeg.localRotation = _restRightUpperLeg *
                    Quaternion.AngleAxis(_sittingLegAngle * blend, Vector3.right);
            }

            // Bend knees (lower legs hang down)
            if (_leftLowerLeg != null)
            {
                _leftLowerLeg.localRotation = _restLeftLowerLeg *
                    Quaternion.AngleAxis(-_sittingKneeAngle * blend, Vector3.right);
            }

            if (_rightLowerLeg != null)
            {
                _rightLowerLeg.localRotation = _restRightLowerLeg *
                    Quaternion.AngleAxis(-_sittingKneeAngle * blend, Vector3.right);
            }

            // Straighten feet (toes point down naturally)
            if (_leftFoot != null)
            {
                _leftFoot.localRotation = _restLeftFoot *
                    Quaternion.AngleAxis(15f * blend, Vector3.right);
            }

            if (_rightFoot != null)
            {
                _rightFoot.localRotation = _restRightFoot *
                    Quaternion.AngleAxis(15f * blend, Vector3.right);
            }
        }

        // ── Drag Hand IK: Arms reach upward when grabbed ────────

        private void ApplyDragHandIK(float blend)
        {
            if (!_hasCompleteArmRig) return;

            // Raise shoulders slightly
            if (_leftShoulder != null)
            {
                _leftShoulder.localRotation = _restLeftShoulder *
                    Quaternion.AngleAxis(-20f * blend, Vector3.forward);
            }

            if (_rightShoulder != null)
            {
                _rightShoulder.localRotation = _restRightShoulder *
                    Quaternion.AngleAxis(20f * blend, Vector3.forward);
            }

            // Raise arms upward (reach toward grab point)
            if (_leftHand != null)
            {
                _leftHand.localRotation = _restLeftHand *
                    Quaternion.AngleAxis(-40f * blend, Vector3.forward);
            }

            if (_rightHand != null)
            {
                _rightHand.localRotation = _restRightHand *
                    Quaternion.AngleAxis(40f * blend, Vector3.forward);
            }
        }
    }
}
