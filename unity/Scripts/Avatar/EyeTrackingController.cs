using UnityEngine;
using UnityEngine.InputSystem;
using UniVRM10;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Tracking modes for state-aware tracking permissions (Feature #10).
    /// </summary>
    public enum TrackingMode
    {
        Normal,     // Full tracking — idle, active
        Reduced,    // Reduced speed — dancing (eyes follow beat, not cursor)
        Disabled,   // No tracking — sleeping
        LookUp      // Look toward cursor above — being dragged
    }

    /// <summary>
    /// Controls eye, head, and upper body tracking to follow the mouse cursor.
    /// Uses VRM LookAt for eyes/head, and cascading spine rotation for body.
    /// Mate Engine-style per-component track speeds: eyes lead, head follows, body last.
    /// </summary>
    public class EyeTrackingController : MonoBehaviour
    {
        [Header("Eye Tracking Settings")]
        [SerializeField] private bool enableEyeTracking = true;
        [SerializeField] private float maxHorizontalAngle = 30f;
        [SerializeField] private float maxVerticalAngle = 20f;
        [SerializeField] private float lookAtDistance = 2f;

        [Header("Per-Component Speeds (Mate Engine style)")]
        [SerializeField] private float eyeSpeed = 8f;          // Fast — eyes lead
        [SerializeField] private float headSpeed = 4f;          // Medium — head follows
        [SerializeField] private float bodySpeed = 2f;          // Slow — body last

        [Header("Spine/Body Tracking")]
        [SerializeField] private float spineBlend = 0.5f;       // 0..1 blend for body lean
        [SerializeField] private float chestMultiplier = 0.8f;   // Cascading: chest gets 80%
        [SerializeField] private float upperChestMultiplier = 0.6f; // Upper chest gets 60%
        [SerializeField] private float spineMaxYaw = 15f;        // Max horizontal rotation degrees
        [SerializeField] private float spineMaxPitch = 10f;      // Max vertical rotation degrees

        [Header("References")]
        [SerializeField] private Camera mainCamera;

        private Vrm10Instance _vrm;
        private Transform _lookAtTarget;
        private Vector3 _currentLookAt;
        private Vector3 _targetLookAt;
        private Transform _headBone;

        // Spine tracking bones
        private Transform _spine;
        private Transform _chest;
        private Transform _upperChest;
        private Quaternion _origSpine;
        private Quaternion _origChest;
        private Quaternion _origUpperChest;

        // Separate interpolation states (per-component speed)
        private Vector3 _eyeLookAt;   // Fastest — drives VRM LookAt target
        private Vector3 _headLookAt;  // Medium — unused directly but kept for head speed
        private float _currentBodyYaw;
        private float _currentBodyPitch;

        // Feature #10: State-aware tracking
        private TrackingMode _trackingMode = TrackingMode.Normal;
        private float _savedEyeSpeed, _savedHeadSpeed, _savedBodySpeed, _savedSpineBlend;

        public void Initialize(Vrm10Instance vrm)
        {
            _vrm = vrm;

            if (mainCamera == null)
                mainCamera = Camera.main;

            var animator = vrm?.GetComponent<Animator>();
            _headBone = animator?.GetBoneTransform(HumanBodyBones.Head);

            // Cache spine chain for body tracking
            _spine = animator?.GetBoneTransform(HumanBodyBones.Spine);
            _chest = animator?.GetBoneTransform(HumanBodyBones.Chest);
            _upperChest = animator?.GetBoneTransform(HumanBodyBones.UpperChest);
            if (_upperChest == null) _upperChest = _chest; // Fallback

            if (_spine) _origSpine = _spine.localRotation;
            if (_chest) _origChest = _chest.localRotation;
            if (_upperChest && _upperChest != _chest) _origUpperChest = _upperChest.localRotation;

            // Create look-at target (not parented to avatar)
            var targetObj = new GameObject("EyeTrackTarget");
            _lookAtTarget = targetObj.transform;

            // Initialize positions to camera to avoid initial jerk
            if (mainCamera != null)
            {
                _eyeLookAt = mainCamera.transform.position;
                _headLookAt = _eyeLookAt;
                _currentLookAt = _eyeLookAt;
                _targetLookAt = _eyeLookAt;
                _lookAtTarget.position = _eyeLookAt;
            }

            // Configure VRM LookAt
            if (_vrm != null)
            {
                _vrm.LookAtTargetType = VRM10ObjectLookAt.LookAtTargetTypes.SpecifiedTransform;
                _vrm.LookAtTarget = _lookAtTarget;
            }
        }

        private void Update()
        {
            if (!enableEyeTracking || _vrm == null || mainCamera == null) return;
            UpdateLookAtTarget();
        }

        private void LateUpdate()
        {
            if (!enableEyeTracking || _vrm == null || spineBlend <= 0f) return;
            ApplySpineTracking();
        }

        private void UpdateLookAtTarget()
        {
            Vector3 mousePos = Mouse.current != null ? Mouse.current.position.ReadValue() : Vector3.zero;

            float normalizedX = (mousePos.x / Screen.width - 0.5f) * 2f;
            float normalizedY = (mousePos.y / Screen.height - 0.5f) * 2f;

            if (_headBone == null) return;

            Vector3 headPos = _headBone.position;
            Vector3 toCamera = (mainCamera.transform.position - headPos).normalized;

            // Negate horizontal: camera faces -Z (180° Y), so camera.right is world -X.
            // Without negation, mouse-right maps to character-right (viewer's left) = inverted.
            float horizontalOffset = -normalizedX * Mathf.Tan(maxHorizontalAngle * Mathf.Deg2Rad) * lookAtDistance;
            float verticalOffset = normalizedY * Mathf.Tan(maxVerticalAngle * Mathf.Deg2Rad) * lookAtDistance;

            _targetLookAt = headPos + toCamera * lookAtDistance +
                           mainCamera.transform.right * horizontalOffset +
                           mainCamera.transform.up * verticalOffset;

            float dt = Time.deltaTime;

            // Per-component speed: eyes lerp fastest (drives VRM LookAt position)
            _eyeLookAt = Vector3.Lerp(_eyeLookAt, _targetLookAt, dt * eyeSpeed);
            _lookAtTarget.position = _eyeLookAt;

            // Head speed — the VRM LookAt system handles head rotation internally,
            // but we control the target position with eye speed. The head naturally
            // lags behind the eyes via VRM's built-in head/eye split.
        }

        /// <summary>
        /// Cascading spine rotation toward cursor. Applied in LateUpdate so it
        /// layers on top of idle/dance animations. Mate Engine-style cascading:
        /// spine gets full amount, chest gets chestMultiplier, upperChest gets upperChestMultiplier.
        /// </summary>
        private void ApplySpineTracking()
        {
            if (_headBone == null || mainCamera == null) return;

            var mouse = Mouse.current;
            if (mouse == null) return;

            Vector2 mousePos = mouse.position.ReadValue();
            float nx = (mousePos.x / Screen.width) * 2f - 1f;  // -1..1
            float ny = (mousePos.y / Screen.height) * 2f - 1f;

            // Target yaw/pitch for spine chain
            // Negate yaw: camera faces -Z, so positive yaw turns character toward +X
            // which appears as LEFT on screen. Negate to match mouse direction.
            float targetYaw = -nx * spineMaxYaw;
            float targetPitch = -ny * spineMaxPitch;

            float dt = Time.deltaTime;
            _currentBodyYaw = Mathf.Lerp(_currentBodyYaw, targetYaw, dt * bodySpeed);
            _currentBodyPitch = Mathf.Lerp(_currentBodyPitch, targetPitch, dt * bodySpeed);

            float w = spineBlend;

            // Apply cascading rotation from original base — each bone gets a fraction
            if (_spine)
            {
                Quaternion rot = Quaternion.Euler(_currentBodyPitch * w, _currentBodyYaw * w, 0f);
                _spine.localRotation = _origSpine * rot;
            }

            if (_chest)
            {
                float cm = chestMultiplier;
                Quaternion rot = Quaternion.Euler(_currentBodyPitch * w * cm, _currentBodyYaw * w * cm, 0f);
                _chest.localRotation = _origChest * rot;
            }

            if (_upperChest && _upperChest != _chest)
            {
                float ucm = upperChestMultiplier;
                Quaternion rot = Quaternion.Euler(_currentBodyPitch * w * ucm, _currentBodyYaw * w * ucm, 0f);
                _upperChest.localRotation = _origUpperChest * rot;
            }
        }

        public void SetEnabled(bool enabled)
        {
            enableEyeTracking = enabled;

            if (!enabled && _vrm?.Vrm.LookAt != null)
            {
                var headBone = _headBone;
                if (headBone != null && _lookAtTarget != null && mainCamera != null)
                {
                    Vector3 toCamera = (mainCamera.transform.position - headBone.position).normalized;
                    _lookAtTarget.position = headBone.position + toCamera * lookAtDistance;
                }
            }
        }

        public void LookAt(Vector3 worldPosition)
        {
            _targetLookAt = worldPosition;
        }

        // ── Settings API ────────────────────────────────────────
        public void SetEyeSpeed(float speed) => eyeSpeed = speed;
        public void SetHeadSpeed(float speed) => headSpeed = speed;
        public void SetBodySpeed(float speed) => bodySpeed = speed;
        public void SetSpineBlend(float blend) => spineBlend = blend;

        /// <summary>Current tracking mode.</summary>
        public TrackingMode CurrentTrackingMode => _trackingMode;

        /// <summary>
        /// Feature #10: State-aware tracking permissions.
        /// Adjusts tracking behavior based on companion state.
        /// </summary>
        public void SetTrackingMode(TrackingMode mode)
        {
            if (_trackingMode == mode) return;

            // Restore saved speeds when leaving a modified mode
            if (_trackingMode != TrackingMode.Normal)
            {
                eyeSpeed = _savedEyeSpeed;
                headSpeed = _savedHeadSpeed;
                bodySpeed = _savedBodySpeed;
                spineBlend = _savedSpineBlend;
            }

            // Save current speeds before modification
            if (mode != TrackingMode.Normal && _trackingMode == TrackingMode.Normal)
            {
                _savedEyeSpeed = eyeSpeed;
                _savedHeadSpeed = headSpeed;
                _savedBodySpeed = bodySpeed;
                _savedSpineBlend = spineBlend;
            }

            _trackingMode = mode;

            switch (mode)
            {
                case TrackingMode.Normal:
                    // Respect the user's saved mouse tracking toggle.
                    // If they disabled tracking in settings, don't re-enable it.
                    var settings = Core.SettingsManager.Instance;
                    enableEyeTracking = settings != null ? settings.data.enableMouseTracking : true;
                    break;
                case TrackingMode.Reduced:
                    enableEyeTracking = true;
                    eyeSpeed *= 0.3f;
                    headSpeed *= 0.2f;
                    bodySpeed = 0f;
                    spineBlend = 0f;
                    break;
                case TrackingMode.Disabled:
                    enableEyeTracking = false;
                    break;
                case TrackingMode.LookUp:
                    enableEyeTracking = true;
                    eyeSpeed = 12f;
                    headSpeed = 6f;
                    bodySpeed = 0f;
                    spineBlend = 0f;
                    break;
            }

            Debug.Log($"[EyeTracking] Tracking mode: {mode}");
        }

        private void OnDestroy()
        {
            if (_lookAtTarget != null)
            {
                Destroy(_lookAtTarget.gameObject);
            }
        }
    }
}
