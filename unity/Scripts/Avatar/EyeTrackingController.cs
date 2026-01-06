using UnityEngine;
using UniVRM10;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Controls eye tracking to follow the mouse cursor.
    /// Uses VRM LookAt system for natural eye movement.
    /// </summary>
    public class EyeTrackingController : MonoBehaviour
    {
        [Header("Eye Tracking Settings")]
        [SerializeField] private bool enableEyeTracking = true;
        [SerializeField] private float lookAtSpeed = 5f;
        [SerializeField] private float maxHorizontalAngle = 30f;
        [SerializeField] private float maxVerticalAngle = 20f;
        [SerializeField] private float lookAtDistance = 2f;

        [Header("References")]
        [SerializeField] private Camera mainCamera;

        private Vrm10Instance _vrm;
        private Transform _lookAtTarget;
        private Vector3 _currentLookAt;
        private Vector3 _targetLookAt;

        public void Initialize(Vrm10Instance vrm)
        {
            _vrm = vrm;

            if (mainCamera == null)
            {
                mainCamera = Camera.main;
            }

            // Create look-at target
            var targetObj = new GameObject("EyeTrackTarget");
            _lookAtTarget = targetObj.transform;
            _lookAtTarget.SetParent(transform);
            _lookAtTarget.localPosition = Vector3.forward * lookAtDistance;

            // Configure VRM LookAt
            if (_vrm != null && _vrm.Vrm.LookAt != null)
            {
                _vrm.Vrm.LookAt.LookAtTargetType = VRM10ObjectLookAt.LookAtTargetTypes.SpecifiedTransform;
                _vrm.Vrm.LookAt.LookAtTarget = _lookAtTarget;
            }
        }

        private void Update()
        {
            if (!enableEyeTracking || _vrm == null || mainCamera == null) return;

            UpdateLookAtTarget();
        }

        private void UpdateLookAtTarget()
        {
            // Get mouse position in screen space
            Vector3 mousePos = Input.mousePosition;
            
            // Convert to normalized screen coordinates (-1 to 1)
            float normalizedX = (mousePos.x / Screen.width - 0.5f) * 2f;
            float normalizedY = (mousePos.y / Screen.height - 0.5f) * 2f;

            // Calculate target position in world space (in front of avatar)
            Transform headBone = GetHeadBone();
            if (headBone == null) return;

            Vector3 headPos = headBone.position;
            Vector3 headForward = headBone.forward;
            Vector3 headRight = headBone.right;
            Vector3 headUp = headBone.up;

            // Calculate horizontal and vertical offsets based on mouse position
            float horizontalOffset = normalizedX * Mathf.Tan(maxHorizontalAngle * Mathf.Deg2Rad) * lookAtDistance;
            float verticalOffset = normalizedY * Mathf.Tan(maxVerticalAngle * Mathf.Deg2Rad) * lookAtDistance;

            // Target position in front of head, offset by mouse position
            _targetLookAt = headPos + headForward * lookAtDistance +
                           headRight * horizontalOffset +
                           headUp * verticalOffset;

            // Smooth movement
            _currentLookAt = Vector3.Lerp(_currentLookAt, _targetLookAt, Time.deltaTime * lookAtSpeed);
            _lookAtTarget.position = _currentLookAt;
        }

        private Transform GetHeadBone()
        {
            if (_vrm == null) return null;
            
            var animator = _vrm.GetComponent<Animator>();
            return animator?.GetBoneTransform(HumanBodyBones.Head);
        }

        /// <summary>
        /// Enable or disable eye tracking.
        /// </summary>
        public void SetEnabled(bool enabled)
        {
            enableEyeTracking = enabled;

            if (!enabled && _vrm?.Vrm.LookAt != null)
            {
                // Reset to looking forward
                var headBone = GetHeadBone();
                if (headBone != null && _lookAtTarget != null)
                {
                    _lookAtTarget.position = headBone.position + headBone.forward * lookAtDistance;
                }
            }
        }

        /// <summary>
        /// Set a specific world position to look at.
        /// </summary>
        public void LookAt(Vector3 worldPosition)
        {
            _targetLookAt = worldPosition;
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
