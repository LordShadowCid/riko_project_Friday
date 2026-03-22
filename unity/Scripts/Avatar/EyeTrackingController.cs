using UnityEngine;
using UnityEngine.InputSystem;
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

            // Create look-at target (not parented to avatar to avoid
            // any influence from model hierarchy)
            var targetObj = new GameObject("EyeTrackTarget");
            _lookAtTarget = targetObj.transform;

            // Initialize _currentLookAt to camera position so we don't
            // lerp from Vector3.zero (would cause initial head jerk)
            if (mainCamera != null)
            {
                _currentLookAt = mainCamera.transform.position;
                _targetLookAt = _currentLookAt;
                _lookAtTarget.position = _currentLookAt;
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

        private void UpdateLookAtTarget()
        {
            Vector3 mousePos = Mouse.current != null ? Mouse.current.position.ReadValue() : Vector3.zero;
            
            float normalizedX = (mousePos.x / Screen.width - 0.5f) * 2f;
            float normalizedY = (mousePos.y / Screen.height - 0.5f) * 2f;

            Transform headBone = GetHeadBone();
            if (headBone == null) return;

            Vector3 headPos = headBone.position;

            // Use direction from head toward camera as the stable base direction.
            // Do NOT use headBone.forward — LookAt rotates the head bone, which
            // moves the target, creating a feedback loop (slow 360° spin).
            Vector3 toCamera = (mainCamera.transform.position - headPos).normalized;

            float horizontalOffset = normalizedX * Mathf.Tan(maxHorizontalAngle * Mathf.Deg2Rad) * lookAtDistance;
            float verticalOffset = normalizedY * Mathf.Tan(maxVerticalAngle * Mathf.Deg2Rad) * lookAtDistance;

            _targetLookAt = headPos + toCamera * lookAtDistance +
                           mainCamera.transform.right * horizontalOffset +
                           mainCamera.transform.up * verticalOffset;

            _currentLookAt = Vector3.Lerp(_currentLookAt, _targetLookAt, Time.deltaTime * lookAtSpeed);
            _lookAtTarget.position = _currentLookAt;
        }

        private Transform GetHeadBone()
        {
            if (_vrm == null) return null;
            
            var animator = _vrm.GetComponent<Animator>();
            return animator?.GetBoneTransform(HumanBodyBones.Head);
        }

        public void SetEnabled(bool enabled)
        {
            enableEyeTracking = enabled;

            if (!enabled && _vrm?.Vrm.LookAt != null)
            {
                // Reset target to in front of face using camera direction (not headBone.forward)
                var headBone = GetHeadBone();
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

        private void OnDestroy()
        {
            if (_lookAtTarget != null)
            {
                Destroy(_lookAtTarget.gameObject);
            }
        }
    }
}
