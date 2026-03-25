using System;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Annabeth.Interaction
{
    /// <summary>
    /// Feature #11: Detect circular mouse motion over avatar's head → pet/pat reaction.
    /// Tracks mouse angle delta; when cumulative angle exceeds 2π, fires OnPetDetected.
    /// </summary>
    public class PetDetectionController : MonoBehaviour
    {
        [Header("Detection")]
        [SerializeField] private float petAngleThreshold = 360f; // degrees for a full circle
        [SerializeField] private float timeoutSeconds = 0.5f;
        [SerializeField] private float cooldownSeconds = 1f;

        public event Action OnPetDetected;

        private Transform _headBone;
        private Camera _cam;
        private float _cumulativeAngle;
        private Vector2 _lastDir;
        private bool _hasLastDir;
        private float _lastMotionTime;
        private float _cooldownTimer;

        public void SetHeadBone(Transform head)
        {
            _headBone = head;
            _cam = Camera.main;
        }

        private void Update()
        {
            if (_headBone == null || _cam == null) return;

            _cooldownTimer -= Time.deltaTime;
            if (_cooldownTimer > 0f) return;

            var mouse = Mouse.current;
            if (mouse == null) return;

            Vector2 mousePos = mouse.position.ReadValue();
            Vector3 headScreen = _cam.WorldToScreenPoint(_headBone.position);
            Vector2 headPos2D = new Vector2(headScreen.x, headScreen.y);

            Vector2 dir = (mousePos - headPos2D);
            if (dir.sqrMagnitude < 100f) // too close to center
            {
                _hasLastDir = false;
                return;
            }

            dir.Normalize();

            if (_hasLastDir)
            {
                // Calculate signed angle between consecutive direction vectors
                float angle = Vector2.SignedAngle(_lastDir, dir);
                _cumulativeAngle += angle;
                _lastMotionTime = Time.time;

                if (Mathf.Abs(_cumulativeAngle) >= petAngleThreshold)
                {
                    OnPetDetected?.Invoke();
                    _cumulativeAngle = 0f;
                    _cooldownTimer = cooldownSeconds;
                    _hasLastDir = false;
                    return;
                }
            }

            _lastDir = dir;
            _hasLastDir = true;

            // Timeout — reset if no motion for a while
            if (Time.time - _lastMotionTime > timeoutSeconds)
            {
                _cumulativeAngle = 0f;
                _hasLastDir = false;
            }
        }
    }
}
