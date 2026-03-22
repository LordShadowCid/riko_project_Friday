using UnityEngine;
using UnityEngine.InputSystem;
using UniVRM10;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Subtle idle animation: breathing (spine) + gentle head drift + body lean toward mouse.
    /// Ported from companion.html applyIdleAnimation() with Mate-Engine style awareness.
    /// Runs in LateUpdate so it layers on top of other controllers.
    /// </summary>
    public class IdleAnimationController : MonoBehaviour
    {
        [Header("Breathing")]
        [SerializeField] private float breathSpeed = 0.8f;
        [SerializeField] private float breathAmplitude = 0.01f;

        [Header("Head Drift")]
        [SerializeField] private float headDriftSpeedY = 0.3f;
        [SerializeField] private float headDriftSpeedX = 0.4f;
        [SerializeField] private float headDriftAmplitudeY = 0.002f;
        [SerializeField] private float headDriftAmplitudeX = 0.001f;

        [Header("Mouse Awareness (Body Lean)")]
        [SerializeField] private float leanMaxAngle = 3f;
        [SerializeField] private float leanSmoothSpeed = 2f;

        private Vrm10Instance _vrm;
        private Animator _animator;
        private Transform _spine;
        private Transform _head;
        private Transform _upperBody;
        private Quaternion _origSpine;
        private Quaternion _origHead;
        private Quaternion _origUpperBody;
        private float _idleTime;
        private float _headPhaseOffset;
        private bool _enabled = true;
        private Camera _cam;
        private float _currentLeanX;
        private float _currentLeanZ;

        public void Initialize(Vrm10Instance vrm)
        {
            _vrm = vrm;
            _animator = vrm?.GetComponent<Animator>();
            _cam = Camera.main;
            if (_animator == null) return;

            _spine = _animator.GetBoneTransform(HumanBodyBones.Spine);
            _head = _animator.GetBoneTransform(HumanBodyBones.Head);
            _upperBody = _animator.GetBoneTransform(HumanBodyBones.UpperChest);
            if (_upperBody == null)
                _upperBody = _animator.GetBoneTransform(HumanBodyBones.Chest);

            if (_spine) _origSpine = _spine.localRotation;
            if (_head) _origHead = _head.localRotation;
            if (_upperBody) _origUpperBody = _upperBody.localRotation;

            // Random phase so it doesn't look mechanical across multiple instances
            _headPhaseOffset = Random.Range(0f, Mathf.PI * 2f);
        }

        private void LateUpdate()
        {
            if (!_enabled || _vrm == null) return;

            _idleTime += Time.deltaTime;

            // Subtle breathing on spine
            if (_spine)
            {
                float breathX = Mathf.Sin(_idleTime * breathSpeed) * breathAmplitude;
                _spine.localRotation = _origSpine * Quaternion.Euler(breathX * Mathf.Rad2Deg, 0, 0);
            }

            // Gentle head drift — apply from captured base rotation to prevent
            // frame-over-frame accumulation (which caused visible head spinning).
            if (_head)
            {
                float driftY = Mathf.Sin(_idleTime * headDriftSpeedY + _headPhaseOffset) * headDriftAmplitudeY;
                float driftX = Mathf.Sin(_idleTime * headDriftSpeedX + _headPhaseOffset) * headDriftAmplitudeX;
                _head.localRotation = _origHead * Quaternion.Euler(driftX * Mathf.Rad2Deg, driftY * Mathf.Rad2Deg, 0);
            }

            // Body lean toward mouse — subtle upper body tilt
            if (_upperBody && _cam != null)
            {
                var mouse = Mouse.current;
                if (mouse != null)
                {
                    Vector2 mousePos = mouse.position.ReadValue();
                    // Normalize to -1..1 (center of screen = 0)
                    float nx = (mousePos.x / Screen.width) * 2f - 1f;
                    float ny = (mousePos.y / Screen.height) * 2f - 1f;

                    // Target lean: tilt toward mouse (Z for left/right, X for forward/back)
                    float targetLeanZ = -nx * leanMaxAngle;
                    float targetLeanX = -ny * leanMaxAngle * 0.5f;

                    _currentLeanZ = Mathf.Lerp(_currentLeanZ, targetLeanZ, Time.deltaTime * leanSmoothSpeed);
                    _currentLeanX = Mathf.Lerp(_currentLeanX, targetLeanX, Time.deltaTime * leanSmoothSpeed);

                    _upperBody.localRotation = _origUpperBody * Quaternion.Euler(_currentLeanX, 0, _currentLeanZ);
                }
            }
        }

        public void SetEnabled(bool enabled)
        {
            _enabled = enabled;
        }
    }
}
