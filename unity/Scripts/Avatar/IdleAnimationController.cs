using UnityEngine;
using UniVRM10;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Subtle idle animation: breathing (spine) + gentle head drift.
    /// Ported from companion.html applyIdleAnimation().
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

        private Vrm10Instance _vrm;
        private Animator _animator;
        private Transform _spine;
        private Transform _head;
        private Quaternion _origSpine;
        private Quaternion _origHead;
        private float _idleTime;
        private float _headPhaseOffset;
        private bool _enabled = true;

        public void Initialize(Vrm10Instance vrm)
        {
            _vrm = vrm;
            _animator = vrm?.GetComponent<Animator>();
            if (_animator == null) return;

            _spine = _animator.GetBoneTransform(HumanBodyBones.Spine);
            _head = _animator.GetBoneTransform(HumanBodyBones.Head);

            if (_spine) _origSpine = _spine.localRotation;
            if (_head) _origHead = _head.localRotation;

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
        }

        public void SetEnabled(bool enabled)
        {
            _enabled = enabled;
        }
    }
}
