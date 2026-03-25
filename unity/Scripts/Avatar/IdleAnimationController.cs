using UnityEngine;
using UnityEngine.InputSystem;
using UniVRM10;

namespace Annabeth.Avatar
{
    /// <summary>Feature #7: Idle behavior variations.</summary>
    public enum IdleVariation { Relaxed, Alert, Bored }

    /// <summary>Feature #7: Parameter profile for each idle variation.</summary>
    public struct VariationProfile
    {
        public float breathSpeed, headDriftSpeed, leanMax, breathAmplitude;

        public static VariationProfile Relaxed => new VariationProfile { breathSpeed = 0.8f, headDriftSpeed = 0.3f, leanMax = 1.5f, breathAmplitude = 0.01f };
        public static VariationProfile Alert => new VariationProfile { breathSpeed = 1.2f, headDriftSpeed = 0.5f, leanMax = 2.5f, breathAmplitude = 0.008f };
        public static VariationProfile Bored => new VariationProfile { breathSpeed = 0.5f, headDriftSpeed = 0.7f, leanMax = 1.0f, breathAmplitude = 0.014f };

        public static VariationProfile Lerp(VariationProfile a, VariationProfile b, float t)
        {
            return new VariationProfile
            {
                breathSpeed = Mathf.Lerp(a.breathSpeed, b.breathSpeed, t),
                headDriftSpeed = Mathf.Lerp(a.headDriftSpeed, b.headDriftSpeed, t),
                leanMax = Mathf.Lerp(a.leanMax, b.leanMax, t),
                breathAmplitude = Mathf.Lerp(a.breathAmplitude, b.breathAmplitude, t)
            };
        }
    }

    /// <summary>
    /// Subtle idle animation: breathing (spine) + gentle head drift + body lean toward mouse.
    /// Ported from companion.html applyIdleAnimation() with Mate-Engine style awareness.
    /// Runs in LateUpdate so it layers on top of other controllers.
    /// </summary>
    public class IdleAnimationController : MonoBehaviour, IBlendableAnimation
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
        [Tooltip("Reduced — EyeTrackingController handles main spine tracking now")]
        [SerializeField] private float leanMaxAngle = 1.5f;
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
        private float _blendWeight = 1f;
        private Camera _cam;
        private float _currentLeanX;
        private float _currentLeanZ;

        // Feature #6: Sleep mode
        private bool _isSleeping;
        private float _sleepBlend; // 0..1, lerps toward sleeping state

        // Feature #7: Idle variation
        private IdleVariation _currentVariation = IdleVariation.Relaxed;
        private float _variationTimer;
        private float _variationInterval = 18f; // seconds between variation changes
        private VariationProfile _currentProfile;
        private VariationProfile _targetProfile;

        // Feature #8: Peek/hide lean
        private float _peekLean; // -1 = lean right (peeking from left edge), +1 = lean left
        private float _currentPeekLean;

        // IBlendableAnimation
        public float BlendWeight => _blendWeight;
        public void SetBlendWeight(float weight) => _blendWeight = weight;
        public void SetBlendActive(bool active)
        {
            _enabled = active;
            if (!active) _blendWeight = 0f;
        }

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

            // Feature #7: Initialize variation profiles
            _currentProfile = VariationProfile.Relaxed;
            _targetProfile = VariationProfile.Relaxed;
            _variationTimer = _variationInterval;
        }

        /// <summary>Feature #6: Put idle animation into sleep mode.</summary>
        public void SetSleeping(bool sleeping)
        {
            _isSleeping = sleeping;
        }

        /// <summary>Feature #8: Set peek lean direction. -1=left edge, +1=right edge, 0=none.</summary>
        public void SetPeekLean(float direction)
        {
            _peekLean = Mathf.Clamp(direction, -1f, 1f);
        }

        private void LateUpdate()
        {
            if (!_enabled || _blendWeight <= 0f || _vrm == null) return;

            _idleTime += Time.deltaTime;
            float w = _blendWeight;

            // Feature #6: Smoothly blend toward/from sleep state
            float sleepTarget = _isSleeping ? 1f : 0f;
            _sleepBlend = Mathf.MoveTowards(_sleepBlend, sleepTarget, Time.deltaTime * 0.5f);

            // Feature #7: Variation timer — cycle through idle variations
            _variationTimer -= Time.deltaTime;
            if (_variationTimer <= 0f)
            {
                _variationTimer = _variationInterval + Random.Range(-3f, 3f);
                _currentProfile = _targetProfile;
                var next = (IdleVariation)(((int)_currentVariation + 1 + Random.Range(0, 2)) % 3);
                _currentVariation = next;
                _targetProfile = next switch
                {
                    IdleVariation.Alert => VariationProfile.Alert,
                    IdleVariation.Bored => VariationProfile.Bored,
                    _ => VariationProfile.Relaxed,
                };
            }
            float variationT = 1f - Mathf.Clamp01(_variationTimer / _variationInterval);
            VariationProfile profile = VariationProfile.Lerp(_currentProfile, _targetProfile, variationT);

            // Apply profile values (modulated by sleep blend — sleeping reduces everything)
            float sleepMul = 1f - _sleepBlend * 0.5f; // 50% reduction when sleeping
            float effectiveBreathSpeed = profile.breathSpeed * sleepMul;
            float effectiveBreathAmp = profile.breathAmplitude * sleepMul;
            float effectiveHeadDriftY = headDriftAmplitudeY * (profile.headDriftSpeed / 0.3f) * sleepMul;
            float effectiveHeadDriftX = headDriftAmplitudeX * (profile.headDriftSpeed / 0.3f) * sleepMul;
            float effectiveLeanMax = profile.leanMax * sleepMul;

            // Feature #8: Smooth peek lean
            _currentPeekLean = Mathf.Lerp(_currentPeekLean, _peekLean, Time.deltaTime * 3f);
            float peekSpineTilt = _currentPeekLean * 20f; // ±20° spine tilt away from edge
            float peekHeadCounter = -_currentPeekLean * 10f; // head counter-tilts toward center

            // Subtle breathing on spine
            if (_spine)
            {
                float breathX = Mathf.Sin(_idleTime * effectiveBreathSpeed) * effectiveBreathAmp;
                Quaternion target = _origSpine * Quaternion.Euler(breathX * Mathf.Rad2Deg, 0, peekSpineTilt);
                _spine.localRotation = Quaternion.Slerp(_origSpine, target, w);
            }

            // Gentle head drift — apply from captured base rotation to prevent
            // frame-over-frame accumulation (which caused visible head spinning).
            // Feature #6: When sleeping, tilt head down ~15°
            // Feature #8: Head counter-tilt during peek
            if (_head)
            {
                float driftY = Mathf.Sin(_idleTime * headDriftSpeedY + _headPhaseOffset) * effectiveHeadDriftY;
                float driftX = Mathf.Sin(_idleTime * headDriftSpeedX + _headPhaseOffset) * effectiveHeadDriftX;
                float sleepTilt = _sleepBlend * 15f; // 15° head nod when sleeping
                Quaternion target = _origHead * Quaternion.Euler(driftX * Mathf.Rad2Deg + sleepTilt, driftY * Mathf.Rad2Deg, peekHeadCounter);
                _head.localRotation = Quaternion.Slerp(_origHead, target, w);
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
                    float targetLeanZ = -nx * effectiveLeanMax;
                    float targetLeanX = -ny * effectiveLeanMax * 0.5f;

                    _currentLeanZ = Mathf.Lerp(_currentLeanZ, targetLeanZ, Time.deltaTime * leanSmoothSpeed);
                    _currentLeanX = Mathf.Lerp(_currentLeanX, targetLeanX, Time.deltaTime * leanSmoothSpeed);

                    Quaternion target = _origUpperBody * Quaternion.Euler(_currentLeanX, 0, _currentLeanZ);
                    _upperBody.localRotation = Quaternion.Slerp(_origUpperBody, target, w);
                }
            }
        }

        public void SetEnabled(bool enabled)
        {
            _enabled = enabled;
        }
    }
}
