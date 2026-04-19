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

        public static VariationProfile Relaxed => new VariationProfile { breathSpeed = 0.8f, headDriftSpeed = 0.3f, leanMax = 2.5f, breathAmplitude = 0.05f };
        public static VariationProfile Alert => new VariationProfile { breathSpeed = 1.2f, headDriftSpeed = 0.5f, leanMax = 3.5f, breathAmplitude = 0.04f };
        public static VariationProfile Bored => new VariationProfile { breathSpeed = 0.5f, headDriftSpeed = 0.7f, leanMax = 2.0f, breathAmplitude = 0.07f };

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
    /// Execution order 100 = runs BEFORE BeatDanceController (200) so dance wins on shared bones.
    /// </summary>
    [DefaultExecutionOrder(100)]
    public class IdleAnimationController : MonoBehaviour, IBlendableAnimation
    {
        [Header("Breathing")]
        [SerializeField] private float breathSpeed = 0.8f;
        [SerializeField] private float breathAmplitude = 0.05f;

        [Header("Head Drift")]
        [SerializeField] private float headDriftSpeedY = 0.3f;
        [SerializeField] private float headDriftSpeedX = 0.4f;
        [SerializeField] private float headDriftAmplitudeY = 0.012f;
        [SerializeField] private float headDriftAmplitudeX = 0.006f;

        [Header("Mouse Awareness (Body Lean)")]
        [Tooltip("Reduced — EyeTrackingController handles main spine tracking now")]
        [SerializeField] private float leanMaxAngle = 2.5f;
        [SerializeField] private float leanSmoothSpeed = 2f;

        [Header("Hip Sway (idle weight shifting)")]
        [SerializeField] private float hipSwaySpeed = 0.15f;
        [SerializeField] private float hipSwayAmplitude = 0.8f;
        [SerializeField] private float hipSwayVertical = 0.003f;

        [Header("Arm Idle Sway")]
        [SerializeField] private float armSwaySpeed = 0.12f;
        [SerializeField] private float armSwayAmplitudeZ = 1.5f;  // ±1.5° subtle pendulum
        [SerializeField] private float armSwayAmplitudeX = 0.8f;  // ±0.8° fore/aft
        [SerializeField] private float elbowDriftAmplitude = 2f;   // ±2° elbow fidget
        [SerializeField] private float shoulderShiftAmplitude = 0.4f; // subtle shoulder micro-movements

        private Vrm10Instance _vrm;
        private Animator _animator;
        private Transform _spine;
        private Transform _head;
        private Transform _upperBody;
        private Transform _hips;
        // Arm bones (via ControlRig for VRM-spec normalized rotations)
        private Transform _leftUpperArm;
        private Transform _rightUpperArm;
        private Transform _leftLowerArm;
        private Transform _rightLowerArm;
        private Transform _leftShoulder;
        private Transform _rightShoulder;
        private Quaternion _origSpine;
        private Quaternion _origHead;
        private Quaternion _origUpperBody;
        private Quaternion _origHips;
        private Vector3 _origHipsPos;
        // Arm/shoulder base rotations (from relaxed pose, not T-pose)
        private Quaternion _origLUA;
        private Quaternion _origRUA;
        private Quaternion _origLLA;
        private Quaternion _origRLA;
        private Quaternion _origLShoulder;
        private Quaternion _origRShoulder;
        private bool _hasArms;
        private float _armPhaseOffset;
        private float _idleTime;
        private float _headPhaseOffset;
        private float _hipSwayPhase;
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
            _hips = _animator.GetBoneTransform(HumanBodyBones.Hips);

            if (_spine) _origSpine = _spine.localRotation;
            if (_head) _origHead = _head.localRotation;
            if (_upperBody) _origUpperBody = _upperBody.localRotation;
            if (_hips) { _origHips = _hips.localRotation; _origHipsPos = _hips.localPosition; }

            // Cache arm + shoulder bones via ControlRig (VRM-spec normalized space)
            var rig = _vrm.Runtime?.ControlRig;
            if (rig != null)
            {
                _leftUpperArm = rig.GetBoneTransform(HumanBodyBones.LeftUpperArm);
                _rightUpperArm = rig.GetBoneTransform(HumanBodyBones.RightUpperArm);
                _leftLowerArm = rig.GetBoneTransform(HumanBodyBones.LeftLowerArm);
                _rightLowerArm = rig.GetBoneTransform(HumanBodyBones.RightLowerArm);
                _leftShoulder = rig.GetBoneTransform(HumanBodyBones.LeftShoulder);
                _rightShoulder = rig.GetBoneTransform(HumanBodyBones.RightShoulder);
            }
            // Snapshot relaxed arm pose (set by AvatarController.ApplyRelaxedPose)
            // so we can layer idle sway on top of it
            if (_leftUpperArm) _origLUA = _leftUpperArm.localRotation;
            if (_rightUpperArm) _origRUA = _rightUpperArm.localRotation;
            if (_leftLowerArm) _origLLA = _leftLowerArm.localRotation;
            if (_rightLowerArm) _origRLA = _rightLowerArm.localRotation;
            if (_leftShoulder) _origLShoulder = _leftShoulder.localRotation;
            if (_rightShoulder) _origRShoulder = _rightShoulder.localRotation;
            _hasArms = _leftUpperArm != null && _rightUpperArm != null;

            // Random phase so it doesn't look mechanical across multiple instances
            _headPhaseOffset = Random.Range(0f, Mathf.PI * 2f);
            _hipSwayPhase = Random.Range(0f, Mathf.PI * 2f);
            _armPhaseOffset = Random.Range(0f, Mathf.PI * 2f);

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

            // Subtle breathing on spine + upper chest
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

            // Body lean toward mouse — subtle upper body tilt + chest breathing
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

                    // Layer chest breathing: slightly delayed from spine for natural wave
                    float chestBreath = Mathf.Sin(_idleTime * effectiveBreathSpeed + 0.4f) * effectiveBreathAmp * 0.6f;
                    Quaternion target = _origUpperBody * Quaternion.Euler(_currentLeanX + chestBreath * Mathf.Rad2Deg, 0, _currentLeanZ);
                    _upperBody.localRotation = Quaternion.Slerp(_origUpperBody, target, w);
                }
            }

            // Hip sway — continuous gentle weight-shifting like a real person standing
            // Mate Engine uses similar idle hip oscillation for liveliness
            if (_hips)
            {
                float swayZ = Mathf.Sin(_idleTime * hipSwaySpeed * Mathf.PI * 2f + _hipSwayPhase) * hipSwayAmplitude;
                float swayY = Mathf.Sin(_idleTime * hipSwaySpeed * Mathf.PI * 2f * 2f + _hipSwayPhase) * hipSwayVertical;
                Quaternion hipTarget = _origHips * Quaternion.Euler(0f, 0f, swayZ * sleepMul);
                _hips.localRotation = Quaternion.Slerp(_origHips, hipTarget, w);
                _hips.localPosition = _origHipsPos + new Vector3(0f, swayY * sleepMul * w, 0f);
            }

            // ── Arm idle sway — gentle pendulum swing + elbow fidget ──
            // This prevents the "arms glued to sides" penguin look.
            // Uses ControlRig bones so it works identically on any VRM model.
            // Skip when blend weight is very low (during crossfade out) to avoid
            // fighting with BeatDanceController for the same ControlRig arm bones.
            if (_hasArms && w > 0.05f)
            {
                float armT = _idleTime * armSwaySpeed * Mathf.PI * 2f + _armPhaseOffset;
                float armMul = sleepMul * w;

                // Upper arms: gentle pendulum swing (Z = side-to-side, X = fore/aft)
                if (_leftUpperArm)
                {
                    float swZ = Mathf.Sin(armT) * armSwayAmplitudeZ;
                    float swX = Mathf.Sin(armT * 0.7f + 0.5f) * armSwayAmplitudeX;
                    Quaternion target = _origLUA * Quaternion.Euler(swX * armMul, 0f, swZ * armMul);
                    _leftUpperArm.localRotation = Quaternion.Slerp(_origLUA, target, w);
                }
                if (_rightUpperArm)
                {
                    // Offset phase so arms don't swing in perfect sync (looks robotic)
                    float swZ = Mathf.Sin(armT + 1.2f) * armSwayAmplitudeZ;
                    float swX = Mathf.Sin(armT * 0.7f + 1.8f) * armSwayAmplitudeX;
                    Quaternion target = _origRUA * Quaternion.Euler(swX * armMul, 0f, -swZ * armMul);
                    _rightUpperArm.localRotation = Quaternion.Slerp(_origRUA, target, w);
                }

                // Lower arms: subtle elbow drift (Y axis = elbow bend/extend)
                if (_leftLowerArm)
                {
                    float drift = Mathf.Sin(armT * 0.5f + 2f) * elbowDriftAmplitude;
                    Quaternion target = _origLLA * Quaternion.Euler(0f, -drift * armMul, 0f);
                    _leftLowerArm.localRotation = Quaternion.Slerp(_origLLA, target, w);
                }
                if (_rightLowerArm)
                {
                    float drift = Mathf.Sin(armT * 0.5f + 3f) * elbowDriftAmplitude;
                    Quaternion target = _origRLA * Quaternion.Euler(0f, drift * armMul, 0f);
                    _rightLowerArm.localRotation = Quaternion.Slerp(_origRLA, target, w);
                }

                // Shoulders: micro-shifts (gives subtle breathing-connected shoulder rise)
                if (_leftShoulder)
                {
                    float rise = Mathf.Sin(_idleTime * effectiveBreathSpeed + 0.3f) * shoulderShiftAmplitude;
                    Quaternion target = _origLShoulder * Quaternion.Euler(0f, 0f, rise * armMul);
                    _leftShoulder.localRotation = Quaternion.Slerp(_origLShoulder, target, w);
                }
                if (_rightShoulder)
                {
                    float rise = Mathf.Sin(_idleTime * effectiveBreathSpeed + 0.8f) * shoulderShiftAmplitude;
                    Quaternion target = _origRShoulder * Quaternion.Euler(0f, 0f, -rise * armMul);
                    _rightShoulder.localRotation = Quaternion.Slerp(_origRShoulder, target, w);
                }
            }
        }

        public void SetEnabled(bool enabled)
        {
            _enabled = enabled;
        }
    }
}
