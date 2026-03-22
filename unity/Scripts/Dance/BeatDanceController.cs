using UnityEngine;
using UniVRM10;

namespace Annabeth.Dance
{
    /// <summary>
    /// Beat-reactive procedural dance animation.
    /// Ported from the JavaScript applyBeatDance() in companion.html.
    /// 13 bones: head, neck, spine, chest, hips, L/R upper arm, L/R lower arm, L/R shoulder.
    /// Always has baseline movement; audio makes it more energetic.
    /// </summary>
    public class BeatDanceController : MonoBehaviour
    {
        [Header("Intensity")]
        [SerializeField] private float baseIntensityNormal = 0.5f;
        [SerializeField] private float baseIntensityFull = 0.7f;
        [SerializeField] private float audioBoostScale = 0.5f;
        [SerializeField] private float energySmoothFactor = 0.85f;

        [Header("Phase Speeds")]
        [SerializeField] private float dancePhaseSpeed = 4f;
        [SerializeField] private float headBobSpeed = 8f;
        [SerializeField] private float armSwaySpeed = 3f;
        [SerializeField] private float hipSwaySpeed = 2f;

        [Header("Happy Expression")]
        [SerializeField] private float happyBase = 0.3f;
        [SerializeField] private float happyScale = 0.4f;

        private Vrm10Instance _vrm;
        private Vrm10RuntimeExpression _expression;
        private Animator _animator;
        private bool _isDancing;
        private bool _isIntense;

        // Phase accumulators
        private float _dancePhase;
        private float _headBobPhase;
        private float _armSwayPhase;
        private float _hipSwayPhase;

        // Audio energy (smoothed)
        private float _danceIntensity;
        private float _bassEnergy;
        private float _midEnergy;
        private float _highEnergy;
        private bool _isBeat;

        // Bone transforms (13 bones)
        private Transform _head;
        private Transform _neck;
        private Transform _spine;
        private Transform _chest;
        private Transform _hips;
        private Transform _leftUpperArm;
        private Transform _rightUpperArm;
        private Transform _leftLowerArm;
        private Transform _rightLowerArm;
        private Transform _leftShoulder;
        private Transform _rightShoulder;

        // Original transforms for reset
        private Vector3 _originalHipsPos;
        private Quaternion _origHead, _origNeck, _origSpine, _origChest, _origHips;
        private Quaternion _origLUA, _origRUA, _origLLA, _origRLA;
        private Quaternion _origLS, _origRS;

        // Shoulder decay tracking
        private float _leftShoulderX;
        private float _rightShoulderX;

        public bool IsDancing => _isDancing;

        public void Initialize(Vrm10Instance vrm)
        {
            _vrm = vrm;
            _expression = vrm?.Runtime?.Expression;
            _animator = vrm?.GetComponent<Animator>();
            if (_animator == null) return;

            // Cache 13 bones
            _head = _animator.GetBoneTransform(HumanBodyBones.Head);
            _neck = _animator.GetBoneTransform(HumanBodyBones.Neck);
            _spine = _animator.GetBoneTransform(HumanBodyBones.Spine);
            _chest = _animator.GetBoneTransform(HumanBodyBones.Chest);
            _hips = _animator.GetBoneTransform(HumanBodyBones.Hips);
            _leftUpperArm = _animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
            _rightUpperArm = _animator.GetBoneTransform(HumanBodyBones.RightUpperArm);
            _leftLowerArm = _animator.GetBoneTransform(HumanBodyBones.LeftLowerArm);
            _rightLowerArm = _animator.GetBoneTransform(HumanBodyBones.RightLowerArm);
            _leftShoulder = _animator.GetBoneTransform(HumanBodyBones.LeftShoulder);
            _rightShoulder = _animator.GetBoneTransform(HumanBodyBones.RightShoulder);

            StoreOriginalTransforms();
        }

        private void StoreOriginalTransforms()
        {
            if (_hips) _originalHipsPos = _hips.localPosition;
            if (_head) _origHead = _head.localRotation;
            if (_neck) _origNeck = _neck.localRotation;
            if (_spine) _origSpine = _spine.localRotation;
            if (_chest) _origChest = _chest.localRotation;
            if (_hips) _origHips = _hips.localRotation;
            if (_leftUpperArm) _origLUA = _leftUpperArm.localRotation;
            if (_rightUpperArm) _origRUA = _rightUpperArm.localRotation;
            if (_leftLowerArm) _origLLA = _leftLowerArm.localRotation;
            if (_rightLowerArm) _origRLA = _rightLowerArm.localRotation;
            if (_leftShoulder) _origLS = _leftShoulder.localRotation;
            if (_rightShoulder) _origRS = _rightShoulder.localRotation;
        }

        private void LateUpdate()
        {
            if (!_isDancing || _vrm == null) return;

            float dt = Time.deltaTime;
            float intenseMult = _isIntense ? 2.5f : 1.0f;
            float speedMult = _isIntense ? 2.0f : 1.0f;

            // Advance phases
            _dancePhase += dt * dancePhaseSpeed * speedMult;
            _headBobPhase += dt * headBobSpeed * speedMult;
            _armSwayPhase += dt * armSwaySpeed * speedMult;
            _hipSwayPhase += dt * hipSwaySpeed * speedMult;

            // Smooth audio intensity
            float audioIntensity = Mathf.Min(1f, (_bassEnergy + _midEnergy) * 2f);
            _danceIntensity = _danceIntensity * energySmoothFactor + audioIntensity * (1f - energySmoothFactor);

            // Only dance when audio energy is present (silence threshold)
            const float silenceThreshold = 0.15f;
            if (_danceIntensity < silenceThreshold && !_isBeat)
            {
                // No music — hold the relaxed resting pose so bones don't
                // drift into T-pose when the idle animation controller is off.
                if (_leftUpperArm) _leftUpperArm.localRotation = Quaternion.Euler(0f, 0f, 55f);
                if (_rightUpperArm) _rightUpperArm.localRotation = Quaternion.Euler(0f, 0f, -55f);
                if (_leftLowerArm) _leftLowerArm.localRotation = Quaternion.Euler(0f, -20f, 0f);
                if (_rightLowerArm) _rightLowerArm.localRotation = Quaternion.Euler(0f, 20f, 0f);
                if (_head) _head.localRotation = _origHead;
                if (_neck) _neck.localRotation = _origNeck;
                if (_spine) _spine.localRotation = _origSpine;
                if (_chest) _chest.localRotation = _origChest;
                if (_hips) { _hips.localPosition = _originalHipsPos; _hips.localRotation = _origHips; }
                if (_leftShoulder) _leftShoulder.localRotation = _origLS;
                if (_rightShoulder) _rightShoulder.localRotation = _origRS;
                return;
            }

            float baseInt = (_isIntense ? baseIntensityFull : baseIntensityNormal) * intenseMult;
            float audioBoost = _danceIntensity * audioBoostScale * intenseMult;
            float eff = baseInt + audioBoost; // effectiveIntensity

            ApplyDance(eff, intenseMult);
        }

        private void ApplyDance(float eff, float im)
        {
            // ── Head bob ──
            if (_head)
            {
                float baseBob = 0.08f;
                float beatBoost = _isBeat ? 0.15f : _bassEnergy * 0.1f;
                float bobAmt = (baseBob + beatBoost) * im;
                float rx = -Mathf.Abs(Mathf.Sin(_headBobPhase)) * bobAmt * eff;
                float rz = Mathf.Sin(_dancePhase * 0.5f) * 0.05f * im * eff;
                _head.localRotation = _origHead * Quaternion.Euler(rx * Mathf.Rad2Deg, 0, rz * Mathf.Rad2Deg);
            }

            // ── Neck ──
            if (_neck)
            {
                float rx = Mathf.Sin(_headBobPhase + 0.2f) * 0.03f * im * eff;
                _neck.localRotation = _origNeck * Quaternion.Euler(rx * Mathf.Rad2Deg, 0, 0);
            }

            // ── Spine ──
            if (_spine)
            {
                float ry = Mathf.Sin(_hipSwayPhase) * 0.08f * im * eff;
                float rx = Mathf.Sin(_dancePhase * 0.5f) * 0.02f * im * eff;
                _spine.localRotation = _origSpine * Quaternion.Euler(rx * Mathf.Rad2Deg, ry * Mathf.Rad2Deg, 0);
            }

            // ── Chest ──
            if (_chest)
            {
                float ry = Mathf.Sin(_hipSwayPhase + 0.3f) * 0.05f * im * eff;
                _chest.localRotation = _origChest * Quaternion.Euler(0, ry * Mathf.Rad2Deg, 0);
            }

            // ── Hips ──
            if (_hips)
            {
                float ry = Mathf.Sin(_hipSwayPhase) * 0.1f * im * eff;
                _hips.localRotation = _origHips * Quaternion.Euler(0, ry * Mathf.Rad2Deg, 0);
                float bounce = Mathf.Sin(_headBobPhase) * 0.005f * im * eff;
                _hips.localPosition = _originalHipsPos + Vector3.up * bounce;
            }

            // ── Arms ──
            float baseArmSway = 0.5f;
            float audioArmBoost = _highEnergy * 0.5f;
            float armE = (baseArmSway + audioArmBoost) * eff;

            if (_leftUpperArm)
            {
                float rz = 1.2f + Mathf.Sin(_armSwayPhase) * 0.3f * armE;
                float rx = -Mathf.Sin(_armSwayPhase * 0.7f) * 0.2f * armE;
                float ry = -1.0f;
                _leftUpperArm.localRotation = _origLUA * Quaternion.Euler(rx * Mathf.Rad2Deg, ry * Mathf.Rad2Deg, rz * Mathf.Rad2Deg);
            }

            if (_rightUpperArm)
            {
                float rz = -1.2f - Mathf.Sin(_armSwayPhase + Mathf.PI) * 0.3f * armE;
                float rx = -Mathf.Sin(_armSwayPhase * 0.7f + 1f) * 0.2f * armE;
                float ry = 1.0f;
                _rightUpperArm.localRotation = _origRUA * Quaternion.Euler(rx * Mathf.Rad2Deg, ry * Mathf.Rad2Deg, rz * Mathf.Rad2Deg);
            }

            if (_leftLowerArm)
            {
                float rz = -0.3f - Mathf.Sin(_armSwayPhase * 1.5f) * 0.2f * im * eff;
                _leftLowerArm.localRotation = _origLLA * Quaternion.Euler(0, 0, rz * Mathf.Rad2Deg);
            }

            if (_rightLowerArm)
            {
                float rz = 0.3f + Mathf.Sin(_armSwayPhase * 1.5f) * 0.2f * im * eff;
                _rightLowerArm.localRotation = _origRLA * Quaternion.Euler(0, 0, rz * Mathf.Rad2Deg);
            }

            // ── Shoulders (bounce on beat, decay) ──
            if (_leftShoulder)
            {
                _leftShoulderX = _isBeat ? 0.08f * im : _leftShoulderX * 0.85f;
                _leftShoulder.localRotation = _origLS * Quaternion.Euler(_leftShoulderX * Mathf.Rad2Deg, 0, 0);
            }

            if (_rightShoulder)
            {
                _rightShoulderX = _isBeat ? 0.08f * im : _rightShoulderX * 0.85f;
                _rightShoulder.localRotation = _origRS * Quaternion.Euler(_rightShoulderX * Mathf.Rad2Deg, 0, 0);
            }

            // ── Happy expression while dancing ──
            if (_expression != null)
            {
                float happyVal = happyBase + eff * happyScale;
                _expression.SetWeight(ExpressionKey.Happy, Mathf.Clamp01(happyVal));
            }
        }

        // ── Public API ──────────────────────────────────────────────

        public void StartDancing(bool intense = false)
        {
            _isDancing = true;
            _isIntense = intense;
            _dancePhase = 0f;
            _headBobPhase = 0f;
            _armSwayPhase = 0f;
            _hipSwayPhase = 0f;
        }

        public void StopDancing()
        {
            _isDancing = false;
            _isBeat = false;

            // Clear happy expression
            _expression?.SetWeight(ExpressionKey.Happy, 0f);

            ResetPose();
        }

        public void SetIntense(bool intense)
        {
            _isIntense = intense;
        }

        /// <summary>
        /// Call from MessageHandler's OnAudioAnalysis.
        /// </summary>
        public void UpdateAudioData(float bass, float mid, float high, bool isBeat)
        {
            _bassEnergy = bass;
            _midEnergy = mid;
            _highEnergy = high;
            _isBeat = isBeat;
        }

        public void SetBeatEnergy(float energy)
        {
            _bassEnergy = energy;
            _midEnergy = energy;
            _isBeat = energy > 0.6f;
        }

        private void ResetPose()
        {
            if (_hips) { _hips.localPosition = _originalHipsPos; _hips.localRotation = _origHips; }
            if (_head) _head.localRotation = _origHead;
            if (_neck) _neck.localRotation = _origNeck;
            if (_spine) _spine.localRotation = _origSpine;
            if (_chest) _chest.localRotation = _origChest;
            if (_leftUpperArm) _leftUpperArm.localRotation = _origLUA;
            if (_rightUpperArm) _rightUpperArm.localRotation = _origRUA;
            if (_leftLowerArm) _leftLowerArm.localRotation = _origLLA;
            if (_rightLowerArm) _rightLowerArm.localRotation = _origRLA;
            if (_leftShoulder) _leftShoulder.localRotation = _origLS;
            if (_rightShoulder) _rightShoulder.localRotation = _origRS;
        }
    }
}
