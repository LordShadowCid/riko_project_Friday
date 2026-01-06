using UnityEngine;
using UniVRM10;
using Annabeth.Core;

namespace Annabeth.Dance
{
    /// <summary>
    /// Controls beat-reactive procedural dance animation.
    /// Ported from the JavaScript implementation in companion.html.
    /// </summary>
    public class BeatDanceController : MonoBehaviour
    {
        [Header("Dance Settings")]
        [SerializeField] private float baseSpeed = 2f;
        [SerializeField] private float speedMultiplier = 3f;
        [SerializeField] private float maxBounceAmount = 0.02f;
        [SerializeField] private float maxArmSwing = 0.15f;
        [SerializeField] private float maxHeadBob = 0.1f;

        [Header("Energy Smoothing")]
        [SerializeField] private float energySmooth = 8f;

        private Vrm10Instance _vrm;
        private Animator _animator;
        private bool _isDancing;
        private float _dancePhase;
        private float _beatEnergy;
        private float _smoothedEnergy;

        // Bone references
        private Transform _hips;
        private Transform _spine;
        private Transform _head;
        private Transform _leftUpperArm;
        private Transform _rightUpperArm;
        private Transform _leftLowerArm;
        private Transform _rightLowerArm;

        // Original positions/rotations for reset
        private Vector3 _originalHipsPos;
        private Quaternion _originalSpineRot;
        private Quaternion _originalHeadRot;
        private Quaternion _originalLeftArmRot;
        private Quaternion _originalRightArmRot;

        public bool IsDancing => _isDancing;

        public void Initialize(Vrm10Instance vrm)
        {
            _vrm = vrm;
            _animator = vrm?.GetComponent<Animator>();

            if (_animator == null) return;

            // Cache bone transforms
            _hips = _animator.GetBoneTransform(HumanBodyBones.Hips);
            _spine = _animator.GetBoneTransform(HumanBodyBones.Spine);
            _head = _animator.GetBoneTransform(HumanBodyBones.Head);
            _leftUpperArm = _animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
            _rightUpperArm = _animator.GetBoneTransform(HumanBodyBones.RightUpperArm);
            _leftLowerArm = _animator.GetBoneTransform(HumanBodyBones.LeftLowerArm);
            _rightLowerArm = _animator.GetBoneTransform(HumanBodyBones.RightLowerArm);

            // Store original transforms
            if (_hips != null) _originalHipsPos = _hips.localPosition;
            if (_spine != null) _originalSpineRot = _spine.localRotation;
            if (_head != null) _originalHeadRot = _head.localRotation;
            if (_leftUpperArm != null) _originalLeftArmRot = _leftUpperArm.localRotation;
            if (_rightUpperArm != null) _originalRightArmRot = _rightUpperArm.localRotation;
        }

        private void LateUpdate()
        {
            if (!_isDancing || _vrm == null) return;

            // Smooth energy changes
            _smoothedEnergy = Mathf.Lerp(_smoothedEnergy, _beatEnergy, Time.deltaTime * energySmooth);

            // Update dance phase
            _dancePhase += Time.deltaTime * (baseSpeed + _smoothedEnergy * speedMultiplier);

            ApplyDanceAnimation();
        }

        private void ApplyDanceAnimation()
        {
            float energy = _smoothedEnergy;

            // Hips bounce (vertical)
            if (_hips != null)
            {
                float bounce = Mathf.Sin(_dancePhase * 4f) * maxBounceAmount * energy;
                _hips.localPosition = _originalHipsPos + Vector3.up * bounce;
            }

            // Spine sway (side to side)
            if (_spine != null)
            {
                float sway = Mathf.Sin(_dancePhase * 2f) * 5f * energy;
                _spine.localRotation = _originalSpineRot * Quaternion.Euler(0, 0, sway);
            }

            // Head bob
            if (_head != null)
            {
                float bob = Mathf.Sin(_dancePhase * 4f) * maxHeadBob * energy;
                float tilt = Mathf.Sin(_dancePhase * 2f) * 3f * energy;
                _head.localRotation = _originalHeadRot * Quaternion.Euler(bob * 10f, 0, tilt);
            }

            // Arm swing (alternating)
            float armPhase = _dancePhase * 2f;
            
            if (_leftUpperArm != null)
            {
                float swing = Mathf.Sin(armPhase) * maxArmSwing * energy * Mathf.Rad2Deg;
                _leftUpperArm.localRotation = _originalLeftArmRot * Quaternion.Euler(swing, 0, 0);
            }

            if (_rightUpperArm != null)
            {
                float swing = Mathf.Sin(armPhase + Mathf.PI) * maxArmSwing * energy * Mathf.Rad2Deg;
                _rightUpperArm.localRotation = _originalRightArmRot * Quaternion.Euler(swing, 0, 0);
            }
        }

        /// <summary>
        /// Start procedural dance animation.
        /// </summary>
        public void StartDancing()
        {
            _isDancing = true;
            _dancePhase = 0f;
        }

        /// <summary>
        /// Stop dancing and reset to original pose.
        /// </summary>
        public void StopDancing()
        {
            _isDancing = false;
            ResetPose();
        }

        /// <summary>
        /// Update beat energy from audio analysis.
        /// </summary>
        public void SetBeatEnergy(float energy)
        {
            _beatEnergy = Mathf.Clamp01(energy);
        }

        /// <summary>
        /// Update with full audio analysis data.
        /// </summary>
        public void UpdateAudioData(float beatEnergy, float bassEnergy, float trebleEnergy)
        {
            // Use beat energy primarily, with bass adding extra bounce
            _beatEnergy = Mathf.Clamp01(beatEnergy + bassEnergy * 0.3f);
        }

        private void ResetPose()
        {
            if (_hips != null) _hips.localPosition = _originalHipsPos;
            if (_spine != null) _spine.localRotation = _originalSpineRot;
            if (_head != null) _head.localRotation = _originalHeadRot;
            if (_leftUpperArm != null) _leftUpperArm.localRotation = _originalLeftArmRot;
            if (_rightUpperArm != null) _rightUpperArm.localRotation = _originalRightArmRot;
        }
    }
}
