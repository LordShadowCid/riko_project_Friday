using UnityEngine;
using UniVRM10;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Feature #3: Procedural walking animation.
    /// Arms swing opposite to same-side leg, hips sway laterally, head bobs.
    /// Implements IBlendableAnimation for smooth crossfade with idle.
    /// </summary>
    public class WalkAnimationController : MonoBehaviour, IBlendableAnimation
    {
        [Header("Walk Cycle")]
        [SerializeField] private float strideSpeed = 6f;
        [SerializeField] private float armSwingAngle = 25f;
        [SerializeField] private float legStrideAngle = 20f;
        [SerializeField] private float hipSwayAngle = 3f;
        [SerializeField] private float headBobAmount = 0.003f;
        [SerializeField] private float spineCounterAngle = 4f;

        private Vrm10Instance _vrm;
        private Animator _animator;
        private float _blendWeight;
        private bool _enabled;
        private float _walkPhase;

        // Bones
        private Transform _hips, _spine, _head;
        private Transform _leftUpperArm, _rightUpperArm;
        private Transform _leftUpperLeg, _rightUpperLeg;
        private Transform _leftLowerLeg, _rightLowerLeg;

        // Original rotations
        private Quaternion _origHips, _origSpine, _origHead;
        private Quaternion _origLUA, _origRUA;
        private Quaternion _origLUL, _origRUL;
        private Quaternion _origLLL, _origRLL;
        private Vector3 _origHipsPos;

        // IBlendableAnimation
        public float BlendWeight => _blendWeight;
        public void SetBlendWeight(float weight) => _blendWeight = weight;
        public void SetBlendActive(bool active)
        {
            _enabled = active;
            if (!active)
            {
                _blendWeight = 0f;
                ResetPose();
            }
        }

        public void Initialize(Vrm10Instance vrm)
        {
            _vrm = vrm;
            _animator = vrm?.GetComponent<Animator>();
            if (_animator == null) return;

            _hips = _animator.GetBoneTransform(HumanBodyBones.Hips);
            _spine = _animator.GetBoneTransform(HumanBodyBones.Spine);
            _head = _animator.GetBoneTransform(HumanBodyBones.Head);
            _leftUpperArm = _animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
            _rightUpperArm = _animator.GetBoneTransform(HumanBodyBones.RightUpperArm);
            _leftUpperLeg = _animator.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
            _rightUpperLeg = _animator.GetBoneTransform(HumanBodyBones.RightUpperLeg);
            _leftLowerLeg = _animator.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
            _rightLowerLeg = _animator.GetBoneTransform(HumanBodyBones.RightLowerLeg);

            if (_hips) { _origHips = _hips.localRotation; _origHipsPos = _hips.localPosition; }
            if (_spine) _origSpine = _spine.localRotation;
            if (_head) _origHead = _head.localRotation;
            if (_leftUpperArm) _origLUA = _leftUpperArm.localRotation;
            if (_rightUpperArm) _origRUA = _rightUpperArm.localRotation;
            if (_leftUpperLeg) _origLUL = _leftUpperLeg.localRotation;
            if (_rightUpperLeg) _origRUL = _rightUpperLeg.localRotation;
            if (_leftLowerLeg) _origLLL = _leftLowerLeg.localRotation;
            if (_rightLowerLeg) _origRLL = _rightLowerLeg.localRotation;
        }

        private void LateUpdate()
        {
            if (!_enabled || _blendWeight <= 0f || _vrm == null) return;

            _walkPhase += Time.deltaTime * strideSpeed;
            float w = _blendWeight;
            float sin = Mathf.Sin(_walkPhase);
            float cos = Mathf.Cos(_walkPhase);

            // Legs: sinusoidal stride (opposite phase)
            if (_leftUpperLeg)
                _leftUpperLeg.localRotation = Quaternion.Slerp(_origLUL,
                    _origLUL * Quaternion.Euler(sin * legStrideAngle, 0, 0), w);
            if (_rightUpperLeg)
                _rightUpperLeg.localRotation = Quaternion.Slerp(_origRUL,
                    _origRUL * Quaternion.Euler(-sin * legStrideAngle, 0, 0), w);

            // Lower legs: knee bend when leg is forward
            float leftKnee = Mathf.Max(0, sin) * legStrideAngle * 0.5f;
            float rightKnee = Mathf.Max(0, -sin) * legStrideAngle * 0.5f;
            if (_leftLowerLeg)
                _leftLowerLeg.localRotation = Quaternion.Slerp(_origLLL,
                    _origLLL * Quaternion.Euler(-leftKnee, 0, 0), w);
            if (_rightLowerLeg)
                _rightLowerLeg.localRotation = Quaternion.Slerp(_origRLL,
                    _origRLL * Quaternion.Euler(-rightKnee, 0, 0), w);

            // Arms: counter-swing opposite to same-side leg
            if (_leftUpperArm)
                _leftUpperArm.localRotation = Quaternion.Slerp(_origLUA,
                    _origLUA * Quaternion.Euler(-sin * armSwingAngle, 0, 0), w);
            if (_rightUpperArm)
                _rightUpperArm.localRotation = Quaternion.Slerp(_origRUA,
                    _origRUA * Quaternion.Euler(sin * armSwingAngle, 0, 0), w);

            // Hips: lateral sway (weight shift) + subtle forward bob
            if (_hips)
            {
                _hips.localRotation = Quaternion.Slerp(_origHips,
                    _origHips * Quaternion.Euler(0, 0, cos * hipSwayAngle), w);
                float bob = Mathf.Abs(Mathf.Sin(_walkPhase * 2f)) * headBobAmount;
                _hips.localPosition = Vector3.Lerp(_origHipsPos,
                    _origHipsPos + new Vector3(0, -bob, 0), w);
            }

            // Spine: counter-rotation to hips
            if (_spine)
                _spine.localRotation = Quaternion.Slerp(_origSpine,
                    _origSpine * Quaternion.Euler(0, sin * spineCounterAngle, 0), w);

            // Head: slight compensation to stay level
            if (_head)
                _head.localRotation = Quaternion.Slerp(_origHead,
                    _origHead * Quaternion.Euler(0, -sin * spineCounterAngle * 0.3f, 0), w);
        }

        private void ResetPose()
        {
            if (_hips) { _hips.localRotation = _origHips; _hips.localPosition = _origHipsPos; }
            if (_spine) _spine.localRotation = _origSpine;
            if (_head) _head.localRotation = _origHead;
            if (_leftUpperArm) _leftUpperArm.localRotation = _origLUA;
            if (_rightUpperArm) _rightUpperArm.localRotation = _origRUA;
            if (_leftUpperLeg) _leftUpperLeg.localRotation = _origLUL;
            if (_rightUpperLeg) _rightUpperLeg.localRotation = _origRUL;
            if (_leftLowerLeg) _leftLowerLeg.localRotation = _origLLL;
            if (_rightLowerLeg) _rightLowerLeg.localRotation = _origRLL;
        }
    }
}
