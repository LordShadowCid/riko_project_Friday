using UnityEngine;
using UniVRM10;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Feature #4: Procedural "held" pose while dragging.
    /// Arms reach up, legs dangle, slight squirm.
    /// Implements IBlendableAnimation for smooth blend in/out.
    /// </summary>
    public class DragPoseController : MonoBehaviour, IBlendableAnimation
    {
        [Header("Held Pose")]
        [SerializeField] private float armRaiseAngle = -40f; // negative = raised in VRM space
        [SerializeField] private float legDangleAngle = 8f;
        [SerializeField] private float squirmSpeed = 3f;
        [SerializeField] private float squirmAmount = 4f;

        private Vrm10Instance _vrm;
        private Animator _animator;
        private float _blendWeight;
        private bool _enabled;
        private float _squirmPhase;

        // Bones
        private Transform _leftUpperArm, _rightUpperArm;
        private Transform _leftLowerArm, _rightLowerArm;
        private Transform _leftUpperLeg, _rightUpperLeg;
        private Transform _leftLowerLeg, _rightLowerLeg;
        private Transform _spine, _head;

        // Original rotations
        private Quaternion _origLUA, _origRUA, _origLLA, _origRLA;
        private Quaternion _origLUL, _origRUL, _origLLL, _origRLL;
        private Quaternion _origSpine, _origHead;

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
            if (_animator == null) return;

            _leftUpperArm = _animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
            _rightUpperArm = _animator.GetBoneTransform(HumanBodyBones.RightUpperArm);
            _leftLowerArm = _animator.GetBoneTransform(HumanBodyBones.LeftLowerArm);
            _rightLowerArm = _animator.GetBoneTransform(HumanBodyBones.RightLowerArm);
            _leftUpperLeg = _animator.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
            _rightUpperLeg = _animator.GetBoneTransform(HumanBodyBones.RightUpperLeg);
            _leftLowerLeg = _animator.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
            _rightLowerLeg = _animator.GetBoneTransform(HumanBodyBones.RightLowerLeg);
            _spine = _animator.GetBoneTransform(HumanBodyBones.Spine);
            _head = _animator.GetBoneTransform(HumanBodyBones.Head);

            if (_leftUpperArm) _origLUA = _leftUpperArm.localRotation;
            if (_rightUpperArm) _origRUA = _rightUpperArm.localRotation;
            if (_leftLowerArm) _origLLA = _leftLowerArm.localRotation;
            if (_rightLowerArm) _origRLA = _rightLowerArm.localRotation;
            if (_leftUpperLeg) _origLUL = _leftUpperLeg.localRotation;
            if (_rightUpperLeg) _origRUL = _rightUpperLeg.localRotation;
            if (_leftLowerLeg) _origLLL = _leftLowerLeg.localRotation;
            if (_rightLowerLeg) _origRLL = _rightLowerLeg.localRotation;
            if (_spine) _origSpine = _spine.localRotation;
            if (_head) _origHead = _head.localRotation;
        }

        private void LateUpdate()
        {
            if (!_enabled || _blendWeight <= 0f || _vrm == null) return;

            _squirmPhase += Time.deltaTime * squirmSpeed;
            float w = _blendWeight;
            float squirm = Mathf.Sin(_squirmPhase) * squirmAmount;

            // Arms: reach upward
            if (_leftUpperArm)
                _leftUpperArm.localRotation = Quaternion.Slerp(_origLUA,
                    _origLUA * Quaternion.Euler(armRaiseAngle, 0, -10f), w);
            if (_rightUpperArm)
                _rightUpperArm.localRotation = Quaternion.Slerp(_origRUA,
                    _origRUA * Quaternion.Euler(armRaiseAngle, 0, 10f), w);

            // Lower arms: slight bend
            if (_leftLowerArm)
                _leftLowerArm.localRotation = Quaternion.Slerp(_origLLA,
                    _origLLA * Quaternion.Euler(0, -15f, 0), w);
            if (_rightLowerArm)
                _rightLowerArm.localRotation = Quaternion.Slerp(_origRLA,
                    _origRLA * Quaternion.Euler(0, 15f, 0), w);

            // Legs: dangle with gentle pendulum
            float legSquirm = squirm * 0.5f;
            if (_leftUpperLeg)
                _leftUpperLeg.localRotation = Quaternion.Slerp(_origLUL,
                    _origLUL * Quaternion.Euler(legDangleAngle + legSquirm, 0, 0), w);
            if (_rightUpperLeg)
                _rightUpperLeg.localRotation = Quaternion.Slerp(_origRUL,
                    _origRUL * Quaternion.Euler(legDangleAngle - legSquirm, 0, 0), w);

            // Lower legs: slight knee bend
            if (_leftLowerLeg)
                _leftLowerLeg.localRotation = Quaternion.Slerp(_origLLL,
                    _origLLL * Quaternion.Euler(-legDangleAngle * 0.6f, 0, 0), w);
            if (_rightLowerLeg)
                _rightLowerLeg.localRotation = Quaternion.Slerp(_origRLL,
                    _origRLL * Quaternion.Euler(-legDangleAngle * 0.6f, 0, 0), w);

            // Spine: slight squirm
            if (_spine)
                _spine.localRotation = Quaternion.Slerp(_origSpine,
                    _origSpine * Quaternion.Euler(0, squirm, 0), w);

            // Head: look up toward cursor
            if (_head)
                _head.localRotation = Quaternion.Slerp(_origHead,
                    _origHead * Quaternion.Euler(-10f, 0, 0), w);
        }
    }
}
