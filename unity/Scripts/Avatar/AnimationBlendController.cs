using UnityEngine;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Interface for animation controllers that support blend weight transitions.
    /// </summary>
    public interface IBlendableAnimation
    {
        float BlendWeight { get; }
        void SetBlendWeight(float weight);
        void SetBlendActive(bool active);
    }

    /// <summary>
    /// Manages smooth crossfade transitions between bone-based animation controllers.
    /// Ramps BlendWeight on outgoing (1→0) and incoming (0→1) controllers using smoothstep.
    /// </summary>
    public class AnimationBlendController : MonoBehaviour
    {
        [SerializeField] private float defaultBlendDuration = 0.5f;

        private IBlendableAnimation _outgoing;
        private IBlendableAnimation _incoming;
        private float _blendTimer;
        private float _blendDuration;
        private bool _isBlending;

        public bool IsBlending => _isBlending;

        /// <summary>
        /// Start a crossfade from one animation controller to another.
        /// </summary>
        public void Crossfade(IBlendableAnimation from, IBlendableAnimation to, float duration = -1f)
        {
            if (duration < 0f) duration = defaultBlendDuration;

            if (_isBlending) FinishBlend();

            _outgoing = from;
            _incoming = to;
            _blendDuration = Mathf.Max(0.01f, duration);
            _blendTimer = 0f;
            _isBlending = true;

            _incoming?.SetBlendWeight(0f);
            _incoming?.SetBlendActive(true);
        }

        private void Update()
        {
            if (!_isBlending) return;

            _blendTimer += Time.deltaTime;
            float t = Mathf.Clamp01(_blendTimer / _blendDuration);
            float smoothT = t * t * (3f - 2f * t); // smoothstep

            _outgoing?.SetBlendWeight(1f - smoothT);
            _incoming?.SetBlendWeight(smoothT);

            if (t >= 1f) FinishBlend();
        }

        private void FinishBlend()
        {
            if (_outgoing != null)
            {
                _outgoing.SetBlendWeight(0f);
                _outgoing.SetBlendActive(false);
            }
            if (_incoming != null)
            {
                _incoming.SetBlendWeight(1f);
            }
            _isBlending = false;
            _outgoing = null;
            _incoming = null;
        }
    }
}
