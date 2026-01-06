using UnityEngine;
using UniVRM10;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Controls automatic blinking animation.
    /// Uses VRM blink BlendShape with randomized timing.
    /// </summary>
    public class BlinkController : MonoBehaviour
    {
        [Header("Blink Settings")]
        [SerializeField] private float minBlinkInterval = 2f;
        [SerializeField] private float maxBlinkInterval = 6f;
        [SerializeField] private float blinkDuration = 0.15f;
        [SerializeField] private bool autoBlinkEnabled = true;

        private Vrm10Instance _vrm;
        private Vrm10RuntimeExpression _expression;
        private float _blinkTimer;
        private float _nextBlinkTime;
        private float _blinkProgress;
        private bool _isBlinking;

        public void Initialize(Vrm10Instance vrm)
        {
            _vrm = vrm;
            _expression = vrm?.Runtime?.Expression;
            ScheduleNextBlink();
        }

        private void Update()
        {
            if (_expression == null || !autoBlinkEnabled) return;

            if (_isBlinking)
            {
                UpdateBlink();
            }
            else
            {
                _blinkTimer += Time.deltaTime;
                if (_blinkTimer >= _nextBlinkTime)
                {
                    StartBlink();
                }
            }
        }

        private void ScheduleNextBlink()
        {
            _nextBlinkTime = Random.Range(minBlinkInterval, maxBlinkInterval);
            _blinkTimer = 0f;
        }

        private void StartBlink()
        {
            _isBlinking = true;
            _blinkProgress = 0f;
        }

        private void UpdateBlink()
        {
            _blinkProgress += Time.deltaTime / blinkDuration;

            float blinkWeight;
            if (_blinkProgress < 0.5f)
            {
                // Closing eyes (0 to 1)
                blinkWeight = _blinkProgress * 2f;
            }
            else
            {
                // Opening eyes (1 to 0)
                blinkWeight = (1f - _blinkProgress) * 2f;
            }

            _expression.SetWeight(ExpressionKey.Blink, Mathf.Clamp01(blinkWeight));

            if (_blinkProgress >= 1f)
            {
                EndBlink();
            }
        }

        private void EndBlink()
        {
            _isBlinking = false;
            _expression.SetWeight(ExpressionKey.Blink, 0f);
            ScheduleNextBlink();
        }

        /// <summary>
        /// Trigger a single blink immediately.
        /// </summary>
        public void TriggerBlink()
        {
            if (!_isBlinking)
            {
                StartBlink();
            }
        }

        /// <summary>
        /// Enable or disable automatic blinking.
        /// </summary>
        public void SetAutoBlinkEnabled(bool enabled)
        {
            autoBlinkEnabled = enabled;
            if (!enabled && _isBlinking)
            {
                _expression?.SetWeight(ExpressionKey.Blink, 0f);
                _isBlinking = false;
            }
        }
    }
}
