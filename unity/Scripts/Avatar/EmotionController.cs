using UnityEngine;
using UniVRM10;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Controls emotion expressions using VRM BlendShapes.
    /// Maps emotion strings to VRM expression presets.
    /// </summary>
    public class EmotionController : MonoBehaviour
    {
        [Header("Emotion Settings")]
        [SerializeField] private float transitionSpeed = 5f;
        [SerializeField] private float defaultWeight = 0.7f;

        private Vrm10Instance _vrm;
        private Vrm10RuntimeExpression _expression;
        private string _currentEmotion = "neutral";
        private ExpressionKey _currentKey = ExpressionKey.Neutral;
        private float _currentWeight;
        private float _targetWeight;

        private static readonly System.Collections.Generic.Dictionary<string, ExpressionKey> EmotionMap = 
            new System.Collections.Generic.Dictionary<string, ExpressionKey>
        {
            { "neutral", ExpressionKey.Neutral },
            { "happy", ExpressionKey.Happy },
            { "joy", ExpressionKey.Happy },
            { "angry", ExpressionKey.Angry },
            { "mad", ExpressionKey.Angry },
            { "sad", ExpressionKey.Sad },
            { "sorrow", ExpressionKey.Sad },
            { "surprised", ExpressionKey.Surprised },
            { "relaxed", ExpressionKey.Relaxed },
            { "fun", ExpressionKey.Happy },
            { "thinking", ExpressionKey.Neutral },
        };

        public void Initialize(Vrm10Instance vrm)
        {
            _vrm = vrm;
            _expression = vrm?.Runtime?.Expression;
        }

        private void Update()
        {
            if (_expression == null) return;

            _currentWeight = Mathf.Lerp(_currentWeight, _targetWeight, Time.deltaTime * transitionSpeed);
            
            if (!_currentKey.Equals(ExpressionKey.Neutral))
            {
                _expression.SetWeight(_currentKey, _currentWeight);
            }
        }

        public void SetEmotion(string emotion)
        {
            emotion = emotion.ToLower().Trim();
            
            if (emotion == _currentEmotion) return;

            if (!_currentKey.Equals(ExpressionKey.Neutral))
            {
                _expression?.SetWeight(_currentKey, 0f);
            }

            _currentEmotion = emotion;
            
            if (EmotionMap.TryGetValue(emotion, out ExpressionKey key))
            {
                _currentKey = key;
                _targetWeight = key.Equals(ExpressionKey.Neutral) ? 0f : defaultWeight;
            }
            else
            {
                Debug.LogWarning($"[EmotionController] Unknown emotion: {emotion}");
                _currentKey = ExpressionKey.Neutral;
                _targetWeight = 0f;
            }

            _currentWeight = 0f;
        }

        public void SetEmotion(string emotion, float weight)
        {
            SetEmotion(emotion);
            _targetWeight = Mathf.Clamp01(weight);
        }

        public void ClearEmotion()
        {
            SetEmotion("neutral");
        }

        public string GetCurrentEmotion() => _currentEmotion;
    }
}
