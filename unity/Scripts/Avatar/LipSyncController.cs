using UnityEngine;
using UniVRM10;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Controls lip sync animation using VRM BlendShapes.
    /// Cycles through vowel shapes (A, I, U, E, O) during speech.
    /// </summary>
    public class LipSyncController : MonoBehaviour
    {
        [Header("Lip Sync Settings")]
        [SerializeField] private float vowelChangeInterval = 0.1f;
        [SerializeField] private float mouthOpenAmount = 0.7f;
        [SerializeField] private float smoothSpeed = 15f;

        private Vrm10Instance _vrm;
        private Vrm10RuntimeExpression _expression;
        private bool _isSpeaking;
        private int _currentVowelIndex;
        private float _vowelTimer;
        private float _targetWeight;
        private float _currentWeight;

        // VRM 1.0 expression keys for lip sync
        private readonly ExpressionKey[] _vowelKeys = new ExpressionKey[]
        {
            ExpressionKey.Aa,
            ExpressionKey.Ih,
            ExpressionKey.Ou,
            ExpressionKey.Ee,
            ExpressionKey.Oh
        };

        public void Initialize(Vrm10Instance vrm)
        {
            _vrm = vrm;
            _expression = vrm?.Runtime?.Expression;
        }

        private void Update()
        {
            if (_expression == null) return;

            if (_isSpeaking)
            {
                UpdateLipSync(Time.deltaTime);
            }
            else
            {
                // Smoothly close mouth when not speaking
                _currentWeight = Mathf.Lerp(_currentWeight, 0f, Time.deltaTime * smoothSpeed);
                ApplyCurrentVowel();
            }
        }

        private void UpdateLipSync(float deltaTime)
        {
            _vowelTimer += deltaTime;

            if (_vowelTimer >= vowelChangeInterval)
            {
                _vowelTimer = 0f;
                
                // Clear previous vowel
                _expression.SetWeight(_vowelKeys[_currentVowelIndex], 0f);
                
                // Move to next vowel
                _currentVowelIndex = (_currentVowelIndex + 1) % _vowelKeys.Length;
            }

            // Smooth transition
            _currentWeight = Mathf.Lerp(_currentWeight, mouthOpenAmount, deltaTime * smoothSpeed);
            ApplyCurrentVowel();
        }

        private void ApplyCurrentVowel()
        {
            // Reset all vowels first
            foreach (var key in _vowelKeys)
            {
                _expression.SetWeight(key, 0f);
            }

            // Apply current vowel with weight
            if (_currentWeight > 0.01f)
            {
                _expression.SetWeight(_vowelKeys[_currentVowelIndex], _currentWeight);
            }
        }

        /// <summary>
        /// Start lip sync animation.
        /// </summary>
        public void StartSpeaking()
        {
            _isSpeaking = true;
            _vowelTimer = 0f;
        }

        /// <summary>
        /// Stop lip sync animation.
        /// </summary>
        public void StopSpeaking()
        {
            _isSpeaking = false;
        }

        /// <summary>
        /// Set a specific vowel shape directly (for more advanced lip sync).
        /// </summary>
        public void SetVowel(string vowel, float weight)
        {
            if (_expression == null) return;

            ExpressionKey key = vowel.ToLower() switch
            {
                "a" or "aa" => ExpressionKey.Aa,
                "i" or "ih" => ExpressionKey.Ih,
                "u" or "ou" => ExpressionKey.Ou,
                "e" or "ee" => ExpressionKey.Ee,
                "o" or "oh" => ExpressionKey.Oh,
                _ => ExpressionKey.Aa
            };

            // Clear other vowels
            foreach (var k in _vowelKeys)
            {
                _expression.SetWeight(k, k == key ? weight : 0f);
            }
        }
    }
}
