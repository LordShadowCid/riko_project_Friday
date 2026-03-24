using System.Collections;
using UnityEngine;
using UnityEngine.UI;

namespace Annabeth.UI
{
    /// <summary>
    /// Speech bubble that floats above the avatar's head and shows AI response text
    /// with a typewriter reveal effect. Builds its own Canvas + UI at runtime.
    /// Off by default — toggled via SettingsManager.data.enableSpeechBubble.
    /// Inspired by Mate-Engine AvatarBubbleHandler.cs (bone attachment + spawn animation).
    /// </summary>
    public class SpeechBubble : MonoBehaviour
    {
        [Header("Attachment")]
        [SerializeField] private HumanBodyBones attachBone = HumanBodyBones.Head;
        [SerializeField] private Vector3 worldOffset = new Vector3(0, 0.35f, 0);

        [Header("Timing")]
        [SerializeField] private float typewriterCharsPerSec = 40f;
        [SerializeField] private float autoDismissDelay = 4f;
        [SerializeField] private float spawnAnimSpeed = 8f;

        private Canvas _canvas;
        private RectTransform _bubbleRect;
        private Text _textField;
        private Image _bgImage;

        private Transform _headBone;
        private Camera _mainCam;

        private string _fullText = "";
        private int _revealedChars;
        private float _charTimer;
        private bool _typingDone;
        private Coroutine _dismissCoroutine;

        // Spawn animation
        private float _scaleT;
        private bool _showing;
        private bool _hiding;

        private static readonly Color BubbleBg = new Color(0.12f, 0.12f, 0.15f, 0.88f);
        private static readonly Color BubbleText = new Color(0.92f, 0.92f, 0.95f, 1f);

        private void Awake()
        {
            BuildUI();
            _mainCam = Camera.main;
            Hide(immediate: true);
        }

        private void BuildUI()
        {
            // Dedicated canvas — WorldSpace would fight with our overlay approach,
            // so we use ScreenSpaceOverlay and reposition each frame via WorldToScreenPoint
            _canvas = UIFactory.CreateCanvas("SpeechBubbleCanvas", 90);
            _canvas.transform.SetParent(transform, false);

            _bubbleRect = UIFactory.CreatePanel(_canvas.transform, "Bubble",
                new Vector2(320, 80), BubbleBg);

            // Anchor center so we can position freely
            _bubbleRect.anchorMin = new Vector2(0.5f, 0.5f);
            _bubbleRect.anchorMax = new Vector2(0.5f, 0.5f);
            _bubbleRect.pivot = new Vector2(0.5f, 0f); // pivot at bottom-center

            _bgImage = _bubbleRect.GetComponent<Image>();

            // Text child — fills the panel with padding
            var textGo = new GameObject("Text");
            textGo.transform.SetParent(_bubbleRect, false);
            var textRt = textGo.AddComponent<RectTransform>();
            textRt.anchorMin = Vector2.zero;
            textRt.anchorMax = Vector2.one;
            textRt.offsetMin = new Vector2(12, 8);
            textRt.offsetMax = new Vector2(-12, -8);

            _textField = textGo.AddComponent<Text>();
            _textField.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf")
                ?? Font.CreateDynamicFontFromOSFont("Segoe UI", 14);
            _textField.fontSize = 15;
            _textField.color = BubbleText;
            _textField.alignment = TextAnchor.UpperLeft;
            _textField.horizontalOverflow = HorizontalWrapMode.Wrap;
            _textField.verticalOverflow = VerticalWrapMode.Overflow;

            // Content size fitter to grow with text
            var csf = _bubbleRect.gameObject.AddComponent<ContentSizeFitter>();
            csf.horizontalFit = ContentSizeFitter.FitMode.Unconstrained;
            csf.verticalFit = ContentSizeFitter.FitMode.PreferredSize;

            // Minimum height via layout element
            var le = _bubbleRect.gameObject.AddComponent<LayoutElement>();
            le.minHeight = 44;
        }

        private void LateUpdate()
        {
            UpdatePosition();
            UpdateTypewriter();
            UpdateScaleAnimation();
        }

        // ── Public API ──────────────────────────────────────────

        /// <summary>Set the head bone reference after VRM loads.</summary>
        public void SetHeadBone(Transform bone)
        {
            _headBone = bone;
        }

        /// <summary>Show text with typewriter effect. Hides any previous text first.</summary>
        public void ShowText(string text)
        {
            if (!IsEnabled()) return;
            if (string.IsNullOrEmpty(text)) return;

            if (_dismissCoroutine != null)
            {
                StopCoroutine(_dismissCoroutine);
                _dismissCoroutine = null;
            }

            _fullText = text;
            _revealedChars = 0;
            _charTimer = 0f;
            _typingDone = false;
            _textField.text = "";

            Show();
        }

        /// <summary>Start the auto-dismiss countdown (call when TTS finishes speaking).</summary>
        public void StartDismissTimer()
        {
            if (!_showing) return;
            if (_dismissCoroutine != null)
                StopCoroutine(_dismissCoroutine);
            _dismissCoroutine = StartCoroutine(DismissAfterDelay());
        }

        public void HideNow() => Hide(immediate: false);

        // ── Internal ────────────────────────────────────────────

        private bool IsEnabled()
        {
            var sm = Core.SettingsManager.Instance;
            return sm != null && sm.data.enableSpeechBubble;
        }

        private void Show()
        {
            if (_canvas != null) _canvas.gameObject.SetActive(true);
            _showing = true;
            _hiding = false;
            _scaleT = 0f;
        }

        private void Hide(bool immediate)
        {
            if (immediate)
            {
                _scaleT = 0f;
                _showing = false;
                _hiding = false;
                if (_bubbleRect != null)
                    _bubbleRect.localScale = Vector3.zero;
                if (_canvas != null)
                    _canvas.gameObject.SetActive(false);
            }
            else
            {
                _hiding = true;
                _showing = false;
            }
        }

        private void UpdatePosition()
        {
            if (_headBone == null || _mainCam == null || _bubbleRect == null) return;
            if (!_canvas.gameObject.activeSelf) return;

            Vector3 worldPos = _headBone.position + worldOffset;
            Vector2 screenPos = _mainCam.WorldToScreenPoint(worldPos);

            // Convert screen position to canvas local position
            RectTransformUtility.ScreenPointToLocalPointInRectangle(
                _canvas.transform as RectTransform, screenPos, null, out Vector2 localPos);

            _bubbleRect.anchoredPosition = localPos;
        }

        private void UpdateTypewriter()
        {
            if (_typingDone || !_showing || string.IsNullOrEmpty(_fullText)) return;

            _charTimer += Time.deltaTime * typewriterCharsPerSec;
            int target = Mathf.Min(Mathf.FloorToInt(_charTimer), _fullText.Length);

            if (target > _revealedChars)
            {
                _revealedChars = target;
                _textField.text = _fullText.Substring(0, _revealedChars);
            }

            if (_revealedChars >= _fullText.Length)
                _typingDone = true;
        }

        private void UpdateScaleAnimation()
        {
            if (_bubbleRect == null) return;

            if (_showing && _scaleT < 1f)
            {
                _scaleT = Mathf.MoveTowards(_scaleT, 1f, Time.deltaTime * spawnAnimSpeed);
                _bubbleRect.localScale = Vector3.one * EaseOutBack(_scaleT);
            }
            else if (_hiding)
            {
                _scaleT = Mathf.MoveTowards(_scaleT, 0f, Time.deltaTime * spawnAnimSpeed * 1.5f);
                _bubbleRect.localScale = Vector3.one * _scaleT;

                if (_scaleT <= 0f)
                {
                    _hiding = false;
                    if (_canvas != null)
                        _canvas.gameObject.SetActive(false);
                }
            }
        }

        private IEnumerator DismissAfterDelay()
        {
            // Wait for typewriter to finish first
            while (!_typingDone)
                yield return null;

            yield return new WaitForSeconds(autoDismissDelay);
            Hide(immediate: false);
            _dismissCoroutine = null;
        }

        /// <summary>Ease-out-back curve for scale-up pop effect (from Mate-Engine spawn anim pattern).</summary>
        private static float EaseOutBack(float t)
        {
            const float c1 = 1.70158f;
            const float c3 = c1 + 1f;
            float t1 = t - 1f;
            return 1f + c3 * t1 * t1 * t1 + c1 * t1 * t1;
        }
    }
}
