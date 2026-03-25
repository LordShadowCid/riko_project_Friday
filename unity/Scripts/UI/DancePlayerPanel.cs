using UnityEngine;
using UnityEngine.UI;
using Annabeth.Dance;

namespace Annabeth.UI
{
    /// <summary>
    /// Feature #17: Minimal dance player UI with play/pause, prev/next,
    /// progress bar, volume, and song name. Built at runtime via UIFactory.
    /// </summary>
    public class DancePlayerPanel : MonoBehaviour
    {
        [SerializeField] private CustomDanceLoader danceLoader;

        private Canvas _canvas;
        private RectTransform _panel;
        private Text _songLabel;
        private Slider _progressSlider;
        private Slider _volumeSlider;
        private Button _playPauseBtn;
        private Text _playPauseText;
        private Toggle _loopToggle;
        private bool _isPaused;
        private bool _isVisible;
        private bool _scrubbing;

        private void Start()
        {
            if (danceLoader == null)
                danceLoader = FindFirstObjectByType<CustomDanceLoader>();

            BuildUI();
            SetVisible(false);

            if (danceLoader != null)
            {
                danceLoader.OnDanceStarted += OnDanceStarted;
                danceLoader.OnDanceStopped += OnDanceStopped;
            }
        }

        private void OnDestroy()
        {
            if (danceLoader != null)
            {
                danceLoader.OnDanceStarted -= OnDanceStarted;
                danceLoader.OnDanceStopped -= OnDanceStopped;
            }
            if (_canvas != null)
                Destroy(_canvas.gameObject);
        }

        private void Update()
        {
            if (!_isVisible || danceLoader == null || !danceLoader.IsPlaying) return;

            // Update progress slider if not being dragged
            if (!_scrubbing)
                _progressSlider.SetValueWithoutNotify(danceLoader.Progress);
        }

        public void SetVisible(bool visible)
        {
            _isVisible = visible;
            if (_canvas != null)
                _canvas.gameObject.SetActive(visible);
        }

        public void Toggle() => SetVisible(!_isVisible);

        private void BuildUI()
        {
            _canvas = UIFactory.CreateCanvas("DancePlayerCanvas", 105);

            var rowSize = new Vector2(320f, 30f);

            // Main panel — bottom center
            _panel = UIFactory.CreatePanel(_canvas.transform, "DancePanel",
                new Vector2(340f, 180f), UIFactory.PanelBg);
            _panel.anchorMin = new Vector2(0.5f, 0f);
            _panel.anchorMax = new Vector2(0.5f, 0f);
            _panel.pivot = new Vector2(0.5f, 0f);
            _panel.anchoredPosition = new Vector2(0f, 10f);

            var layout = _panel.gameObject.AddComponent<VerticalLayoutGroup>();
            layout.padding = new RectOffset(10, 10, 8, 8);
            layout.spacing = 4f;
            layout.childAlignment = TextAnchor.MiddleCenter;
            layout.childControlWidth = true;
            layout.childControlHeight = true;
            layout.childForceExpandWidth = true;
            layout.childForceExpandHeight = false;

            // Song name
            _songLabel = UIFactory.CreateText(_panel, "SongLabel", "No dance loaded",
                14, TextAnchor.MiddleCenter);
            _songLabel.rectTransform.sizeDelta = rowSize;

            // Progress slider
            var (progSlider, _) = UIFactory.CreateSliderRow(_panel, "Progress",
                "Progress", 0f, 1f, false, rowSize);
            _progressSlider = progSlider;
            _progressSlider.onValueChanged.AddListener(OnProgressChanged);

            // Transport buttons row
            var transportRow = new GameObject("TransportRow", typeof(RectTransform));
            transportRow.transform.SetParent(_panel, false);
            var transportRT = transportRow.GetComponent<RectTransform>();
            transportRT.sizeDelta = new Vector2(rowSize.x, 34f);
            var hLayout = transportRow.AddComponent<HorizontalLayoutGroup>();
            hLayout.spacing = 6f;
            hLayout.childAlignment = TextAnchor.MiddleCenter;
            hLayout.childControlWidth = true;
            hLayout.childControlHeight = true;
            hLayout.childForceExpandWidth = true;
            hLayout.childForceExpandHeight = false;

            var btnSize = new Vector2(50f, 30f);
            var prevBtn = UIFactory.CreateButton(transportRow.transform, "PrevBtn", "◀◀", btnSize);
            prevBtn.onClick.AddListener(() => danceLoader?.PlayPrevious());
            _playPauseBtn = UIFactory.CreateButton(transportRow.transform, "PlayPauseBtn", "▶", btnSize);
            _playPauseBtn.onClick.AddListener(OnPlayPauseClicked);
            _playPauseText = _playPauseBtn.GetComponentInChildren<Text>();
            var nextBtn = UIFactory.CreateButton(transportRow.transform, "NextBtn", "▶▶", btnSize);
            nextBtn.onClick.AddListener(() => danceLoader?.PlayNext());
            var stopBtn = UIFactory.CreateButton(transportRow.transform, "StopBtn", "■", btnSize);
            stopBtn.onClick.AddListener(() => danceLoader?.Stop());

            // Volume slider
            var (volSlider, __) = UIFactory.CreateSliderRow(_panel, "Volume",
                "Volume", 0f, 1f, false, rowSize);
            _volumeSlider = volSlider;
            _volumeSlider.SetValueWithoutNotify(1f);
            _volumeSlider.onValueChanged.AddListener(v =>
            {
                if (danceLoader != null) danceLoader.Volume = v;
            });
        }

        private void OnPlayPauseClicked()
        {
            if (danceLoader == null) return;
            if (!danceLoader.IsPlaying)
            {
                if (danceLoader.AvailableDances.Count > 0)
                    danceLoader.PlayDance(0);
                return;
            }

            _isPaused = !_isPaused;
            if (_isPaused)
                danceLoader.Pause();
            else
                danceLoader.Resume();

            if (_playPauseText != null)
                _playPauseText.text = _isPaused ? "▶" : "❚❚";
        }

        private void OnProgressChanged(float value)
        {
            _scrubbing = true;
            danceLoader?.SetTime(value);
            _scrubbing = false;
        }

        private void OnDanceStarted(string name)
        {
            SetVisible(true);
            _isPaused = false;
            if (_songLabel != null) _songLabel.text = name;
            if (_playPauseText != null) _playPauseText.text = "❚❚";
        }

        private void OnDanceStopped()
        {
            _isPaused = false;
            if (_songLabel != null) _songLabel.text = "No dance loaded";
            if (_playPauseText != null) _playPauseText.text = "▶";
            if (_progressSlider != null) _progressSlider.SetValueWithoutNotify(0f);
        }
    }
}
