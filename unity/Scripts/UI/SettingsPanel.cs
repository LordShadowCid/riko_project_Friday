using UnityEngine;
using UnityEngine.UI;

namespace Annabeth.UI
{
    /// <summary>
    /// Settings panel with sliders and toggles for all user preferences.
    /// Builds its own scroll-view + controls at runtime via UIFactory — no Editor wiring needed.
    /// </summary>
    public class SettingsPanel : MonoBehaviour
    {
        // ── Runtime-created references ──────────────────────────
        private RectTransform _panelRect;

        // Display
        private Slider _sliderFPS;
        private Text _labelFPS;
        private Toggle _toggleAlwaysOnTop;
        private Slider _sliderAvatarSize;
        private Text _labelAvatarSize;

        // Tracking
        private Toggle _toggleMouseTracking;
        private Slider _sliderEyeBlend;
        private Text _labelEyeBlend;
        private Slider _sliderHeadBlend;
        private Text _labelHeadBlend;

        // Interaction
        private Toggle _toggleParticles;
        private Toggle _toggleTouchSounds;
        private Slider _sliderSFXVolume;
        private Text _labelSFXVolume;

        // AI
        private Toggle _toggleSpeechBubble;

        // System
        private Toggle _toggleSleepMode;
        private Slider _sliderSleepTimer;
        private Text _labelSleepTimer;
        private Toggle _toggleAutoMemoryTrim;
        private Toggle _toggleMinimizeToTray;

        private Button _btnResetDefaults;
        private Button _btnClose;

        private RadialMenu _radialMenu;
        private bool _loading;

        /// <summary>Called by RadialMenu after AddComponent.</summary>
        public void SetRadialMenu(RadialMenu menu) => _radialMenu = menu;

        private void Awake()
        {
            BuildUI();
        }

        private void OnEnable()
        {
            WireListeners();
            LoadSettings();
        }

        private void OnDisable()
        {
            UnwireListeners();
        }

        // ── Build ───────────────────────────────────────────────

        private void BuildUI()
        {
            var panelSize = new Vector2(420, 520);
            _panelRect = UIFactory.CreatePanel(transform, "SettingsBg", panelSize);
            _panelRect.anchorMin = new Vector2(0.5f, 0.5f);
            _panelRect.anchorMax = new Vector2(0.5f, 0.5f);
            _panelRect.anchoredPosition = Vector2.zero;

            // Scroll view fills the panel
            var (_, content) = UIFactory.CreateScrollView(_panelRect, panelSize - new Vector2(8, 60));
            var svRt = content.parent.parent.GetComponent<RectTransform>();
            svRt.anchorMin = new Vector2(0.5f, 0.5f);
            svRt.anchorMax = new Vector2(0.5f, 0.5f);
            svRt.anchoredPosition = new Vector2(0, 18);

            var rowSize = new Vector2(380, 28);
            var toggleSize = new Vector2(380, 26);

            // ── Display ─────────────
            UIFactory.CreateSectionHeader(content, "Display", rowSize);
            (_sliderFPS, _labelFPS) = UIFactory.CreateSliderRow(content, "FPS", "FPS Limit", 15, 165, true, rowSize);
            _toggleAlwaysOnTop = UIFactory.CreateToggle(content, "AlwaysOnTop", "Always On Top", toggleSize);
            (_sliderAvatarSize, _labelAvatarSize) = UIFactory.CreateSliderRow(content, "AvatarSize", "Avatar Size", 0.5f, 2f, false, rowSize);

            UIFactory.CreateSeparator(content, rowSize.x);

            // ── Tracking ────────────
            UIFactory.CreateSectionHeader(content, "Tracking", rowSize);
            _toggleMouseTracking = UIFactory.CreateToggle(content, "MouseTracking", "Mouse Tracking", toggleSize);
            (_sliderEyeBlend, _labelEyeBlend) = UIFactory.CreateSliderRow(content, "EyeBlend", "Eye Blend", 0f, 1f, false, rowSize);
            (_sliderHeadBlend, _labelHeadBlend) = UIFactory.CreateSliderRow(content, "HeadBlend", "Head Blend", 0f, 1f, false, rowSize);

            UIFactory.CreateSeparator(content, rowSize.x);

            // ── Interaction ─────────
            UIFactory.CreateSectionHeader(content, "Interaction", rowSize);
            _toggleParticles = UIFactory.CreateToggle(content, "Particles", "Particles", toggleSize);
            _toggleTouchSounds = UIFactory.CreateToggle(content, "TouchSounds", "Touch Sounds", toggleSize);
            (_sliderSFXVolume, _labelSFXVolume) = UIFactory.CreateSliderRow(content, "SFXVolume", "SFX Volume", 0f, 1f, false, rowSize);

            UIFactory.CreateSeparator(content, rowSize.x);

            // ── AI / Speech ─────────
            UIFactory.CreateSectionHeader(content, "AI / Speech", rowSize);
            _toggleSpeechBubble = UIFactory.CreateToggle(content, "SpeechBubble", "Speech Bubble", toggleSize);

            UIFactory.CreateSeparator(content, rowSize.x);

            // ── System ──────────────
            UIFactory.CreateSectionHeader(content, "System", rowSize);
            _toggleSleepMode = UIFactory.CreateToggle(content, "SleepMode", "Sleep Mode", toggleSize);
            (_sliderSleepTimer, _labelSleepTimer) = UIFactory.CreateSliderRow(content, "SleepTimer", "Sleep Timer", 30, 360, true, rowSize);
            _toggleAutoMemoryTrim = UIFactory.CreateToggle(content, "AutoMemTrim", "Auto Memory Trim", toggleSize);
            _toggleMinimizeToTray = UIFactory.CreateToggle(content, "MinToTray", "Minimize to Tray", toggleSize);

            UIFactory.CreateSeparator(content, rowSize.x);

            // ── Action Buttons ──────
            var btnSize = new Vector2(180, 32);
            _btnResetDefaults = UIFactory.CreateButton(content, "BtnReset", "Reset Defaults", btnSize);
            _btnClose = UIFactory.CreateButton(content, "BtnClose", "Close", btnSize);
        }

        // ── Wiring ──────────────────────────────────────────────

        private void WireListeners()
        {
            _sliderFPS?.onValueChanged.AddListener(OnFPSChanged);
            _toggleAlwaysOnTop?.onValueChanged.AddListener(OnAlwaysOnTopChanged);
            _sliderAvatarSize?.onValueChanged.AddListener(OnAvatarSizeChanged);

            _toggleMouseTracking?.onValueChanged.AddListener(OnMouseTrackingChanged);
            _sliderEyeBlend?.onValueChanged.AddListener(OnEyeBlendChanged);
            _sliderHeadBlend?.onValueChanged.AddListener(OnHeadBlendChanged);

            _toggleParticles?.onValueChanged.AddListener(OnParticlesChanged);
            _toggleTouchSounds?.onValueChanged.AddListener(OnTouchSoundsChanged);
            _sliderSFXVolume?.onValueChanged.AddListener(OnSFXVolumeChanged);

            _toggleSpeechBubble?.onValueChanged.AddListener(OnSpeechBubbleChanged);

            _toggleSleepMode?.onValueChanged.AddListener(OnSleepModeChanged);
            _sliderSleepTimer?.onValueChanged.AddListener(OnSleepTimerChanged);
            _toggleAutoMemoryTrim?.onValueChanged.AddListener(OnAutoMemoryTrimChanged);
            _toggleMinimizeToTray?.onValueChanged.AddListener(OnMinimizeToTrayChanged);

            _btnResetDefaults?.onClick.AddListener(OnResetDefaults);
            _btnClose?.onClick.AddListener(OnClose);
        }

        private void UnwireListeners()
        {
            _sliderFPS?.onValueChanged.RemoveListener(OnFPSChanged);
            _toggleAlwaysOnTop?.onValueChanged.RemoveListener(OnAlwaysOnTopChanged);
            _sliderAvatarSize?.onValueChanged.RemoveListener(OnAvatarSizeChanged);

            _toggleMouseTracking?.onValueChanged.RemoveListener(OnMouseTrackingChanged);
            _sliderEyeBlend?.onValueChanged.RemoveListener(OnEyeBlendChanged);
            _sliderHeadBlend?.onValueChanged.RemoveListener(OnHeadBlendChanged);

            _toggleParticles?.onValueChanged.RemoveListener(OnParticlesChanged);
            _toggleTouchSounds?.onValueChanged.RemoveListener(OnTouchSoundsChanged);
            _sliderSFXVolume?.onValueChanged.RemoveListener(OnSFXVolumeChanged);

            _toggleSpeechBubble?.onValueChanged.RemoveListener(OnSpeechBubbleChanged);

            _toggleSleepMode?.onValueChanged.RemoveListener(OnSleepModeChanged);
            _sliderSleepTimer?.onValueChanged.RemoveListener(OnSleepTimerChanged);
            _toggleAutoMemoryTrim?.onValueChanged.RemoveListener(OnAutoMemoryTrimChanged);
            _toggleMinimizeToTray?.onValueChanged.RemoveListener(OnMinimizeToTrayChanged);

            _btnResetDefaults?.onClick.RemoveListener(OnResetDefaults);
            _btnClose?.onClick.RemoveListener(OnClose);
        }

        // ── Load / Apply ────────────────────────────────────────

        private void LoadSettings()
        {
            var sm = Core.SettingsManager.Instance;
            if (sm == null) return;
            var d = sm.data;

            _loading = true;

            // Display
            _sliderFPS?.SetValueWithoutNotify(d.fpsLimit);
            UpdateLabel(_labelFPS, $"{d.fpsLimit} FPS");
            _toggleAlwaysOnTop?.SetIsOnWithoutNotify(d.alwaysOnTop);
            _sliderAvatarSize?.SetValueWithoutNotify(d.avatarSize);
            UpdateLabel(_labelAvatarSize, $"{d.avatarSize:F1}x");

            // Tracking
            _toggleMouseTracking?.SetIsOnWithoutNotify(d.enableMouseTracking);
            _sliderEyeBlend?.SetValueWithoutNotify(d.eyeBlend);
            UpdateLabel(_labelEyeBlend, $"{d.eyeBlend:P0}");
            _sliderHeadBlend?.SetValueWithoutNotify(d.headBlend);
            UpdateLabel(_labelHeadBlend, $"{d.headBlend:P0}");

            // Interaction
            _toggleParticles?.SetIsOnWithoutNotify(d.enableParticles);
            _toggleTouchSounds?.SetIsOnWithoutNotify(d.enableTouchSounds);
            _sliderSFXVolume?.SetValueWithoutNotify(d.sfxVolume);
            UpdateLabel(_labelSFXVolume, $"{d.sfxVolume:P0}");

            // AI
            _toggleSpeechBubble?.SetIsOnWithoutNotify(d.enableSpeechBubble);

            // System
            _toggleSleepMode?.SetIsOnWithoutNotify(d.enableSleepMode);
            _sliderSleepTimer?.SetValueWithoutNotify(d.sleepTimerSeconds);
            UpdateLabel(_labelSleepTimer, $"{d.sleepTimerSeconds:F0}s");
            _toggleAutoMemoryTrim?.SetIsOnWithoutNotify(d.enableAutoMemoryTrim);
            _toggleMinimizeToTray?.SetIsOnWithoutNotify(d.minimizeToTray);

            _loading = false;
        }

        // ── Value Changed Handlers ──────────────────────────────

        private void OnFPSChanged(float val)
        {
            if (_loading) return;
            int fps = Mathf.RoundToInt(val);
            Data.fpsLimit = fps;
            UpdateLabel(_labelFPS, $"{fps} FPS");
            SaveAndApply();
        }

        private void OnAlwaysOnTopChanged(bool val)
        {
            if (_loading) return;
            Data.alwaysOnTop = val;
            SaveAndApply();
        }

        private void OnAvatarSizeChanged(float val)
        {
            if (_loading) return;
            Data.avatarSize = val;
            UpdateLabel(_labelAvatarSize, $"{val:F1}x");
            SaveAndApply();
        }

        private void OnMouseTrackingChanged(bool val)
        {
            if (_loading) return;
            Data.enableMouseTracking = val;
            SaveAndApply();
        }

        private void OnEyeBlendChanged(float val)
        {
            if (_loading) return;
            Data.eyeBlend = val;
            UpdateLabel(_labelEyeBlend, $"{val:P0}");
            SaveAndApply();
        }

        private void OnHeadBlendChanged(float val)
        {
            if (_loading) return;
            Data.headBlend = val;
            UpdateLabel(_labelHeadBlend, $"{val:P0}");
            SaveAndApply();
        }

        private void OnParticlesChanged(bool val)
        {
            if (_loading) return;
            Data.enableParticles = val;
            SaveAndApply();
        }

        private void OnTouchSoundsChanged(bool val)
        {
            if (_loading) return;
            Data.enableTouchSounds = val;
            SaveAndApply();
        }

        private void OnSFXVolumeChanged(float val)
        {
            if (_loading) return;
            Data.sfxVolume = val;
            UpdateLabel(_labelSFXVolume, $"{val:P0}");
            SaveAndApply();
        }

        private void OnSpeechBubbleChanged(bool val)
        {
            if (_loading) return;
            Data.enableSpeechBubble = val;
            SaveAndApply();
        }

        private void OnSleepModeChanged(bool val)
        {
            if (_loading) return;
            Data.enableSleepMode = val;
            SaveAndApply();
        }

        private void OnSleepTimerChanged(float val)
        {
            if (_loading) return;
            Data.sleepTimerSeconds = val;
            UpdateLabel(_labelSleepTimer, $"{val:F0}s");
            SaveAndApply();
        }

        private void OnAutoMemoryTrimChanged(bool val)
        {
            if (_loading) return;
            Data.enableAutoMemoryTrim = val;
            SaveAndApply();
        }

        private void OnMinimizeToTrayChanged(bool val)
        {
            if (_loading) return;
            Data.minimizeToTray = val;
            SaveAndApply();
        }

        // ── Button Handlers ─────────────────────────────────────

        private void OnResetDefaults()
        {
            Core.SettingsManager.Instance?.ResetToDefaults();
            LoadSettings();
        }

        private void OnClose()
        {
            if (_radialMenu != null)
                _radialMenu.CloseSettings();
            else
                gameObject.SetActive(false);
        }

        // ── Helpers ─────────────────────────────────────────────

        private Core.SettingsData Data => Core.SettingsManager.Instance.data;

        private void SaveAndApply()
        {
            var sm = Core.SettingsManager.Instance;
            if (sm == null) return;
            sm.SaveAll();
            sm.ApplyAllSettings();
        }

        private static void UpdateLabel(Text label, string text)
        {
            if (label != null)
                label.text = text;
        }
    }
}
