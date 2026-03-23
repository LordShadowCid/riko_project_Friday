using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace Annabeth.UI
{
    /// <summary>
    /// Settings panel with sliders and toggles for all user preferences.
    /// Pattern from Mate-Engine AvatarSettingsMenu.cs:
    ///   1. Wire onValueChanged → write to SettingsData → SaveAll()
    ///   2. LoadSettings() → read from SettingsData → SetValueWithoutNotify
    ///   3. ResetToDefaults() → fresh SettingsData, reload UI
    /// </summary>
    public class SettingsPanel : MonoBehaviour
    {
        [Header("Display")]
        [SerializeField] private Slider sliderFPS;
        [SerializeField] private TextMeshProUGUI labelFPS;
        [SerializeField] private Toggle toggleAlwaysOnTop;
        [SerializeField] private Slider sliderAvatarSize;
        [SerializeField] private TextMeshProUGUI labelAvatarSize;

        [Header("Tracking")]
        [SerializeField] private Toggle toggleMouseTracking;
        [SerializeField] private Slider sliderEyeBlend;
        [SerializeField] private TextMeshProUGUI labelEyeBlend;
        [SerializeField] private Slider sliderHeadBlend;
        [SerializeField] private TextMeshProUGUI labelHeadBlend;

        [Header("Interaction")]
        [SerializeField] private Toggle toggleParticles;
        [SerializeField] private Toggle toggleTouchSounds;
        [SerializeField] private Slider sliderSFXVolume;
        [SerializeField] private TextMeshProUGUI labelSFXVolume;

        [Header("AI / Speech")]
        [SerializeField] private Toggle toggleSpeechBubble;

        [Header("System")]
        [SerializeField] private Toggle toggleSleepMode;
        [SerializeField] private Slider sliderSleepTimer;
        [SerializeField] private TextMeshProUGUI labelSleepTimer;
        [SerializeField] private Toggle toggleAutoMemoryTrim;
        [SerializeField] private Toggle toggleMinimizeToTray;

        [Header("Actions")]
        [SerializeField] private Button btnResetDefaults;
        [SerializeField] private Button btnClose;

        [Header("References")]
        [SerializeField] private RadialMenu radialMenu;

        private bool _loading; // Guard to prevent save during LoadSettings()

        private void OnEnable()
        {
            WireListeners();
            LoadSettings();
        }

        private void OnDisable()
        {
            UnwireListeners();
        }

        // ── Wiring ──────────────────────────────────────────────

        private void WireListeners()
        {
            // Display
            sliderFPS?.onValueChanged.AddListener(OnFPSChanged);
            toggleAlwaysOnTop?.onValueChanged.AddListener(OnAlwaysOnTopChanged);
            sliderAvatarSize?.onValueChanged.AddListener(OnAvatarSizeChanged);

            // Tracking
            toggleMouseTracking?.onValueChanged.AddListener(OnMouseTrackingChanged);
            sliderEyeBlend?.onValueChanged.AddListener(OnEyeBlendChanged);
            sliderHeadBlend?.onValueChanged.AddListener(OnHeadBlendChanged);

            // Interaction
            toggleParticles?.onValueChanged.AddListener(OnParticlesChanged);
            toggleTouchSounds?.onValueChanged.AddListener(OnTouchSoundsChanged);
            sliderSFXVolume?.onValueChanged.AddListener(OnSFXVolumeChanged);

            // AI
            toggleSpeechBubble?.onValueChanged.AddListener(OnSpeechBubbleChanged);

            // System
            toggleSleepMode?.onValueChanged.AddListener(OnSleepModeChanged);
            sliderSleepTimer?.onValueChanged.AddListener(OnSleepTimerChanged);
            toggleAutoMemoryTrim?.onValueChanged.AddListener(OnAutoMemoryTrimChanged);
            toggleMinimizeToTray?.onValueChanged.AddListener(OnMinimizeToTrayChanged);

            // Buttons
            btnResetDefaults?.onClick.AddListener(OnResetDefaults);
            btnClose?.onClick.AddListener(OnClose);
        }

        private void UnwireListeners()
        {
            sliderFPS?.onValueChanged.RemoveListener(OnFPSChanged);
            toggleAlwaysOnTop?.onValueChanged.RemoveListener(OnAlwaysOnTopChanged);
            sliderAvatarSize?.onValueChanged.RemoveListener(OnAvatarSizeChanged);

            toggleMouseTracking?.onValueChanged.RemoveListener(OnMouseTrackingChanged);
            sliderEyeBlend?.onValueChanged.RemoveListener(OnEyeBlendChanged);
            sliderHeadBlend?.onValueChanged.RemoveListener(OnHeadBlendChanged);

            toggleParticles?.onValueChanged.RemoveListener(OnParticlesChanged);
            toggleTouchSounds?.onValueChanged.RemoveListener(OnTouchSoundsChanged);
            sliderSFXVolume?.onValueChanged.RemoveListener(OnSFXVolumeChanged);

            toggleSpeechBubble?.onValueChanged.RemoveListener(OnSpeechBubbleChanged);

            toggleSleepMode?.onValueChanged.RemoveListener(OnSleepModeChanged);
            sliderSleepTimer?.onValueChanged.RemoveListener(OnSleepTimerChanged);
            toggleAutoMemoryTrim?.onValueChanged.RemoveListener(OnAutoMemoryTrimChanged);
            toggleMinimizeToTray?.onValueChanged.RemoveListener(OnMinimizeToTrayChanged);

            btnResetDefaults?.onClick.RemoveListener(OnResetDefaults);
            btnClose?.onClick.RemoveListener(OnClose);
        }

        // ── Load / Apply ────────────────────────────────────────

        /// <summary>
        /// Read from SettingsData and push values into UI controls
        /// using SetValueWithoutNotify to avoid triggering save loops.
        /// </summary>
        private void LoadSettings()
        {
            var sm = Core.SettingsManager.Instance;
            if (sm == null) return;
            var d = sm.data;

            _loading = true;

            // Display
            if (sliderFPS != null)
            {
                sliderFPS.minValue = 15;
                sliderFPS.maxValue = 165;
                sliderFPS.wholeNumbers = true;
                sliderFPS.SetValueWithoutNotify(d.fpsLimit);
            }
            UpdateLabel(labelFPS, $"{d.fpsLimit} FPS");

            if (toggleAlwaysOnTop != null)
                toggleAlwaysOnTop.SetIsOnWithoutNotify(d.alwaysOnTop);

            if (sliderAvatarSize != null)
            {
                sliderAvatarSize.minValue = 0.5f;
                sliderAvatarSize.maxValue = 2f;
                sliderAvatarSize.SetValueWithoutNotify(d.avatarSize);
            }
            UpdateLabel(labelAvatarSize, $"{d.avatarSize:F1}x");

            // Tracking
            if (toggleMouseTracking != null)
                toggleMouseTracking.SetIsOnWithoutNotify(d.enableMouseTracking);

            if (sliderEyeBlend != null)
            {
                sliderEyeBlend.minValue = 0f;
                sliderEyeBlend.maxValue = 1f;
                sliderEyeBlend.SetValueWithoutNotify(d.eyeBlend);
            }
            UpdateLabel(labelEyeBlend, $"{d.eyeBlend:P0}");

            if (sliderHeadBlend != null)
            {
                sliderHeadBlend.minValue = 0f;
                sliderHeadBlend.maxValue = 1f;
                sliderHeadBlend.SetValueWithoutNotify(d.headBlend);
            }
            UpdateLabel(labelHeadBlend, $"{d.headBlend:P0}");

            // Interaction
            if (toggleParticles != null)
                toggleParticles.SetIsOnWithoutNotify(d.enableParticles);
            if (toggleTouchSounds != null)
                toggleTouchSounds.SetIsOnWithoutNotify(d.enableTouchSounds);

            if (sliderSFXVolume != null)
            {
                sliderSFXVolume.minValue = 0f;
                sliderSFXVolume.maxValue = 1f;
                sliderSFXVolume.SetValueWithoutNotify(d.sfxVolume);
            }
            UpdateLabel(labelSFXVolume, $"{d.sfxVolume:P0}");

            // AI
            if (toggleSpeechBubble != null)
                toggleSpeechBubble.SetIsOnWithoutNotify(d.enableSpeechBubble);

            // System
            if (toggleSleepMode != null)
                toggleSleepMode.SetIsOnWithoutNotify(d.enableSleepMode);

            if (sliderSleepTimer != null)
            {
                sliderSleepTimer.minValue = 30f;
                sliderSleepTimer.maxValue = 360f;
                sliderSleepTimer.wholeNumbers = true;
                sliderSleepTimer.SetValueWithoutNotify(d.sleepTimerSeconds);
            }
            UpdateLabel(labelSleepTimer, $"{d.sleepTimerSeconds:F0}s");

            if (toggleAutoMemoryTrim != null)
                toggleAutoMemoryTrim.SetIsOnWithoutNotify(d.enableAutoMemoryTrim);
            if (toggleMinimizeToTray != null)
                toggleMinimizeToTray.SetIsOnWithoutNotify(d.minimizeToTray);

            _loading = false;
        }

        // ── Value Changed Handlers ──────────────────────────────

        private void OnFPSChanged(float val)
        {
            if (_loading) return;
            int fps = Mathf.RoundToInt(val);
            Data.fpsLimit = fps;
            UpdateLabel(labelFPS, $"{fps} FPS");
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
            UpdateLabel(labelAvatarSize, $"{val:F1}x");
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
            UpdateLabel(labelEyeBlend, $"{val:P0}");
            SaveAndApply();
        }

        private void OnHeadBlendChanged(float val)
        {
            if (_loading) return;
            Data.headBlend = val;
            UpdateLabel(labelHeadBlend, $"{val:P0}");
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
            UpdateLabel(labelSFXVolume, $"{val:P0}");
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
            UpdateLabel(labelSleepTimer, $"{val:F0}s");
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
            if (radialMenu != null)
                radialMenu.CloseSettings();
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

        private static void UpdateLabel(TextMeshProUGUI label, string text)
        {
            if (label != null)
                label.text = text;
        }
    }
}
