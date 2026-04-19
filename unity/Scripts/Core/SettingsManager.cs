using System;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Annabeth.Core
{
    /// <summary>
    /// Singleton that persists all user preferences to JSON.
    /// Architecture based on Mate-Engine's SaveLoadHandler:
    /// - SettingsData class with all fields and sane defaults
    /// - SaveToDisk / LoadFromDisk / MigrateAfterLoad
    /// - ApplyAllSettings pushes data to live controllers
    /// </summary>
    public class SettingsManager : MonoBehaviour
    {
        public static SettingsManager Instance { get; private set; }

        public SettingsData data = new SettingsData();

        /// <summary>Fired after settings are loaded or reset, so controllers can re-read values.</summary>
        public event Action OnSettingsChanged;

        private string SavePath => Path.Combine(Application.persistentDataPath, "settings.json");

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
            DontDestroyOnLoad(gameObject);

            LoadFromDisk();
        }

        private void OnApplicationQuit()
        {
            SaveToDisk();
        }

        // ── Public API ──────────────────────────────────────────

        /// <summary>Save current data to disk and notify listeners.</summary>
        public void SaveAll()
        {
            SaveToDisk();
            OnSettingsChanged?.Invoke();
        }

        /// <summary>Reload from disk (discarding unsaved in-memory changes).</summary>
        public void Reload()
        {
            LoadFromDisk();
            OnSettingsChanged?.Invoke();
        }

        /// <summary>Reset everything to defaults, save, and notify.</summary>
        public void ResetToDefaults()
        {
            data = new SettingsData();
            SaveToDisk();
            OnSettingsChanged?.Invoke();
            Debug.Log("[SettingsManager] Reset to defaults.");
        }

        // ── Cached controller references (avoid repeated FindFirstObjectByType) ──
        private FPSController _cachedFps;
        private MemoryOptimizer _cachedMem;
        private TransparentWindowController _cachedWin;
        private SleepController _cachedSleep;
        private Avatar.EyeTrackingController _cachedEye;
        private Avatar.DragAnimationController _cachedSway;
        private WebSocketClient _cachedWs;
        private DesktopAmbientProbe _cachedAmbient;
        private Avatar.AvatarController _cachedAvatar;
        private UI.ThemeManager _cachedTheme;
        private Avatar.IKController _cachedIk;

        private T GetCached<T>(ref T cache) where T : MonoBehaviour
        {
            if (cache == null) cache = FindFirstObjectByType<T>();
            return cache;
        }

        /// <summary>Call when a VRM is loaded or controllers are recreated to clear stale caches.</summary>
        public void InvalidateControllerCache()
        {
            _cachedFps = null; _cachedMem = null; _cachedWin = null;
            _cachedSleep = null; _cachedEye = null; _cachedSway = null;
            _cachedWs = null; _cachedAmbient = null; _cachedAvatar = null;
            _cachedTheme = null; _cachedIk = null;
        }

        /// <summary>
        /// Push current settings values to all live controllers.
        /// Called once after VRM loads and whenever settings change.
        /// </summary>
        public void ApplyAllSettings()
        {
            // FPS
            GetCached(ref _cachedFps)?.ApplySettings();

            // Memory optimizer
            GetCached(ref _cachedMem)?.ApplySettings();

            // Always-on-top
            var win = GetCached(ref _cachedWin);
            if (win != null)
                win.SetTopmost(data.alwaysOnTop);

            // Sleep controller
            GetCached(ref _cachedSleep)?.ApplySettings();

            // Eye / spine tracking (per-component speeds + spine blend)
            var eye = GetCached(ref _cachedEye);
            if (eye != null)
            {
                eye.SetEnabled(data.enableMouseTracking);
                eye.SetEyeSpeed(data.eyeTrackSpeed);
                eye.SetHeadSpeed(data.headTrackSpeed);
                eye.SetBodySpeed(data.bodyTrackSpeed);
                eye.SetSpineBlend(data.spineBlend);
            }

            // Sway physics (drag animation)
            var sway = GetCached(ref _cachedSway);
            if (sway != null)
            {
                sway.SetSwayEnabled(data.enableSway);
                sway.SetIntensity(data.swayIntensity);
                sway.SetSpringFrequency(data.swaySpringFrequency);
                sway.SetDampingRatio(data.swayDampingRatio);
            }

            // Start with Windows
            ApplyStartWithWindows();

            // Graphics quality (Feature #23)
            QualitySettings.SetQualityLevel(data.graphicsQuality, true);

            // Feature #24: Send audio threshold/filter to Python backend
            var ws = GetCached(ref _cachedWs);
            if (ws != null)
            {
                ws.Send("audio_config", new System.Collections.Generic.Dictionary<string, object>
                {
                    { "sound_threshold", data.soundThreshold },
                    { "filter_apps", data.soundFilterApps }
                });
            }

            // Feature #9: Ambient probe settings
            var ambient = GetCached(ref _cachedAmbient);
            if (ambient != null)
            {
                ambient.SetEnabled(data.enableAmbientProbe);
                ambient.SetIntensity(data.ambientProbeIntensity);
            }

            // Avatar size + character opacity
            var avatarCtrl = GetCached(ref _cachedAvatar);
            if (avatarCtrl != null)
            {
                avatarCtrl.ApplyAvatarSize(data.avatarSize);
                avatarCtrl.ApplyCharacterOpacity(data.characterOpacity);
            }

            // v5: Theme manager
            var theme = GetCached(ref _cachedTheme);
            if (theme != null)
                theme.ApplyTheme(data.uiHueShift, data.uiSaturation);

            // v5: IK controller
            var ik = GetCached(ref _cachedIk);
            if (ik != null)
                ik.SetEnabled(data.enableIK);

            Debug.Log("[SettingsManager] Applied all settings.");
        }

        // ── Start with Windows ────────────────────────────────────

#if UNITY_STANDALONE_WIN
        [DllImport("advapi32.dll", CharSet = CharSet.Unicode)]
        static extern int RegOpenKeyExW(IntPtr hKey, string subKey, uint options, uint sam, out IntPtr result);
        [DllImport("advapi32.dll", CharSet = CharSet.Unicode)]
        static extern int RegSetValueExW(IntPtr hKey, string name, uint reserved, uint type, byte[] data, uint cbData);
        [DllImport("advapi32.dll", CharSet = CharSet.Unicode)]
        static extern int RegDeleteValueW(IntPtr hKey, string name);
        [DllImport("advapi32.dll")]
        static extern int RegCloseKey(IntPtr hKey);

        static readonly IntPtr HKCU = new IntPtr(unchecked((int)0x80000001));
#endif

        private void ApplyStartWithWindows()
        {
#if UNITY_STANDALONE_WIN
            try
            {
                if (RegOpenKeyExW(HKCU, @"SOFTWARE\Microsoft\Windows\CurrentVersion\Run",
                    0, 0x0002 /* KEY_SET_VALUE */, out IntPtr hKey) == 0)
                {
                    try
                    {
                        if (data.startWithWindows)
                        {
                            string exe = System.Diagnostics.Process.GetCurrentProcess().MainModule?.FileName ?? "";
                            if (exe.Length > 0)
                            {
                                string val = $"\"{exe}\"";
                                byte[] bytes = System.Text.Encoding.Unicode.GetBytes(val + "\0");
                                RegSetValueExW(hKey, "Annabeth", 0, 1 /* REG_SZ */, bytes, (uint)bytes.Length);
                            }
                        }
                        else
                        {
                            RegDeleteValueW(hKey, "Annabeth");
                        }
                    }
                    finally { RegCloseKey(hKey); }
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[SettingsManager] Startup registry: {e.Message}");
            }
#endif
        }

        // ── Persistence ─────────────────────────────────────────

        private void SaveToDisk()
        {
            try
            {
                string json = JsonUtility.ToJson(data, true);
                File.WriteAllText(SavePath, json);
                Debug.Log($"[SettingsManager] Saved to {SavePath}");
            }
            catch (Exception e)
            {
                Debug.LogError($"[SettingsManager] Save failed: {e.Message}");
            }
        }

        private void LoadFromDisk()
        {
            if (!File.Exists(SavePath))
            {
                Debug.Log("[SettingsManager] No settings file found, using defaults.");
                return;
            }

            try
            {
                string json = File.ReadAllText(SavePath);
                JsonUtility.FromJsonOverwrite(json, data);
                MigrateAfterLoad();
                Debug.Log("[SettingsManager] Loaded settings from disk.");
            }
            catch (Exception e)
            {
                Debug.LogError($"[SettingsManager] Load failed, using defaults: {e.Message}");
                data = new SettingsData();
            }
        }

        /// <summary>
        /// Handle version upgrades — add new fields, fix ranges, etc.
        /// Bump settingsVersion in SettingsData when adding migrations.
        /// </summary>
        private void MigrateAfterLoad()
        {
            // ── Version 1 → 2: Add spine/body tracking, sway physics, dance transitions ──
            if (data.settingsVersion < 2)
            {
                data.spineBlend = 0.5f;
                data.eyeTrackSpeed = 8f;
                data.headTrackSpeed = 4f;
                data.bodyTrackSpeed = 2f;
                data.enableSway = true;
                data.swayIntensity = 1.0f;
                data.swaySpringFrequency = 2.6f;
                data.swayDampingRatio = 0.35f;
                data.danceTransitionSpeed = 0.6f;
                data.settingsVersion = 2;
                Debug.Log("[SettingsManager] Migrated v1 → v2 (spine tracking, sway, dance transitions).");
            }

            // ── Version 2 → 3: Feature Gap Plan additions ──
            if (data.settingsVersion < 3)
            {
                data.graphicsQuality = 1;
                data.enableRandomMessages = false;
                data.randomMessageIntervalMinutes = 10f;
                data.enableAmbientProbe = false;
                data.ambientProbeIntensity = 0.5f;
                data.soundThreshold = 0.02f;
                data.soundFilterApps = "";
                data.settingsVersion = 3;
                Debug.Log("[SettingsManager] Migrated v2 → v3 (graphics quality, random messages, ambient, audio filter).");
            }

            // ── Version 3 → 4: Character opacity ──
            if (data.settingsVersion < 4)
            {
                data.characterOpacity = 1.0f;
                data.settingsVersion = 4;
                Debug.Log("[SettingsManager] Migrated v3 → v4 (character opacity).");
            }

            // ── Version 4 → 5: Theme, IK, Accessories ──
            if (data.settingsVersion < 5)
            {
                data.uiHueShift = 0f;
                data.uiSaturation = 1f;
                data.enableIK = true;
                data.settingsVersion = 5;
                Debug.Log("[SettingsManager] Migrated v4 → v5 (theme, IK, accessories).");
            }
            // ── Version 5 → 6: Speech bubble on by default ──
            if (data.settingsVersion < 6)
            {
                data.enableSpeechBubble = true;
                data.settingsVersion = 6;
                Debug.Log("[SettingsManager] Migrated v5 → v6 (speech bubble default on).");
            }
            // Range clamping (always runs)
            data.fpsLimit = Mathf.Clamp(data.fpsLimit, 15, 165);
            data.sleepTimerSeconds = Mathf.Clamp(data.sleepTimerSeconds, 30f, 360f);
            data.sfxVolume = Mathf.Clamp01(data.sfxVolume);
            data.avatarSize = Mathf.Clamp(data.avatarSize, 0.5f, 2f);
            data.eyeBlend = Mathf.Clamp01(data.eyeBlend);
            data.headBlend = Mathf.Clamp01(data.headBlend);
            data.spineBlend = Mathf.Clamp01(data.spineBlend);
            data.eyeTrackSpeed = Mathf.Clamp(data.eyeTrackSpeed, 1f, 20f);
            data.headTrackSpeed = Mathf.Clamp(data.headTrackSpeed, 0.5f, 15f);
            data.bodyTrackSpeed = Mathf.Clamp(data.bodyTrackSpeed, 0.5f, 10f);
            data.swayIntensity = Mathf.Clamp(data.swayIntensity, 0f, 2f);
            data.swaySpringFrequency = Mathf.Clamp(data.swaySpringFrequency, 0.5f, 10f);
            data.swayDampingRatio = Mathf.Clamp(data.swayDampingRatio, 0.05f, 1f);
            data.danceTransitionSpeed = Mathf.Clamp(data.danceTransitionSpeed, 0.1f, 3f);

            // v3 clamping
            data.graphicsQuality = Mathf.Clamp(data.graphicsQuality, 0, 2);
            data.randomMessageIntervalMinutes = Mathf.Clamp(data.randomMessageIntervalMinutes, 5f, 30f);
            data.ambientProbeIntensity = Mathf.Clamp01(data.ambientProbeIntensity);
            data.soundThreshold = Mathf.Clamp(data.soundThreshold, 0f, 0.5f);

            // v4 clamping
            data.characterOpacity = Mathf.Clamp(data.characterOpacity, 0.1f, 1f);

            // v5 clamping
            data.uiHueShift = Mathf.Repeat(data.uiHueShift, 360f);
            data.uiSaturation = Mathf.Clamp(data.uiSaturation, 0f, 2f);
        }
    }

    /// <summary>
    /// All user-configurable settings. Serialized to JSON via JsonUtility.
    /// Add new fields with sensible defaults — old save files will get defaults for missing fields.
    /// </summary>
    [Serializable]
    public class SettingsData
    {
        // Version for migration — bump when adding new fields
        public int settingsVersion = 6;

        // ── Display ─────────────────────────────────────────
        public int fpsLimit = 60;
        public bool alwaysOnTop = true;
        public bool hideFromTaskbar = true;

        // ── Avatar ──────────────────────────────────────────
        public string selectedModelPath = "";   // Empty = default bundled model
        public float avatarSize = 1.0f;
        public float characterOpacity = 1.0f;    // 0.1..1.0, how opaque the character appears

        // ── Animation / Tracking ────────────────────────────
        public bool enableMouseTracking = true;
        public float eyeBlend = 1.0f;
        public float headBlend = 0.7f;

        // v2: Per-component track speeds (Mate Engine style)
        public float spineBlend = 0.5f;          // Spine/body lean toward cursor
        public float eyeTrackSpeed = 8f;          // Fast — eyes lead
        public float headTrackSpeed = 4f;         // Medium — head follows
        public float bodyTrackSpeed = 2f;         // Slow — body last

        // v2: Sway physics (window drag momentum)
        public bool enableSway = true;
        public float swayIntensity = 1.0f;        // 0..2 multiplier
        public float swaySpringFrequency = 2.6f;  // Hz — Mate Engine default
        public float swayDampingRatio = 0.35f;     // Under-damped for bounce

        // v2: Dance transition blend speed
        public float danceTransitionSpeed = 0.6f; // Seconds for style crossfade

        // ── Interaction ─────────────────────────────────────
        public bool enableParticles = true;
        public bool enableTouchSounds = true;
        public float sfxVolume = 1.0f;

        // ── AI / Speech ─────────────────────────────────────
        public bool enableSpeechBubble = true;  // On by default

        // ── System ──────────────────────────────────────────
        public bool enableSleepMode = false;
        public float sleepTimerSeconds = 120f;
        public bool enableAutoMemoryTrim = false;
        public bool minimizeToTray = true;
        public bool startWithWindows = false;

        // ── v3: Feature Gap Plan additions ──────────────────
        public int graphicsQuality = 1;               // 0=Low, 1=Medium, 2=High
        public bool enableRandomMessages = false;      // AI random messages (#12)
        public float randomMessageIntervalMinutes = 10f; // 5-30 min range
        public bool enableAmbientProbe = false;        // Desktop ambient lighting (#9)
        public float ambientProbeIntensity = 0.5f;     // 0..1
        public float soundThreshold = 0.02f;           // Audio threshold for dance (#24)
        public string soundFilterApps = "";            // Comma-separated app names (#24)

        // ── v5: Theme / IK / Accessories ────────────────────
        public float uiHueShift = 0f;                 // 0..360 degrees
        public float uiSaturation = 1f;               // 0..2 multiplier
        public bool enableIK = true;                   // Inverse kinematics (sit/drag poses)
    }
}
