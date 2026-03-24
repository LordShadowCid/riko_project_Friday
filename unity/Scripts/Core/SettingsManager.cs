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

        /// <summary>
        /// Push current settings values to all live controllers.
        /// Called once after VRM loads and whenever settings change.
        /// </summary>
        public void ApplyAllSettings()
        {
            // FPS
            var fps = FindFirstObjectByType<FPSController>();
            fps?.ApplySettings();

            // Memory optimizer
            var mem = FindFirstObjectByType<MemoryOptimizer>();
            mem?.ApplySettings();

            // Always-on-top
            var win = FindFirstObjectByType<TransparentWindowController>();
            if (win != null)
                win.SetTopmost(data.alwaysOnTop);

            // Sleep controller
            var sleep = FindFirstObjectByType<SleepController>();
            sleep?.ApplySettings();

            // Start with Windows
            ApplyStartWithWindows();

            Debug.Log("[SettingsManager] Applied all settings.");
        }

        // ── Start with Windows ────────────────────────────────────

#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
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
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
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
                                string val = $"\"{ exe}\"";
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
            // Example: if (data.settingsVersion < 2) { /* migrate */ data.settingsVersion = 2; }
            data.fpsLimit = Mathf.Clamp(data.fpsLimit, 15, 165);
            data.sleepTimerSeconds = Mathf.Clamp(data.sleepTimerSeconds, 30f, 360f);
            data.sfxVolume = Mathf.Clamp01(data.sfxVolume);
            data.avatarSize = Mathf.Clamp(data.avatarSize, 0.5f, 2f);
            data.eyeBlend = Mathf.Clamp01(data.eyeBlend);
            data.headBlend = Mathf.Clamp01(data.headBlend);
        }
    }

    /// <summary>
    /// All user-configurable settings. Serialized to JSON via JsonUtility.
    /// Add new fields with sensible defaults — old save files will get defaults for missing fields.
    /// </summary>
    [Serializable]
    public class SettingsData
    {
        // Version for migration
        public int settingsVersion = 1;

        // ── Display ─────────────────────────────────────────
        public int fpsLimit = 60;
        public bool alwaysOnTop = true;
        public bool hideFromTaskbar = true;

        // ── Avatar ──────────────────────────────────────────
        public string selectedModelPath = "";   // Empty = default bundled model
        public float avatarSize = 1.0f;

        // ── Animation / Tracking ────────────────────────────
        public bool enableMouseTracking = true;
        public float eyeBlend = 1.0f;
        public float headBlend = 0.7f;

        // ── Interaction ─────────────────────────────────────
        public bool enableParticles = true;
        public bool enableTouchSounds = true;
        public float sfxVolume = 1.0f;

        // ── AI / Speech ─────────────────────────────────────
        public bool enableSpeechBubble = false; // Off by default, toggle-on feature

        // ── System ──────────────────────────────────────────
        public bool enableSleepMode = false;
        public float sleepTimerSeconds = 120f;
        public bool enableAutoMemoryTrim = false;
        public bool minimizeToTray = true;
        public bool startWithWindows = false;
    }
}
