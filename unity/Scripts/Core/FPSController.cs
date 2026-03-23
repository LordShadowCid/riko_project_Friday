using UnityEngine;

namespace Annabeth.Core
{
    /// <summary>
    /// Limits Application.targetFrameRate to save power for a long-running desktop process.
    /// Ported from Mate-Engine FPSLimiter.cs — same logic, reads from SettingsManager.
    /// </summary>
    public class FPSController : MonoBehaviour
    {
        private void Start()
        {
            // Disable VSync so targetFrameRate actually works
            QualitySettings.vSyncCount = 0;
            ApplySettings();
        }

        /// <summary>Called by SettingsManager.ApplyAllSettings() and SettingsPanel.</summary>
        public void ApplySettings()
        {
            int target = SettingsManager.Instance != null
                ? SettingsManager.Instance.data.fpsLimit
                : 60;
            SetFPSLimit(target);
        }

        public void SetFPSLimit(int fps)
        {
            fps = Mathf.Clamp(fps, 15, 165);
            Application.targetFrameRate = fps;
        }

        /// <summary>
        /// Temporarily drop FPS (e.g., during sleep mode) then restore later.
        /// </summary>
        public void SetTemporaryLimit(int fps)
        {
            Application.targetFrameRate = Mathf.Clamp(fps, 5, 165);
        }

        public void RestoreLimit()
        {
            ApplySettings();
        }
    }
}
