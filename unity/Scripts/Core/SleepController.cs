using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Annabeth.Core
{
    /// <summary>
    /// Detects system-wide idle time and puts the companion to sleep.
    /// Reduces FPS and disables tracking when sleeping. Wakes on any input.
    /// Uses Win32 GetLastInputInfo for accurate idle detection on Windows.
    /// </summary>
    public class SleepController : MonoBehaviour
    {
#if UNITY_STANDALONE_WIN
        [DllImport("user32.dll")]
        static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);

        [StructLayout(LayoutKind.Sequential)]
        struct LASTINPUTINFO { public uint cbSize, dwTime; }
#endif

        public bool IsSleeping { get; private set; }
        public event Action<bool> OnSleepStateChanged;

        float _fallbackLastInput;

        void Start() => _fallbackLastInput = Time.unscaledTime;

        void Update()
        {
            var sm = SettingsManager.Instance;
            if (sm == null || !sm.data.enableSleepMode) return;

            float idle = GetSystemIdleSeconds();

            if (IsSleeping)
            {
                if (idle < 2f) SetSleeping(false);
            }
            else
            {
                if (idle >= sm.data.sleepTimerSeconds) SetSleeping(true);
            }
        }

        public void ToggleSleep() => SetSleeping(!IsSleeping);

        public void ApplySettings()
        {
            if (SettingsManager.Instance != null && !SettingsManager.Instance.data.enableSleepMode)
                SetSleeping(false);
        }

        void SetSleeping(bool sleeping)
        {
            if (IsSleeping == sleeping) return;
            IsSleeping = sleeping;

            var fps = FindFirstObjectByType<FPSController>();
            if (sleeping) fps?.SetTemporaryLimit(10);
            else fps?.RestoreLimit();

            var eyes = FindFirstObjectByType<Annabeth.Avatar.EyeTrackingController>();
            eyes?.SetEnabled(!sleeping);

            OnSleepStateChanged?.Invoke(sleeping);
            Debug.Log($"[SleepController] {(sleeping ? "Sleeping" : "Awake")}");
        }

        float GetSystemIdleSeconds()
        {
#if UNITY_STANDALONE_WIN
            var info = new LASTINPUTINFO { cbSize = (uint)Marshal.SizeOf<LASTINPUTINFO>() };
            if (GetLastInputInfo(ref info))
            {
                uint elapsed = unchecked((uint)Environment.TickCount - info.dwTime);
                return elapsed / 1000f;
            }
#endif
            // Fallback: detect Unity-level input
            if (UnityEngine.Input.anyKey
                || Mathf.Abs(UnityEngine.Input.GetAxis("Mouse X")) > 0.01f
                || Mathf.Abs(UnityEngine.Input.GetAxis("Mouse Y")) > 0.01f)
                _fallbackLastInput = Time.unscaledTime;
            return Time.unscaledTime - _fallbackLastInput;
        }
    }
}
