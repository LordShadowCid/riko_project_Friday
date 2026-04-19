using System;
using System.Runtime.InteropServices;
using UnityEngine;
using Annabeth.Avatar;
using Annabeth.UI;
using Annabeth.Core;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Detects user inactivity via Win32 GetLastInputInfo and drives idle/sleep states.
    /// Phase 7: Idle / Screensaver Mode.
    /// </summary>
    public class IdleController : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private Animator animator;
        [SerializeField] private EyeTrackingController eyeTrackingController;
        [SerializeField] private IdleBubbleController idleBubbleController;

        [Header("Performance")]
        [Tooltip("Components to disable during sleep (e.g. post-processing) to save GPU.")]
        [SerializeField] private MonoBehaviour[] disableOnSleep;

        [Header("Settings")]
        [Tooltip("Seconds of inactivity before entering idle. 0 = read from SettingsManager.")]
        [SerializeField] private float idleTimeoutSeconds = 300f;

        // ── Events (used by Phase 9 Discord and Phase 6 IdleBubble) ──────────
        public event Action<bool> OnIdleStateChanged;
        public event Action<bool> OnSleepStateChanged;

        // ── Win32 input detection ─────────────────────────────────────────────
        [DllImport("user32.dll")]
        private static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);

        [StructLayout(LayoutKind.Sequential)]
        private struct LASTINPUTINFO
        {
            public uint cbSize;
            public uint dwTime;
        }

        // ── State ─────────────────────────────────────────────────────────────
        private bool _isIdle    = false;
        private bool _isSleeping = false;

        public bool IsIdle     => _isIdle;
        public bool IsSleeping => _isSleeping;

        // ── Lifecycle ─────────────────────────────────────────────────────────

        private void Awake()
        {
#if !UNITY_EDITOR_WIN && !UNITY_STANDALONE_WIN
            // GetLastInputInfo is Windows-only; disable on other platforms silently.
            enabled = false;
            return;
#endif
            // Read timeout from SettingsManager if set to default sentinel
            if (idleTimeoutSeconds <= 0f && SettingsManager.Instance != null)
                idleTimeoutSeconds = SettingsManager.Instance.data.idleTimeoutSeconds;
            if (idleTimeoutSeconds <= 0f)
                idleTimeoutSeconds = 300f;  // Hard fallback: 5 minutes
        }

        private void Update()
        {
            float idle = GetIdleSeconds();
            float sleepThreshold = idleTimeoutSeconds * 2f;

            if (idle >= sleepThreshold)
            {
                if (!_isSleeping) SetSleeping(true);
            }
            else if (idle >= idleTimeoutSeconds)
            {
                if (_isSleeping) SetSleeping(false);
                if (!_isIdle) SetIdle(true);
            }
            else
            {
                // User is active (idle < 1s confirms actual movement)
                if (_isSleeping) SetSleeping(false);
                if (_isIdle)     SetIdle(false);
            }
        }

        // ── Helpers ───────────────────────────────────────────────────────────

        private float GetIdleSeconds()
        {
            var info = new LASTINPUTINFO { cbSize = (uint)Marshal.SizeOf<LASTINPUTINFO>() };
            if (!GetLastInputInfo(ref info)) return 0f;
            // Environment.TickCount wraps at ~25 days; safe for normal use
            return (Environment.TickCount - (int)info.dwTime) / 1000f;
        }

        private void SetIdle(bool idle)
        {
            _isIdle = idle;
            if (animator != null) animator.SetBool("isIdle", idle);
            eyeTrackingController?.SetTrackingMode(idle ? TrackingMode.Reduced : TrackingMode.Normal);
            idleBubbleController?.SetIdleMode(idle);
            OnIdleStateChanged?.Invoke(idle);
            Debug.Log($"[IdleController] Idle: {idle}");
        }

        private void SetSleeping(bool sleeping)
        {
            _isSleeping = sleeping;
            if (animator != null) animator.SetBool("isSleeping", sleeping);
            eyeTrackingController?.SetTrackingMode(sleeping ? TrackingMode.Disabled : TrackingMode.Normal);
            if (disableOnSleep != null)
            {
                foreach (var comp in disableOnSleep)
                    if (comp != null) comp.enabled = !sleeping;
            }
            OnSleepStateChanged?.Invoke(sleeping);
            // Waking from sleep also clears idle flag
            if (!sleeping && _isIdle) SetIdle(false);
            Debug.Log($"[IdleController] Sleeping: {sleeping}");
        }

        /// <summary>Override the idle timeout at runtime (e.g. from SettingsPanel).</summary>
        public void SetTimeout(float seconds)
        {
            idleTimeoutSeconds = Mathf.Max(10f, seconds);
        }
    }
}
