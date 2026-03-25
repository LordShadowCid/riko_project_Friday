using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Annabeth.Core
{
    /// <summary>
    /// Random-walk locomotion across the desktop by moving the Unity window via Win32 MoveWindow.
    /// Inspired by Mate-Engine's AvatarLocomotionController (746 lines) — simplified.
    ///
    /// Features:
    /// - Random walk with configurable distance and speed
    /// - Monitor-bounds-aware direction picking (avoids walking off-screen)
    /// - Walking animation integration via Animator "IsWalking" bool + facing direction
    /// - Screen-edge hiding / peeking
    /// - Can be toggled on/off via hotkey or settings
    /// </summary>
    public class DesktopLocomotionController : MonoBehaviour
    {
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
        [StructLayout(LayoutKind.Sequential)]
        struct RECT { public int Left, Top, Right, Bottom; }

        [StructLayout(LayoutKind.Sequential)]
        struct MONITORINFO
        {
            public int cbSize;
            public RECT rcMonitor;
            public RECT rcWork;
            public uint dwFlags;
        }

        [DllImport("user32.dll")] static extern IntPtr GetActiveWindow();
        [DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
        [DllImport("user32.dll")] static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int w, int h, bool repaint);
        [DllImport("user32.dll")] static extern IntPtr MonitorFromWindow(IntPtr hwnd, uint dwFlags);
        [DllImport("user32.dll")] static extern bool GetMonitorInfo(IntPtr hMonitor, ref MONITORINFO lpmi);

        const uint MONITOR_DEFAULTTOPRIMARY = 1;
#endif

        [Header("Walk Settings")]
        [SerializeField] private int minWalkDistance = 250;
        [SerializeField] private int maxWalkDistance = 550;
        [SerializeField] private int walkSpeed = 2; // pixels per frame
        [SerializeField] private float decisionInterval = 10f;

        [Header("Screen-Edge Peeking")]
        [SerializeField] private bool peekingEnabled = true;
        [SerializeField] private int peekVisiblePixels = 40; // how much of window stays visible when hiding

        [Header("State")]
        [SerializeField] private bool locomotionEnabled;

        public bool IsWalking { get; private set; }
        public bool IsPeeking { get; private set; }
        public bool IsEnabled => locomotionEnabled;

        /// <summary>Fired when walking state changes. Bool = isWalking.</summary>
        public event Action<bool> OnWalkStateChanged;
        /// <summary>Fired when the avatar starts peeking from screen edge.</summary>
        public event Action<bool> OnPeekStateChanged;

        // Walk direction: -1 = left, 0 = idle, 1 = right
        private int _walkDirection;
        private int _remainingPixels;
        private float _decisionTimer;
        private bool _initialized;

#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
        private IntPtr _hwnd;
        private RECT _workArea; // usable screen area (excludes taskbar)

        private void Start()
        {
            _hwnd = GetActiveWindow();
            if (_hwnd == IntPtr.Zero) return;
            RefreshMonitorBounds();
            _decisionTimer = decisionInterval;
            _initialized = true;
            Debug.Log($"[DesktopLocomotion] Initialized. Work area: ({_workArea.Left},{_workArea.Top})-({_workArea.Right},{_workArea.Bottom})");
        }

        private void Update()
        {
            if (!_initialized || !locomotionEnabled || _hwnd == IntPtr.Zero) return;

            // Decision timer — pick a new walk direction periodically
            _decisionTimer -= Time.deltaTime;
            if (_decisionTimer <= 0f)
            {
                _decisionTimer = decisionInterval;
                if (!IsWalking && !IsPeeking)
                    DecideWalk();
            }

            // Execute walk
            if (IsWalking)
                ExecuteWalk();
        }

        private void RefreshMonitorBounds()
        {
            var hMon = MonitorFromWindow(_hwnd, MONITOR_DEFAULTTOPRIMARY);
            var mi = new MONITORINFO { cbSize = Marshal.SizeOf<MONITORINFO>() };
            if (GetMonitorInfo(hMon, ref mi))
                _workArea = mi.rcWork;
        }

        private void DecideWalk()
        {
            GetWindowRect(_hwnd, out RECT r);
            int w = r.Right - r.Left;

            // Determine which direction has more room
            int spaceLeft = r.Left - _workArea.Left;
            int spaceRight = _workArea.Right - r.Right;

            // Pick direction — prefer the side with more room, with randomness
            if (spaceLeft < 50 && spaceRight < 50) return; // no room

            if (spaceLeft < 50)
                _walkDirection = 1; // only right
            else if (spaceRight < 50)
                _walkDirection = -1; // only left
            else
                _walkDirection = UnityEngine.Random.value > 0.5f ? 1 : -1;

            int maxDist = _walkDirection == 1 ? spaceRight : spaceLeft;
            int desired = UnityEngine.Random.Range(minWalkDistance, maxWalkDistance + 1);
            _remainingPixels = Mathf.Min(desired, maxDist - 10);

            if (_remainingPixels < 30) return; // too short, skip

            SetWalking(true);
            Debug.Log($"[DesktopLocomotion] Walking {(_walkDirection > 0 ? "right" : "left")} {_remainingPixels}px");
        }

        private void ExecuteWalk()
        {
            GetWindowRect(_hwnd, out RECT r);
            int w = r.Right - r.Left;
            int h = r.Bottom - r.Top;
            int step = Mathf.Min(walkSpeed, _remainingPixels);

            int newX = r.Left + (_walkDirection * step);

            // Clamp to work area
            newX = Mathf.Clamp(newX, _workArea.Left, _workArea.Right - w);

            MoveWindow(_hwnd, newX, r.Top, w, h, true);
            _remainingPixels -= step;

            if (_remainingPixels <= 0)
                SetWalking(false);
        }

        /// <summary>
        /// Walk to the nearest screen edge and peek out (partial hide).
        /// </summary>
        public void StartPeek()
        {
            if (!_initialized || !peekingEnabled || _hwnd == IntPtr.Zero) return;
            if (IsPeeking) { StopPeek(); return; } // toggle

            GetWindowRect(_hwnd, out RECT r);
            int w = r.Right - r.Left;

            // Determine nearest edge
            int distLeft = r.Left - _workArea.Left;
            int distRight = _workArea.Right - r.Right;

            if (distLeft <= distRight)
            {
                // Hide to left edge
                _walkDirection = -1;
                _remainingPixels = distLeft + w - peekVisiblePixels;
            }
            else
            {
                // Hide to right edge
                _walkDirection = 1;
                _remainingPixels = distRight + w - peekVisiblePixels;
            }

            if (_remainingPixels > 0)
            {
                SetWalking(true);
                IsPeeking = true;
                OnPeekStateChanged?.Invoke(true);
                Debug.Log("[DesktopLocomotion] Peeking at screen edge");
            }
        }

        /// <summary>
        /// Return from peeking to the nearest fully-visible position.
        /// </summary>
        public void StopPeek()
        {
            if (!IsPeeking || _hwnd == IntPtr.Zero) return;

            GetWindowRect(_hwnd, out RECT r);
            int w = r.Right - r.Left;
            int h = r.Bottom - r.Top;

            // Snap back to fully visible on the nearest edge
            int newX;
            if (r.Left < _workArea.Left)
                newX = _workArea.Left; // was hiding on left
            else
                newX = _workArea.Right - w; // was hiding on right

            MoveWindow(_hwnd, newX, r.Top, w, h, true);

            IsPeeking = false;
            SetWalking(false);
            OnPeekStateChanged?.Invoke(false);
            Debug.Log("[DesktopLocomotion] Stopped peeking");
        }

        /// <summary>
        /// Enable/disable random walk locomotion.
        /// </summary>
        public void SetEnabled(bool enabled)
        {
            locomotionEnabled = enabled;
            if (!enabled)
            {
                SetWalking(false);
                if (IsPeeking) StopPeek();
            }
            Debug.Log($"[DesktopLocomotion] Enabled: {enabled}");
        }

        public void ToggleEnabled()
        {
            SetEnabled(!locomotionEnabled);
        }

        private void SetWalking(bool walking)
        {
            if (IsWalking == walking) return;
            IsWalking = walking;
            if (!walking) _remainingPixels = 0;
            OnWalkStateChanged?.Invoke(walking);
        }

        /// <summary>
        /// Get current walk direction for facing: -1 left, 0 idle, 1 right.
        /// </summary>
        public int GetWalkDirection() => IsWalking ? _walkDirection : 0;
#else
        private void Start()
        {
            Debug.Log("[DesktopLocomotion] Only active in Windows standalone builds.");
        }

        public void SetEnabled(bool enabled) { locomotionEnabled = enabled; }
        public void ToggleEnabled() { locomotionEnabled = !locomotionEnabled; }
        public void StartPeek() { }
        public void StopPeek() { }
        public int GetWalkDirection() => 0;
#endif
    }
}
