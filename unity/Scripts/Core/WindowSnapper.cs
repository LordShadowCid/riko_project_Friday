using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Annabeth.Core
{
    /// <summary>
    /// Snaps the standalone window to screen edges, taskbar, and other window title bars.
    /// Windows-only via Win32 P/Invoke. Only active in standalone builds.
    ///
    /// Behaviors:
    /// - Snap to screen edges when dragged close (magnetic 20px threshold)
    /// - Double-right-click to "sit" on taskbar (bottom of screen)
    /// - Press Home to cycle snap positions: bottom-right, bottom-left, top-right, top-left
    /// - Window-top sitting: detect foreground window title bars and sit on top
    /// - Gravity/falling: when a sitting surface disappears, fall to taskbar with physics
    /// </summary>
    public class WindowSnapper : MonoBehaviour
    {
#if UNITY_STANDALONE_WIN
        [StructLayout(LayoutKind.Sequential)]
        struct RECT { public int Left, Top, Right, Bottom; }

        [StructLayout(LayoutKind.Sequential)]
        struct POINT { public int X, Y; }

        [StructLayout(LayoutKind.Sequential)]
        struct APPBARDATA
        {
            public int cbSize;
            public IntPtr hWnd;
            public uint uCallbackMessage;
            public uint uEdge;
            public RECT rc;
            public int lParam;
        }

        const uint ABM_GETTASKBARPOS = 5;

        [DllImport("user32.dll")] static extern IntPtr GetActiveWindow();
        [DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
        [DllImport("user32.dll")] static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int w, int h, bool repaint);
        [DllImport("user32.dll")] static extern int GetSystemMetrics(int nIndex);
        [DllImport("shell32.dll")] static extern uint SHAppBarMessage(uint dwMessage, ref APPBARDATA pData);
        [DllImport("user32.dll")] static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
        [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr hWnd);
        [DllImport("user32.dll")] static extern bool IsIconic(IntPtr hWnd);
        [DllImport("user32.dll")] static extern uint GetWindowLong(IntPtr hWnd, int nIndex);

        delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
        const int GWL_EXSTYLE = -20;
        const uint WS_EX_TOOLWINDOW = 0x00000080;
#endif

        [Header("Snap Settings")]
        [SerializeField] private int snapThreshold = 20;
        [SerializeField] private KeyCode snapCycleKey = KeyCode.Home;

        [Header("Window Sitting")]
        [SerializeField] private int sittingProbeRadius = 30; // pixels around avatar center to detect windows
        [SerializeField] private float sittingCheckInterval = 0.5f;

        [Header("Gravity")]
        [SerializeField] private float gravityAcceleration = 1200f; // pixels/sec²
        [SerializeField] private float maxFallSpeed = 800f;

        /// <summary>True when avatar is sitting (on taskbar or window top).</summary>
        public bool IsSitting { get; private set; }
#if UNITY_STANDALONE_WIN
        public System.IntPtr SittingOnWindowHandle => _sittingOnWindow;
#else
        public System.IntPtr SittingOnWindowHandle => System.IntPtr.Zero;
#endif
        /// <summary>True when avatar is falling due to gravity.</summary>
        public bool IsFalling { get; private set; }

        public event Action<bool> OnSittingChanged;
        public event Action OnFallStarted;
        public event Action OnFallLanded;

        private int _snapIndex;

#if UNITY_STANDALONE_WIN
        private IntPtr _hwnd;
        private int _screenW, _screenH;
        private int _taskbarH;
        private float _doubleClickTimer;
        private const float DoubleClickWindow = 0.4f;
        private float _snapCheckTimer;
        private const float SnapCheckInterval = 0.05f; // 20 Hz

        // Window sitting state
        private IntPtr _sittingOnWindow;
        private RECT _sittingWindowRect;
        private float _sittingCheckTimer;

        // Drag-hold sit guard (Feature #5)
        private float _dragHoldTimer;
        private const float MinDragHoldToSit = 1.0f;
        private bool _isDragActive;

        // SmoothDamp following (Feature #2)
        private float _followVelX, _followVelY;
        private const float FollowSmoothTime = 0.08f;

        // Gravity state
        private float _fallVelocity;

        private void Start()
        {
            _hwnd = GetActiveWindow();
            if (_hwnd == IntPtr.Zero) return;

            _screenW = GetSystemMetrics(0);
            _screenH = GetSystemMetrics(1);

            var abd = new APPBARDATA { cbSize = Marshal.SizeOf<APPBARDATA>() };
            SHAppBarMessage(ABM_GETTASKBARPOS, ref abd);
            _taskbarH = abd.rc.Bottom - abd.rc.Top;
            if (_taskbarH < 20) _taskbarH = 48;

            Debug.Log($"[WindowSnapper] Screen: {_screenW}x{_screenH}, Taskbar: {_taskbarH}px");
        }

        private void Update()
        {
            if (_hwnd == IntPtr.Zero) return;

            // Gravity takes priority
            if (IsFalling)
            {
                ExecuteFall();
                return;
            }

            // Drag-hold timer for sit guard (Feature #5)
            if (_isDragActive) _dragHoldTimer += Time.deltaTime;

            // Double right-click to sit on taskbar
            if (UnityEngine.Input.GetMouseButtonDown(1))
            {
                if (_doubleClickTimer > 0f)
                {
                    SitOnTaskbar();
                    _doubleClickTimer = 0f;
                }
                else
                {
                    _doubleClickTimer = DoubleClickWindow;
                }
            }
            if (_doubleClickTimer > 0f)
                _doubleClickTimer -= Time.deltaTime;

            // Home key to cycle snap positions
            if (UnityEngine.Input.GetKeyDown(snapCycleKey))
                CycleSnapPosition();

            // Check if the window we're sitting on is still there
            if (IsSitting && _sittingOnWindow != IntPtr.Zero)
            {
                _sittingCheckTimer -= Time.deltaTime;
                if (_sittingCheckTimer <= 0f)
                {
                    _sittingCheckTimer = sittingCheckInterval;
                    CheckSittingSurface();
                }
            }
        }

        private void LateUpdate()
        {
            if (_hwnd == IntPtr.Zero || IsFalling) return;
            _snapCheckTimer -= Time.deltaTime;
            if (_snapCheckTimer <= 0f)
            {
                _snapCheckTimer = SnapCheckInterval;
                ApplyEdgeSnap();
            }
        }

        private void ApplyEdgeSnap()
        {
            GetWindowRect(_hwnd, out RECT r);
            int w = r.Right - r.Left;
            int h = r.Bottom - r.Top;
            int x = r.Left;
            int y = r.Top;
            bool snapped = false;

            if (Math.Abs(x) < snapThreshold) { x = 0; snapped = true; }
            if (Math.Abs(r.Right - _screenW) < snapThreshold) { x = _screenW - w; snapped = true; }
            if (Math.Abs(y) < snapThreshold) { y = 0; snapped = true; }
            int bottomTarget = _screenH - _taskbarH - h;
            if (Math.Abs(r.Top - bottomTarget) < snapThreshold) { y = bottomTarget; snapped = true; }

            if (snapped)
                MoveWindow(_hwnd, x, y, w, h, true);
        }

        private void SitOnTaskbar()
        {
            GetWindowRect(_hwnd, out RECT r);
            int w = r.Right - r.Left;
            int h = r.Bottom - r.Top;
            int x = _screenW - w - 10;
            int y = _screenH - _taskbarH - h;
            MoveWindow(_hwnd, x, y, w, h, true);

            _sittingOnWindow = IntPtr.Zero; // taskbar sit, no tracked window
            SetSitting(true);
            Debug.Log("[WindowSnapper] Sitting on taskbar (bottom-right)");
        }

        /// <summary>
        /// Try to sit on the nearest visible window title bar beneath the avatar.
        /// Called externally (e.g., by hotkey or drag-end near a window top).
        /// Respects drag-hold guard: only sits after holding drag for MinDragHoldToSit seconds.
        /// </summary>
        public void TrySitOnNearestWindow()
        {
            // Feature #5: Drag-hold guard — prevent accidental sits during quick repositioning
            if (_isDragActive && _dragHoldTimer < MinDragHoldToSit) return;
            GetWindowRect(_hwnd, out RECT myRect);
            int myCenterX = (myRect.Left + myRect.Right) / 2;
            int myBottom = myRect.Bottom;
            int myW = myRect.Right - myRect.Left;
            int myH = myRect.Bottom - myRect.Top;

            IntPtr bestWnd = IntPtr.Zero;
            int bestTop = int.MaxValue;
            RECT bestRect = default;

            // Find the nearest window whose top edge is just below our bottom
            EnumWindows((hWnd, _) =>
            {
                if (hWnd == _hwnd) return true;
                if (!IsWindowVisible(hWnd) || IsIconic(hWnd)) return true;

                uint exStyle = GetWindowLong(hWnd, GWL_EXSTYLE);
                if ((exStyle & WS_EX_TOOLWINDOW) != 0) return true;

                GetWindowRect(hWnd, out RECT wr);
                int wWidth = wr.Right - wr.Left;
                int wHeight = wr.Bottom - wr.Top;
                if (wWidth < 100 || wHeight < 50) return true; // skip tiny windows

                // Check if our center X is within the window's horizontal range
                if (myCenterX >= wr.Left && myCenterX <= wr.Right)
                {
                    // Window top must be near or below our bottom
                    int dist = Math.Abs(wr.Top - myBottom);
                    if (dist < sittingProbeRadius && wr.Top < bestTop)
                    {
                        bestTop = wr.Top;
                        bestWnd = hWnd;
                        bestRect = wr;
                    }
                }
                return true;
            }, IntPtr.Zero);

            if (bestWnd != IntPtr.Zero)
            {
                // Place our avatar on top of the detected window
                int sitX = Mathf.Clamp(myCenterX - myW / 2, bestRect.Left, bestRect.Right - myW);
                int sitY = bestRect.Top - myH;
                MoveWindow(_hwnd, sitX, sitY, myW, myH, true);

                _sittingOnWindow = bestWnd;
                _sittingWindowRect = bestRect;
                SetSitting(true);
                Debug.Log($"[WindowSnapper] Sitting on window at ({bestRect.Left},{bestRect.Top})");
            }
        }

        /// <summary>
        /// Check if the window we're sitting on has moved or been closed.
        /// If so, trigger gravity/falling.
        /// </summary>
        private void CheckSittingSurface()
        {
            if (_sittingOnWindow == IntPtr.Zero)
            {
                // Sitting on taskbar — check if we're still above taskbar
                GetWindowRect(_hwnd, out RECT r);
                int expectedY = _screenH - _taskbarH - (r.Bottom - r.Top);
                if (Math.Abs(r.Top - expectedY) > 5) return; // user dragged away
                return; // taskbar is always there
            }

            // Check if window is still visible
            if (!IsWindowVisible(_sittingOnWindow) || IsIconic(_sittingOnWindow))
            {
                StartFalling();
                return;
            }

            // Check if window moved significantly
            GetWindowRect(_sittingOnWindow, out RECT wr);
            if (Math.Abs(wr.Top - _sittingWindowRect.Top) > 50 ||
                Math.Abs(wr.Left - _sittingWindowRect.Left) > 200)
            {
                StartFalling();
                return;
            }

            // Window moved a bit — follow it with SmoothDamp (Feature #2)
            if (wr.Top != _sittingWindowRect.Top || wr.Left != _sittingWindowRect.Left)
            {
                GetWindowRect(_hwnd, out RECT myRect);
                int myW = myRect.Right - myRect.Left;
                int myH = myRect.Bottom - myRect.Top;
                int offsetX = myRect.Left - _sittingWindowRect.Left;
                int targetX = wr.Left + offsetX;
                int targetY = wr.Top - myH;
                targetX = Mathf.Clamp(targetX, wr.Left, wr.Right - myW);

                float smoothX = Mathf.SmoothDamp(myRect.Left, targetX, ref _followVelX, FollowSmoothTime);
                float smoothY = Mathf.SmoothDamp(myRect.Top, targetY, ref _followVelY, FollowSmoothTime);
                MoveWindow(_hwnd, (int)smoothX, (int)smoothY, myW, myH, false);
                _sittingWindowRect = wr;
            }
        }

        private void StartFalling()
        {
            SetSitting(false);
            IsFalling = true;
            _fallVelocity = 0f;
            _sittingOnWindow = IntPtr.Zero;
            OnFallStarted?.Invoke();
            Debug.Log("[WindowSnapper] Surface lost — falling!");
        }

        private void ExecuteFall()
        {
            GetWindowRect(_hwnd, out RECT r);
            int w = r.Right - r.Left;
            int h = r.Bottom - r.Top;

            _fallVelocity += gravityAcceleration * Time.deltaTime;
            if (_fallVelocity > maxFallSpeed) _fallVelocity = maxFallSpeed;

            int fallPixels = Mathf.CeilToInt(_fallVelocity * Time.deltaTime);
            int newY = r.Top + fallPixels;

            // Check for landing on taskbar (or bottom of work area)
            int taskbarTop = _screenH - _taskbarH - h;
            if (newY >= taskbarTop)
            {
                newY = taskbarTop;
                IsFalling = false;
                _fallVelocity = 0f;
                SetSitting(true);
                OnFallLanded?.Invoke();
                Debug.Log("[WindowSnapper] Landed on taskbar!");
            }

            MoveWindow(_hwnd, r.Left, newY, w, h, true);
        }

        private void SetSitting(bool sitting)
        {
            if (IsSitting == sitting) return;
            IsSitting = sitting;
            OnSittingChanged?.Invoke(sitting);
        }

        private void CycleSnapPosition()
        {
            GetWindowRect(_hwnd, out RECT r);
            int w = r.Right - r.Left;
            int h = r.Bottom - r.Top;
            int x, y;

            _snapIndex = (_snapIndex + 1) % 4;
            switch (_snapIndex)
            {
                case 0:
                    x = _screenW - w - 10;
                    y = _screenH - _taskbarH - h;
                    break;
                case 1:
                    x = 10;
                    y = _screenH - _taskbarH - h;
                    break;
                case 2:
                    x = _screenW - w - 10;
                    y = 10;
                    break;
                default:
                    x = 10;
                    y = 10;
                    break;
            }

            MoveWindow(_hwnd, x, y, w, h, true);
            SetSitting(false);
            _sittingOnWindow = IntPtr.Zero;
            string[] names = { "bottom-right", "bottom-left", "top-right", "top-left" };
            Debug.Log($"[WindowSnapper] Snapped to {names[_snapIndex]}");
        }

        /// <summary>
        /// Get the taskbar height in pixels.
        /// </summary>
        public int GetTaskbarHeight() => _taskbarH;

        // ── Drag-Hold Sit Guard (Feature #5) ────────────────────

        /// <summary>Call from CompanionManager when drag starts.</summary>
        public void NotifyDragStart()
        {
            _isDragActive = true;
            _dragHoldTimer = 0f;
        }

        /// <summary>Call from CompanionManager when drag ends.</summary>
        public void NotifyDragEnd()
        {
            if (_isDragActive && _dragHoldTimer >= MinDragHoldToSit)
                TrySitOnNearestWindow();
            _isDragActive = false;
            _dragHoldTimer = 0f;
        }
#else
        private void Start()
        {
            Debug.Log("[WindowSnapper] Only active in Windows standalone builds.");
        }

        public void TrySitOnNearestWindow() { }
        public int GetTaskbarHeight() => 0;
        public void NotifyDragStart() { }
        public void NotifyDragEnd() { }
#endif
    }
}
