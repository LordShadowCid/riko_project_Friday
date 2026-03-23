using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Annabeth.Core
{
    /// <summary>
    /// Snaps the standalone window to screen edges and the taskbar.
    /// Windows-only via Win32 P/Invoke. Only active in standalone builds.
    ///
    /// Behaviors:
    /// - Snap to screen edges when dragged close (magnetic 20px threshold)
    /// - Double-right-click to "sit" on taskbar (bottom of screen)
    /// - Press Home to cycle snap positions: bottom-right, bottom-left, top-right
    /// </summary>
    public class WindowSnapper : MonoBehaviour
    {
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
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
#endif

        [Header("Snap Settings")]
        [SerializeField] private int snapThreshold = 20;
        [SerializeField] private KeyCode snapCycleKey = KeyCode.Home;

        private int _snapIndex;

#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
        private IntPtr _hwnd;
        private int _screenW, _screenH;
        private int _taskbarH;
        private float _doubleClickTimer;
        private const float DoubleClickWindow = 0.4f;
        private float _snapCheckTimer;
        private const float SnapCheckInterval = 0.05f; // 20 Hz instead of every frame

        private void Start()
        {
            _hwnd = GetActiveWindow();
            if (_hwnd == IntPtr.Zero) return;

            _screenW = GetSystemMetrics(0); // SM_CXSCREEN
            _screenH = GetSystemMetrics(1); // SM_CYSCREEN

            // Get taskbar height
            var abd = new APPBARDATA { cbSize = Marshal.SizeOf<APPBARDATA>() };
            SHAppBarMessage(ABM_GETTASKBARPOS, ref abd);
            _taskbarH = abd.rc.Bottom - abd.rc.Top;
            if (_taskbarH < 20) _taskbarH = 48; // fallback

            Debug.Log($"[WindowSnapper] Screen: {_screenW}x{_screenH}, Taskbar: {_taskbarH}px");
        }

        private void Update()
        {
            if (_hwnd == IntPtr.Zero) return;

            // Double right-click to sit on taskbar
            if (Input.GetMouseButtonDown(1))
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
            if (Input.GetKeyDown(snapCycleKey))
            {
                CycleSnapPosition();
            }
        }

        private void LateUpdate()
        {
            if (_hwnd == IntPtr.Zero) return;
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

            // Snap to left edge
            if (Math.Abs(x) < snapThreshold) { x = 0; snapped = true; }
            // Snap to right edge
            if (Math.Abs(r.Right - _screenW) < snapThreshold) { x = _screenW - w; snapped = true; }
            // Snap to top edge
            if (Math.Abs(y) < snapThreshold) { y = 0; snapped = true; }
            // Snap to bottom (above taskbar)
            int bottomTarget = _screenH - _taskbarH - h;
            if (Math.Abs(r.Top - bottomTarget) < snapThreshold) { y = bottomTarget; snapped = true; }

            if (snapped)
            {
                MoveWindow(_hwnd, x, y, w, h, true);
            }
        }

        private void SitOnTaskbar()
        {
            GetWindowRect(_hwnd, out RECT r);
            int w = r.Right - r.Left;
            int h = r.Bottom - r.Top;
            // Place at bottom-right, just above taskbar
            int x = _screenW - w - 10;
            int y = _screenH - _taskbarH - h;
            MoveWindow(_hwnd, x, y, w, h, true);
            Debug.Log("[WindowSnapper] Sitting on taskbar (bottom-right)");
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
                case 0: // Bottom-right
                    x = _screenW - w - 10;
                    y = _screenH - _taskbarH - h;
                    break;
                case 1: // Bottom-left
                    x = 10;
                    y = _screenH - _taskbarH - h;
                    break;
                case 2: // Top-right
                    x = _screenW - w - 10;
                    y = 10;
                    break;
                default: // Top-left
                    x = 10;
                    y = 10;
                    break;
            }

            MoveWindow(_hwnd, x, y, w, h, true);
            string[] names = { "bottom-right", "bottom-left", "top-right", "top-left" };
            Debug.Log($"[WindowSnapper] Snapped to {names[_snapIndex]}");
        }
#else
        private void Start()
        {
            Debug.Log("[WindowSnapper] Only active in Windows standalone builds.");
        }
#endif
    }
}
