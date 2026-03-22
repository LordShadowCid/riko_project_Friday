using System;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace Annabeth.Core
{
    /// <summary>
    /// Makes the Unity window transparent, frameless, and always-on-top.
    /// Windows-only via Win32 P/Invoke. Only active in standalone builds.
    /// 
    /// Requirements:
    /// - Camera: Background Type = Solid Color, Color = (0,0,0,0)
    /// - URP: HDR off on the camera (alpha passthrough)
    /// - Player Settings: Use DXGI Flip Model = false (if available)
    /// </summary>
    public class TransparentWindowController : MonoBehaviour
    {
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
        // ── Win32 Constants ─────────────────────────────────────────
        const int GWL_STYLE = -16;
        const int GWL_EXSTYLE = -20;
        const uint WS_POPUP = 0x80000000;
        const uint WS_VISIBLE = 0x10000000;
        const uint WS_EX_LAYERED = 0x00080000;
        const uint WS_EX_TRANSPARENT = 0x00000020;
        const uint WS_EX_TOOLWINDOW = 0x00000080;
        const int HWND_TOPMOST = -1;
        const int HWND_NOTOPMOST = -2;
        const uint SWP_NOMOVE = 0x0002;
        const uint SWP_NOSIZE = 0x0001;
        const uint SWP_FRAMECHANGED = 0x0020;
        const uint SWP_SHOWWINDOW = 0x0040;

        [StructLayout(LayoutKind.Sequential)]
        struct MARGINS
        {
            public int cxLeftWidth;
            public int cxRightWidth;
            public int cyTopHeight;
            public int cyBottomHeight;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct POINT { public int X; public int Y; }

        [StructLayout(LayoutKind.Sequential)]
        struct RECT { public int Left; public int Top; public int Right; public int Bottom; }

        // ── P/Invoke ────────────────────────────────────────────────
        [DllImport("user32.dll")] static extern IntPtr GetActiveWindow();
        [DllImport("user32.dll")] static extern uint GetWindowLong(IntPtr hWnd, int nIndex);
        [DllImport("user32.dll")] static extern int SetWindowLong(IntPtr hWnd, int nIndex, uint dwNewLong);
        [DllImport("user32.dll")] static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);
        [DllImport("user32.dll")] static extern bool GetCursorPos(out POINT lpPoint);
        [DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
        [DllImport("user32.dll")] static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int nWidth, int nHeight, bool bRepaint);
        [DllImport("dwmapi.dll")] static extern int DwmExtendFrameIntoClientArea(IntPtr hwnd, ref MARGINS pMarInset);
#endif

        [Header("Window Settings")]
        [SerializeField] private bool transparent = true;
        [SerializeField] private bool alwaysOnTop = true;
        [SerializeField] private bool clickThroughTransparent = false;
        [SerializeField] private bool hideFromTaskbar = true;

        [Header("Drag Settings")]
        [SerializeField] private KeyCode dragKey = KeyCode.Mouse1; // right-click

#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
        private IntPtr _hwnd;
        private bool _dragging;
        private POINT _dragStartCursor;
        private RECT _dragStartRect;
#endif

        private void Start()
        {
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
            _hwnd = GetActiveWindow();
            if (_hwnd == IntPtr.Zero)
            {
                Debug.LogError("[TransparentWindow] Could not get window handle.");
                return;
            }

            ConfigureCamera();
            ApplyWindowStyle();
            Debug.Log("[TransparentWindow] Window configured: transparent=" + transparent +
                      " topmost=" + alwaysOnTop + " clickThrough=" + clickThroughTransparent);
#else
            Debug.Log("[TransparentWindow] Only active in Windows standalone builds.");
#endif
        }

        private void Update()
        {
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
            HandleDrag();
#endif
        }

#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
        void ConfigureCamera()
        {
            var cam = Camera.main;
            if (cam == null) return;

            // Transparent background
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0, 0, 0, 0);

            // Disable HDR on camera for proper alpha passthrough
            if (cam.TryGetComponent<UniversalAdditionalCameraData>(out var urpData))
            {
                urpData.renderPostProcessing = false;
            }
        }

        void ApplyWindowStyle()
        {
            if (transparent)
            {
                // Remove title bar and borders
                SetWindowLong(_hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);

                // Set extended style
                uint exStyle = WS_EX_LAYERED;
                if (clickThroughTransparent)
                    exStyle |= WS_EX_TRANSPARENT;
                if (hideFromTaskbar)
                    exStyle |= WS_EX_TOOLWINDOW;
                SetWindowLong(_hwnd, GWL_EXSTYLE, exStyle);

                // DWM: extend frame to cover entire client area → alpha = transparent
                var margins = new MARGINS
                {
                    cxLeftWidth = -1,
                    cxRightWidth = -1,
                    cyTopHeight = -1,
                    cyBottomHeight = -1
                };
                DwmExtendFrameIntoClientArea(_hwnd, ref margins);
            }

            SetTopmost(alwaysOnTop);
        }

        void HandleDrag()
        {
            if (UnityEngine.Input.GetKeyDown(dragKey))
            {
                GetCursorPos(out _dragStartCursor);
                GetWindowRect(_hwnd, out _dragStartRect);
                _dragging = true;
            }

            if (_dragging && UnityEngine.Input.GetKey(dragKey))
            {
                GetCursorPos(out POINT current);
                int dx = current.X - _dragStartCursor.X;
                int dy = current.Y - _dragStartCursor.Y;
                int w = _dragStartRect.Right - _dragStartRect.Left;
                int h = _dragStartRect.Bottom - _dragStartRect.Top;
                MoveWindow(_hwnd, _dragStartRect.Left + dx, _dragStartRect.Top + dy, w, h, true);
            }

            if (UnityEngine.Input.GetKeyUp(dragKey))
            {
                _dragging = false;
            }
        }
#endif

        // ── Public API ──────────────────────────────────────────────

        public void SetTopmost(bool topmost)
        {
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
            if (_hwnd == IntPtr.Zero) return;
            alwaysOnTop = topmost;
            IntPtr insert = topmost ? new IntPtr(HWND_TOPMOST) : new IntPtr(HWND_NOTOPMOST);
            SetWindowPos(_hwnd, insert, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_FRAMECHANGED | SWP_SHOWWINDOW);
#endif
        }

        public void SetClickThrough(bool enabled)
        {
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
            if (_hwnd == IntPtr.Zero) return;
            clickThroughTransparent = enabled;
            uint exStyle = GetWindowLong(_hwnd, GWL_EXSTYLE);
            if (enabled)
                exStyle |= WS_EX_TRANSPARENT;
            else
                exStyle &= ~WS_EX_TRANSPARENT;
            SetWindowLong(_hwnd, GWL_EXSTYLE, exStyle);
#endif
        }
    }
}
