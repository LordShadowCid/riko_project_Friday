using System;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering.Universal;

namespace Annabeth.Core
{
    /// <summary>
    /// Makes the Unity window transparent, frameless, and always-on-top (Mate-Engine style).
    /// Windows-only via Win32 P/Invoke. Only active in standalone builds.
    ///
    /// Key behaviors:
    /// - Transparent background: only the VRM character is visible on screen.
    /// - Click-through on empty areas: clicks on transparent pixels pass through to the desktop.
    /// - Click capture on character: uses mesh-bounds raycast to detect cursor over VRM.
    /// - Left-click drag: grab the character anywhere on her body to move the window.
    /// - Always-on-top: stays above other windows.
    ///
    /// Requirements:
    /// - Camera: Background Type = Solid Color, Color = (0,0,0,0)
    /// - URP: HDR off on the camera (alpha passthrough)
    /// - Player Settings: Use DXGI Flip Model = false
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
        [SerializeField] private bool hideFromTaskbar = true;

#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
        private IntPtr _hwnd;
        private Camera _cam;
        private SkinnedMeshRenderer[] _renderers;
        private bool _dragging;
        private POINT _dragStartCursor;
        private RECT _dragStartRect;
        private bool _cursorOverCharacter;
        private bool _clickThroughActive;
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

            _cam = Camera.main;
            ConfigureCamera();
            ApplyWindowStyle();
            Debug.Log("[TransparentWindow] Window configured: transparent, topmost, click-through on empty areas.");
#else
            Debug.Log("[TransparentWindow] Only active in Windows standalone builds.");
#endif
        }

#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
        private void Update()
        {
            CacheRenderersIfNeeded();
            UpdateClickThrough();
            HandleDrag();
        }

        /// <summary>
        /// Late-init: grab renderers from VRM once it's loaded.
        /// </summary>
        void CacheRenderersIfNeeded()
        {
            if (_renderers != null && _renderers.Length > 0) return;
            // The VRM is loaded dynamically, so we check periodically
            var vrm = FindFirstObjectByType<UniVRM10.Vrm10Instance>();
            if (vrm != null)
                _renderers = vrm.GetComponentsInChildren<SkinnedMeshRenderer>();
        }

        /// <summary>
        /// Raycast mouse against VRM mesh bounds each frame.
        /// Over character → capture clicks (can drag). Over empty → pass through to desktop.
        /// </summary>
        void UpdateClickThrough()
        {
            if (_hwnd == IntPtr.Zero || _cam == null) return;

            // While dragging, always capture clicks
            if (_dragging)
            {
                if (_clickThroughActive)
                    SetClickThroughInternal(false);
                return;
            }

            bool overCharacter = IsMouseOverCharacter();
            _cursorOverCharacter = overCharacter;

            if (overCharacter && _clickThroughActive)
            {
                // Mouse is over the character — capture clicks for drag/touch
                SetClickThroughInternal(false);
            }
            else if (!overCharacter && !_clickThroughActive)
            {
                // Mouse is over transparent area — let clicks pass through to desktop
                SetClickThroughInternal(true);
            }
        }

        bool IsMouseOverCharacter()
        {
            if (_renderers == null || _renderers.Length == 0 || _cam == null) return false;

            Vector3 mousePos = UnityEngine.Input.mousePosition;
            Ray ray = _cam.ScreenPointToRay(mousePos);

            foreach (var r in _renderers)
            {
                if (r != null && r.bounds.IntersectRay(ray))
                    return true;
            }
            return false;
        }

        void SetClickThroughInternal(bool passThrough)
        {
            _clickThroughActive = passThrough;
            uint exStyle = GetWindowLong(_hwnd, GWL_EXSTYLE);
            if (passThrough)
                exStyle |= WS_EX_TRANSPARENT;
            else
                exStyle &= ~WS_EX_TRANSPARENT;
            SetWindowLong(_hwnd, GWL_EXSTYLE, exStyle);
        }

        void ConfigureCamera()
        {
            if (_cam == null) return;

            // Transparent background — clear to fully transparent black
            _cam.clearFlags = CameraClearFlags.SolidColor;
            _cam.backgroundColor = new Color(0, 0, 0, 0);

            // Disable HDR and post-processing for proper alpha passthrough in URP
            if (_cam.TryGetComponent<UniversalAdditionalCameraData>(out var urpData))
            {
                urpData.renderPostProcessing = false;
            }
        }

        void ApplyWindowStyle()
        {
            if (transparent)
            {
                // Remove title bar and borders — borderless popup
                SetWindowLong(_hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);

                // Layered window for per-pixel alpha + hide from taskbar
                uint exStyle = WS_EX_LAYERED;
                if (hideFromTaskbar)
                    exStyle |= WS_EX_TOOLWINDOW;
                // Start with click-through ON (mouse not over character initially)
                exStyle |= WS_EX_TRANSPARENT;
                _clickThroughActive = true;
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

        /// <summary>
        /// Left-click drag: grab the character to move the window.
        /// </summary>
        void HandleDrag()
        {
            // Only start drag if mouse is over the character
            if (UnityEngine.Input.GetMouseButtonDown(0) && _cursorOverCharacter)
            {
                GetCursorPos(out _dragStartCursor);
                GetWindowRect(_hwnd, out _dragStartRect);
                _dragging = true;
            }

            if (_dragging && UnityEngine.Input.GetMouseButton(0))
            {
                GetCursorPos(out POINT current);
                int dx = current.X - _dragStartCursor.X;
                int dy = current.Y - _dragStartCursor.Y;
                int w = _dragStartRect.Right - _dragStartRect.Left;
                int h = _dragStartRect.Bottom - _dragStartRect.Top;
                MoveWindow(_hwnd, _dragStartRect.Left + dx, _dragStartRect.Top + dy, w, h, true);
            }

            if (UnityEngine.Input.GetMouseButtonUp(0))
            {
                _dragging = false;
            }
        }
#endif

        // ── Public API ──────────────────────────────────────────────

        /// <summary>Whether the cursor is currently hovering over the character.</summary>
        public bool IsCursorOverCharacter
        {
            get
            {
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
                return _cursorOverCharacter;
#else
                return false;
#endif
            }
        }

        /// <summary>Whether the window is currently being dragged.</summary>
        public bool IsDragging
        {
            get
            {
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
                return _dragging;
#else
                return false;
#endif
            }
        }

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
            SetClickThroughInternal(enabled);
#endif
        }
    }
}
