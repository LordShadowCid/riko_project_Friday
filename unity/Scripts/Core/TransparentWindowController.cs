using System;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering.Universal;
using Annabeth.UI;

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

        // Feature #13: File drop
        [DllImport("shell32.dll")] static extern void DragAcceptFiles(IntPtr hWnd, bool fAccept);
        [DllImport("shell32.dll")] static extern uint DragQueryFileW(IntPtr hDrop, uint iFile, [Out] StringBuilder lpszFile, uint cch);
        [DllImport("shell32.dll")] static extern void DragFinish(IntPtr hDrop);
        [DllImport("comctl32.dll")] static extern bool SetWindowSubclass(IntPtr hWnd, SubclassProc pfnSubclass, UIntPtr uIdSubclass, IntPtr dwRefData);
        [DllImport("comctl32.dll")] static extern bool RemoveWindowSubclass(IntPtr hWnd, SubclassProc pfnSubclass, UIntPtr uIdSubclass);
        [DllImport("comctl32.dll")] static extern IntPtr DefSubclassProc(IntPtr hWnd, uint uMsg, IntPtr wParam, IntPtr lParam);

        delegate IntPtr SubclassProc(IntPtr hWnd, uint uMsg, IntPtr wParam, IntPtr lParam, UIntPtr uIdSubclass, IntPtr dwRefData);

        const uint WM_DROPFILES = 0x0233;
#endif

        /// <summary>Fired when the user starts dragging the window.</summary>
        public event Action OnDragStart;
        /// <summary>Fired when the user stops dragging the window.</summary>
        public event Action OnDragEnd;
        /// <summary>Feature #13: Fired when a file is dropped onto the window.</summary>
        public event Action<string> OnFileDropped;

        [Header("Window Settings")]
        [SerializeField] private bool transparent = true;
        [SerializeField] private bool alwaysOnTop = true;
        [SerializeField] private bool hideFromTaskbar = true;
        [SerializeField] private bool useOpacityHitTest = false;
        [SerializeField] private float clickThroughAlphaThreshold = 0.1f;

#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
        private IntPtr _hwnd;
        private Camera _cam;
        private SkinnedMeshRenderer[] _renderers;
        private bool _dragging;
        private POINT _dragStartCursor;
        private RECT _dragStartRect;
        private bool _cursorOverCharacter;
        private bool _clickThroughActive;

        // Smooth drag (Feature #21)
        private float _dragVelX, _dragVelY;
        private float _smoothDragX, _smoothDragY;
        private const float DragSmoothTime = 0.04f;

        // Feature #13: File drop subclass
        private SubclassProc _subclassDelegate;
        private string _pendingDropFile;

        // Feature #20: Opacity hit test
        private RenderTexture _hitTestRT;
        private Texture2D _hitTestTex;
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

            // Feature #13: Enable file drop
            DragAcceptFiles(_hwnd, true);
            _subclassDelegate = FileDropSubclassProc;
            SetWindowSubclass(_hwnd, _subclassDelegate, (UIntPtr)1, IntPtr.Zero);

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

            // Feature #13: Process file drops on main thread
            if (_pendingDropFile != null)
            {
                string file = _pendingDropFile;
                _pendingDropFile = null;
                OnFileDropped?.Invoke(file);
            }
        }

        /// <summary>Feature #13: WndProc subclass to intercept WM_DROPFILES messages.</summary>
        private IntPtr FileDropSubclassProc(IntPtr hWnd, uint uMsg, IntPtr wParam, IntPtr lParam, UIntPtr uIdSubclass, IntPtr dwRefData)
        {
            if (uMsg == WM_DROPFILES)
            {
                var sb = new StringBuilder(260);
                uint count = DragQueryFileW(wParam, 0xFFFFFFFF, null, 0);
                if (count > 0)
                {
                    DragQueryFileW(wParam, 0, sb, 260);
                    string path = sb.ToString();
                    if (path.EndsWith(".vrm", StringComparison.OrdinalIgnoreCase))
                        _pendingDropFile = path;
                }
                DragFinish(wParam);
                return IntPtr.Zero;
            }
            return DefSubclassProc(hWnd, uMsg, wParam, lParam);
        }

        private void OnDestroy()
        {
            if (_hwnd != IntPtr.Zero && _subclassDelegate != null)
                RemoveWindowSubclass(_hwnd, _subclassDelegate, (UIntPtr)1);
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

            // While any UI panel is open, always capture clicks so user can interact with UI
            if (UI.RadialMenu.IsAnyPanelOpen)
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
            if (_cam == null) return false;

            // Feature #20: Opacity-based hit test (reads rendered pixel alpha)
            if (useOpacityHitTest)
                return IsMouseOverCharacterOpacity();

            // Default: bounds raycast
            if (_renderers == null || _renderers.Length == 0) return false;

            Vector3 mousePos = UnityEngine.Input.mousePosition;
            Ray ray = _cam.ScreenPointToRay(mousePos);

            foreach (var r in _renderers)
            {
                if (r != null && r.bounds.IntersectRay(ray))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Feature #20: Read rendered pixel alpha at cursor position.
        /// Only captures clicks where avatar alpha exceeds threshold.
        /// </summary>
        bool IsMouseOverCharacterOpacity()
        {
            if (_cam.targetTexture == null) return false;

            Vector3 mousePos = UnityEngine.Input.mousePosition;
            int px = Mathf.Clamp((int)mousePos.x, 0, Screen.width - 1);
            int py = Mathf.Clamp((int)mousePos.y, 0, Screen.height - 1);

            // Create small textures for single-pixel read
            if (_hitTestRT == null)
                _hitTestRT = new RenderTexture(1, 1, 0, RenderTextureFormat.ARGB32);
            if (_hitTestTex == null)
                _hitTestTex = new Texture2D(1, 1, TextureFormat.ARGB32, false);

            // Blit the single pixel from the camera target
            var src = _cam.targetTexture;
            Graphics.CopyTexture(src, 0, 0, px, py, 1, 1, _hitTestRT, 0, 0, 0, 0);

            RenderTexture.active = _hitTestRT;
            _hitTestTex.ReadPixels(new Rect(0, 0, 1, 1), 0, 0, false);
            _hitTestTex.Apply();
            RenderTexture.active = null;

            float alpha = _hitTestTex.GetPixel(0, 0).a;
            return alpha > clickThroughAlphaThreshold;
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
                _smoothDragX = _dragStartRect.Left;
                _smoothDragY = _dragStartRect.Top;
                _dragVelX = 0f;
                _dragVelY = 0f;
                OnDragStart?.Invoke();
            }

            if (_dragging && UnityEngine.Input.GetMouseButton(0))
            {
                GetCursorPos(out POINT current);
                int dx = current.X - _dragStartCursor.X;
                int dy = current.Y - _dragStartCursor.Y;
                int w = _dragStartRect.Right - _dragStartRect.Left;
                int h = _dragStartRect.Bottom - _dragStartRect.Top;
                float targetX = _dragStartRect.Left + dx;
                float targetY = _dragStartRect.Top + dy;
                // Feature #21: SmoothDamp for fluid drag movement
                _smoothDragX = Mathf.SmoothDamp(_smoothDragX, targetX, ref _dragVelX, DragSmoothTime);
                _smoothDragY = Mathf.SmoothDamp(_smoothDragY, targetY, ref _dragVelY, DragSmoothTime);
                MoveWindow(_hwnd, (int)_smoothDragX, (int)_smoothDragY, w, h, true);
            }

            if (UnityEngine.Input.GetMouseButtonUp(0) && _dragging)
            {
                _dragging = false;
                OnDragEnd?.Invoke();
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
