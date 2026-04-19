using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using Annabeth.UI;

namespace Annabeth.Core
{
    /// <summary>
    /// Makes the Unity window transparent, frameless, and always-on-top (Mate-Engine style).
    /// Windows-only via Win32 P/Invoke. Only active in standalone builds.
    ///
    /// Click-through uses WM_NCHITTEST interception (like Kirurobo/UniWindowController):
    /// - Intercepts WM_NCHITTEST in a Win32 subclass proc
    /// - Returns HTTRANSPARENT for transparent areas (clicks pass to desktop)
    /// - Returns HTCLIENT for character areas (window captures clicks + gets focus)
    /// - Never uses WS_EX_TRANSPARENT (avoids focus/input deadlocks)
    ///
    /// Requirements:
    /// - Camera: Background Type = Solid Color, Color = (0,0,0,0)
    /// - URP: HDR off on the camera (alpha passthrough)
    /// - Player Settings: preserveFramebufferAlpha = true, useFlipModelSwapchain = false
    /// </summary>
    public class TransparentWindowController : MonoBehaviour
    {
#if UNITY_STANDALONE_WIN
        // ── Win32 Constants ─────────────────────────────────────────
        const int GWL_STYLE = -16;
        const int GWL_EXSTYLE = -20;
        const uint WS_POPUP = 0x80000000;
        const uint WS_VISIBLE = 0x10000000;
        const uint WS_EX_LAYERED = 0x00080000;
        const uint WS_EX_TOOLWINDOW = 0x00000080;
        const int HWND_TOPMOST = -1;
        const int HWND_NOTOPMOST = -2;
        const uint SWP_NOMOVE = 0x0002;
        const uint SWP_NOSIZE = 0x0001;
        const uint SWP_FRAMECHANGED = 0x0020;
        const uint SWP_SHOWWINDOW = 0x0040;
        const uint SWP_NOACTIVATE = 0x0010;
        const uint SWP_NOOWNERZORDER = 0x0200;

        // WM_NCHITTEST return values
        const int HTCLIENT = 1;
        const int HTTRANSPARENT = -1;
        const uint WM_NCHITTEST = 0x0084;

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

#if UNITY_STANDALONE_WIN
        private IntPtr _hwnd;
        private Camera _cam;
        private SkinnedMeshRenderer[] _renderers;
        private bool _dragging;
        private POINT _dragStartCursor;
        private RECT _dragStartRect;
        private bool _cursorOverCharacter;

        // Smooth drag (Feature #21)
        private float _dragVelX, _dragVelY;
        private float _smoothDragX, _smoothDragY;
        private const float DragSmoothTime = 0.04f;

        // Always-on-top re-assertion timer
        private float _topmostTimer;
        private const float TopmostReassertInterval = 0.5f;

        // Feature #13: File drop subclass
        private SubclassProc _subclassDelegate;
        private string _pendingDropFile;

        // Feature #20: Opacity hit test
        private RenderTexture _hitTestRT;
        private Texture2D _hitTestTex;
#endif

        private void Start()
        {
#if UNITY_STANDALONE_WIN
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

            // Single subclass proc handles WM_NCHITTEST (click-through) and WM_DROPFILES
            _subclassDelegate = WindowSubclassProc;
            SetWindowSubclass(_hwnd, _subclassDelegate, (UIntPtr)1, IntPtr.Zero);

            Debug.Log("[TransparentWindow] Window configured: transparent, topmost, WM_NCHITTEST click-through.");
#else
            Debug.Log("[TransparentWindow] Only active in Windows standalone builds.");
#endif
        }

#if UNITY_STANDALONE_WIN
        private void Update()
        {
            CacheRenderersIfNeeded();
            UpdateCursorOverCharacter();
            HandleDrag();
            ReassertTopmost();

            // Feature #13: Process file drops on main thread
            if (_pendingDropFile != null)
            {
                string file = _pendingDropFile;
                _pendingDropFile = null;
                OnFileDropped?.Invoke(file);
            }
        }

        /// <summary>
        /// Win32 subclass proc: intercepts WM_NCHITTEST for click-through and
        /// WM_DROPFILES for file drop (Feature #13).
        ///
        /// WM_NCHITTEST approach (like Kirurobo/UniWindowController):
        /// - Returns HTTRANSPARENT: click passes through to the window behind
        /// - Returns HTCLIENT: window captures the click and gets keyboard focus
        /// - No WS_EX_TRANSPARENT needed → no focus/input deadlocks
        /// </summary>
        private IntPtr WindowSubclassProc(IntPtr hWnd, uint uMsg, IntPtr wParam, IntPtr lParam, UIntPtr uIdSubclass, IntPtr dwRefData)
        {
            if (uMsg == WM_NCHITTEST)
            {
                // During drag, always capture input
                if (_dragging)
                    return (IntPtr)HTCLIENT;

                // When UI panels are open, always capture input
                if (RadialMenu.IsAnyPanelOpen)
                    return (IntPtr)HTCLIENT;

                // Check if cursor is over the character model
                if (CheckCursorOverCharacter())
                    return (IntPtr)HTCLIENT;

                // Transparent area — let the click pass to the desktop
                return (IntPtr)HTTRANSPARENT;
            }

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

            KillCompanionProcesses();
        }

        private void OnApplicationQuit()
        {
            KillCompanionProcesses();
        }

        /// <summary>
        /// Kill TTS and chat server processes on quit so their console
        /// windows don't linger after the companion shuts down.
        /// Uses cmd.exe + netstat + taskkill /F /T for fast, reliable
        /// process-tree killing. The /T flag kills child processes too,
        /// so Python servers inside PowerShell wrappers are caught.
        /// Guard flag prevents double-execution from OnApplicationQuit + OnDestroy.
        /// </summary>
        private static bool _companionProcessesKilled;
        private const string ManageBackendOnExitEnvVar = "ANNABETH_MANAGE_BACKEND_ON_EXIT";
        static void KillCompanionProcesses()
        {
            if (_companionProcessesKilled) return;
            _companionProcessesKilled = true;

            string manageBackend = Environment.GetEnvironmentVariable(ManageBackendOnExitEnvVar);
            if (!string.Equals(manageBackend, "1", StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(manageBackend, "true", StringComparison.OrdinalIgnoreCase))
            {
                return;
            }

            try
            {
                // Kill process trees listening on TTS (9880) and chat (8765) ports.
                // cmd.exe starts in ~50ms vs ~1s for PowerShell. taskkill /T kills children.
                foreach (int port in new[] { 9880, 8765 })
                {
                    var psi = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = "cmd.exe",
                        Arguments = "/c for /f \"tokens=5\" %a in ('netstat -aon ^| findstr :" + port + " ^| findstr LISTENING') do taskkill /F /T /PID %a",
                        CreateNoWindow = true,
                        UseShellExecute = false
                    };
                    var proc = System.Diagnostics.Process.Start(psi);
                    proc?.WaitForExit(5000);
                }
            }
            catch { }
        }

        /// <summary>
        /// Late-init: grab renderers from VRM once it's loaded.
        /// Re-caches if the previous model was destroyed (e.g. model switch via file drop).
        /// </summary>
        void CacheRenderersIfNeeded()
        {
            // Check existing cache: valid if non-empty AND first entry still alive
            if (_renderers != null && _renderers.Length > 0 && _renderers[0] != null) return;
            // The VRM is loaded dynamically, so we check periodically
            var vrm = FindFirstObjectByType<UniVRM10.Vrm10Instance>();
            if (vrm != null)
                _renderers = vrm.GetComponentsInChildren<SkinnedMeshRenderer>();
        }

        /// <summary>
        /// Update the cached cursor-over-character state each frame.
        /// Used by HandleDrag for drag initiation. The actual click-through
        /// decision is made in WM_NCHITTEST (WindowSubclassProc) which calls
        /// CheckCursorOverCharacter directly for real-time accuracy.
        /// </summary>
        void UpdateCursorOverCharacter()
        {
            _cursorOverCharacter = CheckCursorOverCharacter();
        }

        /// <summary>
        /// Check if the OS cursor is currently over the VRM character.
        /// Uses Win32 GetCursorPos for reliable position (works regardless of
        /// window focus or style flags — unlike Mouse.current.position which
        /// depends on WM_MOUSEMOVE delivery).
        /// Called from both Update (for drag) and WM_NCHITTEST (for click-through).
        /// </summary>
        bool CheckCursorOverCharacter()
        {
            if (_cam == null || _hwnd == IntPtr.Zero) return false;

            // Feature #20: Opacity-based hit test (reads rendered pixel alpha)
            if (useOpacityHitTest)
                return CheckCursorOverCharacterOpacity();

            // Default: bounds raycast
            if (_renderers == null || _renderers.Length == 0) return false;

            // Win32 GetCursorPos always returns the true screen cursor position.
            GetCursorPos(out POINT screenPos);
            GetWindowRect(_hwnd, out RECT winRect);
            int windowHeight = winRect.Bottom - winRect.Top;
            float clientX = screenPos.X - winRect.Left;
            float clientY = windowHeight - (screenPos.Y - winRect.Top); // flip Y: Win32 is top-down, Unity is bottom-up
            Ray ray = _cam.ScreenPointToRay(new Vector3(clientX, clientY, 0f));

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
        bool CheckCursorOverCharacterOpacity()
        {
            if (_cam.targetTexture == null) return false;

            GetCursorPos(out POINT screenPos);
            GetWindowRect(_hwnd, out RECT winRect);
            int px = Mathf.Clamp(screenPos.X - winRect.Left, 0, Screen.width - 1);
            int py = Mathf.Clamp(screenPos.Y - winRect.Top, 0, Screen.height - 1);
            // Flip Y for Unity (bottom-up)
            py = Screen.height - 1 - py;

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

        void ConfigureCamera()
        {
            if (_cam == null) return;

            // Transparent background — clear to fully transparent black
            _cam.clearFlags = CameraClearFlags.SolidColor;
            _cam.backgroundColor = new Color(0, 0, 0, 0);

            // Prevent MSAA and HDR — both force URP to use an intermediate
            // render texture, then blit to backbuffer. That blit clobbers alpha
            // (URP's internal blit shader writes alpha=1). By disabling them,
            // URP renders directly to the backbuffer, preserving the per-pixel
            // alpha set by the camera clear (0) and the character shader (1).
            _cam.allowMSAA = false;
            _cam.allowHDR = false;

            if (_cam.TryGetComponent<UniversalAdditionalCameraData>(out var urpData))
            {
                urpData.renderPostProcessing = false;
                urpData.allowHDROutput = false;
                urpData.requiresColorTexture = false;
                urpData.requiresDepthTexture = false;
                urpData.antialiasing = AntialiasingMode.None;
            }

            // NOTE: The FixAlpha GL callback is intentionally removed.
            // With PlayerSettings.preserveFramebufferAlpha=true (set in build script),
            // AllowPostProcessAlphaOutput=true, and no HDR/MSAA:
            // URP renders to intermediate RT, then blits to backbuffer preserving alpha.
            // Camera clear writes alpha=0 (transparent) for background.
            // MToon opaque mode writes alpha=1 for character pixels.
            // DWM compositor sees: character=opaque, background=transparent.
            Debug.Log("[TransparentWindow] Camera configured for direct-to-backbuffer alpha passthrough.");
        }

        void ApplyWindowStyle()
        {
            if (transparent)
            {
                // Remove title bar and borders — borderless popup
                SetWindowLong(_hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);

                // Layered window for per-pixel alpha compositing + hide from taskbar.
                // NOTE: WS_EX_TRANSPARENT is intentionally NOT used here.
                // Click-through is handled entirely by WM_NCHITTEST returning
                // HTTRANSPARENT for transparent areas. This avoids the focus/input
                // deadlock that WS_EX_TRANSPARENT causes (window can't receive
                // mouse events to know when to remove the transparent flag).
                uint exStyle = WS_EX_LAYERED;
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
                int hr = DwmExtendFrameIntoClientArea(_hwnd, ref margins);
                if (hr != 0)
                    Debug.LogError($"[TransparentWindow] DwmExtendFrameIntoClientArea FAILED: HRESULT 0x{hr:X8}");
                else
                    Debug.Log("[TransparentWindow] DWM glass extended successfully (margins=-1).");
            }

            SetTopmost(alwaysOnTop);
        }

        /// <summary>
        /// Left-click drag: grab the character to move the window.
        /// </summary>
        void HandleDrag()
        {
            // Only start drag if mouse is over the character
            var mouse = Mouse.current;
            if (mouse != null && mouse.leftButton.wasPressedThisFrame && _cursorOverCharacter)
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

            if (_dragging && mouse != null && mouse.leftButton.isPressed)
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

            if (mouse != null && mouse.leftButton.wasReleasedThisFrame && _dragging)
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
#if UNITY_STANDALONE_WIN
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
#if UNITY_STANDALONE_WIN
                return _dragging;
#else
                return false;
#endif
            }
        }

        /// <summary>
        /// Periodically re-assert HWND_TOPMOST so the window stays in front
        /// even when other apps steal focus or request topmost themselves.
        /// </summary>
        void ReassertTopmost()
        {
            if (!alwaysOnTop || _hwnd == IntPtr.Zero) return;
            _topmostTimer -= Time.deltaTime;
            if (_topmostTimer > 0f) return;
            _topmostTimer = TopmostReassertInterval;
            // Use NOACTIVATE + NOOWNERZORDER to avoid stealing focus from other apps.
            // SWP_SHOWWINDOW was causing focus battles that let fullscreen apps push us behind.
            SetWindowPos(_hwnd, new IntPtr(HWND_TOPMOST), 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOOWNERZORDER);
        }

        public void SetTopmost(bool topmost)
        {
#if UNITY_STANDALONE_WIN
            if (_hwnd == IntPtr.Zero) return;
            alwaysOnTop = topmost;
            IntPtr insert = topmost ? new IntPtr(HWND_TOPMOST) : new IntPtr(HWND_NOTOPMOST);
            SetWindowPos(_hwnd, insert, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOOWNERZORDER);
#endif
        }

        public void SetClickThrough(bool enabled)
        {
            // No-op: click-through is now handled by WM_NCHITTEST.
            // Kept for API compatibility.
        }
    }
}
