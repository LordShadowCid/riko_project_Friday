using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Annabeth.Core
{
    /// <summary>
    /// Feature #1: Occluder Quads — creates dynamic depth-only quads that mask the avatar
    /// behind foreground windows when sitting on a window title bar.
    /// Uses Win32 EnumWindows to find overlapping windows and positions invisible quads
    /// that write to the depth buffer (ZWrite On, ColorMask 0) to occlude the avatar.
    /// </summary>
    public class OccluderQuadManager : MonoBehaviour
    {
#if UNITY_STANDALONE_WIN
        [StructLayout(LayoutKind.Sequential)]
        struct RECT { public int Left, Top, Right, Bottom; }

        [DllImport("user32.dll")] static extern IntPtr GetActiveWindow();
        [DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
        [DllImport("user32.dll")] static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
        [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr hWnd);
        [DllImport("user32.dll")] static extern bool IsIconic(IntPtr hWnd);
        [DllImport("user32.dll")] static extern uint GetWindowLong(IntPtr hWnd, int nIndex);
        [DllImport("user32.dll")] static extern IntPtr GetWindow(IntPtr hWnd, uint uCmd);

        delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
        const int GWL_EXSTYLE = -20;
        const uint WS_EX_TOOLWINDOW = 0x00000080;
        const uint WS_EX_TRANSPARENT = 0x00000020;
        const uint WS_EX_LAYERED = 0x00080000;
        const uint GW_HWNDPREV = 3;
#endif

        [Header("Settings")]
        [SerializeField] private float updateInterval = 0.1f; // 10 Hz
        [SerializeField] private int maxOccluders = 8;

        private Camera _cam;
        private Material _occluderMaterial;
        private readonly List<GameObject> _pool = new();
        private int _activeCount;
        private float _updateTimer;
        private bool _enabled;

#if UNITY_STANDALONE_WIN
        private IntPtr _hwnd;
        private IntPtr _sittingOnWindow;

        private void Start()
        {
            _hwnd = GetActiveWindow();
            _cam = Camera.main;
            CreateOccluderMaterial();
            // Pre-allocate pool
            for (int i = 0; i < maxOccluders; i++)
                _pool.Add(CreateQuad(i));
        }

        private void CreateOccluderMaterial()
        {
            // Depth-only material: writes to Z-buffer but renders nothing visible
            _occluderMaterial = new Material(Shader.Find("Unlit/Color"));
            _occluderMaterial.SetColor("_Color", Color.clear);
            _occluderMaterial.SetInt("_ZWrite", 1);
            _occluderMaterial.SetInt("_ColorMask", 0);
            _occluderMaterial.renderQueue = 1900; // Before avatar (2000+)
        }

        private GameObject CreateQuad(int index)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Quad);
            go.name = $"OccluderQuad_{index}";
            go.transform.SetParent(transform);
            go.GetComponent<MeshRenderer>().material = _occluderMaterial;
            var col = go.GetComponent<Collider>();
            if (col) Destroy(col);
            go.SetActive(false);
            return go;
        }

        /// <summary>Enable/disable the occluder system. Call when sitting state changes.</summary>
        public void SetEnabled(bool enabled, IntPtr sittingOnWindow = default)
        {
            _enabled = enabled;
            _sittingOnWindow = sittingOnWindow;
            if (!enabled) HideAll();
        }

        private void Update()
        {
            if (!_enabled || _hwnd == IntPtr.Zero || _cam == null) return;

            _updateTimer -= Time.deltaTime;
            if (_updateTimer > 0f) return;
            _updateTimer = updateInterval;

            UpdateOccluders();
        }

        private void UpdateOccluders()
        {
            if (!GetWindowRect(_hwnd, out RECT myRect)) return;

            _activeCount = 0;
            var foregroundWindows = new List<RECT>();

            // Enumerate windows — collect those that are:
            // 1. Visible and not minimized
            // 2. In front of our sitting window (higher Z-order)
            // 3. Overlapping our avatar window rect
            EnumWindows((hWnd, _) =>
            {
                if (hWnd == _hwnd) return true; // Skip self
                if (!IsWindowVisible(hWnd) || IsIconic(hWnd)) return true;

                uint exStyle = GetWindowLong(hWnd, GWL_EXSTYLE);
                if ((exStyle & WS_EX_TOOLWINDOW) != 0) return true;
                if ((exStyle & WS_EX_TRANSPARENT) != 0) return true;

                if (!GetWindowRect(hWnd, out RECT wr)) return true;

                // Check overlap with our window
                if (wr.Left >= myRect.Right || wr.Right <= myRect.Left ||
                    wr.Top >= myRect.Bottom || wr.Bottom <= myRect.Top)
                    return true; // No overlap

                // Only include windows that are in front of seated window
                // (simplified: any visible overlapping window counts)
                foregroundWindows.Add(wr);
                return foregroundWindows.Count < maxOccluders;
            }, IntPtr.Zero);

            // Position quads for overlapping regions
            for (int i = 0; i < foregroundWindows.Count && i < _pool.Count; i++)
            {
                var wr = foregroundWindows[i];

                // Calculate overlap rectangle
                int overlapL = Mathf.Max(wr.Left, myRect.Left);
                int overlapT = Mathf.Max(wr.Top, myRect.Top);
                int overlapR = Mathf.Min(wr.Right, myRect.Right);
                int overlapB = Mathf.Min(wr.Bottom, myRect.Bottom);

                if (overlapR <= overlapL || overlapB <= overlapT) continue;

                // Convert screen pixels to viewport coordinates relative to our window
                float vpLeft = (float)(overlapL - myRect.Left) / (myRect.Right - myRect.Left);
                float vpRight = (float)(overlapR - myRect.Left) / (myRect.Right - myRect.Left);
                float vpBottom = 1f - (float)(overlapB - myRect.Top) / (myRect.Bottom - myRect.Top);
                float vpTop = 1f - (float)(overlapT - myRect.Top) / (myRect.Bottom - myRect.Top);

                // Convert viewport to world position (at a fixed depth in front of camera)
                float depth = _cam.nearClipPlane + 0.5f;
                Vector3 center = _cam.ViewportToWorldPoint(new Vector3(
                    (vpLeft + vpRight) * 0.5f,
                    (vpBottom + vpTop) * 0.5f,
                    depth));

                Vector3 cornerBL = _cam.ViewportToWorldPoint(new Vector3(vpLeft, vpBottom, depth));
                Vector3 cornerTR = _cam.ViewportToWorldPoint(new Vector3(vpRight, vpTop, depth));
                float scaleX = Mathf.Abs(cornerTR.x - cornerBL.x);
                float scaleY = Mathf.Abs(cornerTR.y - cornerBL.y);

                var quad = _pool[i];
                quad.transform.position = center;
                quad.transform.rotation = _cam.transform.rotation;
                quad.transform.localScale = new Vector3(scaleX, scaleY, 1f);
                quad.SetActive(true);
                _activeCount++;
            }

            // Deactive unused quads
            for (int i = _activeCount; i < _pool.Count; i++)
                _pool[i].SetActive(false);
        }

        private void HideAll()
        {
            foreach (var q in _pool)
                q.SetActive(false);
            _activeCount = 0;
        }

        private void OnDestroy()
        {
            if (_occluderMaterial != null)
                Destroy(_occluderMaterial);
        }
#else
        public void SetEnabled(bool enabled, IntPtr sittingOnWindow = default) { }
#endif
    }
}
