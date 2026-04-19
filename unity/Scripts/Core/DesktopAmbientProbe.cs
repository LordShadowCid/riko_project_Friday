using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Annabeth.Core
{
    /// <summary>
    /// Feature #9: Desktop Ambient Lighting Probe.
    /// Samples the desktop area behind the avatar window and uses the average color
    /// to tint the avatar's ambient lighting, creating environmental integration.
    /// Uses Win32 BitBlt to capture a small region behind the window.
    /// </summary>
    public class DesktopAmbientProbe : MonoBehaviour
    {
#if UNITY_STANDALONE_WIN
        [StructLayout(LayoutKind.Sequential)]
        struct RECT { public int Left, Top, Right, Bottom; }

        [DllImport("user32.dll")] static extern IntPtr GetActiveWindow();
        [DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
        [DllImport("user32.dll")] static extern IntPtr GetDC(IntPtr hWnd);
        [DllImport("user32.dll")] static extern int ReleaseDC(IntPtr hWnd, IntPtr hDC);
        [DllImport("gdi32.dll")] static extern IntPtr CreateCompatibleDC(IntPtr hdc);
        [DllImport("gdi32.dll")] static extern IntPtr CreateCompatibleBitmap(IntPtr hdc, int nWidth, int nHeight);
        [DllImport("gdi32.dll")] static extern IntPtr SelectObject(IntPtr hdc, IntPtr hgdiobj);
        [DllImport("gdi32.dll")] static extern bool BitBlt(IntPtr hdcDest, int xDest, int yDest, int w, int h, IntPtr hdcSrc, int xSrc, int ySrc, uint dwRop);
        [DllImport("gdi32.dll")] static extern uint GetPixel(IntPtr hdc, int x, int y);
        [DllImport("gdi32.dll")] static extern bool DeleteDC(IntPtr hdc);
        [DllImport("gdi32.dll")] static extern bool DeleteObject(IntPtr hObject);

        const uint SRCCOPY = 0x00CC0020;
#endif

        [Header("Settings")]
        [SerializeField] private float probeInterval = 0.5f;
        [SerializeField] private float smoothSpeed = 2f;
        [SerializeField] private int sampleSize = 8; // 8x8 grid sample
        [SerializeField] private float intensity = 0.5f;

        [Header("Light Reference")]
        [SerializeField] private Light avatarLight;

        private Color _currentColor = Color.white;
        private Color _targetColor = Color.white;
        private float _probeTimer;
        private bool _enabled;

        public void SetEnabled(bool enabled)
        {
            _enabled = enabled;
            if (!enabled && avatarLight != null)
                avatarLight.color = Color.white;
        }

        public void SetIntensity(float value)
        {
            intensity = Mathf.Clamp01(value);
        }

#if UNITY_STANDALONE_WIN
        private IntPtr _hwnd;

        private void Start()
        {
            _hwnd = GetActiveWindow();
        }

        private void Update()
        {
            if (!_enabled || _hwnd == IntPtr.Zero) return;

            _probeTimer -= Time.deltaTime;
            if (_probeTimer <= 0f)
            {
                _probeTimer = probeInterval;
                SampleDesktop();
            }

            // Smooth transition
            _currentColor = Color.Lerp(_currentColor, _targetColor, Time.deltaTime * smoothSpeed);

            // Apply to light — blend between white and sampled color based on intensity
            if (avatarLight != null)
                avatarLight.color = Color.Lerp(Color.white, _currentColor, intensity);
        }

        private void SampleDesktop()
        {
            if (!GetWindowRect(_hwnd, out RECT r)) return;

            int winW = r.Right - r.Left;
            int winH = r.Bottom - r.Top;
            if (winW <= 0 || winH <= 0) return;

            // Sample from desktop DC at the window's position
            IntPtr screenDC = GetDC(IntPtr.Zero);
            if (screenDC == IntPtr.Zero) return;

            try
            {
                float totalR = 0, totalG = 0, totalB = 0;
                int count = 0;

                for (int sx = 0; sx < sampleSize; sx++)
                {
                    for (int sy = 0; sy < sampleSize; sy++)
                    {
                        int px = r.Left + (winW * sx / sampleSize);
                        int py = r.Top + (winH * sy / sampleSize);
                        uint pixel = GetPixel(screenDC, px, py);
                        if (pixel == 0xFFFFFFFF) continue; // CLR_INVALID

                        totalR += (pixel & 0xFF) / 255f;
                        totalG += ((pixel >> 8) & 0xFF) / 255f;
                        totalB += ((pixel >> 16) & 0xFF) / 255f;
                        count++;
                    }
                }

                if (count > 0)
                    _targetColor = new Color(totalR / count, totalG / count, totalB / count);
            }
            finally
            {
                ReleaseDC(IntPtr.Zero, screenDC);
            }
        }
#else
        private void Start()
        {
            Debug.Log("[DesktopAmbientProbe] Only active in Windows standalone builds.");
        }
#endif
    }
}
