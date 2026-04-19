using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

namespace Annabeth.UI
{
    /// <summary>
    /// Singleton theme manager — applies HSV hue shift + saturation to all UI colors.
    /// Based on Mate Engine's ThemeManager: uiHueShift (0-360) + uiSaturation (0-2).
    /// Tracks all themed UI elements and recolors them when theme changes.
    /// </summary>
    public class ThemeManager : MonoBehaviour
    {
        public static ThemeManager Instance { get; private set; }

        // ── Base palette (UIFactory defaults before any theme shift) ──
        private static readonly Color BasePanelBg       = new Color(0.10f, 0.10f, 0.12f, 0.92f);
        private static readonly Color BaseButtonNormal   = new Color(0.22f, 0.22f, 0.26f, 1f);
        private static readonly Color BaseButtonHover    = new Color(0.30f, 0.30f, 0.36f, 1f);
        private static readonly Color BaseButtonPress    = new Color(0.18f, 0.18f, 0.22f, 1f);
        private static readonly Color BaseSliderBg       = new Color(0.20f, 0.20f, 0.24f, 1f);
        private static readonly Color BaseSliderFill     = new Color(0.45f, 0.65f, 0.95f, 1f);
        private static readonly Color BaseSliderHandle   = new Color(0.80f, 0.80f, 0.85f, 1f);
        private static readonly Color BaseToggleCheckmark = new Color(0.45f, 0.85f, 0.55f, 1f);
        private static readonly Color BaseTextColor      = new Color(0.90f, 0.90f, 0.92f, 1f);
        private static readonly Color BaseHeaderColor    = new Color(0.55f, 0.75f, 1f, 1f);
        private static readonly Color BaseSeparator      = new Color(0.30f, 0.30f, 0.35f, 0.6f);

        // Current themed colors (computed from base + hue shift + saturation)
        public Color PanelBg { get; private set; }
        public Color ButtonNormal { get; private set; }
        public Color ButtonHover { get; private set; }
        public Color ButtonPress { get; private set; }
        public Color SliderBg { get; private set; }
        public Color SliderFill { get; private set; }
        public Color SliderHandle { get; private set; }
        public Color ToggleCheckmark { get; private set; }
        public Color TextColor { get; private set; }
        public Color HeaderColor { get; private set; }
        public Color Separator { get; private set; }

        // Tracked UI elements for live recoloring
        private readonly List<TrackedElement> _tracked = new List<TrackedElement>();

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;

            // Initialize with default (no shift)
            RecalculateColors(0f, 1f);
        }

        /// <summary>
        /// Apply a new hue shift (0-360) and saturation (0-2) to all themed colors
        /// and update all tracked UI elements.
        /// </summary>
        public void ApplyTheme(float hueShift, float saturation)
        {
            CleanupDestroyed();
            RecalculateColors(hueShift, saturation);
            RecolorAllTracked();
            Debug.Log($"[ThemeManager] Applied theme: hue={hueShift:F0}°, sat={saturation:F2}");
        }

        /// <summary>Register a UI element for live theme updates.</summary>
        public void Track(Graphic graphic, ThemeRole role)
        {
            if (graphic == null) return;
            _tracked.Add(new TrackedElement { graphic = graphic, role = role });
            graphic.color = GetColor(role);
        }

        /// <summary>Register a Button for live theme updates (normal/hover/press colors).</summary>
        public void TrackButton(Button button)
        {
            if (button == null) return;
            var img = button.targetGraphic as Image;
            if (img != null)
                Track(img, ThemeRole.ButtonNormal);

            var cb = button.colors;
            cb.normalColor = ButtonNormal;
            cb.highlightedColor = ButtonHover;
            cb.pressedColor = ButtonPress;
            cb.selectedColor = ButtonNormal;
            button.colors = cb;

            _tracked.Add(new TrackedElement { button = button, role = ThemeRole.ButtonNormal });
        }

        /// <summary>Remove destroyed references from the tracking list.</summary>
        public void CleanupDestroyed()
        {
            _tracked.RemoveAll(t => t.graphic == null && t.button == null);
        }

        // ── Color computation ───────────────────────────────────

        private void RecalculateColors(float hueShift, float saturation)
        {
            PanelBg        = ShiftColor(BasePanelBg, hueShift, saturation);
            ButtonNormal   = ShiftColor(BaseButtonNormal, hueShift, saturation);
            ButtonHover    = ShiftColor(BaseButtonHover, hueShift, saturation);
            ButtonPress    = ShiftColor(BaseButtonPress, hueShift, saturation);
            SliderBg       = ShiftColor(BaseSliderBg, hueShift, saturation);
            SliderFill     = ShiftColor(BaseSliderFill, hueShift, saturation);
            SliderHandle   = ShiftColor(BaseSliderHandle, hueShift, saturation);
            ToggleCheckmark = ShiftColor(BaseToggleCheckmark, hueShift, saturation);
            TextColor      = ShiftColor(BaseTextColor, hueShift, saturation);
            HeaderColor    = ShiftColor(BaseHeaderColor, hueShift, saturation);
            Separator      = ShiftColor(BaseSeparator, hueShift, saturation);
        }

        private static Color ShiftColor(Color input, float hueShift, float saturation)
        {
            float h, s, v;
            Color.RGBToHSV(input, out h, out s, out v);

            // Apply hue shift (wrap 0-1)
            h = Mathf.Repeat(h + hueShift / 360f, 1f);

            // Apply saturation multiplier (clamp 0-1)
            s = Mathf.Clamp01(s * saturation);

            var c = Color.HSVToRGB(h, s, v);
            c.a = input.a; // Preserve original alpha
            return c;
        }

        public Color GetColor(ThemeRole role)
        {
            switch (role)
            {
                case ThemeRole.PanelBg: return PanelBg;
                case ThemeRole.ButtonNormal: return ButtonNormal;
                case ThemeRole.ButtonHover: return ButtonHover;
                case ThemeRole.ButtonPress: return ButtonPress;
                case ThemeRole.SliderBg: return SliderBg;
                case ThemeRole.SliderFill: return SliderFill;
                case ThemeRole.SliderHandle: return SliderHandle;
                case ThemeRole.ToggleCheckmark: return ToggleCheckmark;
                case ThemeRole.TextColor: return TextColor;
                case ThemeRole.HeaderColor: return HeaderColor;
                case ThemeRole.Separator: return Separator;
                default: return Color.white;
            }
        }

        private void RecolorAllTracked()
        {
            for (int i = _tracked.Count - 1; i >= 0; i--)
            {
                var t = _tracked[i];
                if (t.graphic == null && t.button == null)
                {
                    _tracked.RemoveAt(i);
                    continue;
                }

                if (t.graphic != null)
                    t.graphic.color = GetColor(t.role);

                if (t.button != null)
                {
                    var cb = t.button.colors;
                    cb.normalColor = ButtonNormal;
                    cb.highlightedColor = ButtonHover;
                    cb.pressedColor = ButtonPress;
                    cb.selectedColor = ButtonNormal;
                    t.button.colors = cb;
                }
            }
        }

        private struct TrackedElement
        {
            public Graphic graphic;
            public Button button;
            public ThemeRole role;
        }
    }

    public enum ThemeRole
    {
        PanelBg,
        ButtonNormal,
        ButtonHover,
        ButtonPress,
        SliderBg,
        SliderFill,
        SliderHandle,
        ToggleCheckmark,
        TextColor,
        HeaderColor,
        Separator
    }
}
