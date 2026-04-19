using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem.UI;

namespace Annabeth.UI
{
    /// <summary>
    /// Runtime UI factory — creates Canvas, buttons, sliders, toggles, labels, panels, etc.
    /// All Phase 2+ UI scripts use this so nothing needs to be wired in the Editor.
    /// Colors are served by ThemeManager when available, falling back to defaults.
    /// </summary>
    public static class UIFactory
    {
        // ── Default Colors (fallback when ThemeManager not yet initialized) ──
        private static readonly Color DefaultPanelBg       = new Color(0.10f, 0.10f, 0.12f, 0.92f);
        private static readonly Color DefaultButtonNormal   = new Color(0.22f, 0.22f, 0.26f, 1f);
        private static readonly Color DefaultButtonHover    = new Color(0.30f, 0.30f, 0.36f, 1f);
        private static readonly Color DefaultButtonPress    = new Color(0.18f, 0.18f, 0.22f, 1f);
        private static readonly Color DefaultSliderBg       = new Color(0.20f, 0.20f, 0.24f, 1f);
        private static readonly Color DefaultSliderFill     = new Color(0.45f, 0.65f, 0.95f, 1f);
        private static readonly Color DefaultSliderHandle   = new Color(0.80f, 0.80f, 0.85f, 1f);
        private static readonly Color DefaultToggleCheckmark = new Color(0.45f, 0.85f, 0.55f, 1f);
        private static readonly Color DefaultTextColor      = new Color(0.90f, 0.90f, 0.92f, 1f);
        private static readonly Color DefaultHeaderColor    = new Color(0.55f, 0.75f, 1f, 1f);
        private static readonly Color DefaultSeparator      = new Color(0.30f, 0.30f, 0.35f, 0.6f);

        // ── Themed color accessors ──────────────────────────────
        private static ThemeManager TM => ThemeManager.Instance;
        public static Color PanelBg        => TM != null ? TM.PanelBg : DefaultPanelBg;
        public static Color ButtonNormal   => TM != null ? TM.ButtonNormal : DefaultButtonNormal;
        public static Color ButtonHover    => TM != null ? TM.ButtonHover : DefaultButtonHover;
        public static Color ButtonPress    => TM != null ? TM.ButtonPress : DefaultButtonPress;
        public static Color SliderBg       => TM != null ? TM.SliderBg : DefaultSliderBg;
        public static Color SliderFill     => TM != null ? TM.SliderFill : DefaultSliderFill;
        public static Color SliderHandle   => TM != null ? TM.SliderHandle : DefaultSliderHandle;
        public static Color ToggleCheckmark => TM != null ? TM.ToggleCheckmark : DefaultToggleCheckmark;
        public static Color TextColor      => TM != null ? TM.TextColor : DefaultTextColor;
        public static Color HeaderColor    => TM != null ? TM.HeaderColor : DefaultHeaderColor;
        public static Color Separator      => TM != null ? TM.Separator : DefaultSeparator;

        private static Font _cachedFont;
        private static Font DefaultFont
        {
            get
            {
                if (_cachedFont == null)
                    _cachedFont = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
                if (_cachedFont == null)
                    _cachedFont = Font.CreateDynamicFontFromOSFont("Segoe UI", 14);
                return _cachedFont;
            }
        }

        // ── Canvas ──────────────────────────────────────────────

        /// <summary>Create a Screen Space Overlay canvas with raycaster.</summary>
        public static Canvas CreateCanvas(string name, int sortOrder = 100)
        {
            var go = new GameObject(name);
            var canvas = go.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = sortOrder;

            var scaler = go.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080);
            scaler.matchWidthOrHeight = 0.5f;

            go.AddComponent<GraphicRaycaster>();

            // Ensure an EventSystem exists (required for UI clicks)
            EnsureEventSystem();

            return canvas;
        }

        /// <summary>Create an EventSystem with InputSystemUIInputModule if none exists.</summary>
        private static void EnsureEventSystem()
        {
            if (EventSystem.current != null) return;

            var esGo = new GameObject("EventSystem");
            esGo.AddComponent<EventSystem>();
            esGo.AddComponent<InputSystemUIInputModule>();
        }

        // ── Panel ───────────────────────────────────────────────

        public static RectTransform CreatePanel(Transform parent, string name,
            Vector2 size, Color? bgColor = null)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);

            var rt = go.AddComponent<RectTransform>();
            rt.sizeDelta = size;

            var img = go.AddComponent<Image>();
            img.color = bgColor ?? PanelBg;
            img.raycastTarget = true;

            if (bgColor == null)
                TM?.Track(img, ThemeRole.PanelBg);

            return rt;
        }

        // ── Text ────────────────────────────────────────────────

        public static Text CreateText(Transform parent, string name, string content,
            int fontSize = 14, TextAnchor alignment = TextAnchor.MiddleLeft,
            Color? color = null)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var rt = go.AddComponent<RectTransform>();
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;

            var txt = go.AddComponent<Text>();
            txt.text = content;
            txt.font = DefaultFont;
            txt.fontSize = fontSize;
            txt.alignment = alignment;
            txt.color = color ?? TextColor;
            txt.horizontalOverflow = HorizontalWrapMode.Overflow;
            txt.verticalOverflow = VerticalWrapMode.Overflow;

            if (color == null)
                TM?.Track(txt, ThemeRole.TextColor);

            return txt;
        }

        // ── Button ──────────────────────────────────────────────

        public static Button CreateButton(Transform parent, string name, string label,
            Vector2 size, int fontSize = 14)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var rt = go.AddComponent<RectTransform>();
            rt.sizeDelta = size;

            var img = go.AddComponent<Image>();
            img.color = ButtonNormal;

            var btn = go.AddComponent<Button>();
            var colors = btn.colors;
            colors.normalColor = ButtonNormal;
            colors.highlightedColor = ButtonHover;
            colors.pressedColor = ButtonPress;
            colors.selectedColor = ButtonNormal;
            colors.fadeDuration = 0.08f;
            btn.colors = colors;
            btn.targetGraphic = img;

            // Label
            var txt = CreateText(go.transform, "Label", label, fontSize, TextAnchor.MiddleCenter);

            TM?.TrackButton(btn);

            return btn;
        }

        // ── Toggle ──────────────────────────────────────────────

        public static Toggle CreateToggle(Transform parent, string name, string label,
            Vector2 rowSize, int fontSize = 14)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var rt = go.AddComponent<RectTransform>();
            rt.sizeDelta = rowSize;

            // Background box
            var bgGo = new GameObject("Background");
            bgGo.transform.SetParent(go.transform, false);
            var bgRt = bgGo.AddComponent<RectTransform>();
            bgRt.anchorMin = new Vector2(0, 0.5f);
            bgRt.anchorMax = new Vector2(0, 0.5f);
            bgRt.pivot = new Vector2(0, 0.5f);
            bgRt.anchoredPosition = new Vector2(4, 0);
            bgRt.sizeDelta = new Vector2(22, 22);
            var bgImg = bgGo.AddComponent<Image>();
            bgImg.color = SliderBg;

            // Checkmark
            var cmGo = new GameObject("Checkmark");
            cmGo.transform.SetParent(bgGo.transform, false);
            var cmRt = cmGo.AddComponent<RectTransform>();
            cmRt.anchorMin = Vector2.zero;
            cmRt.anchorMax = Vector2.one;
            cmRt.offsetMin = new Vector2(3, 3);
            cmRt.offsetMax = new Vector2(-3, -3);
            var cmImg = cmGo.AddComponent<Image>();
            cmImg.color = ToggleCheckmark;

            // Toggle component
            var toggle = go.AddComponent<Toggle>();
            toggle.targetGraphic = bgImg;
            toggle.graphic = cmImg;
            toggle.isOn = false;

            TM?.Track(bgImg, ThemeRole.SliderBg);
            TM?.Track(cmImg, ThemeRole.ToggleCheckmark);

            // Label text
            var labelGo = new GameObject("Label");
            labelGo.transform.SetParent(go.transform, false);
            var labelRt = labelGo.AddComponent<RectTransform>();
            labelRt.anchorMin = new Vector2(0, 0);
            labelRt.anchorMax = new Vector2(1, 1);
            labelRt.offsetMin = new Vector2(32, 0);
            labelRt.offsetMax = Vector2.zero;

            var txt = labelGo.AddComponent<Text>();
            txt.text = label;
            txt.font = DefaultFont;
            txt.fontSize = fontSize;
            txt.alignment = TextAnchor.MiddleLeft;
            txt.color = TextColor;

            return toggle;
        }

        // ── Slider ──────────────────────────────────────────────

        /// <summary>Creates a labeled slider row: [Label] [====o====] [Value]</summary>
        public static (Slider slider, Text valueLabel) CreateSliderRow(Transform parent,
            string name, string label, float min, float max, bool wholeNumbers,
            Vector2 rowSize, int fontSize = 13)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var rt = go.AddComponent<RectTransform>();
            rt.sizeDelta = rowSize;

            float labelWidth = rowSize.x * 0.35f;
            float valueWidth = 60f;
            float sliderWidth = rowSize.x - labelWidth - valueWidth - 12f;

            // Label
            var labelGo = new GameObject("Label");
            labelGo.transform.SetParent(go.transform, false);
            var labelRt = labelGo.AddComponent<RectTransform>();
            labelRt.anchorMin = new Vector2(0, 0);
            labelRt.anchorMax = new Vector2(0, 1);
            labelRt.pivot = new Vector2(0, 0.5f);
            labelRt.anchoredPosition = new Vector2(4, 0);
            labelRt.sizeDelta = new Vector2(labelWidth, 0);
            var labelTxt = labelGo.AddComponent<Text>();
            labelTxt.text = label;
            labelTxt.font = DefaultFont;
            labelTxt.fontSize = fontSize;
            labelTxt.alignment = TextAnchor.MiddleLeft;
            labelTxt.color = TextColor;

            // Slider
            var sliderGo = new GameObject("Slider");
            sliderGo.transform.SetParent(go.transform, false);
            var sliderRt = sliderGo.AddComponent<RectTransform>();
            sliderRt.anchorMin = new Vector2(0, 0.5f);
            sliderRt.anchorMax = new Vector2(0, 0.5f);
            sliderRt.pivot = new Vector2(0, 0.5f);
            sliderRt.anchoredPosition = new Vector2(labelWidth + 4, 0);
            sliderRt.sizeDelta = new Vector2(sliderWidth, 16);

            // Slider background
            var bgGo = new GameObject("Background");
            bgGo.transform.SetParent(sliderGo.transform, false);
            var bgRtS = bgGo.AddComponent<RectTransform>();
            bgRtS.anchorMin = Vector2.zero;
            bgRtS.anchorMax = Vector2.one;
            bgRtS.offsetMin = Vector2.zero;
            bgRtS.offsetMax = Vector2.zero;
            var bgImgS = bgGo.AddComponent<Image>();
            bgImgS.color = SliderBg;

            TM?.Track(bgImgS, ThemeRole.SliderBg);

            // Fill area
            var fillArea = new GameObject("Fill Area");
            fillArea.transform.SetParent(sliderGo.transform, false);
            var fillAreaRt = fillArea.AddComponent<RectTransform>();
            fillAreaRt.anchorMin = Vector2.zero;
            fillAreaRt.anchorMax = Vector2.one;
            fillAreaRt.offsetMin = new Vector2(5, 2);
            fillAreaRt.offsetMax = new Vector2(-15, -2);

            var fillGo = new GameObject("Fill");
            fillGo.transform.SetParent(fillArea.transform, false);
            var fillRt = fillGo.AddComponent<RectTransform>();
            fillRt.anchorMin = Vector2.zero;
            fillRt.anchorMax = new Vector2(0, 1);
            fillRt.sizeDelta = new Vector2(0, 0);
            var fillImg = fillGo.AddComponent<Image>();
            fillImg.color = SliderFill;

            TM?.Track(fillImg, ThemeRole.SliderFill);

            // Handle area
            var handleArea = new GameObject("Handle Slide Area");
            handleArea.transform.SetParent(sliderGo.transform, false);
            var handleAreaRt = handleArea.AddComponent<RectTransform>();
            handleAreaRt.anchorMin = Vector2.zero;
            handleAreaRt.anchorMax = Vector2.one;
            handleAreaRt.offsetMin = new Vector2(5, 0);
            handleAreaRt.offsetMax = new Vector2(-5, 0);

            var handleGo = new GameObject("Handle");
            handleGo.transform.SetParent(handleArea.transform, false);
            var handleRt = handleGo.AddComponent<RectTransform>();
            handleRt.sizeDelta = new Vector2(14, 20);
            var handleImg = handleGo.AddComponent<Image>();
            handleImg.color = SliderHandle;

            TM?.Track(handleImg, ThemeRole.SliderHandle);

            // Slider component
            var slider = sliderGo.AddComponent<Slider>();
            slider.fillRect = fillRt;
            slider.handleRect = handleRt;
            slider.targetGraphic = handleImg;
            slider.minValue = min;
            slider.maxValue = max;
            slider.wholeNumbers = wholeNumbers;
            slider.direction = Slider.Direction.LeftToRight;

            // Value label
            var valGo = new GameObject("Value");
            valGo.transform.SetParent(go.transform, false);
            var valRt = valGo.AddComponent<RectTransform>();
            valRt.anchorMin = new Vector2(1, 0);
            valRt.anchorMax = new Vector2(1, 1);
            valRt.pivot = new Vector2(1, 0.5f);
            valRt.anchoredPosition = new Vector2(-4, 0);
            valRt.sizeDelta = new Vector2(valueWidth, 0);
            var valTxt = valGo.AddComponent<Text>();
            valTxt.text = "";
            valTxt.font = DefaultFont;
            valTxt.fontSize = fontSize;
            valTxt.alignment = TextAnchor.MiddleRight;
            valTxt.color = TextColor;

            return (slider, valTxt);
        }

        // ── Section Header ──────────────────────────────────────

        public static Text CreateSectionHeader(Transform parent, string text, Vector2 rowSize)
        {
            var go = new GameObject("Header_" + text);
            go.transform.SetParent(parent, false);
            var rt = go.AddComponent<RectTransform>();
            rt.sizeDelta = rowSize;

            var txt = go.AddComponent<Text>();
            txt.text = text;
            txt.font = DefaultFont;
            txt.fontSize = 15;
            txt.fontStyle = FontStyle.Bold;
            txt.alignment = TextAnchor.MiddleLeft;
            txt.color = HeaderColor;

            TM?.Track(txt, ThemeRole.HeaderColor);

            return txt;
        }

        // ── Label (sized row with text) ─────────────────────────

        /// <summary>
        /// Creates a single-line text label inside a layout-compatible row.
        /// </summary>
        public static Text CreateLabel(Transform parent, string name, string text,
            Vector2 rowSize, int fontSize = 13)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var rt = go.AddComponent<RectTransform>();
            rt.sizeDelta = rowSize;

            var txt = go.AddComponent<Text>();
            txt.text = text;
            txt.font = DefaultFont;
            txt.fontSize = fontSize;
            txt.alignment = TextAnchor.MiddleLeft;
            txt.color = TextColor;
            return txt;
        }

        // ── Separator Line ──────────────────────────────────────

        public static void CreateSeparator(Transform parent, float width)
        {
            var go = new GameObject("Separator");
            go.transform.SetParent(parent, false);
            var rt = go.AddComponent<RectTransform>();
            rt.sizeDelta = new Vector2(width, 1);
            var img = go.AddComponent<Image>();
            img.color = Separator;

            TM?.Track(img, ThemeRole.Separator);
        }

        // ── Input Field ─────────────────────────────────────────

        /// <summary>Create a labeled input field row.</summary>
        public static (InputField field, Text label) CreateInputFieldRow(
            Transform parent, string name, string labelText, string placeholder, Vector2 rowSize)
        {
            var row = new GameObject(name + "Row");
            row.transform.SetParent(parent, false);
            var rowRT = row.AddComponent<RectTransform>();
            rowRT.sizeDelta = rowSize;
            var hLayout = row.AddComponent<HorizontalLayoutGroup>();
            hLayout.spacing = 6;
            hLayout.childAlignment = TextAnchor.MiddleLeft;
            hLayout.childControlWidth = true;
            hLayout.childControlHeight = true;
            hLayout.childForceExpandWidth = true;
            hLayout.childForceExpandHeight = false;

            // Label (40% width)
            var lbl = CreateText(row.transform, name + "Label", labelText, 12, TextAnchor.MiddleLeft);
            var lblLE = lbl.gameObject.AddComponent<LayoutElement>();
            lblLE.flexibleWidth = 0.4f;

            // Input field background
            var fieldGO = new GameObject(name + "Field");
            fieldGO.transform.SetParent(row.transform, false);
            var fieldRT = fieldGO.AddComponent<RectTransform>();
            fieldRT.sizeDelta = new Vector2(0, rowSize.y);
            var fieldLE = fieldGO.AddComponent<LayoutElement>();
            fieldLE.flexibleWidth = 0.6f;
            var bg = fieldGO.AddComponent<Image>();
            bg.color = SliderBg;

            TM?.Track(bg, ThemeRole.SliderBg);

            // Text child
            var textGO = new GameObject("Text");
            textGO.transform.SetParent(fieldGO.transform, false);
            var textRT = textGO.AddComponent<RectTransform>();
            textRT.anchorMin = Vector2.zero;
            textRT.anchorMax = Vector2.one;
            textRT.offsetMin = new Vector2(4, 0);
            textRT.offsetMax = new Vector2(-4, 0);
            var textComp = textGO.AddComponent<Text>();
            textComp.font = DefaultFont;
            textComp.fontSize = 12;
            textComp.color = TextColor;
            textComp.alignment = TextAnchor.MiddleLeft;

            // Placeholder child
            var phGO = new GameObject("Placeholder");
            phGO.transform.SetParent(fieldGO.transform, false);
            var phRT = phGO.AddComponent<RectTransform>();
            phRT.anchorMin = Vector2.zero;
            phRT.anchorMax = Vector2.one;
            phRT.offsetMin = new Vector2(4, 0);
            phRT.offsetMax = new Vector2(-4, 0);
            var phText = phGO.AddComponent<Text>();
            phText.font = DefaultFont;
            phText.fontSize = 12;
            phText.color = new Color(TextColor.r, TextColor.g, TextColor.b, 0.4f);
            phText.alignment = TextAnchor.MiddleLeft;
            phText.fontStyle = FontStyle.Italic;
            phText.text = placeholder;

            var inputField = fieldGO.AddComponent<InputField>();
            inputField.textComponent = textComp;
            inputField.placeholder = phText;

            return (inputField, lbl);
        }

        // ── Scroll View ─────────────────────────────────────────

        /// <summary>
        /// Creates a scrollable vertical container. Returns (scrollRect, contentTransform).
        /// Add child items to contentTransform.
        /// </summary>
        public static (ScrollRect scrollRect, RectTransform content) CreateScrollView(
            Transform parent, Vector2 size)
        {
            var go = new GameObject("ScrollView");
            go.transform.SetParent(parent, false);
            var rt = go.AddComponent<RectTransform>();
            rt.sizeDelta = size;
            rt.anchorMin = new Vector2(0.5f, 0.5f);
            rt.anchorMax = new Vector2(0.5f, 0.5f);

            // Viewport
            var viewport = new GameObject("Viewport");
            viewport.transform.SetParent(go.transform, false);
            var vpRt = viewport.AddComponent<RectTransform>();
            vpRt.anchorMin = Vector2.zero;
            vpRt.anchorMax = Vector2.one;
            vpRt.offsetMin = Vector2.zero;
            vpRt.offsetMax = Vector2.zero;
            viewport.AddComponent<RectMask2D>();

            // Content
            var content = new GameObject("Content");
            content.transform.SetParent(viewport.transform, false);
            var cRt = content.AddComponent<RectTransform>();
            cRt.anchorMin = new Vector2(0, 1);
            cRt.anchorMax = new Vector2(1, 1);
            cRt.pivot = new Vector2(0.5f, 1);
            cRt.anchoredPosition = Vector2.zero;
            cRt.sizeDelta = new Vector2(0, 0); // VLG will drive height

            var vlg = content.AddComponent<VerticalLayoutGroup>();
            vlg.spacing = 4;
            vlg.padding = new RectOffset(8, 8, 8, 8);
            vlg.childAlignment = TextAnchor.UpperCenter;
            vlg.childControlWidth = true;
            vlg.childControlHeight = false;
            vlg.childForceExpandWidth = true;
            vlg.childForceExpandHeight = false;

            var csf = content.AddComponent<ContentSizeFitter>();
            csf.verticalFit = ContentSizeFitter.FitMode.PreferredSize;

            var sr = go.AddComponent<ScrollRect>();
            sr.viewport = vpRt;
            sr.content = cRt;
            sr.horizontal = false;
            sr.vertical = true;
            sr.movementType = ScrollRect.MovementType.Clamped;
            sr.scrollSensitivity = 20f;

            return (sr, cRt);
        }

        // ── Vertical Layout ─────────────────────────────────────

        public static VerticalLayoutGroup AddVerticalLayout(RectTransform rt,
            int padL = 8, int padR = 8, int padT = 8, int padB = 8, float spacing = 6)
        {
            var vlg = rt.gameObject.AddComponent<VerticalLayoutGroup>();
            vlg.padding = new RectOffset(padL, padR, padT, padB);
            vlg.spacing = spacing;
            vlg.childAlignment = TextAnchor.UpperCenter;
            vlg.childControlWidth = true;
            vlg.childControlHeight = false;
            vlg.childForceExpandWidth = true;
            vlg.childForceExpandHeight = false;
            return vlg;
        }
    }
}
