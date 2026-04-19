using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;

namespace Annabeth.UI
{
    /// <summary>
    /// Right-click context menu on the avatar. Builds its own Canvas + buttons at runtime.
    /// Quick actions: Settings, Change Character, Toggle Bubble, Sleep, Clear History, Quit.
    /// Inspired by Mate-Engine MenuActions.cs — simplified, no external assets needed.
    /// </summary>
    public class RadialMenu : MonoBehaviour
    {
        /// <summary>Static flag — other scripts check this to block interaction while menu is open.</summary>
        public static bool IsMenuOpen { get; private set; }

        /// <summary>True if any UI panel (menu or settings or library) is open.</summary>
        public static bool IsAnyPanelOpen => IsMenuOpen || _isSettingsOpen || _isLibraryOpen;
        private static bool _isSettingsOpen;
        private static bool _isLibraryOpen;

        private Canvas _canvas;
        private GameObject _menuPanel;
        private RectTransform _menuRect;
        private GameObject _settingsRoot;
        private SettingsPanel _settingsPanel;
        private GameObject _libraryRoot;
        private Avatar.VrmModelLibrary _modelLibrary;

        private Button _btnSettings;
        private Button _btnCharacter;
        private Button _btnBubble;
        private Button _btnSleep;
        private Button _btnClearHistory;
        private Button _btnQuit;
        private Button _btnModeActive;
        private Button _btnModeBeatDance;
        private Button _btnModeFullDance;

        private void Awake()
        {
            BuildUI();
        }

        private void OnDestroy()
        {
            _btnModeActive?.onClick.RemoveListener(OnModeActiveClick);
            _btnModeBeatDance?.onClick.RemoveListener(OnModeBeatDanceClick);
            _btnModeFullDance?.onClick.RemoveListener(OnModeFullDanceClick);
            _btnSettings?.onClick.RemoveListener(OnSettingsClick);
            _btnCharacter?.onClick.RemoveListener(OnCharacterClick);
            _btnBubble?.onClick.RemoveListener(OnBubbleClick);
            _btnSleep?.onClick.RemoveListener(OnSleepClick);
            _btnClearHistory?.onClick.RemoveListener(OnClearHistoryClick);
            _btnQuit?.onClick.RemoveListener(OnQuitClick);

            if (_modelLibrary != null)
                _modelLibrary.OnCloseRequested -= CloseLibrary;

            if (_canvas != null)
                Destroy(_canvas.gameObject);
        }

        private void BuildUI()
        {
            // Shared canvas for all companion UI
            _canvas = UIFactory.CreateCanvas("CompanionUI", 100);
            _canvas.transform.SetParent(transform, false);

            // ── Menu Panel ──────────────────────────────────────
            _menuRect = UIFactory.CreatePanel(_canvas.transform, "RadialMenu",
                new Vector2(180, 380));
            _menuPanel = _menuRect.gameObject;

            UIFactory.AddVerticalLayout(_menuRect, 6, 6, 6, 6, 4);

            var rowSize = new Vector2(168, 32);

            // Mode buttons
            _btnModeActive    = UIFactory.CreateButton(_menuRect, "BtnModeActive",    "Active Mode",     rowSize);
            _btnModeBeatDance = UIFactory.CreateButton(_menuRect, "BtnModeBeatDance", "Beat Dance",      rowSize);
            _btnModeFullDance = UIFactory.CreateButton(_menuRect, "BtnModeFullDance", "Full Dance",      rowSize);

            _btnSettings     = UIFactory.CreateButton(_menuRect, "BtnSettings",     "Settings",        rowSize);
            _btnCharacter    = UIFactory.CreateButton(_menuRect, "BtnCharacter",    "Change Character", rowSize);
            _btnBubble       = UIFactory.CreateButton(_menuRect, "BtnBubble",       "Speech Bubble",    rowSize);
            _btnSleep        = UIFactory.CreateButton(_menuRect, "BtnSleep",        "Sleep / Wake",     rowSize);
            _btnClearHistory = UIFactory.CreateButton(_menuRect, "BtnClearHistory", "Clear History",    rowSize);
            _btnQuit         = UIFactory.CreateButton(_menuRect, "BtnQuit",         "Quit",             rowSize);

            _btnModeActive.onClick.AddListener(OnModeActiveClick);
            _btnModeBeatDance.onClick.AddListener(OnModeBeatDanceClick);
            _btnModeFullDance.onClick.AddListener(OnModeFullDanceClick);
            _btnSettings.onClick.AddListener(OnSettingsClick);
            _btnCharacter.onClick.AddListener(OnCharacterClick);
            _btnBubble.onClick.AddListener(OnBubbleClick);
            _btnSleep.onClick.AddListener(OnSleepClick);
            _btnClearHistory.onClick.AddListener(OnClearHistoryClick);
            _btnQuit.onClick.AddListener(OnQuitClick);

            _menuPanel.SetActive(false);

            // ── Settings Panel (built by SettingsPanel script) ──
            _settingsRoot = new GameObject("SettingsRoot");
            _settingsRoot.transform.SetParent(_canvas.transform, false);
            _settingsPanel = _settingsRoot.AddComponent<SettingsPanel>();
            _settingsPanel.SetRadialMenu(this);
            _settingsRoot.SetActive(false);

            // ── Model Library Panel ─────────────────────────────
            _libraryRoot = new GameObject("LibraryRoot");
            _libraryRoot.transform.SetParent(_canvas.transform, false);
            _modelLibrary = _libraryRoot.AddComponent<Avatar.VrmModelLibrary>();
            _modelLibrary.OnCloseRequested += CloseLibrary;
            _libraryRoot.SetActive(false);
        }

        private void Update()
        {
            var mouse = Mouse.current;
            var kb = Keyboard.current;

            // Right-click toggles menu
            if (mouse != null && mouse.rightButton.wasPressedThisFrame)
            {
                if (IsMenuOpen) CloseMenu();
                else OpenMenuAtCursor();
                return;
            }

            // M key toggles (plain, no modifiers)
            if (kb != null && kb.mKey.wasPressedThisFrame
                && !kb.leftCtrlKey.isPressed && !kb.rightCtrlKey.isPressed
                && !kb.leftShiftKey.isPressed && !kb.rightShiftKey.isPressed)
            {
                ToggleMenu();
                return;
            }

            // Escape closes any open panel
            if (kb != null && kb.escapeKey.wasPressedThisFrame && (IsMenuOpen || _isSettingsOpen || _isLibraryOpen))
            {
                CloseAll();
                return;
            }

            // Left-click outside menu closes it
            if (IsMenuOpen && mouse != null && mouse.leftButton.wasPressedThisFrame)
            {
                if (!IsPointerOverMenu())
                    CloseMenu();
            }
        }

        // ── Public API ──────────────────────────────────────────

        public void ToggleMenu()
        {
            if (IsMenuOpen) CloseMenu();
            else OpenMenuAtCursor();
        }

        public void OpenMenuAtCursor()
        {
            if (_menuPanel == null) return;

            Vector2 mousePos = Mouse.current != null
                ? Mouse.current.position.ReadValue()
                : new Vector2(Screen.width / 2f, Screen.height / 2f);

            RectTransformUtility.ScreenPointToLocalPointInRectangle(
                _canvas.transform as RectTransform,
                mousePos, null, out Vector2 localPoint);

            _menuRect.anchoredPosition = localPoint;
            _menuPanel.SetActive(true);
            IsMenuOpen = true;
        }

        public void CloseMenu()
        {
            if (_menuPanel != null)
                _menuPanel.SetActive(false);
            IsMenuOpen = false;
        }

        public void CloseAll()
        {
            CloseMenu();
            CloseSettings();
            CloseLibrary();
        }

        public void OpenSettings()
        {
            CloseMenu();
            if (_settingsRoot != null)
            {
                _settingsRoot.SetActive(true);
                _isSettingsOpen = true;
            }
        }

        public void CloseSettings()
        {
            if (_settingsRoot != null)
                _settingsRoot.SetActive(false);
            _isSettingsOpen = false;
        }

        public void OpenLibrary()
        {
            CloseMenu();
            if (_libraryRoot != null)
            {
                _libraryRoot.SetActive(true);
                _isLibraryOpen = true;
            }
        }

        public void CloseLibrary()
        {
            if (_libraryRoot != null)
                _libraryRoot.SetActive(false);
            _isLibraryOpen = false;
        }

        // ── Button Handlers ─────────────────────────────────────

        private void OnModeActiveClick()
        {
            var cm = FindFirstObjectByType<CompanionManager>();
            cm?.SetMode(Core.CompanionMode.Active);
            Debug.Log("[RadialMenu] Mode: Active");
            CloseMenu();
        }

        private void OnModeBeatDanceClick()
        {
            var cm = FindFirstObjectByType<CompanionManager>();
            cm?.StartDance(Core.DanceStyle.Procedural);
            Debug.Log("[RadialMenu] Mode: Beat Dance");
            CloseMenu();
        }

        private void OnModeFullDanceClick()
        {
            var cm = FindFirstObjectByType<CompanionManager>();
            cm?.StartDance(Core.DanceStyle.ShikanokoDance);
            Debug.Log("[RadialMenu] Mode: Full Dance");
            CloseMenu();
        }

        private void OnSettingsClick() => OpenSettings();

        private void OnCharacterClick()
        {
            OpenLibrary();
        }

        private void OnBubbleClick()
        {
            if (Core.SettingsManager.Instance != null)
            {
                var data = Core.SettingsManager.Instance.data;
                data.enableSpeechBubble = !data.enableSpeechBubble;
                Core.SettingsManager.Instance.SaveAll();
                Debug.Log($"[RadialMenu] Speech bubble: {data.enableSpeechBubble}");
            }
            CloseMenu();
        }

        private void OnSleepClick()
        {
            CloseMenu();
            var sc = FindFirstObjectByType<Core.SleepController>();
            if (sc != null)
            {
                sc.ToggleSleep();
                Debug.Log($"[RadialMenu] Sleep toggle → {(sc.IsSleeping ? "sleeping" : "awake")}");
            }
            else
            {
                Debug.LogWarning("[RadialMenu] SleepController not found.");
            }
        }

        private void OnClearHistoryClick()
        {
            var ws = FindFirstObjectByType<Core.WebSocketClient>();
            if (ws != null)
            {
                ws.Send("{\"type\":\"clear_history\"}");
                Debug.Log("[RadialMenu] Clear history sent.");
            }
            CloseMenu();
        }

        private void OnQuitClick()
        {
            CloseMenu();
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            // Bypass SystemTrayController's wantsToQuit handler that
            // intercepts Application.Quit() and hides the avatar instead.
            Core.SystemTrayController.forceQuit = true;
            Application.Quit();
#endif
        }

        // ── Helpers ─────────────────────────────────────────────

        private bool IsPointerOverMenu()
        {
            if (_menuRect == null) return false;
            Vector2 mousePos = Mouse.current != null
                ? Mouse.current.position.ReadValue()
                : Vector2.zero;
            return RectTransformUtility.RectangleContainsScreenPoint(_menuRect, mousePos, null);
        }
    }
}
