using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;

namespace Annabeth.UI
{
    /// <summary>
    /// Right-click context menu on the avatar. Opens a panel at cursor position with
    /// quick actions: Settings, Change Character, Toggle Bubble, Sleep, Clear History, Quit.
    /// Inspired by Mate-Engine MenuActions.cs — simplified, no external Tasty Pie Menu asset.
    /// </summary>
    public class RadialMenu : MonoBehaviour
    {
        [Header("UI References")]
        [SerializeField] private GameObject menuPanel;
        [SerializeField] private Button btnSettings;
        [SerializeField] private Button btnCharacter;
        [SerializeField] private Button btnBubble;
        [SerializeField] private Button btnSleep;
        [SerializeField] private Button btnClearHistory;
        [SerializeField] private Button btnQuit;

        [Header("Linked Panels")]
        [SerializeField] private GameObject settingsPanel;

        [Header("Audio (Optional)")]
        [SerializeField] private AudioSource audioSource;
        [SerializeField] private AudioClip openSound;
        [SerializeField] private AudioClip closeSound;

        /// <summary>
        /// Static flag — other scripts check this to block interaction while menu is open.
        /// Pattern from Mate-Engine MenuActions.cs.
        /// </summary>
        public static bool IsMenuOpen { get; private set; }

        /// <summary>True if any UI panel (menu or settings) is open.</summary>
        public static bool IsAnyPanelOpen => IsMenuOpen || _isSettingsOpen;
        private static bool _isSettingsOpen;

        private RectTransform _menuRect;
        private Canvas _parentCanvas;

        private void Awake()
        {
            if (menuPanel != null)
            {
                _menuRect = menuPanel.GetComponent<RectTransform>();
                _parentCanvas = menuPanel.GetComponentInParent<Canvas>();
                menuPanel.SetActive(false);
            }
        }

        private void Start()
        {
            // Wire buttons
            btnSettings?.onClick.AddListener(OnSettingsClick);
            btnCharacter?.onClick.AddListener(OnCharacterClick);
            btnBubble?.onClick.AddListener(OnBubbleClick);
            btnSleep?.onClick.AddListener(OnSleepClick);
            btnClearHistory?.onClick.AddListener(OnClearHistoryClick);
            btnQuit?.onClick.AddListener(OnQuitClick);
        }

        private void OnDestroy()
        {
            btnSettings?.onClick.RemoveListener(OnSettingsClick);
            btnCharacter?.onClick.RemoveListener(OnCharacterClick);
            btnBubble?.onClick.RemoveListener(OnBubbleClick);
            btnSleep?.onClick.RemoveListener(OnSleepClick);
            btnClearHistory?.onClick.RemoveListener(OnClearHistoryClick);
            btnQuit?.onClick.RemoveListener(OnQuitClick);
        }

        private void Update()
        {
            var mouse = Mouse.current;
            var kb = Keyboard.current;

            // Right-click to toggle menu
            if (mouse != null && mouse.rightButton.wasPressedThisFrame)
            {
                if (IsMenuOpen)
                    CloseMenu();
                else
                    OpenMenuAtCursor();
                return;
            }

            // M key to toggle (also handled via HotkeyManager for Ctrl+Shift+M)
            if (kb != null && kb.mKey.wasPressedThisFrame
                && !kb.leftCtrlKey.isPressed && !kb.rightCtrlKey.isPressed
                && !kb.leftShiftKey.isPressed && !kb.rightShiftKey.isPressed)
            {
                ToggleMenu();
                return;
            }

            // Escape closes menu/settings
            if (kb != null && kb.escapeKey.wasPressedThisFrame && (IsMenuOpen || _isSettingsOpen))
            {
                CloseAll();
                return;
            }

            // Close menu if user clicks outside it (left click while menu is open)
            if (IsMenuOpen && mouse != null && mouse.leftButton.wasPressedThisFrame)
            {
                if (!IsPointerOverMenu())
                    CloseMenu();
            }
        }

        // ── Public API ──────────────────────────────────────────

        public void ToggleMenu()
        {
            if (IsMenuOpen)
                CloseMenu();
            else
                OpenMenuAtCursor();
        }

        public void OpenMenuAtCursor()
        {
            if (menuPanel == null) return;

            // Position at mouse cursor
            if (_parentCanvas != null && _menuRect != null)
            {
                Vector2 mousePos = Mouse.current != null
                    ? Mouse.current.position.ReadValue()
                    : Vector2.zero;

                RectTransformUtility.ScreenPointToLocalPointInRectangle(
                    _parentCanvas.transform as RectTransform,
                    mousePos,
                    _parentCanvas.worldCamera,
                    out Vector2 localPoint);

                _menuRect.anchoredPosition = localPoint;
            }

            menuPanel.SetActive(true);
            IsMenuOpen = true;
            PlaySound(openSound);
        }

        public void CloseMenu()
        {
            if (menuPanel != null)
                menuPanel.SetActive(false);
            IsMenuOpen = false;
            PlaySound(closeSound);
        }

        public void CloseAll()
        {
            CloseMenu();
            CloseSettings();
        }

        public void OpenSettings()
        {
            CloseMenu();
            if (settingsPanel != null)
            {
                settingsPanel.SetActive(true);
                _isSettingsOpen = true;
            }
        }

        public void CloseSettings()
        {
            if (settingsPanel != null)
                settingsPanel.SetActive(false);
            _isSettingsOpen = false;
        }

        // ── Button Handlers ─────────────────────────────────────

        private void OnSettingsClick()
        {
            OpenSettings();
        }

        private void OnCharacterClick()
        {
            // Phase 4: Will call VrmModelLibrary.Open()
            CloseMenu();
            Debug.Log("[RadialMenu] Character swap — not yet implemented (Phase 4).");
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
            // Phase 6: Will toggle SleepController
            CloseMenu();
            Debug.Log("[RadialMenu] Sleep toggle — not yet implemented (Phase 6).");
        }

        private void OnClearHistoryClick()
        {
            // Send clear command via WebSocket
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
            return RectTransformUtility.RectangleContainsScreenPoint(
                _menuRect, mousePos, _parentCanvas?.worldCamera);
        }

        private void PlaySound(AudioClip clip)
        {
            if (audioSource != null && clip != null)
                audioSource.PlayOneShot(clip);
        }
    }
}
