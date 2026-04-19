using UnityEngine;
using UnityEngine.InputSystem;
using Annabeth.Core;
using Annabeth.UI;

namespace Annabeth.Input
{
    /// <summary>
    /// Handles keyboard input for controlling the companion.
    /// Maps hotkeys to actions (matching the Python client key mappings).
    /// Keys: 1=Active, 2=Idle, 3=Beat Dance, 4=Shikanoko Dance,
    ///        D=Cycle Dance, S=Silence, Q=Read Pause, R=Read Resume,
    ///        Space=Interrupt, Esc=Idle
    /// </summary>
    public class HotkeyManager : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private CompanionManager companionManager;
        [SerializeField] private MessageHandler messageHandler;
        [SerializeField] private DesktopLocomotionController locomotionController;
        [SerializeField] private WindowSnapper windowSnapper;

        [Header("Settings")]
        [SerializeField] private bool globalHotkeysEnabled = true;

        private DanceStyle _currentDanceStyle = DanceStyle.None;

        private void Start()
        {
            if (companionManager == null)
                companionManager = FindFirstObjectByType<CompanionManager>();
            if (messageHandler == null)
                messageHandler = FindFirstObjectByType<MessageHandler>();
            if (locomotionController == null)
                locomotionController = FindFirstObjectByType<DesktopLocomotionController>();
            if (windowSnapper == null)
                windowSnapper = FindFirstObjectByType<WindowSnapper>();
        }

        private void Update()
        {
            HandleHotkeys();
        }

        private void HandleHotkeys()
        {
            var kb = Keyboard.current;
            if (kb == null) return;

            // Alt+F4 always quits regardless of panel state
            bool alt = kb.leftAltKey.isPressed || kb.rightAltKey.isPressed;
            if (alt && kb.f4Key.wasPressedThisFrame)
            {
                Debug.Log("[HotkeyManager] Alt+F4 — quitting");
                SystemTrayController.forceQuit = true;
                Application.Quit();
                return;
            }

            // Block most hotkeys while UI panels are open
            if (RadialMenu.IsAnyPanelOpen)
                return;

            bool ctrl = kb.leftCtrlKey.isPressed || kb.rightCtrlKey.isPressed;
            bool shift = kb.leftShiftKey.isPressed || kb.rightShiftKey.isPressed;

            // Global hotkeys (Ctrl+Shift+Key)
            if (globalHotkeysEnabled && ctrl && shift)
            {
                if (kb.rKey.wasPressedThisFrame)
                {
                    messageHandler?.SendHotkey("read_aloud");
                    Debug.Log("[HotkeyManager] Read aloud triggered");
                }
                
                if (kb.aKey.wasPressedThisFrame)
                {
                    companionManager?.SetMode(CompanionMode.Active);
                    Debug.Log("[HotkeyManager] Active mode triggered");
                }
                
                if (kb.dKey.wasPressedThisFrame)
                {
                    CycleDanceStyle();
                }
                
                if (kb.mKey.wasPressedThisFrame)
                {
                    companionManager?.ToggleSilence();
                    Debug.Log("[HotkeyManager] Silence toggled");
                }
                
                return;
            }

            // Regular hotkeys (when window focused)
            if (kb.dKey.wasPressedThisFrame)
                CycleDanceStyle();

            if (kb.sKey.wasPressedThisFrame)
            {
                companionManager?.ToggleSilence();
                Debug.Log("[HotkeyManager] Silence toggled");
            }

            if (kb.qKey.wasPressedThisFrame)
            {
                messageHandler?.SendReadPause();
                Debug.Log("[HotkeyManager] Read-aloud paused");
            }

            if (kb.rKey.wasPressedThisFrame)
            {
                messageHandler?.SendReadResume();
                Debug.Log("[HotkeyManager] Read-aloud resumed");
            }

            // 1 = Active mode
            if (kb.digit1Key.wasPressedThisFrame)
            {
                _currentDanceStyle = DanceStyle.None;
                companionManager?.SetMode(CompanionMode.Active);
                Debug.Log("[HotkeyManager] Mode: Active");
            }
            // 2 = Idle mode
            if (kb.digit2Key.wasPressedThisFrame)
            {
                _currentDanceStyle = DanceStyle.None;
                companionManager?.SetMode(CompanionMode.Idle);
                Debug.Log("[HotkeyManager] Mode: Idle");
            }
            // 3 = Beat-reactive procedural dance (dance_beat)
            if (kb.digit3Key.wasPressedThisFrame)
            {
                _currentDanceStyle = DanceStyle.Procedural;
                companionManager?.StartDance(DanceStyle.Procedural);
                Debug.Log("[HotkeyManager] Dance: Procedural (Beat)");
            }
            // 4 = Shikanoko VRMA dance (dance_full)
            if (kb.digit4Key.wasPressedThisFrame)
            {
                _currentDanceStyle = DanceStyle.ShikanokoDance;
                companionManager?.StartDance(DanceStyle.ShikanokoDance);
                Debug.Log("[HotkeyManager] Dance: Shikanoko (Full)");
            }

            if (kb.spaceKey.wasPressedThisFrame)
            {
                messageHandler?.SendHotkey("interrupt");
                Debug.Log("[HotkeyManager] Interrupt");
            }

            if (kb.escapeKey.wasPressedThisFrame)
            {
                companionManager?.SetMode(CompanionMode.Idle);
                Debug.Log("[HotkeyManager] Return to idle");
            }

            // W = Toggle random walk locomotion
            if (kb.wKey.wasPressedThisFrame)
            {
                locomotionController?.ToggleEnabled();
                Debug.Log($"[HotkeyManager] Walk toggled: {locomotionController?.IsEnabled}");
            }

            // P = Screen-edge peek
            if (kb.pKey.wasPressedThisFrame)
            {
                locomotionController?.StartPeek();
                Debug.Log("[HotkeyManager] Peek toggled");
            }

            // G = Try sit on nearest window
            if (kb.gKey.wasPressedThisFrame)
            {
                windowSnapper?.TrySitOnNearestWindow();
                Debug.Log("[HotkeyManager] Try sit on window");
            }
        }

        private void CycleDanceStyle()
        {
            _currentDanceStyle = (DanceStyle)(((int)_currentDanceStyle + 1) % 3);

            if (_currentDanceStyle == DanceStyle.None)
            {
                companionManager?.SetMode(CompanionMode.Active);
            }
            else
            {
                companionManager?.StartDance(_currentDanceStyle);
            }

            Debug.Log($"[HotkeyManager] Dance style: {_currentDanceStyle}");
        }

        public void SetGlobalHotkeysEnabled(bool enabled)
        {
            globalHotkeysEnabled = enabled;
        }
    }
}
