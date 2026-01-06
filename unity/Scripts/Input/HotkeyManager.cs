using UnityEngine;
using Annabeth.Core;

namespace Annabeth.Input
{
    /// <summary>
    /// Handles keyboard input for controlling the companion.
    /// Maps hotkeys to actions (matching the Python/JavaScript implementation).
    /// </summary>
    public class HotkeyManager : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private MessageHandler messageHandler;

        [Header("Settings")]
        [SerializeField] private bool globalHotkeysEnabled = true;

        // Current dance style (for cycling)
        private DanceStyle _currentDanceStyle = DanceStyle.None;

        private void Start()
        {
            if (messageHandler == null)
            {
                messageHandler = FindObjectOfType<MessageHandler>();
            }
        }

        private void Update()
        {
            HandleHotkeys();
        }

        private void HandleHotkeys()
        {
            // Check for modifier keys
            bool ctrl = UnityEngine.Input.GetKey(KeyCode.LeftControl) || UnityEngine.Input.GetKey(KeyCode.RightControl);
            bool shift = UnityEngine.Input.GetKey(KeyCode.LeftShift) || UnityEngine.Input.GetKey(KeyCode.RightShift);

            // Global hotkeys (Ctrl+Shift+Key)
            if (globalHotkeysEnabled && ctrl && shift)
            {
                // Ctrl+Shift+R - Read aloud (OCR capture)
                if (UnityEngine.Input.GetKeyDown(KeyCode.R))
                {
                    messageHandler?.SendHotkey("read_aloud");
                    Debug.Log("[HotkeyManager] Read aloud triggered");
                }
                
                // Ctrl+Shift+A - Toggle active mode
                if (UnityEngine.Input.GetKeyDown(KeyCode.A))
                {
                    messageHandler?.SendModeChange(CompanionMode.Active);
                    Debug.Log("[HotkeyManager] Active mode triggered");
                }
                
                // Ctrl+Shift+D - Toggle dance mode
                if (UnityEngine.Input.GetKeyDown(KeyCode.D))
                {
                    CycleDanceStyle();
                }
                
                // Ctrl+Shift+M - Toggle mute/silence
                if (UnityEngine.Input.GetKeyDown(KeyCode.M))
                {
                    messageHandler?.SendSilenceToggle();
                    Debug.Log("[HotkeyManager] Silence toggled");
                }
                
                return; // Don't process regular keys if modifiers held
            }

            // Regular hotkeys (when window focused)
            
            // D - Cycle dance style
            if (UnityEngine.Input.GetKeyDown(KeyCode.D))
            {
                CycleDanceStyle();
            }

            // S - Toggle silence
            if (UnityEngine.Input.GetKeyDown(KeyCode.S))
            {
                messageHandler?.SendSilenceToggle();
                Debug.Log("[HotkeyManager] Silence toggled");
            }

            // Q - Pause read-aloud
            if (UnityEngine.Input.GetKeyDown(KeyCode.Q))
            {
                messageHandler?.SendReadPause();
                Debug.Log("[HotkeyManager] Read-aloud paused");
            }

            // R - Resume read-aloud
            if (UnityEngine.Input.GetKeyDown(KeyCode.R))
            {
                messageHandler?.SendReadResume();
                Debug.Log("[HotkeyManager] Read-aloud resumed");
            }

            // Number keys 1-4 for mode selection
            if (UnityEngine.Input.GetKeyDown(KeyCode.Alpha1))
            {
                _currentDanceStyle = DanceStyle.None;
                messageHandler?.SendDanceStyle(DanceStyle.None);
                Debug.Log("[HotkeyManager] Dance: None");
            }
            if (UnityEngine.Input.GetKeyDown(KeyCode.Alpha2))
            {
                _currentDanceStyle = DanceStyle.Procedural;
                messageHandler?.SendDanceStyle(DanceStyle.Procedural);
                Debug.Log("[HotkeyManager] Dance: Procedural");
            }
            if (UnityEngine.Input.GetKeyDown(KeyCode.Alpha3))
            {
                _currentDanceStyle = DanceStyle.ShikanokoDance;
                messageHandler?.SendDanceStyle(DanceStyle.ShikanokoDance);
                Debug.Log("[HotkeyManager] Dance: Shikanoko");
            }

            // Space - Interrupt/stop current action
            if (UnityEngine.Input.GetKeyDown(KeyCode.Space))
            {
                messageHandler?.SendHotkey("interrupt");
                Debug.Log("[HotkeyManager] Interrupt");
            }

            // Escape - Return to idle
            if (UnityEngine.Input.GetKeyDown(KeyCode.Escape))
            {
                messageHandler?.SendModeChange(CompanionMode.Idle);
                Debug.Log("[HotkeyManager] Return to idle");
            }
        }

        private void CycleDanceStyle()
        {
            _currentDanceStyle = (DanceStyle)(((int)_currentDanceStyle + 1) % 3);
            messageHandler?.SendDanceStyle(_currentDanceStyle);
            Debug.Log($"[HotkeyManager] Dance style: {_currentDanceStyle}");
        }

        /// <summary>
        /// Enable or disable global hotkeys.
        /// </summary>
        public void SetGlobalHotkeysEnabled(bool enabled)
        {
            globalHotkeysEnabled = enabled;
        }
    }
}
