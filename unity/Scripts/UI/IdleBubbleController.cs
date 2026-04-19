using System.Collections;
using UnityEngine;

namespace Annabeth.UI
{
    /// <summary>
    /// Displays proactive / idle thoughts from the Grillo reflection loop
    /// as floating speech bubbles when the companion is not in an active
    /// conversation.  Receives thoughts via WebSocket as "idle_thought"
    /// messages dispatched by MessageHandler.OnIdleThought.
    ///
    /// Phase 6 — Annabeth idle speech bubbles.
    /// </summary>
    public class IdleBubbleController : MonoBehaviour
    {
        [Header("References")]
        [Tooltip("The shared SpeechBubble component used for all text display.")]
        [SerializeField] private SpeechBubble speechBubble;

        [Header("Timing")]
        [Tooltip("Minimum seconds between idle thought bubbles.")]
        [SerializeField] private float minDelaySeconds = 60f;

        [Tooltip("Maximum seconds between idle thought bubbles.")]
        [SerializeField] private float maxDelaySeconds = 300f;

        // Set to true when the avatar enters idle / sleep, false when active.
        private bool _idleMode = false;

        // ----------------------------------------------------------------
        // Public API
        // ----------------------------------------------------------------

        /// <summary>
        /// Show an idle thought as a speech bubble.
        /// Only visible when idle mode is active.
        /// </summary>
        public void ShowIdleThought(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return;
            if (!_idleMode) return;
            if (speechBubble == null)
            {
                Debug.LogWarning("[IdleBubble] No SpeechBubble reference assigned.");
                return;
            }

            speechBubble.ShowText(text);
            Debug.Log($"[IdleBubble] Showing: {text}");
        }

        /// <summary>
        /// Toggle idle mode on/off.  When off, incoming thoughts are silently discarded.
        /// Called by IdleController (Phase 7) or CompanionManager on mode change.
        /// </summary>
        public void SetIdleMode(bool idle)
        {
            _idleMode = idle;
            if (!idle && speechBubble != null)
                speechBubble.HideNow();
        }

        public bool IsIdleMode => _idleMode;
    }
}
