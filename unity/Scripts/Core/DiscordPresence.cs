using DiscordRPC;
using DiscordRPC.Logging;
using UnityEngine;

namespace Annabeth.Core
{
    /// <summary>
    /// Discord Rich Presence integration for Annabeth.
    /// Phase 9: Shows companion state (Idle, Listening, Thinking, Talking, Dancing, Sleeping)
    /// in the user's Discord profile.
    ///
    /// Requires: com.lachee.discordrpc package in Packages/manifest.json
    /// Setup: Create a Discord application at https://discord.com/developers/applications
    ///        and paste the Application ID into the Inspector field.
    /// </summary>
    public class DiscordPresence : MonoBehaviour
    {
        [Header("Discord Application")]
        [Tooltip("Discord Application ID from https://discord.com/developers/applications")]
        [SerializeField] private string applicationId = "YOUR_APP_ID_HERE";

        [Header("Rich Presence Defaults")]
        [SerializeField] private string defaultDetails = "Hanging out with Annabeth";
        [SerializeField] private string largeImageKey   = "annabeth";
        [SerializeField] private string largeImageText  = "Annabeth AI Companion";

        private DiscordRpcClient _client;
        private bool _initialized;
        private string _currentState  = "Idle";
        private string _currentDetails;

        // ── Lifecycle ──────────────────────────────────────────────────────────

        private void Start()
        {
            _currentDetails = defaultDetails;

            if (string.IsNullOrEmpty(applicationId) || applicationId == "YOUR_APP_ID_HERE")
            {
                Debug.LogWarning("[DiscordPresence] applicationId not set — Discord Rich Presence disabled.");
                return;
            }

            try
            {
                _client = new DiscordRpcClient(applicationId)
                {
                    Logger = new ConsoleLogger { Level = LogLevel.Warning }
                };
                _client.OnReady += (_, e) => Debug.Log($"[DiscordPresence] Connected as {e.User.Username}");
                _client.OnError += (_, e) => Debug.LogWarning($"[DiscordPresence] Error: {e.Message}");
                _client.Initialize();
                _initialized = true;
                SetState("Idle");
                Debug.Log("[DiscordPresence] Initialized.");
            }
            catch (System.Exception ex)
            {
                // Graceful degradation — Discord not running or SDK unavailable
                Debug.LogWarning($"[DiscordPresence] Failed to initialize (Discord may not be running): {ex.Message}");
                _initialized = false;
            }
        }

        /// <summary>Process Discord callbacks — must be called every frame.</summary>
        private void Update()
        {
            if (_initialized) _client?.Invoke();
        }

        private void OnDestroy()
        {
            if (_initialized)
            {
                _client?.ClearPresence();
                _client?.Dispose();
                _initialized = false;
            }
        }

        // ── Public API ────────────────────────────────────────────────────────

        /// <summary>
        /// Update Discord Rich Presence state.
        /// States: "Idle", "Listening", "Thinking", "Talking", "Dancing", "Sleeping"
        /// </summary>
        public void SetState(string state, string details = null)
        {
            if (!_initialized) return;

            _currentState   = state;
            _currentDetails = details ?? defaultDetails;

            try
            {
                _client?.SetPresence(new RichPresence
                {
                    Details    = _currentDetails,
                    State      = state,
                    Timestamps = Timestamps.Now,
                    Assets     = new Assets
                    {
                        LargeImageKey  = largeImageKey,
                        LargeImageText = largeImageText,
                        SmallImageKey  = StateToImageKey(state),
                        SmallImageText = state,
                    }
                });
            }
            catch (System.Exception ex)
            {
                Debug.LogWarning($"[DiscordPresence] SetPresence failed: {ex.Message}");
            }
        }

        // ── Helpers ───────────────────────────────────────────────────────────

        private static string StateToImageKey(string state)
        {
            return state?.ToLower().Replace(" ", "_") ?? "idle";
        }
    }
}
