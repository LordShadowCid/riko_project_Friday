using UnityEngine;
using Annabeth.Core;
using Annabeth.Avatar;
using Annabeth.Dance;
using Annabeth.Interaction;

namespace Annabeth
{
    /// <summary>
    /// Main coordinator for the Annabeth companion.
    /// Wires together all components and handles mode transitions.
    /// </summary>
    public class CompanionManager : MonoBehaviour
    {
        [Header("Core Components")]
        [SerializeField] private WebSocketClient webSocketClient;
        [SerializeField] private MessageHandler messageHandler;

        [Header("Avatar Components")]
        [SerializeField] private AvatarController avatarController;
        [SerializeField] private LipSyncController lipSyncController;
        [SerializeField] private EmotionController emotionController;
        [SerializeField] private BlinkController blinkController;
        [SerializeField] private EyeTrackingController eyeTrackingController;
        [SerializeField] private IdleAnimationController idleAnimationController;

        [Header("Dance Components")]
        [SerializeField] private BeatDanceController beatDanceController;
        [SerializeField] private VrmaAnimationController vrmaAnimationController;

        [Header("Interaction")]
        [SerializeField] private TouchReactionController touchReactionController;

        [Header("State")]
        [SerializeField] private CompanionMode currentMode = CompanionMode.Idle;
        [SerializeField] private DanceStyle currentDanceStyle = DanceStyle.None;
        [SerializeField] private bool isSilenced;
        [SerializeField] private bool isSpeaking;

        // Track whether VRMA animation is audio-paused (no music playing)
        private bool _vrmaAudioPaused;

        public CompanionMode CurrentMode => currentMode;
        public bool IsSpeaking => isSpeaking;
        public bool IsSilenced => isSilenced;

        private void Awake()
        {
            Application.runInBackground = true;

            if (webSocketClient == null) webSocketClient = FindFirstObjectByType<WebSocketClient>();
            if (messageHandler == null) messageHandler = FindFirstObjectByType<MessageHandler>();
            if (avatarController == null) avatarController = FindFirstObjectByType<AvatarController>();
        }

        private void Start()
        {
            if (messageHandler != null)
            {
                messageHandler.OnSpeakStart += HandleSpeakStart;
                messageHandler.OnSpeakEnd += HandleSpeakEnd;
                messageHandler.OnEmotionChange += HandleEmotionChange;
                messageHandler.OnModeChange += HandleModeChange;
                messageHandler.OnSilenceToggle += HandleSilenceToggle;
                messageHandler.OnAudioAnalysis += HandleAudioAnalysis;
                messageHandler.OnDanceStyleChange += HandleDanceStyleChange;
            }

            if (avatarController != null)
            {
                avatarController.OnVrmLoaded += OnVrmLoaded;
            }
        }

        private void OnDestroy()
        {
            if (messageHandler != null)
            {
                messageHandler.OnSpeakStart -= HandleSpeakStart;
                messageHandler.OnSpeakEnd -= HandleSpeakEnd;
                messageHandler.OnEmotionChange -= HandleEmotionChange;
                messageHandler.OnModeChange -= HandleModeChange;
                messageHandler.OnSilenceToggle -= HandleSilenceToggle;
                messageHandler.OnAudioAnalysis -= HandleAudioAnalysis;
                messageHandler.OnDanceStyleChange -= HandleDanceStyleChange;
            }

            if (avatarController != null)
            {
                avatarController.OnVrmLoaded -= OnVrmLoaded;
            }
        }

        private void OnVrmLoaded(UniVRM10.Vrm10Instance vrm)
        {
            Debug.Log("[CompanionManager] VRM loaded, initializing components...");

            // Find and initialize avatar sub-controllers
            lipSyncController = avatarController.GetComponentInChildren<LipSyncController>();
            emotionController = avatarController.GetComponentInChildren<EmotionController>();
            blinkController = avatarController.GetComponentInChildren<BlinkController>();
            eyeTrackingController = avatarController.GetComponentInChildren<EyeTrackingController>();
            idleAnimationController = avatarController.GetComponentInChildren<IdleAnimationController>();

            // Initialize dance controller
            if (beatDanceController != null)
            {
                beatDanceController.Initialize(vrm);
            }

            // Initialize VRMA animation controller
            if (vrmaAnimationController != null)
            {
                vrmaAnimationController.Initialize(vrm);
            }

            // Initialize idle animation
            if (idleAnimationController != null)
            {
                idleAnimationController.Initialize(vrm);
            }

            // Initialize touch reactions
            touchReactionController = avatarController.GetComponentInChildren<TouchReactionController>();
            if (touchReactionController != null)
            {
                touchReactionController.Initialize(vrm);
            }
        }

        // ── Event Handlers ──────────────────────────────────────────

        private void HandleSpeakStart(string text)
        {
            isSpeaking = true;
            lipSyncController?.StartSpeaking();
            Debug.Log($"[CompanionManager] Speaking: {text}");
        }

        private void HandleSpeakEnd()
        {
            isSpeaking = false;
            lipSyncController?.StopSpeaking();
            Debug.Log("[CompanionManager] Stopped speaking");
        }

        private void HandleEmotionChange(string emotion)
        {
            emotionController?.SetEmotion(emotion);
            Debug.Log($"[CompanionManager] Emotion: {emotion}");
        }

        private void HandleModeChange(CompanionMode mode)
        {
            if (currentMode == mode) return;

            var previousMode = currentMode;
            currentMode = mode;

            // Handle mode exit
            switch (previousMode)
            {
                case CompanionMode.Dance:
                    beatDanceController?.StopDancing();
                    vrmaAnimationController?.Stop();
                    break;
            }

            // Handle mode enter
            switch (mode)
            {
                case CompanionMode.Active:
                    eyeTrackingController?.SetEnabled(true);
                    idleAnimationController?.SetEnabled(true);
                    break;

                case CompanionMode.Idle:
                    eyeTrackingController?.SetEnabled(true);
                    idleAnimationController?.SetEnabled(true);
                    emotionController?.ClearEmotion();
                    break;

                case CompanionMode.Dance:
                    idleAnimationController?.SetEnabled(false);
                    if (currentDanceStyle == DanceStyle.Procedural)
                    {
                        beatDanceController?.StartDancing();
                    }
                    break;
            }

            Debug.Log($"[CompanionManager] Mode changed: {previousMode} → {mode}");
        }

        private void HandleSilenceToggle(bool silenced)
        {
            isSilenced = silenced;
            Debug.Log($"[CompanionManager] Silenced: {silenced}");
        }

        private void HandleAudioAnalysis(float bass, float mid, float high, bool isBeat)
        {
            if (currentMode != CompanionMode.Dance) return;

            if (currentDanceStyle == DanceStyle.Procedural)
            {
                beatDanceController?.UpdateAudioData(bass, mid, high, isBeat);
            }
            else if (currentDanceStyle == DanceStyle.ShikanokoDance && vrmaAnimationController != null)
            {
                // Gate VRMA animation on audio — pause when silent, resume when music plays
                float energy = bass * 0.5f + mid * 0.3f + high * 0.2f;
                const float vrmaThreshold = 0.08f;
                if (energy < vrmaThreshold && !_vrmaAudioPaused)
                {
                    vrmaAnimationController.Pause();
                    _vrmaAudioPaused = true;
                }
                else if (energy >= vrmaThreshold && _vrmaAudioPaused)
                {
                    vrmaAnimationController.Resume();
                    _vrmaAudioPaused = false;
                }
            }
        }

        private void HandleDanceStyleChange(DanceStyle style)
        {
            currentDanceStyle = style;

            if (currentMode == CompanionMode.Dance)
            {
                switch (style)
                {
                    case DanceStyle.None:
                        beatDanceController?.StopDancing();
                        vrmaAnimationController?.Stop();
                        break;

                    case DanceStyle.Procedural:
                        vrmaAnimationController?.Stop();
                        beatDanceController?.StartDancing();
                        break;

                    case DanceStyle.ShikanokoDance:
                        beatDanceController?.StopDancing();
                        _vrmaAudioPaused = true; // Start paused until music detected
                        _ = PlayVrmaAnimation("shikanoko_dance.vrma");
                        break;
                }
            }

            Debug.Log($"[CompanionManager] Dance style: {style}");
        }

        // ── Public Methods ──────────────────────────────────────────

        public void SetMode(CompanionMode mode)
        {
            HandleModeChange(mode);
            messageHandler?.SendModeChange(mode);
        }

        public void StartDance(DanceStyle style)
        {
            HandleModeChange(CompanionMode.Dance);
            HandleDanceStyleChange(style);
            messageHandler?.SendDanceStyle(style);
        }

        public void SetDanceStyle(DanceStyle style)
        {
            if (style == DanceStyle.None)
            {
                SetMode(CompanionMode.Active);
            }
            else
            {
                HandleDanceStyleChange(style);
                messageHandler?.SendDanceStyle(style);
            }
        }

        public void ToggleSilence()
        {
            messageHandler?.SendSilenceToggle();
        }

        /// <summary>
        /// Play a VRMA animation by filename (from StreamingAssets/Animations/).
        /// </summary>
        private async System.Threading.Tasks.Task PlayVrmaAnimation(string fileName)
        {
            if (vrmaAnimationController == null)
            {
                Debug.LogWarning("[CompanionManager] VrmaAnimationController not assigned.");
                return;
            }
            await vrmaAnimationController.LoadAndPlay(fileName, destroyCancellationToken);
            // Start paused so animation only plays when music is detected
            if (_vrmaAudioPaused)
            {
                vrmaAnimationController.Pause();
            }
        }
    }
}
