using UnityEngine;
using Annabeth.Core;
using Annabeth.Avatar;
using Annabeth.Dance;

namespace Annabeth
{
    /// <summary>
    /// Main coordinator for the Annabeth companion.
    /// Wires together all components and handles mode transitions.
    /// </summary>
    public class CompanionManager : MonoBehaviour
    {
        [Header("Core Components")]
        [SerializeField] private SocketClient socketClient;
        [SerializeField] private MessageHandler messageHandler;
        
        [Header("Avatar Components")]
        [SerializeField] private AvatarController avatarController;
        [SerializeField] private LipSyncController lipSyncController;
        [SerializeField] private EmotionController emotionController;
        [SerializeField] private BlinkController blinkController;
        [SerializeField] private EyeTrackingController eyeTrackingController;
        
        [Header("Dance Components")]
        [SerializeField] private BeatDanceController beatDanceController;

        [Header("State")]
        [SerializeField] private CompanionMode currentMode = CompanionMode.Idle;
        [SerializeField] private DanceStyle currentDanceStyle = DanceStyle.None;
        [SerializeField] private bool isSilenced;
        [SerializeField] private bool isSpeaking;

        public CompanionMode CurrentMode => currentMode;
        public bool IsSpeaking => isSpeaking;
        public bool IsSilenced => isSilenced;

        private void Awake()
        {
            // Auto-find components if not assigned
            if (socketClient == null) socketClient = FindObjectOfType<SocketClient>();
            if (messageHandler == null) messageHandler = FindObjectOfType<MessageHandler>();
            if (avatarController == null) avatarController = FindObjectOfType<AvatarController>();
        }

        private void Start()
        {
            // Subscribe to events
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

            // Subscribe to VRM loaded event
            if (avatarController != null)
            {
                avatarController.OnVrmLoaded += OnVrmLoaded;
            }
        }

        private void OnDestroy()
        {
            // Unsubscribe from events
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
            
            // Initialize beat dance if present
            if (beatDanceController != null)
            {
                beatDanceController.Initialize(vrm);
            }
        }

        // === Event Handlers ===

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
                    break;
            }

            // Handle mode enter
            switch (mode)
            {
                case CompanionMode.Active:
                    eyeTrackingController?.SetEnabled(true);
                    break;
                    
                case CompanionMode.Idle:
                    eyeTrackingController?.SetEnabled(true);
                    emotionController?.ClearEmotion();
                    break;
                    
                case CompanionMode.Dance:
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

        private void HandleAudioAnalysis(float beatEnergy, float bassEnergy, float trebleEnergy)
        {
            if (currentMode == CompanionMode.Dance && currentDanceStyle == DanceStyle.Procedural)
            {
                beatDanceController?.UpdateAudioData(beatEnergy, bassEnergy, trebleEnergy);
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
                        break;
                        
                    case DanceStyle.Procedural:
                        beatDanceController?.StartDancing();
                        break;
                        
                    case DanceStyle.ShikanokoDance:
                        beatDanceController?.StopDancing();
                        // TODO: Trigger VRMA animation
                        Debug.Log("[CompanionManager] Shikanoko dance - VRMA playback not yet implemented");
                        break;
                }
            }

            Debug.Log($"[CompanionManager] Dance style: {style}");
        }

        // === Public Methods ===

        /// <summary>
        /// Set the current mode programmatically.
        /// </summary>
        public void SetMode(CompanionMode mode)
        {
            HandleModeChange(mode);
            messageHandler?.SendModeChange(mode);
        }

        /// <summary>
        /// Toggle silence state.
        /// </summary>
        public void ToggleSilence()
        {
            messageHandler?.SendSilenceToggle();
        }
    }
}
