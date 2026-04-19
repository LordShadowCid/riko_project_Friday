using UnityEngine;
using Annabeth.Core;
using Annabeth.Avatar;
using Annabeth.Dance;
using Annabeth.Interaction;
using Annabeth.UI;

namespace Annabeth
{
    /// <summary>
    /// Main coordinator for the Annabeth companion.
    /// Wires together all components and handles mode transitions.
    /// </summary>
    public class CompanionManager : MonoBehaviour
    {
        // ── Networking ──────────────────────────────────────────────

        [Header("Networking")]
        [SerializeField] private WebSocketClient webSocketClient;
        [SerializeField] private MessageHandler messageHandler;

        // ── Avatar ──────────────────────────────────────────────────

        [Header("Avatar")]
        [SerializeField] private AvatarController avatarController;
        [SerializeField] private LipSyncController lipSyncController;
        [SerializeField] private EmotionController emotionController;
        [SerializeField] private BlinkController blinkController;
        [SerializeField] private EyeTrackingController eyeTrackingController;
        [SerializeField] private IKController ikController;
        [SerializeField] private AccessoryManager accessoryManager;

        // ── Animation ───────────────────────────────────────────────

        [Header("Animation")]
        [SerializeField] private AnimationBlendController animationBlendController;
        [SerializeField] private IdleAnimationController idleAnimationController;
        [SerializeField] private WalkAnimationController walkAnimController;
        [SerializeField] private DragAnimationController dragAnimController;
        [SerializeField] private DragPoseController dragPoseController;

        // ── Dance ───────────────────────────────────────────────────

        [Header("Dance")]
        [SerializeField] private BeatDanceController beatDanceController;
        [SerializeField] private VrmaAnimationController vrmaAnimationController;
        [SerializeField] private CustomDanceLoader customDanceLoader;

        // ── Interaction ─────────────────────────────────────────────

        [Header("Interaction")]
        [SerializeField] private TouchReactionController touchReactionController;
        [SerializeField] private TouchSoundHandler touchSoundHandler;
        [SerializeField] private ParticleEffectHandler particleEffectHandler;
        [SerializeField] private PetDetectionController petDetectionController;

        // ── Desktop ─────────────────────────────────────────────────

        [Header("Desktop")]
        [SerializeField] private DesktopLocomotionController locomotionController;
        [SerializeField] private WindowSnapper windowSnapper;
        [SerializeField] private OccluderQuadManager occluderQuadManager;
        [SerializeField] private DesktopAmbientProbe ambientProbe;

        // ── UI ──────────────────────────────────────────────────────

        [Header("UI")]
        [SerializeField] private SpeechBubble speechBubble;
        [SerializeField] private IdleBubbleController idleBubbleController;

        // ── System ──────────────────────────────────────────────────

        [Header("System")]
        [SerializeField] private SleepController sleepController;
        [SerializeField] private IdleController idleController;
        [SerializeField] private AlarmTimerManager alarmTimerManager;
        [SerializeField] private DiscordPresence discordPresence;

        // ── State ───────────────────────────────────────────────────

        [Header("State")]
        [SerializeField] private CompanionMode currentMode = CompanionMode.Idle;
        [SerializeField] private DanceStyle currentDanceStyle = DanceStyle.None;
        [SerializeField] private bool isSilenced;
        [SerializeField] private bool isSpeaking;

        private bool _vrmaAudioPaused;
        private float _randomMsgTimer;

        public CompanionMode CurrentMode => currentMode;
        public bool IsSpeaking => isSpeaking;
        public bool IsSilenced => isSilenced;

        // ══════════════════════════════════════════════════════════════
        //  Lifecycle
        // ══════════════════════════════════════════════════════════════

        private void Awake()
        {
            Application.runInBackground = true;

            if (webSocketClient == null) webSocketClient = FindFirstObjectByType<WebSocketClient>();
            if (messageHandler == null) messageHandler = FindFirstObjectByType<MessageHandler>();
            if (avatarController == null) avatarController = FindFirstObjectByType<AvatarController>();

            if (FindFirstObjectByType<ThemeManager>() == null)
            {
                var go = new GameObject("ThemeManager");
                go.AddComponent<ThemeManager>();
            }
        }

        private void Start()
        {
            // Server messages
            if (messageHandler != null)
            {
                messageHandler.OnSpeakStart += HandleSpeakStart;
                messageHandler.OnSpeakEnd += HandleSpeakEnd;
                messageHandler.OnEmotionChange += HandleEmotionChange;
                messageHandler.OnFaceExpression += HandleFaceExpression;
                messageHandler.OnModeChange += HandleModeChange;
                messageHandler.OnSilenceToggle += HandleSilenceToggle;
                messageHandler.OnAudioAnalysis += HandleAudioAnalysis;
                messageHandler.OnDanceStyleChange += HandleDanceStyleChange;
                messageHandler.OnReadHighlight += HandleReadHighlight;
                messageHandler.OnReadClear += HandleReadClear;
                messageHandler.OnIdleThought += HandleIdleThought;
            }

            // Avatar
            if (avatarController != null)
                avatarController.OnVrmLoaded += OnVrmLoaded;

            // Window / drag
            var windowCtrl = FindFirstObjectByType<TransparentWindowController>();
            if (windowCtrl != null)
            {
                windowCtrl.OnDragStart += HandleDragStart;
                windowCtrl.OnDragEnd += HandleDragEnd;
                windowCtrl.OnFileDropped += HandleFileDropped;
            }

            // Desktop
            CacheIfNull(ref sleepController);
            CacheIfNull(ref locomotionController);
            CacheIfNull(ref windowSnapper);

            if (windowSnapper != null)
            {
                windowSnapper.OnSittingChanged += HandleSittingChanged;
                windowSnapper.OnFallStarted += HandleFallStarted;
                windowSnapper.OnFallLanded += HandleFallLanded;
            }

            if (locomotionController != null)
            {
                locomotionController.OnWalkStateChanged += HandleWalkStateChanged;
                locomotionController.OnPeekStateChanged += HandlePeekStateChanged;
            }

            // Sleep
            if (sleepController != null)
            {
                sleepController.OnSleepStart += HandleSleepStart;
                sleepController.OnWakeUp += HandleWakeUp;
            }

            // Interaction
            CacheIfNull(ref petDetectionController);
            if (petDetectionController != null)
                petDetectionController.OnPetDetected += HandlePetDetected;

            // System
            CacheIfNull(ref alarmTimerManager);
            if (alarmTimerManager != null)
                alarmTimerManager.OnTimerFired += HandleTimerFired;

            CacheIfNull(ref idleController);
            if (idleController != null)
            {
                idleController.OnIdleStateChanged  += HandleIdleStateChanged;
                idleController.OnSleepStateChanged += HandleSleepStateChanged;
            }
        }

        private void OnDestroy()
        {
            if (messageHandler != null)
            {
                messageHandler.OnSpeakStart -= HandleSpeakStart;
                messageHandler.OnSpeakEnd -= HandleSpeakEnd;
                messageHandler.OnEmotionChange -= HandleEmotionChange;
                messageHandler.OnFaceExpression -= HandleFaceExpression;
                messageHandler.OnModeChange -= HandleModeChange;
                messageHandler.OnSilenceToggle -= HandleSilenceToggle;
                messageHandler.OnAudioAnalysis -= HandleAudioAnalysis;
                messageHandler.OnDanceStyleChange -= HandleDanceStyleChange;
                messageHandler.OnReadHighlight -= HandleReadHighlight;
                messageHandler.OnReadClear -= HandleReadClear;
                messageHandler.OnIdleThought -= HandleIdleThought;
            }

            if (avatarController != null)
                avatarController.OnVrmLoaded -= OnVrmLoaded;

            var windowCtrl = FindFirstObjectByType<TransparentWindowController>();
            if (windowCtrl != null)
            {
                windowCtrl.OnDragStart -= HandleDragStart;
                windowCtrl.OnDragEnd -= HandleDragEnd;
                windowCtrl.OnFileDropped -= HandleFileDropped;
            }

            if (windowSnapper != null)
            {
                windowSnapper.OnSittingChanged -= HandleSittingChanged;
                windowSnapper.OnFallStarted -= HandleFallStarted;
                windowSnapper.OnFallLanded -= HandleFallLanded;
            }

            if (locomotionController != null)
            {
                locomotionController.OnWalkStateChanged -= HandleWalkStateChanged;
                locomotionController.OnPeekStateChanged -= HandlePeekStateChanged;
            }

            if (sleepController != null)
            {
                sleepController.OnSleepStart -= HandleSleepStart;
                sleepController.OnWakeUp -= HandleWakeUp;
            }

            if (petDetectionController != null)
                petDetectionController.OnPetDetected -= HandlePetDetected;

            if (alarmTimerManager != null)
                alarmTimerManager.OnTimerFired -= HandleTimerFired;

            if (idleController != null)
            {
                idleController.OnIdleStateChanged  -= HandleIdleStateChanged;
                idleController.OnSleepStateChanged -= HandleSleepStateChanged;
            }
        }

        // ══════════════════════════════════════════════════════════════
        //  VRM Load — Initialize All Sub-Controllers
        // ══════════════════════════════════════════════════════════════

        private void OnVrmLoaded(UniVRM10.Vrm10Instance vrm)
        {
            Debug.Log("[CompanionManager] VRM loaded, initializing components...");

            try
            {
                _randomMsgTimer = GetRandomMsgInterval();
                var animator = avatarController.GetComponentInChildren<Animator>();

                // Avatar controllers (live on the VRM hierarchy)
                lipSyncController = avatarController.GetComponentInChildren<LipSyncController>();
                emotionController = avatarController.GetComponentInChildren<EmotionController>();
                blinkController = avatarController.GetComponentInChildren<BlinkController>();
                eyeTrackingController = avatarController.GetComponentInChildren<EyeTrackingController>();
                idleAnimationController = avatarController.GetComponentInChildren<IdleAnimationController>();

                // Animation
                idleAnimationController?.Initialize(vrm);
                beatDanceController?.Initialize(vrm);
                vrmaAnimationController?.Initialize(vrm);

                CacheIfNull(ref dragAnimController);
                dragAnimController?.Initialize(vrm);

                CacheIfNull(ref walkAnimController);
                walkAnimController?.Initialize(vrm);

                CacheIfNull(ref dragPoseController);
                dragPoseController?.Initialize(vrm);

                // IK + Accessories
                CacheIfNull(ref ikController);
                ikController?.Initialize(vrm);

                CacheIfNull(ref accessoryManager);
                accessoryManager?.Initialize(vrm);

                // Dance
                CacheIfNull(ref customDanceLoader);
                if (customDanceLoader != null && animator != null)
                    customDanceLoader.Initialize(animator);

                // Interaction
                touchReactionController = avatarController.GetComponentInChildren<TouchReactionController>();
                if (touchReactionController != null)
                {
                    touchReactionController.Initialize(vrm);
                    touchReactionController.OnTouchReaction += HandleTouchReaction;
                }

                CacheIfNull(ref touchSoundHandler);
                CacheIfNull(ref particleEffectHandler);

                CacheIfNull(ref petDetectionController);
                if (petDetectionController != null)
                {
                    var headBone = animator?.GetBoneTransform(HumanBodyBones.Head);
                    if (headBone != null)
                        petDetectionController.SetHeadBone(headBone);
                }

                // Desktop
                CacheIfNull(ref occluderQuadManager);
                CacheIfNull(ref ambientProbe);

                // UI — attach speech bubble to head bone
                CacheIfNull(ref speechBubble);
                if (speechBubble != null)
                {
                    var headBone = animator?.GetBoneTransform(HumanBodyBones.Head);
                    if (headBone != null)
                        speechBubble.SetHeadBone(headBone);
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[CompanionManager] OnVrmLoaded error: {e}");
            }

            // Apply user settings now that VRM is ready
            if (SettingsManager.Instance != null)
            {
                SettingsManager.Instance.InvalidateControllerCache();
                SettingsManager.Instance.ApplyAllSettings();
            }
        }

        // ══════════════════════════════════════════════════════════════
        //  Random Idle Messages
        // ══════════════════════════════════════════════════════════════

        private void Update()
        {
            var settings = Core.SettingsManager.Instance;
            if (settings == null || !settings.data.enableRandomMessages) return;
            if (currentMode != CompanionMode.Idle) return;
            if (isSpeaking || isSilenced) return;
            if (sleepController != null && sleepController.IsSleeping) return;

            _randomMsgTimer -= Time.deltaTime;
            if (_randomMsgTimer <= 0f)
            {
                _randomMsgTimer = GetRandomMsgInterval();
                string context = $"time={System.DateTime.Now:HH:mm}, idle_minutes={(_randomMsgTimer / 60f):F0}";
                messageHandler?.SendRandomPrompt(context);
            }
        }

        private float GetRandomMsgInterval()
        {
            float minutes = 10f;
            if (Core.SettingsManager.Instance != null)
                minutes = Core.SettingsManager.Instance.data.randomMessageIntervalMinutes;
            // Clamp minimum to 1 minute to prevent rapid-fire, add ±20% jitter
            minutes = Mathf.Max(1f, minutes);
            return minutes * 60f * Random.Range(0.8f, 1.2f);
        }

        // ══════════════════════════════════════════════════════════════
        //  Speech & Emotion Handlers
        // ══════════════════════════════════════════════════════════════

        private void HandleSpeakStart(string text)
        {
            isSpeaking = true;
            lipSyncController?.StartSpeaking();
            speechBubble?.ShowText(text);
            discordPresence?.SetState("Talking");
            Debug.Log($"[CompanionManager] Speaking: {text}");
        }

        private void HandleSpeakEnd()
        {
            isSpeaking = false;
            lipSyncController?.StopSpeaking();
            speechBubble?.StartDismissTimer();
            discordPresence?.SetState("Idle");
            Debug.Log("[CompanionManager] Stopped speaking");
        }

        private void HandleEmotionChange(string emotion)
        {
            emotionController?.SetEmotion(emotion);
            Debug.Log($"[CompanionManager] Emotion: {emotion}");
        }

        private void HandleFaceExpression(string name, float intensity)
        {
            emotionController?.SetExpression(name, intensity);
        }

        private void HandleReadHighlight(string sentence)
        {
            speechBubble?.ShowText(sentence);
            Debug.Log($"[CompanionManager] Read highlight: {sentence}");
        }

        private void HandleReadClear()
        {
            speechBubble?.HideNow();
            Debug.Log("[CompanionManager] Read highlight cleared");
        }

        private void HandleIdleThought(string text)
        {
            idleBubbleController?.ShowIdleThought(text);
        }

        // ══════════════════════════════════════════════════════════════
        //  Drag & Touch Handlers
        // ══════════════════════════════════════════════════════════════

        private void HandleDragStart()
        {
            touchSoundHandler?.PlayDragStart();
            windowSnapper?.NotifyDragStart();
            eyeTrackingController?.SetTrackingMode(TrackingMode.LookUp);
            if (dragPoseController != null)
                TransitionAnimation(idleAnimationController, dragPoseController);
            ikController?.SetDragging(true);
        }

        private void HandleDragEnd()
        {
            touchSoundHandler?.PlayDragEnd();
            windowSnapper?.NotifyDragEnd();
            RestoreTrackingMode();
            if (dragPoseController != null)
                TransitionAnimation(dragPoseController, idleAnimationController);
            ikController?.SetDragging(false);
        }

        private void HandleTouchReaction(Vector3 hitPoint, TouchZone zone)
        {
            touchSoundHandler?.PlayTouchSound();
            bool isHeadZone = zone == TouchZone.Head;
            particleEffectHandler?.PlayAtPosition(hitPoint, isHeadZone);
        }

        // ══════════════════════════════════════════════════════════════
        //  Desktop Interaction Handlers
        // ══════════════════════════════════════════════════════════════

        private void HandleSittingChanged(bool sitting)
        {
            Debug.Log($"[CompanionManager] Sitting: {sitting}");
            if (occluderQuadManager != null && windowSnapper != null)
                occluderQuadManager.SetEnabled(sitting, windowSnapper.SittingOnWindowHandle);
            ikController?.SetSitting(sitting);
        }

        private void HandleFallStarted()
        {
            Debug.Log("[CompanionManager] Falling!");
            emotionController?.SetEmotion("surprised");
        }

        private void HandleFallLanded()
        {
            Debug.Log("[CompanionManager] Landed!");
            emotionController?.ClearEmotion();
        }

        private void HandleWalkStateChanged(bool walking)
        {
            Debug.Log($"[CompanionManager] Walking: {walking}");
            if (walking)
                TransitionAnimation(idleAnimationController, walkAnimController);
            else
                TransitionAnimation(walkAnimController, idleAnimationController);
        }

        private void HandlePeekStateChanged(bool peeking)
        {
            Debug.Log($"[CompanionManager] Peeking: {peeking}");
            if (peeking)
            {
                float leanDir = locomotionController.GetWalkDirection() < 0 ? -1f : 1f;
                idleAnimationController?.SetPeekLean(leanDir);
            }
            else
            {
                idleAnimationController?.SetPeekLean(0f);
            }
        }

        // ══════════════════════════════════════════════════════════════
        //  Interaction Handlers (pet, file drop, alarm)
        // ══════════════════════════════════════════════════════════════

        private void HandlePetDetected()
        {
            Debug.Log("[CompanionManager] Pet detected!");
            emotionController?.SetEmotion("happy");
            speechBubble?.ShowText("~♪");
        }

        private void HandleFileDropped(string filePath)
        {
            Debug.Log($"[CompanionManager] File dropped: {filePath}");
            if (filePath.EndsWith(".vrm", System.StringComparison.OrdinalIgnoreCase) && avatarController != null)
                _ = avatarController.LoadVRM(filePath, isAbsolutePath: true);
        }

        private void HandleTimerFired(TimerEntry timer)
        {
            Debug.Log($"[CompanionManager] Timer fired: {timer.label}");
            emotionController?.SetEmotion("surprised");
            speechBubble?.ShowText($"⏰ {timer.label} — Time's up!");
        }

        // ══════════════════════════════════════════════════════════════
        //  Sleep & Idle Handlers
        // ══════════════════════════════════════════════════════════════

        private void HandleSleepStart()
        {
            Debug.Log("[CompanionManager] Sleep started");
            blinkController?.ForceClose();
            idleAnimationController?.SetSleeping(true);
            eyeTrackingController?.SetTrackingMode(TrackingMode.Disabled);
        }

        private void HandleWakeUp()
        {
            Debug.Log("[CompanionManager] Waking up");
            blinkController?.ForceOpen();
            blinkController?.TriggerRapidBlinks(3);
            idleAnimationController?.SetSleeping(false);
            RestoreTrackingMode();
        }

        private void HandleIdleStateChanged(bool idle)
        {
            Debug.Log($"[CompanionManager] Idle state: {idle}");
            idleBubbleController?.SetIdleMode(idle);
            if (idle) discordPresence?.SetState("Idle");
        }

        private void HandleSleepStateChanged(bool sleeping)
        {
            Debug.Log($"[CompanionManager] Sleep state (IdleController): {sleeping}");
            if (sleeping)
            {
                blinkController?.ForceClose();
                idleAnimationController?.SetSleeping(true);
                discordPresence?.SetState("Sleeping");
            }
            else
            {
                blinkController?.ForceOpen();
                blinkController?.TriggerRapidBlinks(2);
                idleAnimationController?.SetSleeping(false);
                RestoreTrackingMode();
                discordPresence?.SetState("Idle");
            }
        }

        // ══════════════════════════════════════════════════════════════
        //  Mode & Dance
        // ══════════════════════════════════════════════════════════════

        private void RestoreTrackingMode()
        {
            if (sleepController != null && sleepController.IsSleeping)
                eyeTrackingController?.SetTrackingMode(TrackingMode.Disabled);
            else if (currentMode == CompanionMode.Dance)
                eyeTrackingController?.SetTrackingMode(TrackingMode.Reduced);
            else
                eyeTrackingController?.SetTrackingMode(TrackingMode.Normal);
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
                    vrmaAnimationController?.Stop();
                    // beatDanceController cleanup handled by blend completion
                    break;
            }

            // Handle mode enter
            switch (mode)
            {
                case CompanionMode.Active:
                    RestoreTrackingMode();
                    TransitionAnimation(beatDanceController, idleAnimationController);
                    break;

                case CompanionMode.Idle:
                    RestoreTrackingMode();
                    TransitionAnimation(beatDanceController, idleAnimationController);
                    emotionController?.ClearEmotion();
                    break;

                case CompanionMode.Dance:
                    // Feature #10: Reduced tracking during dance
                    eyeTrackingController?.SetTrackingMode(TrackingMode.Reduced);
                    discordPresence?.SetState("Dancing");
                    if (currentDanceStyle == DanceStyle.Procedural)
                    {
                        beatDanceController?.StartDancing();
                        TransitionAnimation(idleAnimationController, beatDanceController);
                    }
                    else
                    {
                        idleAnimationController?.SetEnabled(false);
                    }
                    break;
            }

            Debug.Log($"[CompanionManager] Mode changed: {previousMode} → {mode}");
        }

        /// <summary>
        /// Smooth crossfade between animation controllers, with hard-switch fallback.
        /// </summary>
        private void TransitionAnimation(IBlendableAnimation from, IBlendableAnimation to, float duration = -1f)
        {
            if (animationBlendController != null && from != null && to != null)
            {
                animationBlendController.Crossfade(from, to, duration);
            }
            else
            {
                from?.SetBlendActive(false);
                from?.SetBlendWeight(0f);
                to?.SetBlendActive(true);
                to?.SetBlendWeight(1f);
            }
        }

        private void HandleSilenceToggle(bool silenced)
        {
            isSilenced = silenced;
            Debug.Log($"[CompanionManager] Silenced: {silenced}");
        }

        private void HandleAudioAnalysis(float bass, float mid, float high, bool isBeat)
        {
            if (currentMode != CompanionMode.Dance) return;
            if (sleepController != null && sleepController.IsSleeping) return;

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
            var previousStyle = currentDanceStyle;
            currentDanceStyle = style;

            // Read transition duration from settings
            float transitionDuration = 0.6f;
            if (Core.SettingsManager.Instance != null)
                transitionDuration = Core.SettingsManager.Instance.data.danceTransitionSpeed;

            if (currentMode == CompanionMode.Dance)
            {
                switch (style)
                {
                    case DanceStyle.None:
                        // Crossfade out of whatever is playing → idle
                        if (previousStyle == DanceStyle.Procedural)
                            TransitionAnimation(beatDanceController, idleAnimationController, transitionDuration);
                        else
                        {
                            vrmaAnimationController?.Stop();
                            idleAnimationController?.SetBlendActive(true);
                            idleAnimationController?.SetBlendWeight(1f);
                        }
                        break;

                    case DanceStyle.Procedural:
                        vrmaAnimationController?.Stop();
                        beatDanceController?.StartDancing();
                        // Smooth crossfade: if coming from idle or None, blend in
                        if (previousStyle == DanceStyle.None || previousStyle == DanceStyle.Procedural)
                            TransitionAnimation(idleAnimationController, beatDanceController, transitionDuration);
                        else
                        {
                            // From VRMA → procedural: just blend from idle since VRMA already stopped
                            TransitionAnimation(idleAnimationController, beatDanceController, transitionDuration);
                        }
                        break;

                    case DanceStyle.ShikanokoDance:
                        // Crossfade out of procedural dance if active
                        if (previousStyle == DanceStyle.Procedural && beatDanceController != null)
                        {
                            TransitionAnimation(beatDanceController, idleAnimationController, transitionDuration);
                            // After blend completes, idle will be at 1.0 — VRMA takes over bones directly
                        }
                        else
                        {
                            beatDanceController?.StopDancing();
                            if (animationBlendController != null)
                            {
                                beatDanceController?.SetBlendActive(false);
                                beatDanceController?.SetBlendWeight(0f);
                            }
                        }
                        _vrmaAudioPaused = true; // Start paused until music detected
                        _ = PlayVrmaAnimation("shikanoko_dance.vrma");
                        break;
                }
            }

            Debug.Log($"[CompanionManager] Dance style: {previousStyle} → {style}");
        }

        // ══════════════════════════════════════════════════════════════
        //  Public API
        // ══════════════════════════════════════════════════════════════

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

        private async System.Threading.Tasks.Task PlayVrmaAnimation(string fileName)
        {
            if (vrmaAnimationController == null)
            {
                Debug.LogWarning("[CompanionManager] VrmaAnimationController not assigned.");
                return;
            }
            await vrmaAnimationController.LoadAndPlay(fileName, destroyCancellationToken);
            if (_vrmaAudioPaused)
            {
                vrmaAnimationController.Pause();
            }
        }

        // ══════════════════════════════════════════════════════════════
        //  Helpers
        // ══════════════════════════════════════════════════════════════

        private void OnApplicationQuit()
        {
            // Tell the Python backend to shut down when the Unity app closes
            if (webSocketClient != null)
            {
                webSocketClient.Send(MessageTypes.SHUTDOWN);
            }
        }

        private void CacheIfNull<T>(ref T field) where T : Component
        {
            if (field == null)
                field = FindFirstObjectByType<T>();
        }
    }
}
