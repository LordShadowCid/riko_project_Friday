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

        [Header("Animation Blending")]
        [SerializeField] private AnimationBlendController animationBlendController;

        [Header("Interaction")]
        [SerializeField] private TouchReactionController touchReactionController;
        [SerializeField] private TouchSoundHandler touchSoundHandler;
        [SerializeField] private ParticleEffectHandler particleEffectHandler;

        [Header("UI")]
        [SerializeField] private SpeechBubble speechBubble;

        [Header("Phase 5: Drag + Effects")]
        [SerializeField] private DragAnimationController dragAnimController;

        [Header("Phase 6: System")]
        [SerializeField] private SleepController sleepController;

        [Header("Sprint 3: Walk + Drag Pose + Pet")]
        [SerializeField] private WalkAnimationController walkAnimController;
        [SerializeField] private DragPoseController dragPoseController;
        [SerializeField] private PetDetectionController petDetectionController;

        [Header("Sprint 4: Desktop Integration")]
        [SerializeField] private OccluderQuadManager occluderQuadManager;
        [SerializeField] private DesktopAmbientProbe ambientProbe;

        [Header("Sprint 5: Dance Expansion + Alarms")]
        [SerializeField] private AlarmTimerManager alarmTimerManager;
        [SerializeField] private CustomDanceLoader customDanceLoader;

        [Header("Phase 8: Desktop Interaction")]
        [SerializeField] private DesktopLocomotionController locomotionController;
        [SerializeField] private WindowSnapper windowSnapper;

        [Header("State")]
        [SerializeField] private CompanionMode currentMode = CompanionMode.Idle;
        [SerializeField] private DanceStyle currentDanceStyle = DanceStyle.None;
        [SerializeField] private bool isSilenced;
        [SerializeField] private bool isSpeaking;

        // Track whether VRMA animation is audio-paused (no music playing)
        private bool _vrmaAudioPaused;

        // Feature #12: Random message timer
        private float _randomMsgTimer;

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

            // Wire Phase 5 drag events
            var windowCtrl = FindFirstObjectByType<TransparentWindowController>();
            if (windowCtrl != null)
            {
                windowCtrl.OnDragStart += HandleDragStart;
                windowCtrl.OnDragEnd += HandleDragEnd;
                // Feature #13: Wire file drop
                windowCtrl.OnFileDropped += HandleFileDropped;
            }

            // Cache Phase 6 sleep controller
            if (sleepController == null)
                sleepController = FindFirstObjectByType<SleepController>();

            // Cache Phase 8 desktop interaction controllers
            if (locomotionController == null)
                locomotionController = FindFirstObjectByType<DesktopLocomotionController>();
            if (windowSnapper == null)
                windowSnapper = FindFirstObjectByType<WindowSnapper>();

            // Wire falling events for animation feedback
            if (windowSnapper != null)
            {
                windowSnapper.OnSittingChanged += HandleSittingChanged;
                windowSnapper.OnFallStarted += HandleFallStarted;
                windowSnapper.OnFallLanded += HandleFallLanded;
            }

            // Wire walk state events
            if (locomotionController != null)
            {
                locomotionController.OnWalkStateChanged += HandleWalkStateChanged;
            }

            // Feature #6: Wire sleep events
            if (sleepController != null)
            {
                sleepController.OnSleepStart += HandleSleepStart;
                sleepController.OnWakeUp += HandleWakeUp;
            }

            // Feature #8: Wire peek events
            if (locomotionController != null)
            {
                locomotionController.OnPeekStateChanged += HandlePeekStateChanged;
            }

            // Feature #11: Wire pet detection
            if (petDetectionController == null)
                petDetectionController = FindFirstObjectByType<PetDetectionController>();
            if (petDetectionController != null)
                petDetectionController.OnPetDetected += HandlePetDetected;

            // Feature #15: Wire alarm timer
            if (alarmTimerManager == null)
                alarmTimerManager = FindFirstObjectByType<AlarmTimerManager>();
            if (alarmTimerManager != null)
                alarmTimerManager.OnTimerFired += HandleTimerFired;
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
            }

            // Feature #6: Unwire sleep events
            if (sleepController != null)
            {
                sleepController.OnSleepStart -= HandleSleepStart;
                sleepController.OnWakeUp -= HandleWakeUp;
            }

            // Feature #8: Unwire peek events
            if (locomotionController != null)
            {
                locomotionController.OnPeekStateChanged -= HandlePeekStateChanged;
            }

            // Feature #11: Unwire pet detection
            if (petDetectionController != null)
                petDetectionController.OnPetDetected -= HandlePetDetected;

            // Feature #15: Unwire alarm timer
            if (alarmTimerManager != null)
                alarmTimerManager.OnTimerFired -= HandleTimerFired;
        }

        private void OnVrmLoaded(UniVRM10.Vrm10Instance vrm)
        {
            Debug.Log("[CompanionManager] VRM loaded, initializing components...");

            // Feature #12: Reset random message timer
            _randomMsgTimer = GetRandomMsgInterval();

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
                touchReactionController.OnTouchReaction += HandleTouchReaction;
            }

            // Initialize Phase 5 handlers
            if (touchSoundHandler == null)
                touchSoundHandler = FindFirstObjectByType<TouchSoundHandler>();
            if (particleEffectHandler == null)
                particleEffectHandler = FindFirstObjectByType<ParticleEffectHandler>();

            // Initialize drag animation (Phase 5)
            if (dragAnimController == null)
                dragAnimController = FindFirstObjectByType<DragAnimationController>();
            if (dragAnimController != null)
                dragAnimController.Initialize(vrm);

            // Sprint 3: Initialize walk animation controller
            if (walkAnimController == null)
                walkAnimController = FindFirstObjectByType<WalkAnimationController>();
            if (walkAnimController != null)
                walkAnimController.Initialize(vrm);

            // Sprint 3: Initialize drag pose controller
            if (dragPoseController == null)
                dragPoseController = FindFirstObjectByType<DragPoseController>();
            if (dragPoseController != null)
                dragPoseController.Initialize(vrm);

            // Sprint 3: Initialize pet detection
            if (petDetectionController == null)
                petDetectionController = FindFirstObjectByType<PetDetectionController>();
            if (petDetectionController != null)
            {
                var headBoneForPet = avatarController.GetComponentInChildren<Animator>()
                    ?.GetBoneTransform(HumanBodyBones.Head);
                if (headBoneForPet != null)
                    petDetectionController.SetHeadBone(headBoneForPet);
            }

            // Sprint 4: Initialize occluder quad manager
            if (occluderQuadManager == null)
                occluderQuadManager = FindFirstObjectByType<OccluderQuadManager>();

            // Sprint 4: Initialize ambient probe
            if (ambientProbe == null)
                ambientProbe = FindFirstObjectByType<DesktopAmbientProbe>();

            // Sprint 5: Initialize custom dance loader
            if (customDanceLoader == null)
                customDanceLoader = FindFirstObjectByType<CustomDanceLoader>();
            if (customDanceLoader != null)
            {
                var animator = avatarController.GetComponentInChildren<Animator>();
                if (animator != null)
                    customDanceLoader.Initialize(animator);
            }

            // Wire speech bubble to head bone
            if (speechBubble == null)
                speechBubble = FindFirstObjectByType<SpeechBubble>();
            if (speechBubble != null)
            {
                var headBone = avatarController.GetComponentInChildren<Animator>()
                    ?.GetBoneTransform(HumanBodyBones.Head);
                if (headBone != null)
                    speechBubble.SetHeadBone(headBone);
            }

            // Apply user settings to all controllers now that VRM is ready
            if (SettingsManager.Instance != null)
                SettingsManager.Instance.ApplyAllSettings();
        }

        // ── Feature #12: Random Messages ─────────────────────────────

        private void Update()
        {
            // Feature #12: Fire random messages when idle and enabled
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
            // Add ±20% jitter
            return minutes * 60f * Random.Range(0.8f, 1.2f);
        }

        // ── Event Handlers ──────────────────────────────────────────

        private void HandleSpeakStart(string text)
        {
            isSpeaking = true;
            lipSyncController?.StartSpeaking();
            speechBubble?.ShowText(text);
            Debug.Log($"[CompanionManager] Speaking: {text}");
        }

        private void HandleSpeakEnd()
        {
            isSpeaking = false;
            lipSyncController?.StopSpeaking();
            speechBubble?.StartDismissTimer();
            Debug.Log("[CompanionManager] Stopped speaking");
        }

        private void HandleEmotionChange(string emotion)
        {
            emotionController?.SetEmotion(emotion);
            Debug.Log($"[CompanionManager] Emotion: {emotion}");
        }

        // ── Phase 5: Drag & Touch Handlers ──────────────────────────

        private void HandleDragStart()
        {
            touchSoundHandler?.PlayDragStart();
            // Feature #5: Notify WindowSnapper of drag for sit guard timer
            windowSnapper?.NotifyDragStart();
            // Feature #10: Switch to LookUp tracking during drag
            eyeTrackingController?.SetTrackingMode(TrackingMode.LookUp);
            // Feature #4: Blend in drag pose
            if (dragPoseController != null)
                TransitionAnimation(idleAnimationController, dragPoseController);
        }

        private void HandleDragEnd()
        {
            touchSoundHandler?.PlayDragEnd();
            // Feature #5: Notify WindowSnapper of drag end (may trigger TrySitOnNearestWindow)
            windowSnapper?.NotifyDragEnd();
            // Feature #10: Restore tracking mode based on current state
            RestoreTrackingMode();
            // Feature #4: Blend out drag pose
            if (dragPoseController != null)
                TransitionAnimation(dragPoseController, idleAnimationController);
        }

        private void HandleTouchReaction(Vector3 hitPoint, bool isHead)
        {
            touchSoundHandler?.PlayTouchSound();
            particleEffectHandler?.PlayAtPosition(hitPoint, isHead);
        }

        // ── Phase 8: Desktop Interaction Handlers ────────────────────

        private void HandleSittingChanged(bool sitting)
        {
            Debug.Log($"[CompanionManager] Sitting: {sitting}");
            // Feature #1: Enable/disable occluder quads when sitting on windows
            if (occluderQuadManager != null && windowSnapper != null)
                occluderQuadManager.SetEnabled(sitting, windowSnapper.SittingOnWindowHandle);
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
            // Feature #3: Blend walk animation in/out
            if (walking)
                TransitionAnimation(idleAnimationController, walkAnimController);
            else
                TransitionAnimation(walkAnimController, idleAnimationController);
        }

        // ── Feature #8: Peek/Hide Handlers ──────────────────────────

        private void HandlePeekStateChanged(bool peeking)
        {
            Debug.Log($"[CompanionManager] Peeking: {peeking}");
            if (peeking)
            {
                // Determine peek direction: check if window is near left or right edge
                float leanDir = locomotionController.GetWalkDirection() < 0 ? -1f : 1f;
                idleAnimationController?.SetPeekLean(leanDir);
            }
            else
            {
                idleAnimationController?.SetPeekLean(0f);
            }
        }

        // ── Feature #11: Pet Detection Handler ──────────────────────

        private void HandlePetDetected()
        {
            Debug.Log("[CompanionManager] Pet detected!");
            emotionController?.SetEmotion("happy");
            speechBubble?.ShowText("~♪");
        }

        // ── Feature #13: File Drop Handler ──────────────────────────

        private void HandleFileDropped(string filePath)
        {
            Debug.Log($"[CompanionManager] File dropped: {filePath}");
            if (filePath.EndsWith(".vrm", System.StringComparison.OrdinalIgnoreCase) && avatarController != null)
                _ = avatarController.LoadVRM(filePath, isAbsolutePath: true);
        }

        // ── Feature #15: Alarm/Timer Handler ────────────────────────

        private void HandleTimerFired(TimerEntry timer)
        {
            Debug.Log($"[CompanionManager] Timer fired: {timer.label}");
            emotionController?.SetEmotion("surprised");
            speechBubble?.ShowText($"⏰ {timer.label} — Time's up!");
        }

        // ── Feature #6: Sleep Handlers ──────────────────────────────

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

        /// <summary>
        /// Feature #10: Restore tracking mode based on current companion state.
        /// </summary>
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
                    eyeTrackingController?.SetEnabled(true);
                    eyeTrackingController?.SetTrackingMode(TrackingMode.Normal);
                    TransitionAnimation(beatDanceController, idleAnimationController);
                    break;

                case CompanionMode.Idle:
                    eyeTrackingController?.SetEnabled(true);
                    eyeTrackingController?.SetTrackingMode(TrackingMode.Normal);
                    TransitionAnimation(beatDanceController, idleAnimationController);
                    emotionController?.ClearEmotion();
                    break;

                case CompanionMode.Dance:
                    // Feature #10: Reduced tracking during dance
                    eyeTrackingController?.SetTrackingMode(TrackingMode.Reduced);
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
