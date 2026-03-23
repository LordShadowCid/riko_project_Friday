# Mate-Engine vs Annabeth — Feature Comparison & Adoption Plan

## Complete Feature Matrix

| # | Feature | Mate-Engine | Annabeth | Gap | Priority | Notes |
|---|---------|:-----------:|:--------:|:---:|:--------:|-------|
| **— WINDOW / DESKTOP —** | | | | | | |
| 1 | Transparent window (per-pixel alpha) | ✅ `AvatarWindowHandler` | ✅ `TransparentWindowController` | **DONE** | — | Just implemented, DWM + WS_EX_LAYERED |
| 2 | Click-through on empty areas | ✅ dynamic per-frame | ✅ mesh-bounds raycast | **DONE** | — | Dynamic WS_EX_TRANSPARENT toggle |
| 3 | Left-click drag to move window | ✅ | ✅ | **DONE** | — | |
| 4 | Always-on-top toggle | ✅ `SetWindowPos TOPMOST` | ✅ `SetTopmost()` | **DONE** | — | |
| 5 | Edge snapping (magnetic) | ✅ | ✅ `WindowSnapper` 20px threshold | **DONE** | — | |
| 6 | Window sitting (on other app windows) | ✅ `AvatarTaskbarController` | ❌ | **GAP** | Medium | Detect top window edges, sit avatar on them |
| 7 | Taskbar sitting | ✅ `AvatarTaskbarController` | ✅ partial `WindowSnapper` | **PARTIAL** | Low | We snap to bottom-right, but no "sitting on taskbar" with animation |
| 8 | Hide from taskbar (system tray icon) | ✅ `SystemTray.cs`, `RemoveTaskbarApp.cs` | ❌ | **GAP** | Medium | WS_EX_TOOLWINDOW hides from taskbar already, but no tray icon/menu |
| 9 | System tray icon with context menu | ✅ full tray icon + settings | ❌ | **GAP** | Medium | Right-click tray icon: show/hide, settings, quit |
| 10 | Big Screen mode | ✅ `AvatarBigScreenHandler` | ❌ | **GAP** | Low | Full-screen wallpaper mode |
| 11 | Screensaver mode | ✅ `AvatarBigScreenScreenSaver` | ❌ | **GAP** | Low | Activates after idle timeout |
| 12 | FPS limiter | ✅ `FPSLimiter.cs` | ❌ | **GAP** | Medium | `Application.targetFrameRate` — trivial to add |
| 13 | Move to primary screen | ✅ `MoveToPrimaryScreen.cs` | ❌ | **GAP** | Low | Multi-monitor support |
| 14 | Start with PC | ✅ `SystemStartHandler.cs` | ❌ | **GAP** | Low | Registry/startup folder shortcut |
| 15 | Anti-cheat safe | ✅ (uses no hooks) | ✅ (no hooks either) | **DONE** | — | |
| 16 | Compatible with games (no overlay) | ✅ | ✅ (standalone window) | **DONE** | — | |
| **— CHARACTER / VRM —** | | | | | | |
| 17 | Load custom VRM at runtime | ✅ `VRMLoader.cs` + file picker | ✅ `AvatarController.cs` hardcoded path | **PARTIAL** | **HIGH** | We load VRM but path is hardcoded — need file picker UI |
| 18 | VRM model library / switcher | ✅ `AvatarLibraryMenu.cs` | ❌ | **GAP** | **HIGH** | Scan folder, list available VRMs, switch with one click |
| 19 | Multiple avatars (up to 9) | ✅ `LaunchMateEngineInstance.cs` | ❌ | **GAP** | Low | Multi-instance spawning |
| 20 | Chibi mode (scale toggle) | ✅ `ChibiToggle.cs` + `AvatarScaleController` | ❌ | **GAP** | Medium | Simple scale factor toggle |
| 21 | Blendshape editor UI | ✅ `BlendshapeManager.cs`, `BlendshapeUIBlock` | ❌ | **GAP** | Low | Runtime blendshape slider UI |
| 22 | Custom shaders (MToon, Mochie) | ✅ extensive shader lib | ✅ MToon via UniVRM | **DONE** | — | |
| 23 | Accessories system | ✅ `AccessoiresHandler.cs` | ❌ | **GAP** | Low | Attach props to avatar bones |
| 24 | Cosmetic items (halos, overlays) | ✅ DLC items | ❌ | **GAP** | Low | |
| **— ANIMATION —** | | | | | | |
| 25 | Idle animation (breathing, sway) | ✅ `AvatarSwayController` | ✅ `IdleAnimationController` | **DONE** | — | Both do breathing + sway |
| 26 | Dragging animation (float while held) | ✅ animator state | ❌ | **GAP** | **HIGH** | Play special animation while being dragged |
| 27 | Smooth animation transitions | ✅ blends + `BlendTreeLooper` | ✅ `AnimationBlendController` | **DONE** | — | |
| 28 | VRMA file playback | ✅ (via custom dance player) | ✅ `VrmaAnimationController` | **DONE** | — | |
| 29 | Dance to music (audio-reactive) | ✅ `AvatarDancePlayer` + sync | ✅ `BeatDanceController` | **DONE** | — | We have procedural + VRMA |
| 30 | MMD music animation player | ✅ `MATE ENGINE - Custom Dance Player` | ❌ | **GAP** | Low | Import .vmd dance files |
| 31 | Dance sync (multiple avatars) | ✅ `AvatarDanceSync` | ❌ | **GAP** | Low | Requires multi-avatar first |
| 32 | Masculine animations | ✅ separate male anim set | ❌ | **GAP** | Low | Gender-specific animation selection |
| 33 | Inverse kinematics (IK) | ✅ `IKFix.cs` + `HandHolder` | ❌ | **GAP** | Medium | IK for feet/hands — needed for sitting poses |
| 34 | Locomotion / walking | ✅ `AvatarLocomotionController` | ❌ | **GAP** | Medium | Walk across screen autonomously |
| 35 | Expression based on movement | ✅ animator-driven | ❌ | **GAP** | Low | Happy while dancing, surprised while dragged |
| **— TRACKING —** | | | | | | |
| 36 | Eye tracking (follow cursor) | ✅ `AvatarMouseTracking` | ✅ `EyeTrackingController` | **DONE** | — | |
| 37 | Head tracking (follow cursor) | ✅ spine+head tracking | ✅ `IdleAnimationController` lean | **DONE** | — | |
| 38 | Spine tracking | ✅ upper spine toward mouse | ✅ `IdleAnimationController` lean | **DONE** | — | |
| **— INTERACTION —** | | | | | | |
| 39 | Touch regions (face/head/body) | ✅ `AvatarBigScreenTouchHandler` | ✅ `TouchReactionController` | **DONE** | — | Head zone vs body zone |
| 40 | Touch sound effects | ✅ `AvatarDragSoundHandler` | ❌ | **GAP** | Medium | Play cute sounds on touch/drag |
| 41 | Avatar SFX (general) | ✅ `PetVoiceReactionHandler` | ❌ | **GAP** | Medium | Reaction sounds beyond just touch |
| 42 | Particle effects (hearts, sparkles) | ✅ `AvatarParticleHandler` | ❌ | **GAP** | Medium | Spawn particles on touch/reactions |
| 43 | Food system | ✅ `AvatarFoodController` | ❌ | **GAP** | Low | Feed the avatar — novelty feature |
| **— AI / LLM —** | | | | | | |
| 44 | AI chat (text) | ✅ `LLMUnity` (built-in Qwen 2.5 1.5b) | ✅ **Ollama + llama3.1-8b** | **OURS BETTER** | — | We use much better model + self-hosted |
| 45 | AI chat with markdown | ✅ `MarkdownTextAutoConverter` | ❌ (Python only, no Unity display) | **GAP** | Low | Render markdown in Unity UI |
| 46 | AI system prompt per character | ✅ `AISystemPromptBinder` | ✅ `character_config.yaml` | **DONE** | — | |
| 47 | AI random messages (event-based) | ✅ `AvatarRandomMessages` (Steam) | ❌ | **GAP** | Medium | LLM generates messages on drag, dance, sit events |
| 48 | AI API functions (tool use) | ❌ (listed as missing) | ✅ **tools: web_search, memory, time, personality** | **OURS BETTER** | — | We have tool calling |
| 49 | Delete AI history | ✅ `DeleteAIHistory.cs` | ✅ via chat_history.json | **DONE** | — | |
| **— VOICE / TTS —** | | | | | | |
| 50 | AI voice (TTS) | ❌ (listed as missing) | ✅ **GPT-SoVITS self-hosted** | **OURS BETTER** | — | Full TTS pipeline |
| 51 | Voice packs | ✅ `MEVoicePack.cs` (pre-recorded SFX) | ❌ | **GAP** | Low | Pre-recorded reaction voices |
| 52 | Lip sync | ❌ (no TTS = no lip sync) | ✅ `LipSyncController` (vowel cycling) | **OURS BETTER** | — | |
| **— ASR / INPUT —** | | | | | | |
| 53 | Speech recognition (ASR) | ❌ | ✅ **Faster-Whisper** | **OURS BETTER** | — | Real-time voice input |
| 54 | Read-aloud (screen reader) | ❌ | ✅ **read_aloud pipeline** | **OURS BETTER** | — | Reads selected text / clipboard aloud |
| 55 | Speaker identification | ❌ | ✅ **resemblyzer speaker profiles** | **OURS BETTER** | — | |
| **— UI / SETTINGS —** | | | | | | |
| 56 | Settings menu (right-click / M key) | ✅ `AvatarSettingsMenu` + Tasty Pie Menu | ❌ | **GAP** | **HIGH** | Critical — need a way to access settings in standalone |
| 57 | Pie menu (radial context menu) | ✅ `Tasty Pie Menu` library | ❌ | **GAP** | **HIGH** | Radial menu on right-click — perfect for desktop companion |
| 58 | Settings: sliders, toggles, buttons | ✅ `SettingsHandler*` (7 files) | ❌ | **GAP** | **HIGH** | Scale, opacity, FPS, animations, AI, etc. |
| 59 | Theme manager (color customization) | ✅ `ThemeManager.cs`, `UIThemeApplier` | ❌ | **GAP** | Low | |
| 60 | Debug overlay | ✅ debugging menu | ✅ `DebugOverlay` (F1 key) | **DONE** | — | |
| 61 | Key binding system | ✅ `KeyBindHandler.cs` | ✅ `HotkeyManager` | **DONE** | — | |
| 62 | Screenshot handler | ✅ `MateScreenshotHandler.cs` | ❌ | **GAP** | Low | |
| **— VISUAL QUALITY —** | | | | | | |
| 63 | Post-processing Bloom | ✅ | ❌ (disabled for transparency) | **N/A** | — | Can't use with transparent window |
| 64 | Post-processing AO | ✅ | ❌ (disabled for transparency) | **N/A** | — | Same — incompatible with alpha |
| 65 | MSAA x8 | ✅ | ❌ (disabled for transparency) | **N/A** | — | May cause edge fringing |
| 66 | Desktop ambient probe | ✅ `DesktopAmbientProbe.cs` | ❌ | **GAP** | Medium | Sample desktop colors for lighting match |
| **— MODDING / SDK —** | | | | | | |
| 67 | Mod support (.ME format) | ✅ `MEModHandler`, `MEModLoader` | ❌ | **GAP** | Low | Custom mod packages |
| 68 | Built-in SDK | ✅ `MATE ENGINE - Mod SDK` | ❌ | **GAP** | Low | |
| 69 | Animation modding | ✅ | ✅ (VRMA files in StreamingAssets) | **DONE** | — | |
| 70 | Steam Workshop | ✅ `SteamWorkshopHandler` | ❌ | **N/A** | — | Steam-specific, not applicable |
| **— MISC —** | | | | | | |
| 71 | Alarm / Timer | ✅ `AvatarBigScreenTimer` | ❌ | **GAP** | Medium | Set a timer, avatar alerts you |
| 72 | Sleep mode | ✅ `AvatarSleepController` | ❌ | **GAP** | Medium | Reduce activity after PC idle |
| 73 | Discord Rich Presence | ✅ `DiscordPresence.cs` | ❌ | **GAP** | Low | |
| 74 | Minecraft integration | ✅ `AvatarMinecraftMessages` | ❌ | **GAP** | Low | Read Minecraft chat log |
| 75 | Multi-language UI | ✅ `LanguageDropdownHandler` + Localization | ❌ | **GAP** | Low | |
| 76 | GC / Memory optimizer | ✅ `GCCollect.cs`, `MemoryTrim.cs` | ❌ | **GAP** | Medium | Periodic GC for long-running process |
| 77 | Gravity controller | ✅ `AvatarGravityController` | ❌ | **GAP** | Low | Physics-based fall when not held |
| 78 | Character bubble (speech/thought) | ✅ `AvatarBubbleHandler` | ❌ | **GAP** | **HIGH** | Show AI text in a speech bubble above avatar |
| 79 | Save/Load settings | ✅ `SaveLoadHandler.cs` | ❌ | **GAP** | **HIGH** | Persist user preferences |

---

## Summary Scorecard

| Category | Mate-Engine | Annabeth | Winner |
|----------|:-----------:|:--------:|:------:|
| **Window / Desktop** | 16 features | 6 done + 10 gaps | Mate-Engine |
| **Character / VRM** | 8 features | 3 done + 5 gaps | Mate-Engine |
| **Animation** | 11 features | 5 done + 6 gaps | Mate-Engine |
| **Tracking** | 3 features | 3 done | Tie |
| **Interaction** | 5 features | 1 done + 4 gaps | Mate-Engine |
| **AI / LLM** | 6 features | 5 done (3 ours better) | **Annabeth** |
| **Voice / TTS** | 3 features | 2 ours better | **Annabeth** |
| **ASR / Input** | 0 features | 3 ours better | **Annabeth** |
| **UI / Settings** | 7 features | 2 done + 5 gaps | Mate-Engine |
| **Visual Quality** | 4 features | 1 gap (rest N/A) | Mate-Engine |
| **Modding** | 4 features | 1 done + 3 gaps | Mate-Engine |
| **Misc** | 9 features | 0 done + 9 gaps | Mate-Engine |

**Annabeth's unique advantages (Mate-Engine doesn't have):**
- Self-hosted LLM (Ollama + 8B model vs Mate-Engine's tiny 1.5b)  
- TTS voice synthesis (GPT-SoVITS — Mate-Engine has NO AI voice)
- Real-time ASR (Faster-Whisper speech recognition)
- Speaker identification (resemblyzer profiles)
- Read-aloud pipeline (screen reader)
- LLM tool calling (web search, memory, time, personality adjustment)
- Full conversational pipeline (ASR → LLM → TTS → Avatar)

---

## Recommended Features to Adopt (Prioritized)

### MUST HAVE — Core desktop companion experience
1. **Settings menu / Pie menu** (#56-58) — Without this, users can't configure anything in standalone builds
2. **Character speech bubble** (#78) — Show AI responses visually above the avatar  
3. **VRM file picker / model library** (#17-18) — Easy character swapping is fundamental
4. **Save/Load settings** (#79) — Persist preferences between sessions
5. **Dragging animation** (#26) — Play a float/held animation while dragging
6. **FPS limiter** (#12) — Essential for a long-running desktop process

### SHOULD HAVE — Polish for real daily use
7. **System tray icon** (#8-9) — Proper background app behavior
8. **Chibi mode** (#20) — Fun and practical (smaller footprint)
9. **Touch/drag sound effects** (#40-41) — Audio feedback on interaction
10. **Particle effects** (#42) — Hearts/sparkles on touch
11. **Desktop ambient probe** (#66) — Light avatar to match desktop colors
12. **Timer/Alarm** (#71) — Practical utility feature
13. **Sleep mode** (#72) — Reduce resources when PC is idle
14. **Memory optimizer** (#76) — GC control for long-running process
15. **AI random messages** (#47) — LLM generates contextual messages on events
16. **IK system** (#33) — Needed for sitting poses, hand placement
17. **Locomotion** (#34) — Walking across screen

### NICE TO HAVE — Advanced features for later
18. **Window sitting** (#6) — Sit on top of other app windows
19. **Big screen mode** (#10) — Full wallpaper mode
20. **Screensaver** (#11) — Activate after idle
21. **Start with PC** (#14) — Auto-launch
22. **Multiple avatars** (#19) — Support multiple instances
23. **Blendshape editor** (#21) — Runtime expression tweaking
24. **Gravity system** (#77) — Physics fall when released
25. **Mod support** (#67-68) — Custom mod packages
26. **Discord presence** (#73) — Show in Discord status
27. **Expression on movement** (#35) — Auto-emotion during actions

---

## Phased Implementation Plan

### Phase 1: Essential UI & Settings (Makes it usable as daily desktop companion)
**Scripts to create:**
- `UI/PieMenu.cs` — Radial context menu on right-click (based on Tasty Pie Menu concept)
- `UI/SettingsPanel.cs` — Settings panel with sliders/toggles/buttons
- `UI/SpeechBubble.cs` — World-space Canvas text bubble above avatar's head
- `Core/SettingsManager.cs` — Save/Load settings to JSON (PlayerPrefs or file)
- `Core/FPSController.cs` — Application.targetFrameRate control

**Changes to existing:**
- `CompanionManager.cs` — Route LLM responses to SpeechBubble
- `MessageHandler.cs` — Add bubble text event

**Estimated scope:** 5 new scripts, 2 modifications
**Dependencies:** None — can start immediately

### Phase 2: Character Management & Dragging (Core character experience)
**Scripts to create:**
- `Avatar/VrmFilePicker.cs` — Runtime file browser for VRM selection 
- `Avatar/VrmModelLibrary.cs` — Scan models folder, show thumbnail list, swap VRM
- `Avatar/DragAnimationController.cs` — Special animation while being held/dragged
- `Avatar/ChibiController.cs` — Scale toggle with smooth transition

**Changes to existing:**
- `AvatarController.cs` — Dynamic VRM path, reload support
- `TransparentWindowController.cs` — Hook dragging state to DragAnimationController
- `CompanionManager.cs` — Wire up new controllers

**Estimated scope:** 4 new scripts, 3 modifications
**Dependencies:** Phase 1 (settings panel to hold VRM selector)

### Phase 3: Audio Feedback & Particles (Interaction polish)
**Scripts to create:**
- `Interaction/TouchSoundHandler.cs` — Play audio clips on touch/drag events
- `Interaction/ParticleEffectHandler.cs` — Spawn heart/sparkle particles on touch
- `Interaction/ReactionSoundPlayer.cs` — Short voice clips for reactions (like "hmm", "hehe")
- `Core/AudioManager.cs` — Central audio management for SFX volume control

**Changes to existing:**
- `TouchReactionController.cs` — Trigger sounds and particles
- `CompanionManager.cs` — Wire audio manager

**Estimated scope:** 4 new scripts, 2 modifications  
**Dependencies:** Phase 1 (settings for volume sliders)

### Phase 4: Smart Behavior & System Integration (Daily companion features)
**Scripts to create:**
- `Core/SystemTrayController.cs` — System tray icon with context menu
- `Core/SleepController.cs` — Detect PC idle, reduce activity
- `Core/MemoryOptimizer.cs` — Periodic GC for long-running desktop process
- `Avatar/DesktopAmbientLighting.cs` — Sample screen colors, apply to avatar light
- `UI/TimerAlarmUI.cs` — Simple timer/alarm with LLM notification

**Changes to existing:**
- `CompanionManager.cs` — Sleep/wake transitions
- `IdleAnimationController.cs` — Even subtler activity when sleeping

**Estimated scope:** 5 new scripts, 2 modifications
**Dependencies:** Phase 1 (settings panel for sleep timeout, ambient toggle)

### Phase 5: AI Integration Enhancements (Leverage our LLM advantage)
**Scripts to create:**
- `AI/ContextualMessageGenerator.cs` — LLM generates messages on events (drag, dance, sit)
- `AI/EmotionFromAction.cs` — Auto-set expression based on current action
- `UI/MarkdownRenderer.cs` — Render formatted text in speech bubble

**Changes to existing:**
- `MessageHandler.cs` — New message types for contextual events
- `CompanionManager.cs` — Emit events (started_drag, started_dance, etc.) to Python
- `server/main_chat.py` — Handle contextual event messages, generate short responses

**Estimated scope:** 3 new scripts, 3 modifications
**Dependencies:** Phase 1 (speech bubble), Phase 2 (drag animation)

### Phase 6: Advanced Desktop Features (Power user features)
**Scripts to create:**
- `Core/WindowSittingController.cs` — Detect other windows, sit on their title bars
- `Avatar/IKController.cs` — Inverse kinematics for sitting/hand poses
- `Avatar/LocomotionController.cs` — Walk across screen autonomously
- `Avatar/GravityController.cs` — Physics fall when released from drag

**Changes to existing:**
- `WindowSnapper.cs` — Integration with window sitting
- `TransparentWindowController.cs` — Gravity interaction with window position

**Estimated scope:** 4 new scripts, 2 modifications
**Dependencies:** Phase 2 (drag system), Phase 4 (sleep controller for locomotion triggers)

---

## Total Estimated Effort
| Phase | New Scripts | Modifications | Focus |
|-------|:-----------:|:-------------:|-------|
| Phase 1 | 5 | 2 | UI & Settings |
| Phase 2 | 4 | 3 | Character Management |
| Phase 3 | 4 | 2 | Interaction Polish |
| Phase 4 | 5 | 2 | System Integration |
| Phase 5 | 3 | 3 | AI Enhancement |
| Phase 6 | 4 | 2 | Advanced Desktop |
| **Total** | **25** | **14** | **39 changes** |

Current Annabeth: 18 scripts → After all phases: ~43 scripts
