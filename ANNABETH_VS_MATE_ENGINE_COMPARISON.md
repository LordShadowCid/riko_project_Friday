# Annabeth vs Mate Engine — Deep Dive Comparison

**Date**: January 2026  
**Annabeth Commit**: `6dd2a72` (Phase 8 complete)  
**Mate Engine Source**: https://github.com/shinyflvre/Mate-Engine  

---

## Executive Summary

| Dimension | Annabeth | Mate Engine |
|-----------|----------|-------------|
| **Core Purpose** | AI conversation partner with avatar (ASR→LLM→TTS pipeline) | Standalone desktop pet / companion |
| **Render Pipeline** | URP (Universal) | Built-in RP |
| **C# Scripts** | 32 files, ~5,600 lines | 30+ handlers, ~8,000+ lines |
| **VRM Loading** | UniVRM `Vrm10.LoadPathAsync` direct | Reflection-based component injection (553-line VRMLoader) |
| **Animation** | Procedural (code-driven bones) | Animator state machine + blend trees |
| **Backend** | Python (Whisper + Ollama + GPT-SoVITS) over WebSocket | None (standalone, optional LLM bubble) |
| **Window** | Custom Win32 P/Invoke (301 lines) | UniWindowController library (1155 lines) |
| **Settings** | JsonUtility, 20 fields | Newtonsoft JSON, 50+ fields, reflection reload |
| **Mod Support** | None | Full (AssetBundle .me/.unity3d, Steam Workshop) |
| **Multi-Instance** | Not supported | File-based JSON bus sync |

**Annabeth's Unique Strengths**: Real-time voice AI conversation, speaker identification, read-aloud system, sentence-level TTS pipeline, procedural beat-reactive dance.

**Mate Engine's Unique Strengths**: Polished desktop interaction (spring physics, occluder quads, window sitting with SmoothDamp), modding ecosystem, Animator-based animation with blend trees, food system, accessories, chibi mode.

---

## 1. Window Management

### Annabeth: `TransparentWindowController.cs` (301 lines)
- **Approach**: Direct Win32 P/Invoke from scratch
- **Transparency**: DWM `DwmExtendFrameIntoClientArea` with layered window style
- **Click-through**: Raycast against `SkinnedMeshRenderer` bounds — over character = capture, over empty = pass-through
- **Dragging**: Left-click on character body, `MoveWindow` per frame
- **Always-on-top**: `SetWindowPos(HWND_TOPMOST)` toggled via settings
- **Limitation**: No opacity threshold fallback, no keyboard shortcut for topmost toggle

### Mate Engine: `UniWindowController` (1155 lines) + `AvatarWindowHandler` (860+ lines)
- **Approach**: Third-party UniWindowController library (native plugin `LibUniWinC`)
- **Transparency**: Alpha or ColorKey modes, native plugin handles DWM
- **Click-through**: Two modes — **Opacity** (reads pixel alpha at cursor, configurable threshold 0.1f) or **Raycast**
- **Dragging**: `UniWindowMoveHandle` (Unity UI drag interfaces) with `dragSmooth` parameter and window-sit awareness
- **Hit test**: Auto-switches `isClickThrough` based on pixel opacity
- **Extras**: Window position/size get/set API, monitor selection, zoomed state, file drop support

### Key Gaps in Annabeth
| Gap | Impact | Difficulty |
|-----|--------|------------|
| No opacity-based hit test | Edge artifacts where transparent pixels near character still block clicks | Medium |
| No `dragSmooth` parameter | Dragging feels rigid vs Mate Engine's smoothed movement | Easy |
| No file drop support | Can't drag-and-drop VRM files onto window | Medium |
| No ColorKey transparency fallback | Only alpha mode — some GPUs handle ColorKey better | Low priority |

---

## 2. VRM Loading & Model Management

### Annabeth: `AvatarController.cs` (135 lines) + `VrmModelLibrary.cs` (328 lines)
- **Loading**: `Vrm10.LoadPathAsync(fullPath, canLoadVrm0X: true)` — clean, simple
- **Reload**: Destroy old → load new → call `InitializeControllers()`
- **Library**: Scans `StreamingAssets/Models` folder, shows list with load buttons
- **File Picker**: OS native file dialog via `SFB.StandaloneFileBrowser`
- **Saved Path**: `SettingsManager.Instance.data.selectedModelPath`

### Mate Engine: `VRMLoader.cs` (553 lines) + `AvatarLibraryMenu.cs`
- **Loading**: Format detection (.vrm / .me / DLC), VRM 1.0 with VRM 0.x fallback
- **Component Injection**: `InjectComponentsFromPrefab()` — **copies ALL MonoBehaviours from a template prefab via reflection** onto every loaded model. This is the core architectural pattern.
- **Library**: DLC entries, Steam Workshop integration, `SteamWorkshopAutoLoader`
- **Model Formats**: .vrm, .me (custom encrypted/bundled), DLC prefabs

### Key Gaps in Annabeth
| Gap | Impact | Difficulty |
|-----|--------|------------|
| No `.me` / AssetBundle support | Can't load Mate Engine content packs | Low priority (not needed) |
| No Steam Workshop | No community model sharing | Low priority |
| No component injection pattern | Not needed — Annabeth uses central CompanionManager hub instead | N/A (architectural choice) |
| No model metadata extraction | VRM title/author not displayed | Easy |

**Note**: Annabeth's approach (CompanionManager wires controllers + `OnVrmLoaded` event) is cleaner than Mate Engine's reflection injection. This is not a gap — it's a better pattern.

---

## 3. Mouse / Eye / Head Tracking

### Annabeth: `EyeTrackingController.cs` (118 lines)
- **Approach**: VRM LookAt system with `SpecifiedTransform` target
- **Eyes**: Positions a look-at target based on mouse screen position, VRM's built-in LookAt does the rest
- **Head**: VRM LookAt handles head turn automatically (bundled with eye tracking)
- **Smoothing**: `Vector3.Lerp` at `lookAtSpeed` (5f) for target position
- **Bounds**: ±30° horizontal, ±20° vertical (configurable)
- **Reset**: Returns to forward-facing when disabled

### Mate Engine: `AvatarMouseTracking.cs` (215 lines)
- **Approach**: **Driver pattern** — creates intermediate GameObjects between bone parents and Animator
- **Eyes**: Separate left/right eye drivers with `eyeYawLimit` (12°)
- **Head**: Dedicated head driver with separate speed from eyes, ±45° yaw, ±30° pitch
- **Spine**: Cascading rotation — chest 0.8×, upperChest 0.6× of head rotation
- **Smoothing**: `Quaternion.Slerp` per driver
- **State Awareness**: `TrackingPermission` system — disables tracking in certain Animator states
- **Critical**: Drivers survive Animator writes (direct bone modification gets overridden each frame)

### Key Gaps in Annabeth
| Gap | Impact | Difficulty |
|-----|--------|------------|
| **No spine/upper body rotation** | Character feels stiff — head moves but body doesn't lean toward cursor | Medium |
| No driver pattern | VRM LookAt works fine for now, but bone-level control limited | High (architectural) |
| **No per-component tracking speeds** | Eyes and head move at same rate — eye-lead is more natural | Easy |
| No state-aware tracking permissions | Tracking continues during dance (should reduce/disable) | Medium |
| **No spine lean toward cursor** | Missing the subtle body lean that makes Mate Engine feel alive | Medium |

**Priority**: Spine tracking (body lean toward cursor) is the single highest-impact improvement for visual quality.

---

## 4. Idle Animation

### Annabeth: `IdleAnimationController.cs` (170 lines)
- **Approach**: 100% procedural — code drives bone transforms directly
- **Breathing**: Y-position bob on hips (0.002 units, 0.25 Hz)
- **Head drift**: Random target selection, smooth interpolation at 0.5 units/sec
- **Weight shifting**: Hip lateral sway ±0.003 units at 0.1 Hz
- **Bones manipulated**: Hips, Head (via ControlRig)
- **Speed**: `breathSpeed`, `headDriftSpeed`, `swaySpeed`

### Mate Engine: `AvatarAnimatorController.cs` (243 lines) + Animator + BlendTrees
- **Approach**: Animator state machine with **10 idle blend tree entries**
- **Cycling**: `BlendTreeLooper` (StateMachineBehaviour) auto-cycles through idle variations every 12s
- **Transitions**: Smooth lerp on `IdleIndex` float parameter for seamless blend tree cross-fade
- **Content**: Pre-made animation clips with full-body movement (imported FBX)
- **State machine**: Idle → Dancing → Dragging → Sleeping states with bool-driven transitions

### Key Gaps in Annabeth
| Gap | Impact | Difficulty |
|-----|--------|------------|
| **No animation clips / blend trees** | Procedural idle is functional but limited vs authored animations | High (needs animation assets) |
| **No idle variation cycling** | Same breathing/sway pattern forever — becomes repetitive | Medium |
| No per-joint micro-movements | Missing finger fidgeting, shoulder shifts, random double-blinks | Medium |
| **No Animator state machine** | No formal state transitions (idle→dance→drag→sleep) | High (architectural) |
| No `BlendTreeLooper` equivalent | Timer-based blend tree cycling is elegant and smooth | Medium |

**Note**: This is the most significant architectural difference. Annabeth uses procedural animation; Mate Engine uses authored clip-based animation with blend trees. Procedural is more flexible (works with any VRM) but authored clips look more polished. Both approaches are valid — the choice depends on whether you want to create/source animation clips.

---

## 5. Dance System

### Annabeth: `BeatDanceController.cs` (427 lines) + `VrmaAnimationController.cs` (165 lines)
- **Beat-reactive**: Procedural dance driven by real-time audio analysis (bass/mid/high bands via WASAPI)
- **13 bones**: Hips, spine, chest, head, shoulders, upper arms, upper legs — all manipulated per-frame
- **4 phase accumulators**: Independent oscillation sources for complex patterns
- **Style system**: 8+ dance styles (Bounce, Sway, Pop, Wave, Robot, Groove, Headbang, Gentle) — each has unique per-bone amplitude/frequency profiles
- **Audio source**: Python backend analyzes system audio via WASAPI loopback and broadcasts over WebSocket
- **VRMA**: File-based .vrma animation playback with UniVRM's retargeting
- **Transition**: Hard switch between styles, blend controller manages idle↔dance transitions

### Mate Engine: `AvatarDanceHandler.cs` (1401 lines) + Animator
- **State machine**: Animator-driven with `isDancing` bool and `DanceIndex` float
- **5 built-in dance clips**: Blend tree cycling (lerped float for smooth transitions)
- **Custom dances**: AssetBundle-based (.unity3d / .me), `AnimatorOverrideController` replaces placeholder clip
- **Audio**: NAudio `MasterPeakValue` for sound detection (simpler — just threshold, no frequency bands)
- **Navigation**: Sequential / Shuffle / Loop, full UI (play/pause/prev/next/progress/volume)
- **Multi-instance sync**: File-based JSON bus with Mutex, leader/follower pattern
- **Blendshape forwarding**: `AvatarDanceShapeConverter` using PlayableGraph for MMD facial animation during dance
- **Search/favorites**: Unicode-normalized search, JSON-persisted favorites

### Key Gaps in Annabeth
| Gap | Impact | Difficulty |
|-----|--------|------------|
| No custom dance loading (`.unity3d` / `.me`) | Can't add user-provided dance animations | High |
| No `AnimatorOverrideController` pattern | Missing runtime clip replacement for custom dances | Medium |
| No dance UI (play/pause/progress/volume) | User can't control dance playback beyond hotkeys | Medium |
| No multi-instance sync | Multiple companions can't dance together | Low priority |
| No blendshape forwarding during dance | Facial expressions don't animate during external dances | Medium |
| **No smooth dance transitions** | Hard cut between styles vs Mate Engine's blend tree lerp | Medium |

**Annabeth's Advantage**: The beat-reactive procedural dance is **unique** — Mate Engine's dances are pre-canned clips that don't respond to the actual music's frequency content. Annabeth's dance actually matches the beat. This is a major differentiator.

---

## 6. Desktop Interaction

### Annabeth (Phase 8): `DesktopLocomotionController.cs` (271 lines) + `WindowSnapper.cs` (437 lines)
- **Walk**: Random walk with configurable distance (250-550px), speed (2px/frame), 10s decision interval
- **Direction**: `PickDirectionByEdges` — avoids walking off-screen via monitor work area bounds
- **Window sitting**: `EnumWindows` to find visible windows, probe radius 30px, center-X overlap check
- **Gravity**: Acceleration-based falling (1200 px/s²) when sitting surface disappears
- **Surface tracking**: Follows seated window if it moves; triggers fall on significant displacement
- **Peeking**: Walk to screen edge, show only `peekVisiblePixels` (40px)
- **Edge snap**: Magnetic 20px threshold to screen edges
- **Taskbar sit**: Double-right-click

### Mate Engine: `AvatarLocomotionController.cs` (746 lines) + `AvatarWindowHandler.cs` (860+ lines) + `AvatarHideHandler.cs` (384 lines)
- **Walk**: Random walk with direction avoidance via monitor edges, `WindowSpeed` (2px/frame)
- **Smart Animator**: `ResolveAnimatorSmart()` cascade search, walking animation clips
- **Window sitting**: Hip-based snap probe (24px radius), guard zone (240px), 1s minimum drag hold, `CalibrateSeatAnchorToDesktopY()` (binary search, 20 iterations)
- **Occluder quads**: Dynamically created quad meshes to mask avatar behind foreground windows — **this is a major visual feature**
- **SmoothDamp following**: Prediction-based smooth following of seated window with `SmoothDamp`
- **Screen hiding**: Left/right edge snap with Animator integration (`HideLeft` / `HideRight` bools), grace period, multi-monitor adjacency detection
- **Bone caching**: hips, upper/lower legs, feet, head — for sitting pose adjustment

### Key Gaps in Annabeth
| Gap | Impact | Difficulty |
|-----|--------|------------|
| **No occluder quads** | Avatar doesn't hide behind foreground windows when sitting — breaks immersion | High |
| No `SmoothDamp` window following | Window tracking feels jerky compared to Mate Engine's prediction-smoothed following | Medium |
| No binary search calibration | Sitting position may not be perfectly aligned to desktop Y | Medium |
| No hip-based snap probe | Annabeth uses center-of-window hitbox; Mate Engine uses avatar hip position (more accurate) | Medium |
| No guard zone | Can re-snap immediately after unsnapping — potential jitter | Easy |
| **No walking animation** | Window moves but avatar doesn't play a walk cycle (just translates rigidly) | High (needs clip) |
| No drag-hold minimum (1s) | Can accidentally sit on windows during brief drags | Easy |
| **No Animator `HideLeft`/`HideRight` integration** | No hiding animation when peeking at screen edge | Medium |
| No multi-monitor adjacency detection | Might try to hide at an edge that borders another monitor | Medium |

---

## 7. Drag Sway / Spring Physics

### Annabeth: `DragAnimationController.cs` (57 lines)
- **Approach**: Float-up effect during drag (Y-position offset)
- **Physics**: Simple upward drift while dragging
- **No spring dynamics**: No oscillation, no damping, no overshoot on release

### Mate Engine: `AvatarSwayController.cs` (344 lines)
- **Approach**: Full spring physics on hips/arms/legs during drag
- **Formula**: 2nd-order spring: `acceleration = ω²(target - x) - 2ζωv` where ω = frequency × 2π, ζ = damping ratio
- **Parameters**: `springFrequency` (2.6), `dampingRatio` (0.35), `maxLeanZ` (25°), `maxLeanX` (12°)
- **Joints affected**: Hips (lean), both arms (swing), both legs — with separate max angles per joint
- **Input source**: Window velocity during drag → physics calculation → bone rotation offset
- **State whitelist**: Only active in certain Animator states

### Key Gaps in Annabeth
| Gap | Impact | Difficulty |
|-----|--------|------------|
| **No spring physics** | Dragging feels lifeless — no leaning, swinging, or settling | Medium |
| No window velocity tracking | Can't derive sway direction from drag movement | Easy |
| No per-joint spring parameters | Need separate damping/frequency for arms vs hips | Medium |
| **No overshoot + settle on release** | Missing the satisfying spring-back after releasing drag | Medium |

**Priority**: This is a high-delight, medium-effort improvement. Spring physics + sway adds enormous personality.

---

## 8. Settings System

### Annabeth: `SettingsManager.cs` (230 lines) + `SettingsPanel.cs` (262 lines)
- **Serialization**: `JsonUtility` (built-in, no dependencies)
- **Fields**: ~20 settings covering display, tracking, interaction, AI/speech, system
- **UI**: Runtime-built sliders/toggles via `UIFactory` (no TextMeshPro needed)
- **Reset**: `ResetToDefaults()` method
- **Apply**: `ApplyAllSettings()` pushes to live controllers

### Mate Engine: `SaveLoadHandler.cs` (300 lines) + 6 settings handler scripts
- **Serialization**: `Newtonsoft.Json` (more flexible — supports Dictionaries natively)
- **Fields**: **50+ settings** covering window, sound, idle, dance, avatar, tracking, features (18 bools), performance, visual, audio (3 channels), lights (per-light color/intensity), mods, misc
- **Multi-instance**: CLI args `--savefile` / `--datadir` for separate save files
- **Migration**: Versioned `MigrateAfterLoad()` for forward compatibility
- **Reflection reload**: `SettingsHandlerUtility.ReloadAllSettingsHandlers()` finds and invokes `LoadSettings()` / `ApplySettings()` on ALL MonoBehaviours
- **Handler separation**: 6 dedicated handler scripts (Toggles, Sliders, Dropdowns, Audio, Lights, BigScreen) — each ~100-250 lines

### Key Gaps in Annabeth
| Gap | Impact | Difficulty |
|-----|--------|------------|
| No versioned migration | Settings format changes will break old saves | Easy |
| No per-light color/intensity settings | Less visual customization | Low priority |
| No sound threshold / per-app filtering | Can't configure which apps trigger dance mode | Medium |
| No graphics quality setting | Can't adjust quality level | Easy |
| No hue shift / saturation (theme) | Can't customize UI color theme | Low priority |

---

## 9. Sleep System

### Annabeth: `SleepController.cs` (90 lines)
- **Approach**: Timer-based idle detection → reduce frame rate
- **Sleep action**: Sets `Application.targetFrameRate` to low value
- **Wake triggers**: Any mouse/keyboard input
- **Settings**: `enableSleepMode`, `sleepTimerSeconds`

### Mate Engine: `AvatarSleepController.cs` (117 lines)
- **Approach**: Timer + Animator state integration
- **Sleep action**: Sets `"IsSleeping"` Animator bool → plays sleeping animation
- **Wake triggers**: Configurable `wakeUpBools` (e.g., `"isDragging"`)
- **State awareness**: Only counts time in `allowedStates` (e.g., "Idle", "Sleeping")

### Key Gaps in Annabeth
| Gap | Impact | Difficulty |
|-----|--------|------------|
| **No sleeping animation** | Just reduces FPS — no visual feedback that avatar is sleeping | Medium (needs clip or procedural) |
| No Animator state integration | Can't transition properly between idle→sleep→wake states | Medium |
| No configurable wake triggers | Always wakes on any input — no nuance | Easy |

---

## 10. Features Mate Engine Has That Annabeth Lacks Entirely

| Feature | Mate Engine Implementation | Effort to Add | Priority |
|---------|---------------------------|---------------|----------|
| **Spring physics (sway)** | `AvatarSwayController.cs` (344 lines) — 2nd order spring on 5+ joints | Medium | **HIGH** |
| **Occluder quads** | `AvatarWindowHandler.cs` — dynamic quads mask avatar behind windows | High | **HIGH** |
| **Walking animation** | Animator `IsWalking` bool → walk clip | Medium (needs clip) | **HIGH** |
| **Driver pattern (tracking)** | `AvatarMouseTracking.cs` — intermediate GameObjects for bone control | High | Medium |
| **Chibi mode** | `ChibiToggle.cs` (104 lines) — armature scale + head enlarge + foot ground adjust | Medium | Low |
| **Food system** | `AvatarFoodController.cs` (433 lines) — mouse-follow, head interaction, sounds | High | Low |
| **Mod system** | `MEModHandler.cs` (648 lines) + `MEModLoader.cs` (308 lines) — AssetBundle mods | Very High | Low |
| **Multi-instance sync** | File-based JSON bus, Mutex, leader/follower | High | Low |
| **Accessories** | `AccessoiresHandler.cs` — bone-tracked objects | Medium | Low |
| **Pet/pat detection** | `PetVoiceReactionHandler.cs` — circular mouse motion detection | Medium | Medium |
| **Blend tree cycling** | `BlendTreeLooper.cs` (75 lines) — auto-cycle idle blend tree entries | Medium | Medium |
| **Custom dance loading** | `AvatarDanceHandler.cs` (1401 lines) — .unity3d / .me AssetBundles | Very High | Low |
| **Screen saver** | Big screen mode with timeout | Easy | Low |
| **Discord RPC** | Rich presence integration | Easy | Low |
| **Alarms / Timers** | `AlarmEntry` / `TimerEntry` in settings | Medium | Low |

---

## 11. Features Annabeth Has That Mate Engine Lacks Entirely

| Feature | Annabeth Implementation | Notes |
|---------|------------------------|-------|
| **Real-time voice AI conversation** | Whisper ASR → Ollama LLM → GPT-SoVITS TTS | Core differentiator — Mate Engine has no voice AI |
| **Speaker identification** | resemblyzer voice encoder, .npy profiles | Recognizes who's speaking |
| **Read-aloud system** | Text capture → sentence queue → TTS → browser highlight | Reads web pages aloud |
| **Beat-reactive procedural dance** | 13 bones, 4 phase accumulators, 8 styles | Dances to actual audio frequency content |
| **WASAPI frequency-band analysis** | Bass/mid/high band extraction from system audio | More sophisticated than Mate Engine's peak detection |
| **WebSocket AI communication** | Full duplex messaging with Python backend | Real-time bidirectional state sync |
| **Sentence-level TTS pipeline** | Pre-buffered streaming for low latency | Streams response as sentences, not all-at-once |
| **Emotion system from LLM** | LLM outputs emotion tags → VRM blend shapes | Context-aware emotional expressions |
| **Speech bubble with typewriter** | 40 chars/sec typewriter, auto-dismiss, pop animation | Shows AI response text |
| **System tray integration** | Win32 `Shell_NotifyIcon`, custom context menu | Minimize to tray with quick actions |
| **Audio analyzer broadcast** | Python WASAPI → WebSocket → Unity | Real-time audio data for avatar reactions |

---

## 12. Architectural Comparison

### Annabeth: Hub Coordinator Pattern
```
CompanionManager (481 lines) ← Central coordinator
    ├── Finds all controllers via FindFirstObjectByType
    ├── Wires events: OnSpeakStart, OnEmotionChange, OnDragStart...
    ├── Handles mode transitions: Idle ↔ Active ↔ Dance
    └── Delegates to individual controllers
```
**Pros**: Explicit dependencies, event-driven, easy to debug, clean separation  
**Cons**: CompanionManager grows with each new feature, single point of coupling

### Mate Engine: Template Prefab Injection Pattern
```
VRMLoader → InjectComponentsFromPrefab(templatePrefab, loadedModel)
    └── Reflection copies ALL MonoBehaviours from template → model
         ├── Each handler is self-contained
         ├── Handlers communicate via Animator parameters + static singletons
         └── 30+ handlers automagically injected
```
**Pros**: Zero manual wiring per model, add features by adding to template prefab  
**Cons**: Reflection magic, hard to trace dependencies, implicit coupling via Animator bools

### Verdict
Annabeth's hub pattern is **more maintainable** for a smaller project. Mate Engine's injection pattern scales better for 30+ handlers but is harder to debug. No change needed here.

---

## 13. Recommended Improvements — Prioritized

### Tier 1: High Impact, Achievable
These will make Annabeth feel significantly more alive and polished.

| # | Improvement | What It Does | Est. Lines | Based On |
|---|------------|--------------|-----------|----------|
| **1** | **Spine/Upper Body Tracking** | Add subtle body lean toward cursor (±10° spine, ±5° chest) alongside existing eye tracking | ~80 | `AvatarMouseTracking.cs` spine logic |
| **2** | **Spring Physics Sway** | Add 2nd-order spring to hips/arms during window drag — lean, swing, settle | ~200 | `AvatarSwayController.cs` |
| **3** | **Smooth Dance Transitions** | Crossfade between dance styles instead of hard-cutting | ~60 | Existing `AnimationBlendController` extension |
| **4** | **Per-Component Track Speeds** | Eyes lead (fast), head follows (medium), body last (slow) | ~40 | `AvatarMouseTracking.cs` separate speeds |
| **5** | **Settings Version Migration** | Add `settingsVersion` field + `MigrateAfterLoad()` to prevent breaking old saves | ~30 | `SaveLoadHandler.cs` migration |

### Tier 2: Medium Impact, Medium Effort
These address noticeable missing polish.

| # | Improvement | What It Does | Est. Lines | Based On |
|---|------------|--------------|-----------|----------|
| **6** | **Occluder Quads** | Create dynamic quads behind foreground windows to mask seated avatar | ~200 | `AvatarWindowHandler.cs` occluder logic |
| **7** | **SmoothDamp Window Following** | Replace jerky `MoveWindow` tracking with prediction-smoothed following for seated avatar | ~60 | `AvatarWindowHandler.PinToTarget()` |
| **8** | **Walking Animation** | Procedural walk cycle (arm swing + leg stride) so avatar doesn't just slide across screen | ~120 | New procedural (Annabeth style) |
| **9** | **Sleep Animation** | Close eyes + tilt head + slow breathing when sleeping instead of just reducing FPS | ~60 | Extend `SleepController` + `BlinkController` |
| **10** | **Idle Variation System** | 2-3 different idle behaviors (relaxed, alert, bored) that cycle over time | ~100 | `BlendTreeLooper.cs` concept, procedural |

### Tier 3: Nice-to-Have, Lower Priority
These are fun features but not critical.

| # | Improvement | What It Does | Est. Lines | Based On |
|---|------------|--------------|-----------|----------|
| **11** | **Pet/Pat Detection** | Detect circular mouse motion over avatar → trigger pleased reaction | ~80 | `PetVoiceReactionHandler.cs` |
| **12** | **Screen-Edge Hide Animations** | Lean/peek animation when hiding at screen edges | ~60 | `AvatarHideHandler.cs` animator bools |
| **13** | **Drag-Hold Window Sit Guard** | Require 1s drag hold before allowing window sit (prevents accidental sits) | ~10 | `AvatarWindowHandler.minDragHoldSecondsToSit` |
| **14** | **VRM Metadata Display** | Show VRM title/author in library + settings | ~30 | `VRMLoader.cs` metadata extraction |
| **15** | **Multi-Monitor Adjacency** | Don't try to hide at a screen edge that borders another monitor | ~60 | `AvatarHideHandler.GetAllowedEdgesForMonitor()` |

### NOT Recommended
These Mate Engine features don't fit Annabeth's design:

| Feature | Why Skip |
|---------|----------|
| **Chibi Mode** | User explicitly removed this ("we dont need") |
| **Food System** | Doesn't align with AI conversation partner design |
| **Mod System** | Massive scope (956+ lines), requires AssetBundle pipeline — not needed for a personal companion |
| **Animator State Machine** | Would require replacing Annabeth's procedural animation system entirely — too disruptive |
| **Multi-Instance Sync** | Single companion design, not multi-pet |
| **Discord RPC** | Low value, privacy concern |
| **Steam Workshop** | No Steam distribution planned |

---

## 14. Code Quality & Architecture Notes

### Things Annabeth Does Better
1. **Async VRM loading** — `Vrm10.LoadPathAsync` with `destroyCancellationToken` vs Mate Engine's synchronous path
2. **Event-driven architecture** — Clean C# events vs Mate Engine's implicit Animator bool polling
3. **No reflection magic** — All wiring is explicit in CompanionManager
4. **Runtime UI factory** — No TextMeshPro dependency, no Editor-wired prefabs
5. **`#if UNITY_STANDALONE_WIN && !UNITY_EDITOR`** guards — Clean editor/build separation
6. **WebSocket client** — Proper async WebSocket with reconnection logic

### Things Mate Engine Does Better
1. **Animator parameter hashing** — Pre-hashed `Animator.StringToHash()` instead of string lookups
2. **State whitelist pattern** — Feature activation gated by Animator state
3. **SmoothDamp everywhere** — More polished motion than raw `Lerp`
4. **Guard timing** — `EnforceHold` / `FreezeAnimator` prevents accidental state changes
5. **Bone caching** — Caches `GetBoneTransform()` results instead of re-querying

### Quick Wins from Mate Engine's Code Style
- Cache `Animator.StringToHash()` results (Annabeth doesn't use Animator params much, but if added)
- Add `SmoothDamp` to window following instead of raw position sets
- Add bone caching to `IdleAnimationController` (currently queries `GetBoneTransform` each frame via `ControlRig`)

---

## Summary

Annabeth and Mate Engine serve fundamentally different purposes. Annabeth is an **AI conversation partner** with an avatar — the voice AI pipeline is the product, the avatar is the interface. Mate Engine is a **desktop pet** — the visual interaction IS the product.

The highest-value improvements (#1-#5 above) would bring Annabeth's **visual presence** closer to Mate Engine's polish while preserving its unique AI-driven architecture. The spring physics sway and spine tracking alone would make the biggest perceived difference.
