# Annabeth — Master Implementation Plan

**Created:** March 21, 2026  
**Status:** Active  

---

## Current State Summary

### What's DONE and Working

| Component | Status | Details |
|-----------|--------|---------|
| **ASR (Whisper)** | ✅ Complete | Faster-Whisper, VAD hands-free + push-to-talk, GPU w/ CPU fallback |
| **LLM (Ollama)** | ✅ Complete | mannix/llama3.1-8b-abliterated, streaming with sentence-level pipeline |
| **TTS (GPT-SoVITS)** | ✅ Complete | Pre-buffered sentence pipeline, interrupt support |
| **Speaker ID** | ✅ Complete | resemblyzer voice encoder, per-speaker .npy profiles |
| **Read-Aloud** | ✅ Phases 1-3 | Text capture → sentence queue → TTS → browser highlight extension |
| **Memory System** | ✅ Complete | Conversation summarizer, feedback logging, self-eval, memory store |
| **Audio Analyzer** | ✅ Complete | WASAPI loopback, bass/mid/high bands via WebSocket |
| **Avatar Server** | ✅ Complete | aiohttp WebSocket on :8765, mode/silence/read-aloud messages |
| **Shared State** | ✅ Complete | Thread-safe CompanionState, AudioState, ReadAloudState |
| **Config System** | ✅ Complete | YAML config → dataclass hierarchy, shared enums |
| **Server Utilities** | ✅ Complete | Centralized audio device resolution, Ollama settings |
| **Unity Scripts (14)** | ✅ Compiled | AvatarController, LipSync, Emotion, Blink, EyeTracking, IdleAnim, BeatDance, VrmaAnim, WebSocket, MessageHandler, TransparentWindow, CompanionManager, HotkeyManager, BuildConfigurator |
| **Unity Scene** | ✅ Wired | VRM loads ~1.5s, right-side-up, relaxed pose, camera correct |
| **Mate-Engine Reference** | ✅ Created | MATE_ENGINE_REFERENCE.md — full architecture documented |

### What's NOT Done

| Item | Priority | Blocked By |
|------|----------|------------|
| Full integration test (Python ↔ Unity) | **P0** | Nothing — ready to test |
| VRMA playback test (press 3 key) | **P0** | Integration test |
| Transparent window standalone build | **P1** | Nothing |
| Mouse/head tracking (look-at cursor) | **P1** | Nothing |
| Window dragging improvements | **P2** | Transparent window |
| Subtitle display | **P2** | Integration test |
| Animation blend trees / idle improvements | **P2** | Integration test |
| Chibi mode toggle | **P3** | Nothing |
| Window sitting (taskbar) | **P3** | Transparent window |
| Touch/click reactions | **P3** | Mouse tracking |
| Sleep/idle behaviors | **P3** | Animation system |
| Sway spring physics | **P3** | Window dragging |
| Read-Aloud Phase 4 (overlay for PDFs/Word) | **P4** | Phases 1-3 stable |
| Custom dance player improvements | **P4** | VRMA working |
| Mod/accessory system | **P5** | Most other features |

---

## Optimized Implementation Phases

The phases below are ordered by **dependency chain** and **impact**. Each phase unlocks the next. Within each phase, tasks are ordered to minimize blocked time (e.g., start long builds first, do small tasks while waiting).

---

### Phase 1: Integration & Validation (CRITICAL PATH)

**Goal:** Prove the full ASR → LLM → TTS → Unity pipeline works end-to-end.  
**Why first:** Every subsequent phase depends on a working integration. No point adding features to a pipeline that doesn't connect.

| # | Task | Details | Depends On |
|---|------|---------|------------|
| 1.1 | **Start Python backend + Unity simultaneously** | Run `start_annabeth.ps1` + Unity Play mode | Nothing |
| 1.2 | **Verify WebSocket connects** | Unity console should show `[WebSocket] Connected to ws://127.0.0.1:8765/ws` | 1.1 |
| 1.3 | **Test speak → lip sync** | Speak to Annabeth, verify Unity avatar does lip sync while TTS plays | 1.2 |
| 1.4 | **Test emotion expressions** | Verify emotion messages arrive and blend shapes change | 1.2 |
| 1.5 | **Test mode switching** | Press D key → verify dance mode toggles, hotkeys work | 1.2 |
| 1.6 | **Test beat dance with audio** | Play music → verify avatar reacts to audio_analysis messages | 1.5 |
| 1.7 | **Test VRMA playback** | Press 3 → verify Shikanoko dance loads and plays | 1.2 |
| 1.8 | **Test silence toggle** | Press S → verify mute/unmute cycle works both sides | 1.2 |
| 1.9 | **Fix any connection/messaging bugs** | Debug and fix issues found in steps above | 1.2-1.8 |

**Exit Criteria:** Full voice chat loop works — you speak → Annabeth responds with voice + lip sync + emotion on the Unity avatar.

---

### Phase 2: Standalone Window & Build

**Goal:** Unity runs as a transparent always-on-top desktop companion (not just in-editor).  
**Why second:** This is the whole point of the Unity migration — a proper desktop companion window. Needed before any desktop-interaction features (window sitting, drag, etc.).

| # | Task | Details | Depends On |
|---|------|---------|------------|
| 2.1 | **Test TransparentWindowController in-editor** | Verify Win32 P/Invoke calls work (DWM extend, layered) in Play mode | Phase 1 |
| 2.2 | **Configure build settings** | Use BuildConfigurator menu: Annabeth > Configure Build Settings | Nothing |
| 2.3 | **Build standalone .exe** | Annabeth > Build Standalone — outputs to `Builds/` folder | 2.2 |
| 2.4 | **Test standalone transparency** | Run .exe, verify transparent background, always-on-top | 2.3 |
| 2.5 | **Test standalone WebSocket** | Run .exe alongside Python backend, verify voice chat works | 2.3, Phase 1 |
| 2.6 | **Fix build-only issues** | Address any differences between editor and standalone | 2.4-2.5 |
| 2.7 | **Update start_annabeth.ps1** | Add option to launch Unity build instead of/alongside PyQt client | 2.5 |
| 2.8 | **Window drag via right-click** | Verify TransparentWindowController drag works in standalone | 2.4 |

**Exit Criteria:** `Annabeth.exe` runs as a transparent desktop overlay, connects to Python backend, full voice chat works.

---

### Phase 3: Mouse Tracking & Look-At

**Goal:** Avatar eyes and head follow the mouse cursor naturally.  
**Why third:** Highest visual impact for lowest effort. Makes the avatar feel alive. Mate-Engine's driver pattern is the reference implementation.

| # | Task | Details | Depends On |
|---|------|---------|------------|
| 3.1 | **Improve EyeTrackingController** | Current: uses VRM LookAt target. Enhance with smooth interpolation and dead-zone (Mate-Engine uses driver pattern with separate head/eye speeds) | Nothing |
| 3.2 | **Add head tracking to mouse** | Head turns toward cursor with separate speed from eyes (slower). Use ControlRig `GetBoneTransform(HumanBodyBones.Head)` with clamped angles | 3.1 |
| 3.3 | **Add upper body lean** | Slight spine rotation toward cursor (like Mate-Engine's `AvatarMouseTracking` spine tracking). Very subtle, ±5° max | 3.2 |
| 3.4 | **Screen-edge awareness** | If cursor goes off-screen, gradually return to center/forward look | 3.1 |
| 3.5 | **Tune look-at weights** | Eyes 100%, Head 60%, Spine 20% — adjust until natural | 3.2-3.3 |

**Reference:** Mate-Engine `AvatarMouseTracking.cs` — driver pattern with `mouseSensitivity`, `headTurnSpeed(3f)`, `returnSpeed(2f)`, `headTurnLimit(25°)`, `bodyTurnLimit(10°)`

**Exit Criteria:** Avatar smoothly looks at cursor with eyes leading, head following, subtle body lean.

---

### Phase 4: Idle Animation & Breathing Improvements

**Goal:** Rich idle behavior that makes the avatar feel alive when not speaking.  
**Why fourth:** Builds on mouse tracking (Phase 3). Currently idle is just breathing + head drift. Mate-Engine shows what's possible.

| # | Task | Details | Depends On |
|---|------|---------|------------|
| 4.1 | **Improve breathing animation** | Add chest rise/fall and shoulder movement. Current only does Y-position bob. Add ControlRig chest bone rotation | Nothing |
| 4.2 | **Add weight shifting** | Subtle hip sway side-to-side, shift weight between feet periodically | Nothing |
| 4.3 | **Add random micro-movements** | Small random head tilts, eye blinks with occasional double-blink, hand fidgeting | Phase 3 |
| 4.4 | **Add idle state variations** | After N seconds of no interaction: posture relaxes more, maybe look around, sigh animation | 4.1-4.3 |
| 4.5 | **Blend tree for idle states** | Create 2-3 idle animation states with smooth cross-fade (Mate-Engine uses `AnimatorStateInfo` + `NormalizedTime` for transitions) | 4.1-4.4 |

**Reference:** Mate-Engine `AvatarAnimatorController.cs` — NAudio-driven idle/dance/action state machine with blend tree transitions

**Exit Criteria:** Avatar looks naturally alive at idle — breathing, shifting weight, occasional glances.

---

### Phase 5: Subtitle Display & UI

**Goal:** Show what Annabeth is saying and current mode status on screen.  
**Why fifth:** Quality-of-life feature. Helps when audio isn't clear. Needed for read-aloud highlighting.

| # | Task | Details | Depends On |
|---|------|---------|------------|
| 5.1 | **Create SubtitleController.cs** | TextMeshPro world-space or screen-space text above/below avatar. Receive `speak_start` text from WebSocket | Phase 1 |
| 5.2 | **Auto-hide subtitles** | Fade in on speak_start, fade out 2s after speak_end | 5.1 |
| 5.3 | **Word-by-word highlight** | Highlight current word during read-aloud (use `read_highlight` messages already sent by Python backend) | 5.1 |
| 5.4 | **Mode indicator** | Small icon/text showing current mode (Active/Idle/Dance/Muted). Could be a colored dot near avatar | Phase 1 |
| 5.5 | **Speech bubble style** | Style subtitles as a speech bubble near avatar's head (Mate-Engine's `AvatarBubbleHandler` pattern) | 5.1 |

**Reference:** Mate-Engine `AvatarBubbleHandler.cs` — `bubblePanel`, `bubbleText`, `bubbleCanvasGroup` with fade

**Exit Criteria:** Spoken text appears as speech bubble near avatar, fades after speaking stops. Mode indicator visible.

---

### Phase 6: Sway Physics & Drag Feel

**Goal:** Avatar sways naturally when window is dragged, and settles with spring physics.  
**Why sixth:** Builds on standalone window (Phase 2). High delight factor. Mate-Engine's spring physics is the reference.

| # | Task | Details | Depends On |
|---|------|---------|------------|
| 6.1 | **Create SwayController.cs** | Track window velocity during drag. Apply rotation offset to hips/spine based on velocity | Phase 2 |
| 6.2 | **Spring physics settling** | When drag stops, overshoot and oscillate back to center. Damped spring: `velocity += -springK * offset - damping * velocity` | 6.1 |
| 6.3 | **Hair/clothes bonus** | VRM spring bones should react naturally. Verify UniVRM spring bone settings are reasonable | 6.1 |
| 6.4 | **Drag momentum** | If thrown (fast release), continue sliding in direction with deceleration | 6.2 |

**Reference:** Mate-Engine `AvatarSwayController.cs` — `swayAmount(15f)`, `swaySpeed(5f)`, `returnSpeed(3f)`, spring-based offset applied to upper body

**Exit Criteria:** Dragging the window makes the avatar swing/lean in the drag direction, then bounce back smoothly.

---

### Phase 7: Click/Touch Reactions

**Goal:** Avatar reacts when user clicks on her.  
**Why seventh:** Builds on mouse tracking (Phase 3) and emotion system. Adds personality.

| # | Task | Details | Depends On |
|---|------|---------|------------|
| 7.1 | **Detect click on avatar** | Raycast from mouse position → check if hits VRM mesh collider | Phase 3 |
| 7.2 | **Region-based reactions** | Head pat → happy/embarrassed expression. Poke body → surprised. Different zones = different reactions | 7.1 |
| 7.3 | **Visual feedback** | Brief expression change + head tilt or step back. Maybe particle effect (heart, star) | 7.2 |
| 7.4 | **Voice reactions** | Send reaction event to Python → short TTS response ("Hey!", "That tickles!", etc.) | 7.2, Phase 1 |
| 7.5 | **Cooldown system** | Don't spam reactions. 2-3s cooldown between touch responses | 7.2 |

**Reference:** Mate-Engine `AvatarMouseTracking.cs` has `isHolding`, `holdDuration`, `screenInteractionRadius(30f)` + context menu system

**Exit Criteria:** Clicking on avatar triggers appropriate reactions (expression + optional voice line).

---

### Phase 8: Desktop Interaction

**Goal:** Avatar interacts with the Windows desktop — sits on taskbar, walks on screen edges.  
**Why eighth:** Advanced feature that needs transparent window (Phase 2) + sway physics (Phase 6). High wow factor.

| # | Task | Details | Depends On |
|---|------|---------|------------|
| 8.1 | **Taskbar detection** | Win32 API: `FindWindow("Shell_TrayWnd", null)` to get taskbar rect | Phase 2 |
| 8.2 | **Window sitting** | Snap avatar to sit on top of taskbar or other window edges. Change to sitting pose | 8.1 |
| 8.3 | **Walking animation** | Basic walk cycle when avatar moves between positions (Mate-Engine's `AvatarLocomotionController`) | Phase 4 |
| 8.4 | **Screen-edge hiding** | Avatar can walk to screen edge and partially hide (peek out) | 8.3 |
| 8.5 | **Gravity / falling** | When "sitting" window closes, avatar falls to taskbar with physics | 8.2, Phase 6 |

**Reference:** Mate-Engine `AvatarWindowHandler.cs` — `ProbeForWindows()`, `FindNearestWindowTop()`, `SitOnWindow()`, `GetWindowAtPoint()` using Win32 `EnumWindows`/`GetWindowRect`  
**Reference:** Mate-Engine `AvatarTaskbarController.cs` — `FindTaskbar()`, `SnapToTaskbar()`

**Exit Criteria:** Avatar can sit on taskbar, walk between positions, react to windows opening/closing.

---

### Phase 9: Chibi Mode

**Goal:** Toggle between normal and chibi (small body, big head) mode.  
**Why ninth:** Fun feature, low complexity once bone system is understood. Direct port from Mate-Engine.

| # | Task | Details | Depends On |
|---|------|---------|------------|
| 9.1 | **Create ChibiController.cs** | Scale armature root, head, upper legs per Mate-Engine values | Nothing |
| 9.2 | **Add hotkey toggle** | Assign a key (C) to toggle chibi mode | 9.1 |
| 9.3 | **Ground adjustment** | After scaling, adjust Y position so feet touch ground | 9.1 |
| 9.4 | **Transition animation** | Smooth scale interpolation over 0.3s instead of snap | 9.1 |
| 9.5 | **Sound/particle effects** | Optional: pop sound + particle burst on toggle (as Mate-Engine does) | 9.4 |

**Reference:** Mate-Engine `ChibiToggle.cs` — `chibiArmatureScale(0.3,0.3,0.3)`, `chibiHeadScale(2.7,2.7,2.7)`, `chibiUpperLegScale(0.6,0.6,0.6)`, `AdjustFeetToGround` coroutine

**Exit Criteria:** Press C → avatar shrinks to chibi proportions with smooth transition, press C again → returns to normal.

---

### Phase 10: Read-Aloud Phase 4 & Polish

**Goal:** Complete the read-aloud feature set and polish all systems.  
**Why last:** Nice-to-have. Core experience is complete by Phase 9.

| # | Task | Details | Depends On |
|---|------|---------|------------|
| 10.1 | **Read-Aloud overlay for non-browser** | Transparent overlay window showing text + highlight for PDFs, Word, etc. | Phase 2 |
| 10.2 | **Custom dance improvements** | Better dance loading, more animations, blend between dances | Phase 1 (VRMA test) |
| 10.3 | **Sleep mode** | After long inactivity, avatar transitions to sleep pose. Wake on interaction | Phase 4, Phase 7 |
| 10.4 | **Settings persistence (Unity side)** | Save window position, chibi state, volume, etc. across sessions (PlayerPrefs or JSON) | Various |
| 10.5 | **Performance profiling** | Profile CPU/GPU usage, optimize hot paths | All |
| 10.6 | **Error recovery** | Auto-reconnect WebSocket, graceful degradation if Python backend dies | Phase 1 |
| 10.7 | **Documentation update** | Update README with Unity setup instructions, hotkey reference | All |

**Exit Criteria:** Feature-complete desktop companion with all systems polished and documented.

---

## Quick Reference: Hotkey Map

| Key | Current Function | Phase |
|-----|-----------------|-------|
| **S** | Toggle silence (mute/unmute) | ✅ Done |
| **D** | Cycle dance mode (off → beat → full → off) | ✅ Done |
| **Q** | Pause read-aloud to ask question | ✅ Done |
| **R** | Resume read-aloud | ✅ Done |
| **1** | Active mode | ✅ Done |
| **2** | Idle mode | ✅ Done |
| **3** | Play Shikanoko VRMA | ✅ Done |
| **4** | Stop VRMA, return to active | ✅ Done |
| **Space** | Push-to-talk (if VAD disabled) | ✅ Done |
| **C** | Toggle chibi mode | Phase 9 |
| **Ctrl+Shift+R** | Trigger read-aloud | ✅ Done (Python) |
| **Ctrl+Shift+A** | Toggle active mode | ✅ Done (Python) |
| **Ctrl+Shift+D** | Cycle dance | ✅ Done (Python) |
| **Ctrl+Shift+M** | Toggle mute | ✅ Done (Python) |

---

## Architecture After All Phases

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Backend (UNCHANGED)                    │
│  ASR (Whisper) → LLM (Ollama) → TTS (GPT-SoVITS)              │
│  + Speaker ID, Read-Aloud, Memory, Audio Analyzer               │
│                              │                                   │
│              WebSocket Server (avatar_server.py :8765)           │
└──────────────────────────────┼──────────────────────────────────┘
                               │ WebSocket
┌──────────────────────────────▼──────────────────────────────────┐
│                    Unity Standalone (.exe)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─── Window Layer ───────────────────────────────────────────┐ │
│  │ TransparentWindowController (Win32 DWM/Layered)            │ │
│  │ • Transparent background, always-on-top, click-through     │ │
│  │ • Right-click drag, window position persistence            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─── Avatar Layer ───────────────────────────────────────────┐ │
│  │ AvatarController (VRM 1.0 via UniVRM)                      │ │
│  │ ├── LipSyncController (vowel cycling on speak)             │ │
│  │ ├── EmotionController (expression blend shapes)            │ │
│  │ ├── BlinkController (randomized auto-blink)                │ │
│  │ ├── EyeTrackingController (mouse → eye + head + spine)     │ │
│  │ ├── IdleAnimationController (breathing, weight shift, etc) │ │
│  │ ├── SwayController (spring physics on drag)                │ │
│  │ ├── ChibiController (scale toggle)                         │ │
│  │ └── TouchReactionController (click → reaction zones)       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─── Dance Layer ────────────────────────────────────────────┐ │
│  │ BeatDanceController (procedural, audio-reactive)           │ │
│  │ VrmaAnimationController (choreographed VRMA files)         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─── Desktop Interaction Layer ──────────────────────────────┐ │
│  │ WindowSitController (taskbar + window top sitting)         │ │
│  │ LocomotionController (walk between positions)              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─── UI Layer ───────────────────────────────────────────────┐ │
│  │ SubtitleController (speech bubble + read-aloud highlight)  │ │
│  │ ModeIndicator (status dot/icon)                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─── Core Layer ─────────────────────────────────────────────┐ │
│  │ WebSocketClient + MessageHandler + HotkeyManager           │ │
│  │ CompanionManager (state machine, coordinates all above)    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Technical Decisions

1. **Keep Python backend 100% unchanged** — It works. Don't touch working infrastructure.
2. **WebSocket, not TCP** — Already implemented on both sides. No reason to change transport.
3. **ControlRig for bone transforms** — VRM10 runtime overrides direct bone access. Always use `vrm.Runtime.ControlRig.GetBoneTransform()`.
4. **MCP `script-execute` for Unity automation** — `gameobject-find` and `component-add` timeout. Script execution is reliable.
5. **MCP screenshots are 180° rotated** — Trust user reports over MCP screenshots for visual verification.
6. **URP stays** — Our project uses URP 17.3.0. Mate-Engine uses Built-in RP. Adapt patterns, don't adopt their render pipeline.
7. **Spring bones from UniVRM** — No need for custom spring physics on hair/clothes. UniVRM handles it.

---

## Files to Create (Future Phases)

| File | Phase | Purpose |
|------|-------|---------|
| `unity/Scripts/Avatar/SwayController.cs` | 6 | Spring physics on window drag |
| `unity/Scripts/Avatar/TouchReactionController.cs` | 7 | Click zone detection + reactions |
| `unity/Scripts/Avatar/ChibiController.cs` | 9 | Chibi mode bone scaling |
| `unity/Scripts/UI/SubtitleController.cs` | 5 | Speech bubble TTS text display |
| `unity/Scripts/UI/ModeIndicator.cs` | 5 | Current mode status display |
| `unity/Scripts/Desktop/WindowSitController.cs` | 8 | Sit on taskbar/windows |
| `unity/Scripts/Desktop/LocomotionController.cs` | 8 | Walk between desktop positions |
| `unity/Scripts/Desktop/SleepController.cs` | 10 | Sleep after inactivity |

## Files to Modify (Future Phases)

| File | Phase | Changes |
|------|-------|---------|
| `unity/Scripts/Avatar/EyeTrackingController.cs` | 3 | Add head/spine tracking, smooth interp, dead-zone |
| `unity/Scripts/Avatar/IdleAnimationController.cs` | 4 | Add weight shift, micro-movements, idle variations |
| `unity/Scripts/CompanionManager.cs` | 5+ | Wire new controllers, add states |
| `unity/Scripts/Input/HotkeyManager.cs` | 9 | Add C key for chibi toggle |
| `unity/Scripts/Core/MessageHandler.cs` | 5, 7 | Add subtitle + touch reaction message types |
| `start_annabeth.ps1` | 2 | Add Unity .exe launch option |

---

*This plan replaces UNITY_MIGRATION_PLAN.md and READ_ALOUD_IMPLEMENTATION_PLAN.md as the single source of truth for all remaining work.*
