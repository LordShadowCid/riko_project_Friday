# Annabeth Feature Gap Closure Plan — 24 Features

> **Base commit**: `17e5e76` (Tier 1 improvements complete, 134/135 tests pass)  
> **Skipped by user**: Chibi Mode, Discord Rich Presence  
> **Mate Engine reference**: `MATE_ENGINE_REFERENCE.md` (architecture + code patterns)  
> **Goal**: Feature parity with Mate Engine + Annabeth's unique voice AI pipeline

---

## Score Card (before this plan)

| Category | Done | Gap |
|----------|------|-----|
| Window Management | 5 | 4 |
| VRM/Avatar | 6 | 1 |
| Animation (Idle/Dance) | 8 | 5 |
| Tracking | 6 | 1 |
| Desktop Interaction | 7 | 6 |
| Drag/Sway | 4 | 1 |
| Sleep | 2 | 2 |
| Settings | 8 | 2 |
| Interaction | 3 | 2 |
| **TOTAL** | **49** | **24** |

After this plan: **73 features** — full Mate Engine parity (minus Chibi + Discord).

---

## Phase 1: Desktop Presence Polish
> **Goal**: Make the avatar feel physically grounded on the desktop.  
> **Priority**: HIGH — these are the most visible missing behaviors.  
> **Est. total**: ~650 new/modified lines across 6 files

### Feature 1: Occluder Quads (avatar hides behind foreground windows)
**What**: When the avatar is sitting on a window's title bar, foreground windows that overlap should visually occlude (hide) parts of the avatar. Dynamic quad meshes are created at runtime to mask the avatar behind those windows.

**Mate Engine reference**: `AvatarWindowHandler.cs` → `UpdateOccluderQuadsFrameSync()`
- Enumerate visible windows via `EnumWindows` + `GetWindowRect`
- For each window whose Z-order is in front of the seated window AND whose rect overlaps the avatar, create a black unlit quad mesh at matching screen position
- Quads use a shader that writes to depth buffer but not color buffer (ZWrite On, ColorMask 0) — this masks the avatar via depth testing
- Quads are recycled via a pool to avoid GC allocation

**Files to modify**:
- `Core/WindowSnapper.cs` — Add occluder quad management (pooling, positioning, Z-sorting)
- `CompanionManager.cs` — Wire occluder enable/disable to sitting state

**New code needed**:
- `OccluderQuad` pool: `List<GameObject>` with MeshRenderer + unlit depth-only material
- `UpdateOccluders()` called each frame while sitting — enumerate windows, calculate screen rects, position quads
- Win32: reuse existing `EnumWindows`, add `GetWindowLong(GWL_EXSTYLE)` to skip transparent/tool windows

**Estimated lines**: ~180  
**Difficulty**: Medium-High

---

### Feature 2: SmoothDamp Window Following
**What**: When the avatar is sitting on a window that moves, the avatar should follow with prediction-smoothed movement instead of jerky frame-by-frame `MoveWindow`.

**Mate Engine reference**: `AvatarWindowHandler.cs` → `PinToTarget()`
- Uses `Mathf.SmoothDamp` with separate velocity tracking for X and Y
- Predicts target position based on velocity (0.5× `deltaTime` ahead)
- `smoothTime` ~0.08f for responsive but smooth following

**Files to modify**:
- `Core/WindowSnapper.cs` — Replace direct `MoveWindow` tracking with SmoothDamp

**Key pattern**:
```csharp
private float _followVelX, _followVelY;
private const float SmoothTime = 0.08f;

void PinToTarget(int targetX, int targetY) {
    float smoothX = Mathf.SmoothDamp(currentX, targetX, ref _followVelX, SmoothTime);
    float smoothY = Mathf.SmoothDamp(currentY, targetY, ref _followVelY, SmoothTime);
    MoveWindow(hwnd, (int)smoothX, (int)smoothY, width, height, false);
}
```

**Estimated lines**: ~40  
**Difficulty**: Easy

---

### Feature 3: Procedural Walking Animation
**What**: When the avatar walks across the desktop (via `DesktopLocomotionController`), play a procedural walk cycle — arm swing, leg stride, head bob — instead of just sliding the window.

**Mate Engine reference**: `AvatarLocomotionController.cs` — Sets `"IsWalking"` Animator bool → plays walk clip. Since we use procedural animation (no Animator clips), we'll create our own procedural walk cycle matching our code-driven bone approach.

**New file**: `Avatar/WalkAnimationController.cs`

**Procedural walk cycle (13 bones)**:
- **Legs**: Left/right upper+lower leg sinusoidal stride (opposite phase)
- **Arms**: Counter-swing opposite to same-side leg (natural gait)
- **Hips**: Lateral sway (weight shift) + subtle vertical bob
- **Spine/Chest**: Counter-rotation to hips (torso twist)
- **Head**: Slight compensation to keep head level
- Implements `IBlendableAnimation` for smooth transition in/out
- `walkSpeed` parameter synced with actual window movement speed

**Files to modify**:
- `Core/DesktopLocomotionController.cs` — Fire `OnWalkStart`/`OnWalkStop` events
- `CompanionManager.cs` — Wire walk events to `WalkAnimationController` + blend controller

**Estimated lines**: ~200  
**Difficulty**: Medium

---

### Feature 4: Dragging Held Pose
**What**: When the user grabs and holds the avatar (dragging the window), play a "held" pose — arms reach up, legs dangle, slight squirm. Distinct from the spring sway which handles the physics.

**Mate Engine reference**: `AvatarAnimatorController.cs` sets `isDragging` → Animator plays drag state clip. We'll do it procedurally.

**New file**: `Avatar/DragPoseController.cs`

**Procedural held pose**:
- **Arms**: Raise toward grab point (IK-like reach toward top of screen)
- **Legs**: Relax/dangle (slight knee bend, gentle pendulum sway)
- **Head**: Look up toward cursor
- Implements `IBlendableAnimation` — blend in on drag start, blend out on release
- Trigger on `TransparentWindowController.OnDragStart`, release on `OnDragEnd`

**Files to modify**:
- `CompanionManager.cs` — Wire drag events to DragPoseController + blend controller

**Estimated lines**: ~130  
**Difficulty**: Medium

---

### Feature 5: Drag-Hold Window Sit Guard
**What**: Require the user to hold the drag for at least 1 second before the avatar can "sit" on a window title bar. Prevents accidental sits during quick repositioning.

**Mate Engine reference**: `AvatarWindowHandler.cs` → `minDragHoldSecondsToSit = 1.0f`

**Files to modify**:
- `Core/WindowSnapper.cs` — Add `_dragHoldTimer` that starts counting on drag begin. Only allow `TryWindowSit()` when timer exceeds threshold.

**Key pattern**:
```csharp
private float _dragHoldTimer;
private const float MinDragHoldToSit = 1.0f;

// In drag tracking:
_dragHoldTimer += Time.deltaTime;
if (_dragHoldTimer < MinDragHoldToSit) return; // Don't snap yet
```

**Estimated lines**: ~10  
**Difficulty**: Easy

---

## Phase 2: Visual Quality & Feel
> **Goal**: Polish the avatar's visual behavior so it feels alive.  
> **Priority**: MEDIUM — these add delight and reduce repetitiveness.  
> **Est. total**: ~400 new/modified lines across 5 files

### Feature 6: Sleep Animation
**What**: When the avatar enters sleep mode, close eyes, tilt head down, slow breathing — not just reduce FPS. On wake, play a gentle wake-up sequence (blink, stretch, head lift).

**Mate Engine reference**: `AvatarSleepController.cs` → `IsSleeping` Animator bool → sleep clip. We'll extend our existing `SleepController.cs` with procedural sleep behavior.

**Files to modify**:
- `Core/SleepController.cs` — Add `OnSleepStart` / `OnWakeUp` events, configurable wake triggers
- `Avatar/BlinkController.cs` — Add `ForceClose()` / `ForceOpen()` methods for sleep
- `Avatar/IdleAnimationController.cs` — Sleep mode: reduce breathing, tilt head down 15°
- `CompanionManager.cs` — Wire sleep events to blink + idle controllers

**Sleep behavior**:
- On sleep: Blend eye close (VRM expression `Blink` → 1.0), head pitch down 15°, reduce breath amplitude 50%
- On wake: Rapid blinks (3×), head returns to center, normal breathing resumes
- Wake triggers: Mouse within window bounds, keyboard input, incoming WebSocket message

**Estimated lines**: ~90  
**Difficulty**: Medium

---

### Feature 7: Idle Variation System
**What**: Instead of the same breathing/sway forever, cycle between 2-3 idle behaviors every 15-20 seconds. Variations: **Relaxed** (current default), **Alert** (faster head tracking, slightly wider eyes), **Bored** (slow blinks, head drift speed up, occasional sigh — hips drop slightly).

**Mate Engine reference**: `BlendTreeLooper.cs` (75 lines) — StateMachineBehaviour that cycles through blend tree entries via lerped float parameter. Since we're procedural, we implement the concept differently: adjust parameter profiles in `IdleAnimationController` on a timer.

**Files to modify**:
- `Avatar/IdleAnimationController.cs` — Add `IdleVariation` enum, timer-based cycling, per-variation parameter profiles

**Key pattern**:
```csharp
enum IdleVariation { Relaxed, Alert, Bored }
struct VariationProfile { float breathSpeed, headDriftSpeed, leanMax, blinkRate; }
// Timer cycles every 15-20s, smooth-lerp between profiles
```

**Estimated lines**: ~80  
**Difficulty**: Medium

---

### Feature 8: Screen-Edge Hide Animations
**What**: When the avatar walks to or is dragged to a screen edge and "peeks," play a lean/tilt animation — body leans away from the edge, head turns to peek back toward screen center.

**Mate Engine reference**: `AvatarHideHandler.cs` → Sets `HideLeft` / `HideRight` Animator bools → plays lean clip. We'll do it procedurally in the existing `DesktopLocomotionController`.

**Files to modify**:
- `Core/DesktopLocomotionController.cs` — Add `OnPeekStart(Side)` / `OnPeekEnd` events
- `Avatar/IdleAnimationController.cs` or new peek handler — Procedural lean: spine tilt ±20° away from edge, head counter-tilt ±10° toward center

**Estimated lines**: ~60  
**Difficulty**: Easy-Medium

---

### Feature 9: Desktop Ambient Lighting Probe
**What**: Sample the dominant color from the desktop area behind the avatar and use it to tint the avatar's lighting. Creates the illusion that the avatar exists in the same light environment as the desktop.

**Mate Engine reference**: Not in Mate Engine (this is an enhancement beyond both). Uses `ReadPixels` or a separate low-res screenshot sampled at the avatar window position.

**New file**: `Core/DesktopAmbientProbe.cs`

**Approach**:
- Every 0.5s, use Win32 `BitBlt` to capture a small region (64×64) behind the avatar window position
- Calculate average RGB → set as ambient light color on the avatar's Light component
- Fade smoothly between probes to avoid flicker
- Settings: `enableAmbientProbe`, `probeIntensity`

**Files to modify**:
- `CompanionManager.cs` — Add Light reference, wire probe updates
- `Core/SettingsManager.cs` — Add `enableAmbientProbe`, `ambientProbeIntensity` fields

**Estimated lines**: ~120  
**Difficulty**: Medium

---

### Feature 10: State-Aware Tracking Permissions
**What**: Reduce or disable eye/head/body tracking during certain states (dancing, sleeping, being dragged). During dance: eyes should follow beat, not cursor. During sleep: eyes closed, no tracking. During drag: look up at cursor (different tracking mode).

**Mate Engine reference**: `AvatarMouseTracking.cs` → `TrackingPermission` system with per-state `allowHead`/`allowSpine`/`allowEye` bools.

**Files to modify**:
- `Avatar/EyeTrackingController.cs` — Add `SetTrackingMode(TrackingMode mode)` where mode = Normal, Reduced (dance), Disabled (sleep), LookUp (drag)
- `CompanionManager.cs` — Wire mode changes to state transitions

**Estimated lines**: ~50  
**Difficulty**: Easy

---

## Phase 3: Interaction Enrichment
> **Goal**: Make the avatar feel reactive and interactive beyond voice AI.  
> **Priority**: MEDIUM — fun features that add personality.  
> **Est. total**: ~480 new/modified lines across 7 files

### Feature 11: Pet/Pat Detection
**What**: Detect circular mouse motion over the avatar's head → trigger a pleased reaction (happy expression, slight head tilt into the petting motion, optional purring sound).

**Mate Engine reference**: `PetVoiceReactionHandler.cs` — Tracks mouse angle delta over time. If cumulative angle exceeds threshold (360°+), triggers pet reaction.

**New file**: `Interaction/PetDetectionController.cs`

**Algorithm**:
```
Track mouse positions in a ring buffer (last 20 frames)
Calculate angle between consecutive positions relative to avatar head center
Sum angles — if |sum| > 2π (full circle), fire OnPetDetected event
Reset accumulator after firing or on timeout (0.5s no motion)
```

**Files to modify**:
- `CompanionManager.cs` — Wire `OnPetDetected` → happy expression + head tilt
- `Interaction/TouchReactionController.cs` — Add pet reaction type

**Estimated lines**: ~90  
**Difficulty**: Medium

---

### Feature 12: AI Random Messages
**What**: Periodically, the avatar can generate a spontaneous comment via the LLM — reacting to time of day, how long the user has been idle, or random thoughts. Displayed in the speech bubble.

**Mate Engine reference**: `AvatarRandomMessages.cs` — Uses `LLMUnitySamples.Bubble` for LLM integration, allowed states whitelist. Our version sends a WebSocket request to the Python backend (which already has Ollama) with a special prompt.

**Files to modify**:
- `Core/MessageHandler.cs` — Add `SendRandomPrompt()` method with context (time of day, idle duration)
- `CompanionManager.cs` — Add timer (every 5-15 min configurable), fire random prompt in idle state only
- `Core/SettingsManager.cs` — Add `enableRandomMessages`, `randomMessageIntervalMinutes`

**Random prompt templates**:
```
"It's {time}. You've been quiet for {minutes} minutes. Say something brief and in-character."
"Comment on something fun or interesting. Keep it to one sentence."
```

**Estimated lines**: ~70  
**Difficulty**: Easy

---

### Feature 13: File Drop Support (drag VRM onto window)
**What**: Allow the user to drag-and-drop a `.vrm` file onto the avatar window to load it immediately.

**Mate Engine reference**: UniWindowController supports `onFilesDropped` callback natively. Since we use custom Win32, we need to implement `DragAcceptFiles` + `WM_DROPFILES` message handling.

**Files to modify**:
- `Core/TransparentWindowController.cs` — Add `DragAcceptFiles(hwnd, true)` in init, handle `WM_DROPFILES` in a message loop or via `WndProc` subclass
- `CompanionManager.cs` — Wire dropped file path → `AvatarController.LoadModel(path)`

**Win32 additions**:
```csharp
[DllImport("shell32.dll")] static extern void DragAcceptFiles(IntPtr hWnd, bool fAccept);
[DllImport("shell32.dll")] static extern uint DragQueryFile(IntPtr hDrop, uint iFile, StringBuilder lpszFile, uint cch);
[DllImport("shell32.dll")] static extern void DragFinish(IntPtr hDrop);
// Need WndProc hook or polling via GetMessage
```

**Estimated lines**: ~80  
**Difficulty**: Medium-High (WndProc subclassing in Unity is tricky)

---

### Feature 14: VRM Metadata Display
**What**: When a VRM model is loaded, extract and display its metadata (title, author, version) in the model library UI and optionally in the settings panel.

**Mate Engine reference**: `VRMLoader.cs` → Extracts `Vrm10Instance.Vrm.Meta` after loading.

**Files to modify**:
- `Avatar/VrmModelLibrary.cs` — After scanning, extract metadata via `Vrm10.LoadPathAsync` (lightweight) or parse VRM JSON header directly
- `UI/SettingsPanel.cs` — Show "Current Model: {title} by {author}" label

**Key pattern** (UniVRM):
```csharp
var vrm10 = loadedInstance.GetComponent<Vrm10Instance>();
string title = vrm10.Vrm.Meta.Name;
string author = vrm10.Vrm.Meta.Authors[0];
```

**Estimated lines**: ~40  
**Difficulty**: Easy

---

### Feature 15: Alarms & Timers
**What**: Set countdown timers or alarms. When triggered, the avatar reacts — speech bubble shows "Time's up!", plays a sound, shows alert expression.

**Mate Engine reference**: `SaveLoadHandler.cs` → `AlarmEntry` / `TimerEntry` in settings data.

**New file**: `Core/AlarmTimerManager.cs`

**Data model**:
```csharp
[Serializable]
public class TimerEntry {
    public string label;
    public float durationSeconds;
    public float remainingSeconds;
    public bool isRunning;
}
```

**Files to modify**:
- `Core/SettingsManager.cs` — Add `List<TimerEntry>` to settings data
- `UI/SettingsPanel.cs` — Add timer section (add/remove/start/stop)
- `CompanionManager.cs` — Wire timer fires to speech bubble + expression

**Estimated lines**: ~150  
**Difficulty**: Medium

---

## Phase 4: Dance & Animation Expansion
> **Goal**: Expand the dance system to support user-provided animations.  
> **Priority**: MEDIUM-LOW — nice-to-have but complex.  
> **Est. total**: ~550 new/modified lines across 5 files

### Feature 16: Custom Dance Loading (.unity3d AssetBundles)
**What**: Load custom dance animation files from a `StreamingAssets/CustomDances/` folder. Each `.unity3d` AssetBundle contains an AnimationClip + AudioClip.

**Mate Engine reference**: `AvatarDanceHandler.cs` (1401 lines) — `TryAddUnity3D()` reads AssetBundle, extracts clips. Uses `AnimatorOverrideController` to replace a placeholder clip.

**Since Annabeth uses procedural animation (no Animator), our approach differs**:
- Load AssetBundle → extract `AnimationClip` + `AudioClip`
- Create a temporary `Animation` component (legacy) or `PlayableGraph` to play the clip
- Or: Add a minimal Animator Controller with one state, override its clip

**New file**: `Dance/CustomDanceLoader.cs`

**Approach (PlayableGraph — cleanest for our architecture)**:
```csharp
var graph = PlayableGraph.Create("CustomDance");
var clipPlayable = AnimationClipPlayable.Create(graph, loadedClip);
var output = AnimationPlayableOutput.Create(graph, "Output", animator);
output.SetSourcePlayable(clipPlayable);
graph.Play();
// AudioSource for music
```

**Files to modify**:
- `Dance/VrmaAnimationController.cs` — Extend to support .unity3d in addition to .vrma
- `CompanionManager.cs` — Add custom dance scanning + selection
- `UI/SettingsPanel.cs` or `UI/RadialMenu.cs` — Dance selection list

**Estimated lines**: ~200  
**Difficulty**: High

---

### Feature 17: Dance UI Player
**What**: A minimal dance player UI with play/pause, next/prev, progress bar, and volume control for custom dances.

**Mate Engine reference**: `AvatarDanceHandler.cs` → Full UI with buttons, slider, text labels.

**New file**: `UI/DancePlayerPanel.cs`

**UI elements** (built at runtime via UIFactory):
- Play/Pause toggle button
- Previous / Next buttons
- Progress slider (bound to `AudioSource.time / clip.length`)
- Volume slider (bound to `AudioSource.volume`)
- Song name label
- Shuffle / Loop toggle

**Files to modify**:
- `UI/UIFactory.cs` — May need a horizontal button group helper

**Estimated lines**: ~160  
**Difficulty**: Medium

---

### Feature 18: Blendshape Forwarding During Dance
**What**: When playing a custom dance clip that contains blendshape/expression animation curves (common in MMD dances), forward those curves to the VRM model's SkinnedMeshRenderers.

**Mate Engine reference**: `AvatarDanceShapeConverter.cs` (163 lines) — Creates a proxy Animator with PlayableGraph that reads curves from dance clip and applies them to the VRM's SkinnedMeshRenderer. Searches candidate paths: "Body", "Model/Body", "Face".

**Files to modify**:
- `Dance/CustomDanceLoader.cs` — After loading a clip, scan for blendshape curves and create a forwarding PlayableGraph
- Or create `Dance/DanceBlendshapeForwarder.cs`

**Key pattern**:
```csharp
// Check if clip has blendshape curves
foreach (var binding in AnimationUtility.GetCurveBindings(clip)) {
    if (binding.type == typeof(SkinnedMeshRenderer) && binding.propertyName.StartsWith("blendShape."))
        // Has blendshape animations → set up forwarding
}
```

**Estimated lines**: ~100  
**Difficulty**: Medium-High

---

### Feature 19: VMD Motion Player
**What**: Load and play `.vmd` (Vocaloid Motion Data) files — the standard format for MMD dance motions. Convert VMD bone data to Unity Humanoid rig.

**Mate Engine reference**: References `VroidMMDTools` for VMD→Unity conversion.

**New file**: `Dance/VmdPlayer.cs`

**Approach**:
- Parse VMD binary format (header, bone frames, morph frames)
- Map MMD bone names → Unity HumanBodyBones (standard mapping table)
- Convert local rotations from MMD coordinate system (left-handed, Y-up) to Unity
- Create AnimationClip at runtime from parsed frames

**Or**: Use existing open-source VMD parser (e.g., UniVMD) as a package reference.

**Estimated lines**: ~200 (parser) or ~30 (if using package)  
**Difficulty**: High (from scratch) / Easy (with package)

---

## Phase 5: Window & Input Refinement
> **Goal**: Polish edge cases and add quality-of-life improvements.  
> **Priority**: LOW — these fix minor issues and add nice-to-haves.  
> **Est. total**: ~320 new/modified lines across 6 files

### Feature 20: Opacity-Based Hit Test
**What**: Instead of testing clicks against the mesh bounding box, read the actual rendered pixel alpha at the cursor position. Only capture clicks where the avatar is visually opaque (alpha > 0.1). This prevents edge artifacts where transparent pixels near the character still block clicks.

**Mate Engine reference**: `UniWindowController` → `isClickThrough` based on pixel alpha at cursor via `GetPixel()` native call. Configurable `clickThroughAlphaThreshold = 0.1f`.

**Files to modify**:
- `Core/TransparentWindowController.cs` — Add optional `ReadPixels`-based hit test alongside existing bounds raycast. Use `RenderTexture.ReadPixels()` at cursor position (1×1 pixel read) and check alpha.

**Fallback**: Keep existing bounds+raycast as default, opacity test as opt-in (it's slightly more expensive).

**Estimated lines**: ~60  
**Difficulty**: Medium

---

### Feature 21: Smooth Drag Movement
**What**: Add smoothing to window movement during drag so the avatar doesn't rigidly snap to cursor position every frame.

**Mate Engine reference**: `UniWindowMoveHandle.cs` → `dragSmooth` parameter.

**Files to modify**:
- `Core/TransparentWindowController.cs` — Apply `Vector2.SmoothDamp` or lerp between current window position and target cursor position during drag

**Estimated lines**: ~20  
**Difficulty**: Easy

---

### Feature 22: Multi-Monitor Adjacency Detection
**What**: When deciding whether the avatar can "hide" at a screen edge, check if another monitor is adjacent to that edge. If so, don't hide there — the avatar would appear to clip into the other monitor.

**Mate Engine reference**: `AvatarHideHandler.cs` → `GetAllowedEdgesForMonitor()` with `adjacencyTolerancePx` (6) and `adjacencyMinVerticalOverlapPx` (32).

**Files to modify**:
- `Core/DesktopLocomotionController.cs` — Add `GetAllowedHideEdges()` using `EnumDisplayMonitors` + rect overlap check

**Win32**:
```csharp
[DllImport("user32.dll")] static extern bool EnumDisplayMonitors(IntPtr hdc, IntPtr lprcClip, MonitorEnumProc lpfnEnum, IntPtr dwData);
delegate bool MonitorEnumProc(IntPtr hMonitor, IntPtr hdcMonitor, ref RECT lprcMonitor, IntPtr dwData);
```

**Estimated lines**: ~70  
**Difficulty**: Medium

---

### Feature 23: Graphics Quality Setting
**What**: Allow the user to adjust Unity's quality level (Low/Medium/High) from settings. Affects shadow resolution, anti-aliasing, LOD bias.

**Files to modify**:
- `Core/SettingsManager.cs` — Add `graphicsQuality` field (0=Low, 1=Medium, 2=High)
- `UI/SettingsPanel.cs` — Add quality dropdown/slider
- Apply via `QualitySettings.SetQualityLevel(level)`

**Estimated lines**: ~30  
**Difficulty**: Easy

---

### Feature 24: Sound Threshold & Per-App Audio Filtering
**What**: Configure the audio level threshold that triggers dance mode, and optionally filter which applications' audio is detected (e.g., only react to Spotify, not system sounds).

**Mate Engine reference**: `AvatarAnimatorController.cs` → `SOUND_THRESHOLD` (0.02f) + `allowedApps` list. Uses NAudio to enumerate audio sessions per-process.

**This applies to the Python backend** (which does WASAPI analysis) rather than Unity:

**Files to modify**:
- `Core/SettingsManager.cs` — Add `soundThreshold`, `soundFilterApps` (comma-separated string)
- `UI/SettingsPanel.cs` — Add threshold slider + text field for app filter
- `Core/WebSocketClient.cs` — Send threshold/filter config to Python backend on settings change
- **Python side**: `client/audio_analyzer.py` — Add per-session filtering using `pycaw` (IAudioSessionManager2)

**Estimated lines**: ~80 (Unity) + ~40 (Python)  
**Difficulty**: Medium

---

## Implementation Order (Recommended)

```
Sprint 1 (Quick Wins — Easy features first):
  ├─ Feature 5:  Drag-Hold Sit Guard .............. ~10 lines, Easy
  ├─ Feature 14: VRM Metadata Display ............. ~40 lines, Easy
  ├─ Feature 21: Smooth Drag Movement ............. ~20 lines, Easy  
  ├─ Feature 23: Graphics Quality Setting ......... ~30 lines, Easy
  └─ Feature 10: State-Aware Tracking ............. ~50 lines, Easy
   Total: ~150 lines, all Easy

Sprint 2 (SmoothDamp + Sleep + Idle Variations):
  ├─ Feature 2:  SmoothDamp Window Following ...... ~40 lines, Easy
  ├─ Feature 6:  Sleep Animation .................. ~90 lines, Medium
  ├─ Feature 7:  Idle Variation System ............ ~80 lines, Medium
  └─ Feature 12: AI Random Messages ............... ~70 lines, Easy
   Total: ~280 lines, Easy-Medium mix

Sprint 3 (Walk + Drag Pose + Interaction):
  ├─ Feature 3:  Procedural Walking Animation ..... ~200 lines, Medium
  ├─ Feature 4:  Dragging Held Pose ............... ~130 lines, Medium
  ├─ Feature 8:  Screen-Edge Hide Animations ...... ~60 lines, Easy-Medium
  └─ Feature 11: Pet/Pat Detection ................ ~90 lines, Medium
   Total: ~480 lines, Medium

Sprint 4 (Desktop Integration):
  ├─ Feature 1:  Occluder Quads ................... ~180 lines, Medium-High
  ├─ Feature 9:  Desktop Ambient Lighting ......... ~120 lines, Medium
  ├─ Feature 13: File Drop Support ................ ~80 lines, Medium-High
  └─ Feature 22: Multi-Monitor Adjacency .......... ~70 lines, Medium
   Total: ~450 lines, Medium-High

Sprint 5 (Dance Expansion):
  ├─ Feature 16: Custom Dance Loading ............. ~200 lines, High
  ├─ Feature 17: Dance UI Player .................. ~160 lines, Medium
  ├─ Feature 18: Blendshape Forwarding ............ ~100 lines, Medium-High
  └─ Feature 15: Alarms & Timers .................. ~150 lines, Medium
   Total: ~610 lines, High complexity

Sprint 6 (Advanced Polish):
  ├─ Feature 19: VMD Motion Player ................ ~200 lines, High
  ├─ Feature 20: Opacity-Based Hit Test ........... ~60 lines, Medium
  └─ Feature 24: Sound Threshold/Per-App .......... ~120 lines, Medium
   Total: ~380 lines, High

GRAND TOTAL: ~2,350 new/modified lines across 24 features
```

---

## New Files to Create (8)

| File | Feature | Lines |
|------|---------|-------|
| `Avatar/WalkAnimationController.cs` | #3 Walking | ~200 |
| `Avatar/DragPoseController.cs` | #4 Drag Pose | ~130 |
| `Interaction/PetDetectionController.cs` | #11 Pet/Pat | ~90 |
| `Core/DesktopAmbientProbe.cs` | #9 Ambient | ~120 |
| `Core/AlarmTimerManager.cs` | #15 Timers | ~150 |
| `Dance/CustomDanceLoader.cs` | #16 Custom Dance | ~200 |
| `UI/DancePlayerPanel.cs` | #17 Dance UI | ~160 |
| `Dance/VmdPlayer.cs` | #19 VMD | ~200 |

## Existing Files to Modify (14)

| File | Features |
|------|----------|
| `CompanionManager.cs` | #1, #3, #4, #6, #9, #10, #11, #12, #13, #15 |
| `Core/WindowSnapper.cs` | #1, #2, #5 |
| `Core/TransparentWindowController.cs` | #13, #20, #21 |
| `Core/DesktopLocomotionController.cs` | #3, #8, #22 |
| `Core/SettingsManager.cs` | #9, #12, #15, #23, #24 |
| `Core/SleepController.cs` | #6 |
| `Avatar/EyeTrackingController.cs` | #10 |
| `Avatar/IdleAnimationController.cs` | #6, #7, #8 |
| `Avatar/BlinkController.cs` | #6 |
| `Avatar/VrmModelLibrary.cs` | #14 |
| `UI/SettingsPanel.cs` | #14, #15, #23, #24 |
| `Core/MessageHandler.cs` | #12 |
| `Core/WebSocketClient.cs` | #24 |
| `Dance/VrmaAnimationController.cs` | #16 |

---

## Mate Engine Patterns Being Reused

| Pattern | From | Used In |
|---------|------|---------|
| Spring physics formula | `AvatarSwayController.cs` | Already done (Tier 1) |
| SmoothDamp following | `AvatarWindowHandler.PinToTarget()` | Feature #2 |
| Drag hold guard | `AvatarWindowHandler.minDragHoldSecondsToSit` | Feature #5 |
| Occluder quad pooling | `AvatarWindowHandler.UpdateOccluderQuadsFrameSync()` | Feature #1 |
| Tracking permissions | `AvatarMouseTracking.TrackingPermission` | Feature #10 |
| Pet angle accumulation | `PetVoiceReactionHandler` | Feature #11 |
| Blend tree cycling concept | `BlendTreeLooper.cs` | Feature #7 (procedural version) |
| VRM metadata extraction | `VRMLoader.cs` | Feature #14 |
| AnimatorOverrideController | `AvatarDanceHandler.cs` | Feature #16 |
| PlayableGraph blendshapes | `AvatarDanceShapeConverter.cs` | Feature #18 |
| NAudio per-app filtering | `AvatarAnimatorController.cs` | Feature #24 |
| HideLeft/HideRight concept | `AvatarHideHandler.cs` | Feature #8 |
| Monitor adjacency detection | `AvatarHideHandler.GetAllowedEdgesForMonitor()` | Feature #22 |
| Random LLM messages | `AvatarRandomMessages.cs` | Feature #12 |

---

## What's NOT Being Ported (and why)

| Mate Engine Feature | Reason |
|----|--------|
| Chibi Mode | User request: skip |
| Discord Rich Presence | User request: skip |
| Food System | Doesn't fit AI companion design |
| Mod System (AssetBundle .me) | Massive scope, not needed for personal companion |
| Multi-Instance Sync | Single companion design |
| Steam Workshop | No Steam distribution |
| Animator State Machine | Would require replacing entire procedural animation system |
| Component Injection (reflection) | Annabeth's hub pattern is better for this project |
| Accessories System | Low value, can add later if desired |
| Big Screen / Screen Saver | Low value for desktop companion |
