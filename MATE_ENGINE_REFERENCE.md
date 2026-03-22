# Mate Engine — Comprehensive Unity Reference

> **Source**: https://github.com/shinyflvre/Mate-Engine  
> **Render Pipeline**: Built-in (NOT URP)  
> **Languages**: ShaderLab 84.3%, C# 12.5%  
> **License**: Mixed GNU AGPL v3 & MateProv2  
> **Main Scene**: `Scenes - USED FOR MATE ENGINE > Mate Engine Main`  
> **Scripts Folder**: `Assets/MATE ENGINE - Scripts/`  
> **Packages Folder**: `Assets/MATE ENGINE - Packages/`  
> **Shaders Folder**: `Assets/MATE ENGINE - Shaders/`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [VRM Loading System](#2-vrm-loading-system)
3. [Avatar Handler System](#3-avatar-handler-system)
4. [Animation System](#4-animation-system)
5. [Custom Dance Player](#5-custom-dance-player)
6. [Window Management](#6-window-management)
7. [Input & Interaction](#7-input--interaction)
8. [Settings & Save System](#8-settings--save-system)
9. [Mod System](#9-mod-system)
10. [UniWindowController (Kirurobo)](#10-uniwindowcontroller-kirurobo)
11. [Key Patterns & Architecture](#11-key-patterns--architecture)
12. [Animator Parameters Reference](#12-animator-parameters-reference)
13. [Win32 / P/Invoke Reference](#13-win32--pinvoke-reference)

---

## 1. Project Overview

### Features
- Desktop companion with transparent window (walks on desktop, sits on windows/taskbar)
- VRM 0.x and VRM 1.0 model loading
- Sound-reactive idle ↔ dance transitions (NAudio)
- Mouse/eye/head/spine tracking
- Drag sway spring physics
- Window sitting with snap detection
- Screen-edge hiding (left/right)
- Chibi mode (shrink body, enlarge head)
- Custom dance player (.unity3d / .me format, multi-instance sync)
- Mod system (AssetBundle-based objects and dances)
- Sleep mode, food system, locomotion (random walk), accessories
- Persistent settings (JSON), Discord RPC, LLM chat bubbles
- Steam Workshop integration

### Architecture Overview
```
UniWindowController (Kirurobo)   ← Transparent borderless window
    └── Camera (Built-in RP)
         └── VRMLoader.cs        ← Loads .vrm / .me / DLC models
              └── componentTemplatePrefab
                   └── 30+ handler MonoBehaviours injected via reflection
                        ├── AvatarAnimatorController (NAudio sound detection, state machine)
                        ├── AvatarMouseTracking (driver pattern, head/spine/eye)
                        ├── AvatarWindowHandler (window sitting, snap probe)
                        ├── AvatarSwayController (spring physics on drag)
                        ├── AvatarDanceHandler (custom dance player, 1400+ lines)
                        ├── SaveLoadHandler (singleton, JSON persistence)
                        └── ... 20+ more handlers
```

---

## 2. VRM Loading System

### VRMLoader.cs (553 lines)
**Path**: `Assets/MATE ENGINE - Scripts/VRMLoader/VRMLoader.cs`

#### Fields
| Field | Type | Description |
|-------|------|-------------|
| `loadVRMButton` | Button | UI trigger |
| `mainModel` | GameObject | Default model reference |
| `customModelOutput` | Transform | Parent for loaded models |
| `animatorController` | RuntimeAnimatorController | Assigned to loaded model |
| `componentTemplatePrefab` | GameObject | **KEY** — prefab with all handler components |
| `currentModel` | GameObject | Currently active model |
| `isLoading` | bool | Loading guard |

#### Loading Pipeline
1. **Start()**: Loads saved model path from `SaveLoadHandler.Instance.data.selectedModelPath`
2. **LoadVRM(path)**: Determines format:
   - `.me` files → AssetBundle extract
   - DLC prefabs → Direct instantiate
   - `.vrm` files → VRM1.0 first (`Vrm10Data.Parse` → `Vrm10Importer`), VRM0.x fallback
3. **FinalizeLoadedModel()**: Critical finalization sequence:
   ```
   DisableMainModel()
   → ClearPreviousCustomModel()
   → Parent to customModelOutput
   → EnableSkinnedMeshRenderers()
   → AssignAnimatorController()
   → InjectComponentsFromPrefab(componentTemplatePrefab, currentModel)  ← REFLECTION
   → Extract metadata
   → MEModLoader.Instance.AssignHandlersForCurrentAvatar()
   → ReleaseRam()
   ```

#### InjectComponentsFromPrefab — Component Injection Pattern
The **most important architectural pattern** in Mate Engine. Uses reflection to copy ALL MonoBehaviour components from `componentTemplatePrefab` onto the loaded VRM model:
- Iterates all Components on the prefab
- For each MonoBehaviour, adds the same type to the target model
- Copies all serializable fields via reflection
- This injects 30+ handler scripts onto every loaded avatar

### AvatarLibraryMenu.cs
**Path**: `Assets/MATE ENGINE - Scripts/VRMLoader/AvatarLibraryMenu.cs`
- `DLCEntry` / `AvatarEntry` classes for model catalog
- Steam Workshop integration via `SteamWorkshopAutoLoader`
- Scans `StreamingAssets` and `persistentDataPath` for models

---

## 3. Avatar Handler System

**Path**: `Assets/MATE ENGINE - Scripts/AvatarHandlers/`

All handlers are injected onto loaded avatars via the `componentTemplatePrefab` pattern.

### AvatarMouseTracking.cs (215 lines)
**Mouse/head/spine/eye tracking with driver pattern**

#### TrackingPermission System
```csharp
[System.Serializable]
public class TrackingPermission {
    public string stateOrParameterName;  // Animator state or param
    public bool isParameter;             // true = check bool param, false = check state name
    public bool allowHead = true;
    public bool allowSpine = true;
    public bool allowEye = true;
}
```

#### Driver Pattern (Critical Architecture)
Instead of modifying bones directly (which gets overridden by Animator), creates **driver GameObjects**:
```
headDriver    → parented to head bone's parent
spineDriver   → parented to spine bone's parent
leftEyeDriver → parented to left eye bone's parent
rightEyeDriver→ parented to right eye bone's parent
```
- Each driver is an empty GameObject with its own transform
- LateUpdate reads cursor position, calculates target rotation
- Applies rotation to driver transforms using `Quaternion.Slerp` smoothing
- Blends with base rotation: `delta * baseRot`

#### Settings
| Field | Default | Description |
|-------|---------|-------------|
| `headYawLimit` | 45° | Max horizontal head rotation |
| `headPitchLimit` | 30° | Max vertical head rotation |
| `spineMinRotation` | -15° | Min spine yaw |
| `spineMaxRotation` | 15° | Max spine yaw |
| `eyeYawLimit` | 12° | Max eye horizontal rotation |

#### Key Details
- VRM1.0 LookAt integration with `YawPitchValue`
- LateUpdate checks animator state hash against `TrackingPermission` list
- Cascading spine rotation: chest 0.8×, upperChest 0.6×
- Smoothing via `Quaternion.Slerp`

---

### AvatarAnimatorController.cs (243 lines)
**Sound detection and animation state management**

#### NAudio Integration
```csharp
using NAudio.CoreAudioApi;
// MMDeviceEnumerator → GetDefaultAudioEndpoint → AudioMeterInformation.MasterPeakValue
```
- `CheckSoundContinuously` coroutine (2s interval)
- Checks `allowedApps` list for per-app sound filtering
- `SOUND_THRESHOLD` (0.02f) triggers dance mode

#### Static Hashed Parameters
```csharp
private static readonly int danceIndexParam = Animator.StringToHash("DanceIndex");
private static readonly int isIdleParam = Animator.StringToHash("isIdle");
private static readonly int isDraggingParam = Animator.StringToHash("isDragging");
private static readonly int isDancingParam = Animator.StringToHash("isDancing");
private static readonly int idleIndexParam = Animator.StringToHash("IdleIndex");
private static readonly int isMaleParam = Animator.StringToHash("isMale");
private static readonly int isFemaleParam = Animator.StringToHash("isFemale");
```

#### Settings
| Field | Default | Description |
|-------|---------|-------------|
| `SOUND_THRESHOLD` | 0.02f | Sound level to trigger dancing |
| `allowedApps` | List\<string\> | Apps to check for sound |
| `totalIdleAnimations` | 10 | Number of idle blend tree entries |
| `IDLE_SWITCH_TIME` | 12s | Time between idle switches |
| `DANCE_CLIP_COUNT` | 5 | Number of built-in dance clips |
| `DANCE_SWITCH_TIME` | 15s | Time between dance switches |
| `BlockDraggingOverride` | bool | Prevents drag state |

#### Smooth Transitions
- `SmoothIdleTransition` / `SmoothDanceTransition` — lerps float parameters for BlendTree transitions

---

### AvatarWindowHandler.cs (860+ lines)
**Window sitting with snap detection and occluder quads**

#### Key Concepts
- **Snap Probe**: Hip-based ray probe to detect window title bars
- **Guard Zone**: Prevents re-snapping too quickly
- `probeRadiusPx` (24): Detection radius in pixels
- `probeGuardPx` (240): Guard zone radius
- `minDragHoldSecondsToSit` (1.0): Must hold drag for 1s before sitting

#### Bone Caching
Caches: hips, leftUpperLeg, rightUpperLeg, leftLowerLeg, rightLowerLeg, leftFoot, rightFoot, head

#### Critical Methods
| Method | Description |
|--------|-------------|
| `TrySnap()` | Sets `isTaskbarSit` / `isWindowSit` bools |
| `CalibrateSeatAnchorToDesktopY()` | Binary search (20 iterations) for Desktop Y position |
| `PinToTarget()` | SmoothDamp with prediction for smooth following |
| `UpdateOccluderQuadsFrameSync()` | Creates quads to mask avatar behind windows |

#### Win32 API Usage
- `Kirurobo.WinApi` for window enumeration
- Window rect detection for sitting targets
- Monitor info for multi-monitor support

---

### AvatarHideHandler.cs (384 lines)
**Screen-edge hide/peek (left/right snap)**

#### Settings
| Field | Default | Description |
|-------|---------|-------------|
| `snapThresholdPx` | 12 | Pixels from edge to trigger snap |
| `unsnapThresholdPx` | 24 | Pixels from edge to unsnap |
| `smoothingTime` | 0.10f | SmoothDamp time |
| `keepTopmostWhileSnapped` | true | Force topmost while hidden |
| `unsnapGraceTime` | 0.12f | Grace period before unsnapping |

#### State
```csharp
enum Side { None, Left, Right }
Side snappedSide = Side.None;
```

#### Animator Integration
Sets `"HideLeft"` / `"HideRight"` animator bools

#### Multi-Monitor
- `GetAllowedEdgesForMonitor()` — detects monitor adjacency
- `adjacencyTolerancePx` (6), `adjacencyMinVerticalOverlapPx` (32)

---

### AvatarSwayController.cs (344 lines)
**Drag-sway spring physics on hips/arms/legs**

#### Spring Physics Formula
```csharp
static void Spring(ref float x, ref float v, float xt, float f, float z, float dt) {
    float w = f * 2f * Mathf.PI;
    float a = w * w * (xt - x) - 2f * z * w * v;
    v += a * dt;
    x += v * dt;
}
```

#### Settings
| Field | Default | Description |
|-------|---------|-------------|
| `useWindowVelocity` | true | Use window movement for sway |
| `maxLeanZ` | 25° | Max forward/back lean |
| `maxLeanX` | 12° | Max left/right lean |
| `springFrequency` | 2.6 | Spring oscillation frequency |
| `dampingRatio` | 0.35 | Spring damping (0 = no damping, 1 = critical) |

#### Joints Affected
- Hips (lean Z/X)
- Left/Right arms (additive swing, separate arm max angles)
- Left/Right legs (additive rotations)
- State whitelist pattern — only active in certain animator states

#### Win32
- `GetWindowRect` for window position tracking
- Calculates window velocity for physics input

---

### AvatarLocomotionController.cs (746 lines)
**Random walk across desktop — moves the Unity window**

#### Settings
| Field | Default | Description |
|-------|---------|-------------|
| Randomizer interval | 10s | Time between walk decisions |
| `MinWalkCycle` | 250px | Minimum walk distance |
| `MaxWalkCycle` | 550px | Maximum walk distance |
| `WindowSpeed` | 2 | Pixels per frame |

#### Key Methods
- `ResolveAnimatorSmart()` — cascade search for Animator component
- `PickDirectionByEdges()` — avoids walking off monitor
- Monitor/avatar bounds blocking

#### Win32 API
```csharp
[DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
[DllImport("user32.dll")] static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int nWidth, int nHeight, bool bRepaint);
[DllImport("user32.dll")] static extern IntPtr MonitorFromWindow(IntPtr hwnd, uint dwFlags);
[DllImport("user32.dll")] static extern bool GetMonitorInfo(IntPtr hMonitor, ref MONITORINFO lpmi);
```

---

### AvatarSleepController.cs (117 lines)
**Idle timeout → sleep state**

| Field | Default | Description |
|-------|---------|-------------|
| `enableSleep` | false | Master enable |
| `sleepTimer` | 60s | Seconds idle before sleeping |
| `allowedStates` | ["Idle","Sleeping"] | States that count toward sleep timer |
| `wakeUpBools` | ["isDragging"] | Bools that wake up from sleep |

Sets `"IsSleeping"` animator bool.

---

### AvatarFoodController.cs (433 lines)
**Food/treat system with mouse-follow and head interaction**

- Food entries follow mouse cursor
- Head interaction detection (proximity to head bone)
- Sway physics on food items (shares `Spring()` pattern with SwayController)
- `headLocalOffset`, `springFrequency` (2.6), `avatarProbeInterval` (0.25s)
- Random sound playback on interaction

---

### ChibiToggle.cs (104 lines)
**Chibi mode — shrink body, enlarge head**

```csharp
[RequireComponent(typeof(Animator))]
public class ChibiToggle : MonoBehaviour
```

#### Fields
| Field | Default | Description |
|-------|---------|-------------|
| `chibiArmatureScale` | (0.3, 0.3, 0.3) | Armature shrink scale |
| `chibiHeadScale` | (2.7, 2.7, 2.7) | Head enlarge scale |
| `chibiUpperLegScale` | (0.6, 0.6, 0.6) | Upper leg adjust scale |
| `audioSource` | AudioSource | Sound effects source |
| `chibiEnterSounds` | List\<AudioClip\> | Sounds on chibi enter |
| `chibiExitSounds` | List\<AudioClip\> | Sounds on chibi exit |
| `particleEffectObject` | GameObject | Particle effect on toggle |
| `particleDuration` | 4f | How long particles play |

#### Bones Cached
`armatureRoot`, `head`, `leftFoot`, `rightFoot`, `leftUpperLeg`, `rightUpperLeg`

#### ToggleChibiMode()
```csharp
armatureRoot.localScale = becomingChibi ? chibiArmatureScale : Vector3.one;
head.localScale = becomingChibi ? chibiHeadScale : Vector3.one;
leftUpperLeg.localScale = becomingChibi ? chibiUpperLegScale : Vector3.one;
rightUpperLeg.localScale = becomingChibi ? chibiUpperLegScale : Vector3.one;
```
- After scaling: `AdjustFeetToGround(originalFootY)` coroutine repositions avatar so feet stay on ground
- Plays random sound, triggers particles

#### ChibiSettingsData (Serializable, in MEModLoader.cs)
```csharp
public class ChibiSettingsData {
    public Vector3 chibiArmatureScale = new Vector3(0.3f, 0.3f, 0.3f);
    public Vector3 chibiHeadScale = new Vector3(2.7f, 2.7f, 2.7f);
    public Vector3 chibiUpperLegScale = new Vector3(0.6f, 0.6f, 0.6f);
    public float screenInteractionRadius = 30f;
    public float holdDuration = 2f;
}
```

---

### AvatarBubbleHandler.cs
**Speech/thought bubble attachment**

```csharp
[ExecuteAlways]
public class AvatarBubbleHandler : MonoBehaviour
```

| Field | Type | Description |
|-------|------|-------------|
| `avatarAnimator` | Animator | Reference |
| `animatorParameter` | string | "isSitting" — controls visibility |
| `attachTarget` | GameObject | The bubble object |
| `attachBone` | HumanBodyBones | Head (default) |
| `keepOriginalRotation` | bool | Don't rotate with bone |
| `activationKey` | KeyCode | Space (toggle) |
| `spawnAnimationSpeed` | float | 0–1 lerp speed |

- Static `ActiveHandlers` list
- Scale lerp animation (spawn/despawn)
- Checks `"isDragging"` bool to disable during drag
- Checks `"isBigScreen"` param

---

### AvatarTaskbarController.cs
**Taskbar sitting with pink zone detection**

```csharp
[ExecuteAlways]
public class AvatarTaskbarController : MonoBehaviour
```

- Pink zone overlap detection (5px from taskbar)
- `"IsSitting"` animator bool (hashed)
- Bone attachment with scale animation (lerp in/out)
- `OnDrawGizmos()` for editor visualization

---

### HandHolder.cs
**IK hand holding system**

Caches: `leftHand`, `rightHand`, `chest`, `leftShoulder`, `rightShoulder`
- `SetAnimator()` rebinds after model swap

---

### AccessoiresHandler.cs
**Bone-tracked accessories**

- Per-bone attachment with smoothness lerp
- Steam DRM check for premium accessories
- Tracks accessories to bone transforms each frame

---

### PetVoiceReactionHandler.cs
**Pat/pet detection using angle tracking**

- Detects circular mouse motion over avatar (petting gesture)
- Triggers voice reaction audio clips

---

### AvatarRebindHandler.cs (69 lines)
**Animator rebinding utility**

```csharp
public static void RebindTree() {
    // 1. Set culling to AlwaysAnimate
    // 2. Soft controller nudge (re-assign controller)
    // 3. Hard rebind: Rebind() + Update(0)
    // 4. Also rebinds UniversalBlendshapes
}
```

---

### UniversalBlendshapes.cs
**VRM expression/blendshape system**

```csharp
[DisallowMultipleComponent]
```
- Supports VRM0 proxy and VRM1 expressions
- Standard expressions: `Blink`, `Blink_L`, `Blink_R`, `LookUp/Down/Left/Right`, `Neutral`, `A/I/U/E/O`, `Joy/Angry/Sorrow/Fun`
- State machine with `fadeSpeed`, `safeTimeout`, `minHoldTime`

---

### AvatarRandomMessages.cs
**Random LLM chat bubbles**

- LLM integration via `LLMUnitySamples.Bubble`
- Stream speed setting
- Allowed states whitelist (only shows in certain states)
- Enabled/disabled via `SaveLoadHandler`

---

### BlendTreeLooper.cs (75 lines)
**StateMachineBehaviour for blend tree auto-cycling**

```csharp
public class BlendTreeLooper : StateMachineBehaviour {
    public string blendParam = "Index";
    public int animationCount = 6;
    public float animationDuration = 2f;
    public float transitionDuration = 1f;
}
```
- Timer-based forward cycling with lerp + `Mathf.Repeat`
- Cycles through blend tree entries automatically

---

### AvatarStateObjector.cs
**Show/hide objects based on animator state**

- `ObjectorRule` list: state name → target object
- Checks both bool parameters and state names
- Scale lerp for smooth show/hide transitions
- `spawnAnimationSpeed` controls lerp speed

---

### AvatarScaleController.cs
**Mouse scroll scaling**

| Field | Type | Description |
|-------|------|-------------|
| `avatarSizeSlider` | Slider | UI binding |
| `scrollSensitivity` | float | Scroll wheel speed |
| `smoothFactor` | float | Smooth lerp factor |

- Checks `UniWindowController.current.isClickThrough` — disables when click-through
- Checks `MenuActions.IsMovementBlocked()`
- Checks `controller.isDragging` — disables during drag

---

### SwingController.cs
**UI elements follow bones on screen**

- `SwingFollowEntry` list: bone → UI element mapping
- Uses `Camera.WorldToScreenPoint()` for bone-to-screen projection

---

## 4. Animation System

### Animator Controller Structure
- **Blend Trees** for idle animations (10 entries) and dances (5 built-in)
- **Float parameters** (`DanceIndex`, `IdleIndex`) lerped for smooth transitions
- **Bool parameters** for state switches (`isDancing`, `isIdle`, `isDragging`, etc.)
- **BlendTreeLooper** StateMachineBehaviour for automatic blend tree cycling

### Sound-Reactive State Machine
```
IDLE (sound < threshold)
  ↓ (sound detected, allowed app)
DANCING (auto-switch dance clips)
  ↓ (sound stops)
IDLE
```

### NAudio Sound Detection
```csharp
using NAudio.CoreAudioApi;
var enumerator = new MMDeviceEnumerator();
var device = enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);
float peak = device.AudioMeterInformation.MasterPeakValue;
```
- 2-second polling interval (`CheckSoundContinuously` coroutine)
- Per-app filtering via `allowedApps` list

---

## 5. Custom Dance Player

### AvatarDanceHandler.cs (1401 lines)
**Path**: `Assets/MATE ENGINE - Scripts/AvatarHandlers/AvatarDancePlayer.cs`  
**Namespace**: `CustomDancePlayer`

#### Core Architecture
```
StreamingAssets/CustomDances/  ← .unity3d and .me dance files
persistentDataPath/Mods/       ← User-added dance mods
     ↓
LoadAllSources() scans both folders
     ↓
DanceEntry[] (id, path, clip, audio, bundle, meta)
     ↓
AnimatorOverrideController replaces placeholder clip
     ↓
SmoothPlayFlow coroutine manages transitions
```

#### UI Fields
| Field | Type | Description |
|-------|------|-------------|
| `playButton` | Button | Play/pause |
| `stopButton` | Button | Stop |
| `prevButton` | Button | Previous track |
| `nextButton` | Button | Next track |
| `progressSlider` | Slider | Playback progress |
| `playingNowText` | TMP/Text | Current song name |
| `authorText` | TMP/Text | Song author |
| `volumeSlider` | Slider | Audio volume |

#### Animator Integration
| Field | Default | Description |
|-------|---------|-------------|
| `danceLayerName` | "Dance Layer" | Animator layer name |
| `danceStateName` | "Custom Dance" | State name to override |
| `placeholderClipName` | "CUSTOM_DANCE" | Clip replaced by AnimatorOverrideController |
| `customDancingParam` | "isCustomDancing" | Bool param |
| `waitingParam` | "isWaitingForDancing" | Bool param |

#### DanceEntry Data Class
```csharp
class DanceEntry {
    string id, path, bundlePath;
    AnimationClip clip;
    AudioClip audio;
    AssetBundle bundle;
    bool fromME;
    string extractedDir, author, stableId;
}
```

#### DanceMeta (JSON from .me files)
```csharp
class DanceMeta {
    string songName, songAuthor, mmdAuthor;
    float songLength;
}
```

#### Loading Formats
1. **`.unity3d`** — Direct AssetBundle load (`TryAddUnity3D`)
2. **`.me`** — ZIP extract → cache dir (timestamp comparison) → load bundle → `dance_meta.json` / `dance.json` → SHA1 stableId (`TryAddME`)

#### Playback System
- **AnimatorOverrideController pattern**: Creates override from base controller, replaces placeholder clip with loaded dance clip
- **SmoothPlayFlow coroutine**:
  ```
  Freeze animator → Set "isWaitingForDancing" = true
  → Wait for waiting state → Load AssetBundle → Load clip + audio
  → Create AnimatorOverrideController → Replace placeholder
  → Set "isCustomDancing" = true → Unfreeze
  ```
- **SmoothStopFlow**: Reverse process

#### Navigation Modes
| Mode | Description |
|------|-------------|
| Sequential | Play in order |
| Shuffle | Random next |
| Loop | Repeat current |

`filteredQueue` system for search-filtered playback.

#### Multi-Instance Sync (Leader/Follower)
- **File-based JSON bus**: `avatar_dance_play_bus.json`
- **Mutex**: Thread-safe file access
- **Leader** broadcasts: `{cmd, sid, index, title, atUtc}`
- **Follower** polls file at `pollInterval` (0.05s)
- `leadSeconds` (1.5s): Leader starts ahead for sync
- `LeaderAutoNextWatcher`: Auto-advances for all instances

#### MMD Blendshape Detection
HashSet of Japanese MMD blendshape names for bypass detection. If model has MMD blendshapes, AvatarDanceShapeConverter handles them differently.

#### Guard Timing
- `EnforceHold` / `FreezeAnimator` / `UnfreezeAnimator`
- Prevents accidental state changes during transitions

### AvatarDancePlayerTools.cs (261 lines)
**Search/filter and favorites UI**

| Feature | Details |
|---------|---------|
| Search | Text filter with Unicode normalization |
| Favorites | JSON persistence (`favorite_songs.json`) |
| Toggles | Loop, shuffle |
| Methods | `ReindexAndWire()`, `ApplyFilter()`, `Normalize()` |

### AvatarDancePlayerUtils.cs (33 lines)
Volume slider ↔ `danceAudioSource` binding.

### AvatarDanceShapeConverter.cs (163 lines)
**PlayableGraph-based blendshape forwarding during dance**

- Creates proxy animator with `PlayableGraph`
- `candidatePaths`: "Body", "Model/Body", "Face"
- Detects VRMLoader-based model via `FindAndBindAnimator`
- `bypassForThisAvatar` if `HasMmdBlendshapes`
- `TearDownGraph` / `TearDownProxy` lifecycle management

### Custom Dance Player Prefab Structure
```
CustomDancePlayer.prefab
├── DummyBlendshapeMesh.asset   ← For MMD facial compatibility
├── CUSTOM_DANCE.anim           ← Placeholder clip (overridden)
├── DANCE_END.anim              ← End transition clip
├── CustomDanceAvatarController.controller
└── DanceAvatarHelper.cs        ← Helper script
```

---

## 6. Window Management

### AvatarWindowHandler.cs (860+ lines)
See [Section 3](#avatarwindowhandlercs-860-lines) for full details.

Key window management features:
- **Window sitting**: Detects title bars via Win32 EnumWindows
- **Snap probe**: Hip-based, configurable radius
- **Occluder quads**: Dynamically created to mask avatar behind foreground windows
- **Binary search calibration**: 20-iteration search for correct Desktop Y position
- **SmoothDamp**: Smooth following with prediction for seated position

### AvatarHideHandler.cs (384 lines)
See [Section 3](#avatarhidehandlercs-384-lines) for full details.

### AvatarLocomotionController.cs (746 lines)
See [Section 3](#avatarlocomotioncontrollercs-746-lines) for full details.

Moves the actual Unity window across the desktop using Win32 `MoveWindow`.

---

## 7. Input & Interaction

### Mouse Tracking — Driver Pattern
See [AvatarMouseTracking.cs](#avatarmousetrackingcs-215-lines) for complete details.

**Summary**: Creates intermediate "driver" GameObjects between bone parents and the Animator. These drivers are rotated toward the mouse cursor with Slerp smoothing. The Animator controls the base pose while the drivers add the tracking overlay.

### Dragging
- `AvatarAnimatorController.isDragging` bool tracks drag state
- Multiple handlers check this bool to disable features during drag
- `AvatarSwayController` applies spring physics during drag movement
- `BlockDraggingOverride` can disable dragging

### Food Interaction
- `AvatarFoodController` handles mouse-follow food items
- Head proximity detection triggers eating animation
- Spring physics on food items for natural movement

### Pet/Pat Detection
- `PetVoiceReactionHandler` detects circular mouse motion
- Angle tracking algorithm identifies petting gesture
- Triggers voice reaction clips

### Click-Through
- `UniWindowController.isClickThrough` — OS-level click-through
- `AvatarScaleController` disables scroll-to-scale when click-through active
- Hit test based on opacity threshold or raycast

---

## 8. Settings & Save System

### SaveLoadHandler.cs (300 lines)
**Path**: `Assets/MATE ENGINE - Scripts/Settings/SaveLoadHandler.cs`

#### Singleton Pattern
```csharp
public static SaveLoadHandler Instance;
void Awake() {
    Instance = this;
    DontDestroyOnLoad(gameObject);
}
```

#### Persistence
- **Serialization**: `Newtonsoft.Json` (`JsonConvert.SerializeObject` / `DeserializeObject`)
- **File Path**: `Application.persistentDataPath/[customDataDir]/settings.json`
- **Multi-instance**: CLI args `--savefile` and `--datadir` for separate save files
- `SaveToDisk()` / `LoadFromDisk()` methods
- `MigrateAfterLoad()` — versioned migration based on `settingsVersion`

#### Static Helpers
- `SyncAllowedAppsToAllAvatars()` — propagates app list to all instances
- `ApplyAllSettingsToAllAvatars()` — applies settings to all active avatars

### SettingsData — All Fields

```csharp
public class SettingsData {
    // Window
    public WindowSizeState windowSizeState;  // Normal/Big/Small
    public bool isTopmost = true;
    
    // Sound Detection
    public float soundThreshold = 0.2f;
    public List<string> allowedApps;
    
    // Idle
    public float idleSwitchTime = 10f;
    public float idleTransitionTime = 1f;
    
    // Dancing
    public bool enableDancing = true;
    public bool enableDanceSwitch = false;
    public float danceSwitchTime = 15f;
    public float danceTransitionTime = 2f;
    
    // Avatar
    public float avatarSize = 1.0f;
    public string selectedModelPath = "";
    
    // Tracking Blends
    public float headBlend = 0.7f;
    public float eyeBlend = 1f;
    public float spineBlend = 0.5f;
    
    // Features
    public bool enableMouseTracking = true;
    public bool enableHandHolding = true;
    public bool enableWindowSitting = false;
    public bool enableIK = true;
    public bool enableLocomotion = false;
    public bool enableParticles = true;
    public bool enableRandomMessages = false;
    public bool enableRandomAvatar = false;
    public bool enableFeedSystem = false;
    public bool enableMinecraftMessages = false;
    
    // Performance
    public int fpsLimit = 90;
    public int graphicsQualityLevel = 1;
    
    // Visual
    public bool bloom = false;
    public bool dayNight = true;
    public bool ambientOcclusion = false;
    public float uiHueShift = 0f;
    public float uiSaturation = 1f;
    public string selectedParticleTheme = "Standard";
    
    // Audio
    public float petVolume = 1f;
    public float effectsVolume = 1f;
    public float menuVolume = 1f;
    
    // Lights
    public Dictionary<string, float> lightIntensities;
    public Dictionary<string, float> lightSaturations;
    public Dictionary<string, float> lightHues;
    
    // Mod/Accessory State
    public Dictionary<string, bool> groupToggles;
    public Dictionary<string, bool> modStates;
    public Dictionary<string, bool> accessoryStates;
    
    // Misc
    public bool enableDiscordRPC = true;
    public bool tutorialDone = false;
    public string selectedLocaleCode = "en";
    public bool startWithWindows = false;
    public int contextLength = 4096;
    public bool enableHusbandoMode = false;
    public bool enableAutoMemoryTrim = false;
    public int settingsVersion = 0;
    public bool alarmsEnabled = true;
    public float windowSitYOffset = 0f;
    
    // Big Screen
    public int bigScreenScreenSaverTimeoutIndex = 0;
    public bool bigScreenScreenSaverEnabled = false;
    
    // Nested Classes
    public class AlarmEntry { id, enabled, hour, minute, daysMask, text, lastTriggeredUnixMinute }
    public class TimerEntry { id, enabled, hours, minutes, presetSeconds, running, targetUnix, text }
}
```

### Settings Handler Scripts

#### SettingsHandlerUtility.cs (17 lines)
```csharp
public static class SettingsHandlerUtility {
    public static void ReloadAllSettingsHandlers() {
        // Reflection: finds all MonoBehaviours
        // Invokes LoadSettings() and ApplySettings() if they exist
    }
}
```

#### SettingsHandlerToggles.cs (245 lines)
Toggle bindings for 18+ boolean settings:
`enableDancing`, `enableMouseTracking`, `isTopmost`, `enableParticles`, `bloom`, `dayNight`, `enableWindowSitting`, `enableDiscordRPC`, `enableHandHolding`, `ambientOcclusion`, `enableIK`, `enableDanceSwitch`, `enableRandomMessages`, `enableHusbandoMode`, `enableAutoMemoryTrim`, `enableMinecraftMessages`, `enableFeedSystem`, `enableRandomAvatar`, `enableLocomotion`

- `ApplySettings()` propagates to `AvatarRandomMessages`, `MemoryTrim`, visual objects (bloom/dayNight/AO GameObjects), window settings
- `ResetToDefaults()` method

#### SettingsHandlerSliders.cs (192 lines)
Slider bindings: `soundThreshold`, `idleSwitchTime`, `idleTransitionTime`, `avatarSize`, `fpsLimit`, `headBlend`, `spineBlend`, `eyeBlend`, `hueShift`, `saturation`, `windowSitYOffset`, `danceSwitchTime`, `danceTransitionTime`

Propagates to: `FPSLimiter`, `AvatarScaleController`, `ThemeManager`, `AvatarWindowHandler`

#### SettingsHandlerDropdowns.cs
- Particle themes
- Graphics quality (`QualitySettings.SetQualityLevel`)
- Context length for LLM (`contextOptions` array)

#### SettingsHandlerAudio.cs
- `petVolume`, `effectsVolume`, `menuVolume` sliders → `SaveLoadHandler`
- `UpdateAllCategoryVolumes()`

#### SettingsHandlerLights.cs
- Per-light intensity/saturation/hue sliders
- Dictionary storage: `lightIntensities[lightName]`, `lightSaturations[lightName]`, `lightHues[lightName]`

#### SettingsHandlerBigScreen.cs
- Screen saver enable toggle + timeout slider

---

## 9. Mod System

### MEModHandler.cs (648 lines)
**Path**: `Assets/MATE ENGINE - Scripts/Settings/MEModHandler.cs`

#### Fields
| Field | Type | Description |
|-------|------|-------------|
| `loadModButton` | Button | UI trigger |
| `modListContainer` | Transform | UI scroll list |
| `modEntryPrefab` | GameObject | Entry template |
| `modFolderPath` | string | `persistentDataPath/Mods` |

#### Mod Formats
1. **`.unity3d`** — Direct AssetBundle load
2. **`.me`** — ZIP extract → `ME_Cache` (timestamp comparison) → load bundle

#### ModType Enum
```csharp
enum ModType { MEObject, Unity3D, MEDance }
```

#### Loading Pipeline (MEObject)
```
Extract .me ZIP → ME_Cache/
→ Load AssetBundle → Load prefab
→ Instantiate
→ Apply reference_paths.json (field wiring)
→ Apply scene_links.json (scene reference binding)
→ PreloadAudioClips()
→ AddToModListUI()
```

#### Key Data Classes
```csharp
class ModInfo { name, author, description, weblink, buildTarget, timestamp }
class RefPathMap { /* JSON mapping for prefab field references */ }
class SceneLinkMap { /* JSON mapping for scene object references */ }
```

#### State Persistence
- `GlobalInstances` dict (static) — tracks all loaded mod instances
- `PersistState → SaveLoadHandler.Instance.data.modStates`
- `RemoveMod()`: Unloads bundle, destroys instance, deletes files

#### Steam Workshop
- Steam Workshop ID resolution
- `SteamWorkshopAutoLoader` integration

### MEModLoader.cs (308 lines)
**Path**: `Assets/MATE ENGINE - Scripts/Settings/MEModLoader.cs`

#### Singleton
```csharp
public static MEModLoader Instance;
void Awake() { Instance = this; DontDestroyOnLoad(gameObject); }
```

#### References
- `ChibiToggle chibiToggle`
- `AvatarDragSoundHandler`
- `PetVoiceReactionHandler`

#### Folder Structure
```
StreamingAssets/Mods/ModLoader/
├── Chibi Mode/Sounds/    ← chibiEnterSounds, chibiExitSounds
├── Drag Mode/Sounds/     ← drag sounds
└── Hover Reactions/      ← voice reaction clips
```

#### Key Methods
- `EnsureFolderStructure()` — creates folders if missing
- `LoadChibiSounds()` / `LoadDragSounds()` — coroutines
- `AssignHandlersForCurrentAvatar()` — wires handlers to current model
- `ApplyChibiSettings()` — loads chibi settings from JSON

---

## 10. UniWindowController (Kirurobo)

**Package Path**: `Assets/MATE ENGINE - Packages/Kirurobo/UniWindowController/`  
**Namespace**: `Kirurobo`

### Overview
Third-party library for transparent borderless windows on Windows/macOS. Uses native plugin `LibUniWinC`.

### UniWindowController.cs (~1155 lines)
**Main MonoBehaviour** — singleton via `UniWindowController.current`

#### Key Properties
| Property | Type | Description |
|----------|------|-------------|
| `current` | static | Singleton accessor |
| `isTransparent` | bool | Transparent window |
| `isTopmost` | bool | Always on top |
| `isBottommost` | bool | Always on bottom |
| `isZoomed` | bool | Maximized |
| `isClickThrough` | bool | Mouse passes through |
| `isHitTestEnabled` | bool | Auto click-through based on opacity |
| `alphaValue` | float | Window alpha (0–1) |
| `windowPosition` | Vector2 | Window position on desktop |
| `windowSize` | Vector2 | Window size |
| `shouldFitMonitor` | bool | Fit to monitor |
| `monitorToFit` | int | Target monitor |

#### Enums
```csharp
public enum TransparentType { None = 0, Alpha = 1, ColorKey = 2 }
public enum HitTestType { None = 0, Opacity = 1, Raycast = 2 }
```

#### Hit Test
- **Opacity** mode: Reads pixel alpha at cursor position, compares with `opacityThreshold` (0.1f)
- **Raycast** mode: Physics raycast from camera through cursor position
- If on transparent pixel → `isClickThrough = true` (mouse passes to desktop)

#### Camera Background
- `autoSwitchCameraBackground` (true) — auto-switches camera clear flags when transparent
- Stores original `CameraClearFlags` and `backgroundColor`
- Sets to `SolidColor` + `Color.clear` when transparent (or `keyColor` for ColorKey mode)

#### Important Settings
```csharp
public Camera currentCamera;                    // Main camera
public TransparentType transparentType = Alpha; // Windows only
public Color32 keyColor;                        // For ColorKey mode
public float opacityThreshold = 0.1f;           // Hit test threshold
public bool autoSwitchCameraBackground = true;  // Auto-set camera bg
public bool forceWindowed = false;              // Force windowed mode
```

#### Player Settings Requirements
- **DXGI Flip Mode Swapchain** must be DISABLED for transparency to work
- Editor validates this and offers auto-fix

### UniWinCore.cs (Internal, ~760 lines)
**Low-level native plugin wrapper**

#### Native DLL Methods (LibUniWinC)
```csharp
// Window state
IsActive(), IsTransparent(), IsBorderless(), IsTopmost(), IsBottommost(), IsMaximized()

// Window control
AttachMyWindow(), AttachMyOwnerWindow(), DetachWindow()
SetTransparent(bool), SetBorderless(bool), SetAlphaValue(float)
SetClickThrough(bool), SetTopmost(bool), SetBottommost(bool)

// Window geometry
SetPosition(float x, float y), GetPosition(out float, out float)
SetSize(float w, float h), GetSize(out float, out float)

// Cursor
GetCursorPosition(out float x, out float y)

// File drop
SetAllowDrop(bool), OnDropFiles callback

// Transparency
SetTransparentType(int)  // Alpha or ColorKey
```

### UniWindowMoveHandle.cs
**Drag-to-move handler**

```csharp
public class UniWindowMoveHandle : MonoBehaviour, IDragHandler, IBeginDragHandler, IEndDragHandler, IPointerUpHandler
```
- Implements Unity's drag interfaces
- `dragSmooth` (0–100): Smoothing for drag movement
- `disableOnZoomed`: Disables drag when maximized
- Tracks `isWindowSit` animator parameter

---

## 11. Key Patterns & Architecture

### 1. Component Template Prefab Injection
**The core architecture**: A prefab (`componentTemplatePrefab`) contains ALL handler MonoBehaviours. When a VRM is loaded, all components are copied via reflection from the prefab onto the loaded model. This means every handler is pre-configured on the prefab and doesn't need per-model setup.

### 2. Driver Pattern (Mouse Tracking)
Instead of modifying bones directly (overridden by Animator each frame), create intermediate "driver" GameObjects parented to bone parents. Rotate drivers in LateUpdate. This survives Animator writes.

### 3. Spring Physics Pattern
Shared across `AvatarSwayController` and `AvatarFoodController`:
```csharp
static void Spring(ref float x, ref float v, float xt, float f, float z, float dt) {
    float w = f * 2f * Mathf.PI;        // angular frequency
    float a = w * w * (xt - x) - 2f * z * w * v;  // spring + damping
    v += a * dt;
    x += v * dt;
}
```

### 4. Animator Parameter Hashing
All frequently-accessed parameters are pre-hashed:
```csharp
private static readonly int isDraggingParam = Animator.StringToHash("isDragging");
```

### 5. State Whitelist Pattern
Many handlers only activate in certain animator states:
```csharp
List<string> allowedStates = new List<string> { "Idle", "Dancing" };
bool IsInAllowedState() {
    var stateInfo = animator.GetCurrentAnimatorStateInfo(0);
    return allowedStates.Any(s => stateInfo.IsName(s));
}
```

### 6. AnimatorOverrideController Pattern (Dance Player)
Replace placeholder clips at runtime without modifying the base controller:
```csharp
var overrideController = new AnimatorOverrideController(baseController);
overrideController["CUSTOM_DANCE"] = loadedDanceClip;
animator.runtimeAnimatorController = overrideController;
```

### 7. Singleton Pattern
Multiple systems use singleton + DontDestroyOnLoad:
- `SaveLoadHandler.Instance`
- `MEModLoader.Instance`
- `UniWindowController.current`

### 8. Win32 P/Invoke Pattern
Direct Windows API calls for window management:
```csharp
[DllImport("user32.dll")]
static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
[DllImport("user32.dll")]
static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int nWidth, int nHeight, bool bRepaint);
```

### 9. Leader/Follower Sync (Dance Player)
Multi-instance sync via file-based JSON bus with Mutex for thread safety. Leader broadcasts play state, followers poll and sync.

### 10. BlendTree Cycling
`BlendTreeLooper` (StateMachineBehaviour) automatically cycles through blend tree entries using timer + lerp on a float parameter.

### 11. Reflection-Based Settings Reload
`SettingsHandlerUtility.ReloadAllSettingsHandlers()` uses reflection to find and invoke `LoadSettings()` / `ApplySettings()` on ALL MonoBehaviours in the scene.

---

## 12. Animator Parameters Reference

### Bool Parameters
| Parameter | Set By | Description |
|-----------|--------|-------------|
| `isIdle` | AvatarAnimatorController | Currently idle |
| `isDancing` | AvatarAnimatorController | Sound-reactive dancing |
| `isDragging` | AvatarAnimatorController | Being dragged |
| `isMale` | AvatarAnimatorController | Male model |
| `isFemale` | AvatarAnimatorController | Female model |
| `isWindowSit` | AvatarWindowHandler | Sitting on window |
| `isTaskbarSit` | AvatarWindowHandler | Sitting on taskbar |
| `isSitting` | AvatarTaskbarController | Alt sitting state |
| `IsSleeping` | AvatarSleepController | Sleep mode active |
| `HideLeft` | AvatarHideHandler | Hidden on left edge |
| `HideRight` | AvatarHideHandler | Hidden on right edge |
| `isCustomDancing` | AvatarDanceHandler | Custom dance playing |
| `isWaitingForDancing` | AvatarDanceHandler | Waiting for dance state |
| `isBigScreen` | AvatarBubbleHandler | Big screen mode |

### Float Parameters
| Parameter | Set By | Description |
|-----------|--------|-------------|
| `DanceIndex` | AvatarAnimatorController | Current dance blend tree index |
| `IdleIndex` | AvatarAnimatorController | Current idle blend tree index |
| `Index` | BlendTreeLooper | Generic blend tree cycling |

---

## 13. Win32 / P/Invoke Reference

### Commonly Used Functions
```csharp
// Window position/size
[DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
[DllImport("user32.dll")] static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int W, int H, bool bRepaint);

// Monitor info
[DllImport("user32.dll")] static extern IntPtr MonitorFromWindow(IntPtr hwnd, uint dwFlags);
[DllImport("user32.dll")] static extern bool GetMonitorInfo(IntPtr hMonitor, ref MONITORINFO lpmi);

// Window enumeration (via Kirurobo.WinApi)
// Used by AvatarWindowHandler for detecting title bars

// LibUniWinC (native plugin)
SetTransparent, SetBorderless, SetClickThrough, SetTopmost, SetBottommost
SetAlphaValue, SetPosition, GetPosition, SetSize, GetSize
GetCursorPosition, SetAllowDrop
```

### RECT Structure
```csharp
[StructLayout(LayoutKind.Sequential)]
struct RECT {
    public int left, top, right, bottom;
}
```

---

## Quick Reference: File Locations

| System | Path |
|--------|------|
| VRM Loader | `Assets/MATE ENGINE - Scripts/VRMLoader/` |
| Avatar Handlers | `Assets/MATE ENGINE - Scripts/AvatarHandlers/` |
| Settings | `Assets/MATE ENGINE - Scripts/Settings/` |
| Custom Dance Player | `Assets/MATE ENGINE - Scripts/AvatarHandlers/AvatarDancePlayer.cs` |
| Dance Player Tools | `Assets/MATE ENGINE - Scripts/AvatarHandlers/AvatarDancePlayerTools.cs` |
| UniWindowController | `Assets/MATE ENGINE - Packages/Kirurobo/UniWindowController/` |
| UniWinCore (native) | `Assets/MATE ENGINE - Packages/Kirurobo/UniWindowController/Runtime/Scripts/LowLevel/` |
| Shaders | `Assets/MATE ENGINE - Shaders/` (lilToon, Poiyomi, Mochie) |
| Editor Tools | `Assets/Editor/MEModInitializer.cs` |
