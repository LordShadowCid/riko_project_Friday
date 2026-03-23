# Annabeth Feature Adoption Plan — Detailed Implementation Guide

> **Selected features (from user):**
> 1. Settings Menu / UI in standalone builds
> 2. Speech Bubble (toggle-on feature) to show AI text
> 3. VRM file picker / model library for easy character swapping
> 4. Save/Load for user preferences
> 5. Dragging animation, particle effects, and sound feedback
> 6. System tray, sleep mode, FPS limiter

> **Source reference:** Mate-Engine (https://github.com/shinyflvre/Mate-Engine)
> All Mate-Engine code referenced below has been fully read and analyzed.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     NEW SCRIPTS TO CREATE                        │
├──────────────────────────────────────────────────────────────────┤
│ Core/                                                            │
│   SettingsManager.cs      ← Save/Load JSON (from SaveLoadHandler)│
│   FPSController.cs        ← FPSLimiter port                     │
│   SleepController.cs      ← AvatarSleepController port          │
│   SystemTrayController.cs ← Windows system tray                  │
│   MemoryOptimizer.cs      ← MemoryTrim port                     │
│                                                                  │
│ UI/                                                              │
│   RadialMenu.cs           ← Right-click pie menu                 │
│   SettingsPanel.cs        ← Settings sliders/toggles             │
│   SpeechBubble.cs         ← AI text above avatar head           │
│                                                                  │
│ Avatar/                                                          │
│   VrmFilePicker.cs        ← File browser dialog                  │
│   VrmModelLibrary.cs      ← Scan folder, list, thumbnail        │
│   DragAnimationController.cs ← Float animation while dragged    │
│   GravityController.cs    ← Spring bone physics on drag          │
│                                                                  │
│ Interaction/                                                     │
│   TouchSoundHandler.cs    ← Play sound on touch/drag            │
│   ParticleEffectHandler.cs← Hearts/sparkles on touch             │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    EXISTING SCRIPTS TO MODIFY                    │
├──────────────────────────────────────────────────────────────────┤
│ CompanionManager.cs       ← Wire new controllers, drag state     │
│ MessageHandler.cs         ← Add speak text → bubble event        │
│ AvatarController.cs       ← Dynamic VRM path, reload support     │
│ TransparentWindowController.cs ← Expose isDragging, drag events  │
│ HotkeyManager.cs          ← Add settings toggle key (M / F1)    │
│ TouchReactionController.cs ← Trigger sounds + particles          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Settings Infrastructure (Save/Load + FPS + Memory)

> **Goal:** Persistent preferences system that every future feature will depend on.
> **Mate-Engine reference:** `SaveLoadHandler.cs`, `FPSLimiter.cs`, `MemoryTrim.cs`

### Phase 1A: SettingsManager.cs (Core/SettingsManager.cs)
**What it does:** Singleton that holds all user preferences in a serializable data class, saves to JSON, loads on startup. Every future setting (FPS, sleep, bubble, particles, etc.) gets a field here.

**Ported from:** `SaveLoadHandler.cs` — We take its exact architecture:
- Singleton pattern with `Instance` + `DontDestroyOnLoad`
- `SettingsData` serializable class with all fields and sane defaults
- `SaveToDisk()` → `JsonUtility.ToJson()` → `File.WriteAllText()` to `Application.persistentDataPath/settings.json`
- `LoadFromDisk()` → read JSON → deserialize → `MigrateAfterLoad()` for version upgrades
- `ApplyAllSettings()` static method that pushes data to all live controllers

**Key differences from Mate-Engine:**
- We use `JsonUtility` (built-in) instead of Newtonsoft for simpler dependency — for Dictionary fields we use a serializable list-of-pairs wrapper
- We don't need Mate-Engine's `--savefile` / `--datadir` command-line arg parsing
- We don't need their Steam-specific fields

**SettingsData fields (initial set):**
```csharp
[Serializable]
public class SettingsData
{
    // Version for migration
    public int settingsVersion = 1;
    
    // Display
    public int fpsLimit = 60;
    public bool alwaysOnTop = true;
    public bool hideFromTaskbar = true;
    
    // Avatar
    public string selectedModelPath = "";      // Empty = default model
    public float avatarSize = 1.0f;
    
    // Animation  
    public bool enableMouseTracking = true;
    public float eyeBlend = 1.0f;
    public float headBlend = 0.7f;
    
    // Interaction
    public bool enableParticles = true;
    public bool enableTouchSounds = true;
    public float sfxVolume = 1.0f;
    
    // AI / Speech  
    public bool enableSpeechBubble = false;    // Toggle-on feature as requested
    
    // System
    public bool enableSleepMode = false;
    public float sleepTimerSeconds = 120f;
    public bool enableAutoMemoryTrim = false;
    public bool startWithWindows = false;
    public bool minimizeToTray = true;
}
```

**Integration points:**
- `CompanionManager.Start()` → Call `SettingsManager.Instance.ApplyAllSettings()`
- Every controller reads from `SettingsManager.Instance.data.xxx` on init

---

### Phase 1B: FPSController.cs (Core/FPSController.cs)
**What it does:** Limits `Application.targetFrameRate` to save power for a long-running desktop process.

**Ported directly from:** `FPSLimiter.cs` — This is almost a 1:1 copy. The Mate-Engine version is clean and complete:
```
Core logic: Application.targetFrameRate = targetFPS; QualitySettings.vSyncCount = 0;
Range: 15–165 FPS
Reads from: SettingsManager.Instance.data.fpsLimit
```

**Simplifications vs Mate-Engine:**
- Remove the UI slider/label references (those go in SettingsPanel instead)
- Mate-Engine stores to PlayerPrefs — we use SettingsManager instead
- Keep the `SetFPSLimit(float)` public method for SettingsPanel to call

**~30 lines of code. Trivial port.**

---

### Phase 1C: MemoryOptimizer.cs (Core/MemoryOptimizer.cs)
**What it does:** Periodic GC collection and working set trim for a long-running desktop process.

**Ported directly from:** `MemoryTrim.cs` — We take:
- Coroutine-based `TrimRoutine()`: GC.Collect → Resources.UnloadUnusedAssets → EmptyWorkingSet (Win32)
- Auto-trim every 600s (10 min) if enabled
- Startup trim after 10s delay
- `TrimNow()` public method for manual trigger (e.g., after VRM swap)

**From `GCCollect.cs` we take:**
- Optional memory logging before/after: `Profiler.GetMonoUsedSizeLong()`

**~50 lines. Almost direct copy.**

---

### Phase 1 Files Summary

| File | Lines (est.) | Based on | Changes to existing |
|------|:------------:|----------|-------------------|
| `Core/SettingsManager.cs` | ~150 | `SaveLoadHandler.cs` | `CompanionManager.cs`: call ApplyAllSettings on Start |
| `Core/FPSController.cs` | ~30 | `FPSLimiter.cs` | — |
| `Core/MemoryOptimizer.cs` | ~60 | `MemoryTrim.cs` + `GCCollect.cs` | — |

---

## Phase 2: Settings UI (Radial Menu + Settings Panel)

> **Goal:** Right-click radial menu → opens settings panel. Users can configure everything in standalone builds.
> **Mate-Engine reference:** `MenuActions.cs` (radial menu), `AvatarSettingsMenu.cs` (settings panel), `Tasty Pie Menu/` (3rd-party radial)

### Phase 2A: RadialMenu.cs (UI/RadialMenu.cs)

**What it does:** Right-click on avatar → show a simple radial/context menu with options like: Settings, VRM Library, Dance Mode, Sleep, Quit.

**Mate-Engine approach:** Uses a 3rd-party `Tasty Pie Menu` asset (`Xamin.CircleSelector`). Their `MenuActions.cs` orchestrates:
- F1 key toggles the radial menu
- Menu follows the avatar's head bone position via `WorldToScreenPoint`
- While any menu is open, it blocks movement/tracking/reactions

**Our approach — simpler than Mate-Engine:**
We do NOT need the Tasty Pie Menu asset. Instead, build a lightweight right-click context menu using Unity UI (Canvas):
- Screen-Space Overlay panel that appears at cursor position
- 6-8 buttons in a circle or vertical list
- Right-click (or configurable key, default M) toggles it
- While open, blocks dragging and touch reactions

**Menu items:**
1. ⚙️ Settings → Opens SettingsPanel
2. 👤 Change Character → Opens VRM Library
3. 💬 Toggle Speech Bubble → Quick toggle
4. 💤 Sleep/Wake → Toggle sleep mode
5. 🗑️ Clear Chat History → Wipe chat_history.json via WS command
6. ❌ Quit → Application.Quit()

**Key behaviors from MenuActions.cs we adopt:**
- `IsMenuOpen` static bool → other scripts check this to block interaction
- `IsMovementBlocked()` static method
- Close menu automatically when dragging starts
- Open/close sounds (AudioSource references, optional)

**Integration:**
- `TransparentWindowController.cs`: When menu is open, DON'T set click-through
- `TouchReactionController.cs`: Skip reactions when `RadialMenu.IsMenuOpen`
- `HotkeyManager.cs`: Add M key to toggle menu

**~120 lines.**

---

### Phase 2B: SettingsPanel.cs (UI/SettingsPanel.cs)

**What it does:** Scrollable panel with sliders and toggles for all settings. Opened from the radial menu.

**Mate-Engine reference:** `AvatarSettingsMenu.cs` — This is a massive ~500-line file because it handles 30+ settings. We take the PATTERN but with only our settings:

**Pattern from Mate-Engine (what we copy):**
```
1. All UI controls are [SerializeField] public references (Slider, Toggle, etc.)
2. Start() → wire onValueChanged listeners → each writes to SettingsManager.Instance.data.XXX → call SaveAll()  
3. LoadSettings() → read from SettingsManager.Instance.data → SetValueWithoutNotify on each control
4. ApplySettings() → push live values to all controllers
5. ResetToDefaults() → create fresh SettingsData, reload UI
```

**Settings to expose (grouped):**

| Group | Control | Type | Field |
|-------|---------|------|-------|
| Display | FPS Limit | Slider (15-165) | fpsLimit |
| Display | Always On Top | Toggle | alwaysOnTop |
| Display | Avatar Size | Slider (0.5-2.0) | avatarSize |
| Tracking | Mouse Tracking | Toggle | enableMouseTracking |
| Tracking | Eye Blend | Slider (0-1) | eyeBlend |
| Tracking | Head Blend | Slider (0-1) | headBlend |
| Interaction | Particles | Toggle | enableParticles |
| Interaction | Touch Sounds | Toggle | enableTouchSounds |
| Interaction | SFX Volume | Slider (0-1) | sfxVolume |
| AI | Speech Bubble | Toggle | enableSpeechBubble |
| System | Sleep Mode | Toggle | enableSleepMode |
| System | Sleep Timer | Slider (30-360s) | sleepTimerSeconds |
| System | Memory Auto-Trim | Toggle | enableAutoMemoryTrim |
| System | Start with Windows | Toggle | startWithWindows |
| System | Minimize to Tray | Toggle | minimizeToTray |

**~180 lines (much smaller than Mate-Engine's because we have fewer settings).**

---

### Phase 2 Unity Scene Setup

Both the RadialMenu and SettingsPanel need a **UI Canvas** in the scene:

```
Scene Hierarchy:
  CompanionUI (Canvas - Screen Space Overlay)
    ├── RadialMenu (Panel, hidden by default)
    │     ├── BtnSettings
    │     ├── BtnCharacter
    │     ├── BtnBubble
    │     ├── BtnSleep
    │     ├── BtnClearHistory
    │     └── BtnQuit
    ├── SettingsPanel (Panel, hidden by default, scrollable)
    │     ├── Header ("Settings")
    │     ├── ScrollView
    │     │     ├── SliderFPS
    │     │     ├── ToggleAlwaysOnTop
    │     │     ├── ... (all settings controls)
    │     │     └── BtnResetDefaults
    │     └── BtnClose
    └── SpeechBubble (see Phase 3)
```

**Note:** The Canvas and UI GameObjects must be created in the Unity Editor (not code). The scripts just reference them via `[SerializeField]`. We'll provide a scene setup guide at implementation time.

---

### Phase 2 Files Summary

| File | Lines (est.) | Based on | Changes to existing |
|------|:------------:|----------|-------------------|
| `UI/RadialMenu.cs` | ~120 | `MenuActions.cs` pattern | `HotkeyManager.cs`: add M key |
| `UI/SettingsPanel.cs` | ~180 | `AvatarSettingsMenu.cs` pattern | `TransparentWindowController.cs`: check IsMenuOpen |

---

## Phase 3: Speech Bubble (Toggle-On Feature)

> **Goal:** Show AI response text in a floating bubble above the avatar's head. Off by default, user toggles it on in settings.
> **Mate-Engine reference:** `AvatarBubbleHandler.cs`

### Phase 3A: SpeechBubble.cs (UI/SpeechBubble.cs)

**What it does:** UI element positioned above the avatar's head bone. Shows AI text with a typing animation and auto-dismiss.

**What we take from AvatarBubbleHandler.cs:**
1. **Bone attachment:** `animator.GetBoneTransform(HumanBodyBones.Head)` → position bubble above it. Mate-Engine parents the bubble to the bone transform
2. **Spawn animation:** Smooth scale lerp from `Vector3.zero` to `originalScale` using `spawnAnimationSpeed`. Clean scale-up/scale-down effect
3. **Audio feedback:** Optional `enableSound` / `disableSound` AudioClips played via `AudioSource.PlayOneShot`
4. **Toggle:** Mate-Engine ties to an animator bool. For us: `SettingsManager.Instance.data.enableSpeechBubble`

**What we do differently (new behavior):**
- Mate-Engine's bubble is a generic GameObject attachment. Ours is specifically a **TextMeshPro UI element** showing AI responses
- We add a **typewriter text effect** (type characters one by one at ~30 chars/sec)  
- We add **auto-dismiss** after the text finishes plus a delay (e.g., 5 seconds)
- We **receive text from `MessageHandler.OnSpeakStart`** event

**Implementation sketch:**
```csharp
public class SpeechBubble : MonoBehaviour
{
    [Header("Attachment")]
    [SerializeField] private HumanBodyBones attachBone = HumanBodyBones.Head;
    [SerializeField] private Vector3 offset = new Vector3(0, 0.3f, 0);
    
    [Header("UI")]  
    [SerializeField] private GameObject bubblePanel;
    [SerializeField] private TMPro.TextMeshProUGUI textField;
    
    [Header("Animation")]  // From Mate-Engine AvatarBubbleHandler
    [SerializeField] [Range(0f, 1f)] private float spawnSpeed = 0.1f;
    [SerializeField] private float autoDismissDelay = 5f;
    [SerializeField] private float typewriterSpeed = 30f;  // chars/sec
    
    [Header("Audio")]  // From Mate-Engine AvatarBubbleHandler
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip showSound;
    [SerializeField] private AudioClip hideSound;
    
    // Spawn animation from AvatarBubbleHandler: smooth scale lerp
    // Position tracking: follow bone each frame via WorldToScreenPoint
    // Typewriter: increment TMP visibleCharacterCount each frame
    // Auto-dismiss: after typing complete + delay, scale back down
}
```

**Integration:**
- `CompanionManager.cs` → On `HandleSpeakStart(text)`, call `speechBubble.ShowText(text)` if enabled
- `CompanionManager.cs` → On `HandleSpeakEnd()`, call `speechBubble.StartDismissTimer()`
- Reads `SettingsManager.Instance.data.enableSpeechBubble` to check if active

**~100 lines.**

---

## Phase 4: VRM File Picker + Model Library

> **Goal:** Browse and swap VRM models at runtime with a visual library.
> **Mate-Engine reference:** `VRMLoader.cs`, `AvatarLibraryMenu.cs`

### Phase 4A: VrmFilePicker.cs (Avatar/VrmFilePicker.cs)

**What it does:** Opens a native Windows file browser to select a .vrm file, then loads it.

**What we take from VRMLoader.cs:**
1. **StandaloneFileBrowser** for native file dialog: `StandaloneFileBrowser.OpenFilePanel("Select Model", "", extensions, false)`
   - We need the **StandaloneFileBrowser** package (https://github.com/gkngkc/UnityStandaloneFileBrowser) — same one Mate-Engine uses. MIT licensed
2. **VRM loading pipeline** — Mate-Engine's `LoadVRM(string path)` tries VRM 1.0 first, falls back to VRM 0.x:
   ```
   GlbFileParser → Vrm10Data.Parse → Vrm10Importer.LoadAsync
   Fallback → GlbBinaryParser → VRMImporterContext → LoadAsync
   ```
   Our `AvatarController.cs` already loads VRM via `Vrm10.LoadPathAsync()`. We just need to make the path dynamic
3. **Model cleanup:** Destroy old instance, null references, GC
4. **Component injection:** Mate-Engine copies MonoBehaviours from template to loaded VRM via reflection. **We don't need this** — our controllers are on parent GameObjects and use `Initialize(vrm)` pattern
5. **Save selected path:** `SettingsManager.Instance.data.selectedModelPath = path; SaveToDisk()`
6. **Auto-load on start:** If `selectedModelPath` is not empty, load that VRM instead of default

**Changes to AvatarController.cs:**
```csharp
// BEFORE: hardcoded path
[SerializeField] private string vrmPath = "Models/claire_avatar.vrm";

// AFTER: check settings for saved path, fall back to default
private async void Start()
{
    string savedPath = SettingsManager.Instance?.data.selectedModelPath;
    if (!string.IsNullOrEmpty(savedPath) && System.IO.File.Exists(savedPath))
        await LoadVRM(savedPath, isAbsolutePath: true);
    else if (loadOnStart)
        await LoadVRM(vrmPath);
}

// Add overload for absolute path (external VRM files)
public async Task LoadVRM(string path, bool isAbsolutePath = false)
```

**Package dependency:** `StandaloneFileBrowser` (MIT, ~5 source files to drop into Plugins/)

**~80 lines for VrmFilePicker + ~20 lines of changes to AvatarController.**

---

### Phase 4B: VrmModelLibrary.cs (Avatar/VrmModelLibrary.cs)

**What it does:** Scans a "models" folder, shows a scrollable list of available VRMs with thumbnails, lets user click to load.

**What we take from AvatarLibraryMenu.cs:**
1. **avatars.json persistence** — Library entries saved to `Application.persistentDataPath/avatars.json`:
   ```json
   [
     { "displayName": "Claire", "author": "Unknown", "version": "1.0", 
       "fileType": "VRM1.X", "filePath": "C:/path/to/file.vrm",
       "thumbnailPath": "C:/AppData/.../Thumbnails/claire_thumb.png",
       "polygonCount": 12000 }
   ]
   ```
2. **Auto-add to library** on first load: `AddAvatarToLibrary()` static method extracts VRM metadata (name, author, version from `Vrm10Instance.Vrm.Meta`) and saves thumbnail as PNG
3. **Thumbnail generation:** `MakeReadableCopy(texture)` → RenderTexture blit → ReadPixels → save PNG
4. **Polygon count:** `GetTotalPolygons()` iterates MeshFilter + SkinnedMeshRenderer triangle counts
5. **UI list:** Instantiate prefab items in a ScrollView, each with: thumbnail (RawImage), name, author, Load button, Remove button
6. **Default model entry:** Built-in model always shows as first entry with "Load Default" button

**What we remove vs Mate-Engine:**
- All Steam Workshop code
- DLC system  
- NSFW toggle, upload button  
- .ME AssetBundle loading (we only support .vrm)
- Live-update coroutine (no Workshop polling)

**What remains is clean:** Scan folder, load JSON, display entries, load on click, add new VRMs, remove entries

**~200 lines (vs Mate-Engine's ~400).**

**Data flow:**
```
User clicks "Change Character" in RadialMenu
  → VrmModelLibrary panel opens
  → Shows: [Default Model] + [entries from avatars.json]
  → User clicks "Load" on an entry
    → AvatarController.LoadVRM(entry.filePath, isAbsolutePath: true)
    → CompanionManager re-wires all sub-controllers via OnVrmLoaded
    → SettingsManager saves selected path
  → User clicks "Add New" button  
    → StandaloneFileBrowser opens
    → Select .vrm file → loads → metadata extracted → added to library  
    → Library UI refreshes
```

---

### Phase 4 Files Summary

| File | Lines (est.) | Based on | Changes to existing |
|------|:------------:|----------|-------------------|
| `Avatar/VrmFilePicker.cs` | ~80 | `VRMLoader.cs` (loading + file dialog) | `AvatarController.cs`: dynamic path + reload support |
| `Avatar/VrmModelLibrary.cs` | ~200 | `AvatarLibraryMenu.cs` (simplified) | `CompanionManager.cs`: OnVrmLoaded re-wiring |

**External dependency:** StandaloneFileBrowser (MIT, ~5 files)

---

## Phase 5: Dragging Animation, Particles, and Sound

> **Goal:** When dragging the avatar, play a float animation. Touch triggers heart particles and sounds.
> **Mate-Engine reference:** `AvatarGravityController.cs`, `AvatarDragSoundHandler.cs`, `AvatarParticleHandler.cs`

### Phase 5A: DragAnimationController.cs (Avatar/DragAnimationController.cs)

**What it does:** While user drags the window (holding click on avatar), VRM spring bones react to movement and apply a "floating" pose.

**What we take from AvatarGravityController.cs:**
1. **Window motion → spring bone force:** Track window position via `GetWindowRect()` each frame, compute delta. Feed as external force to VRM spring bones:
   ```csharp
   // VRM 1.0 spring bone joints
   foreach (var joint in springBoneJoints)
   {
       joint.m_gravityDir = force.normalized;
       joint.m_gravityPower = force.magnitude;
       vrm10Instance.Runtime.SpringBone.SetJointLevel(joint.transform, joint.Blittable);
   }
   ```
   This makes hair/clothes sway dynamically when dragging — satisfying physics feedback
2. **P/Invoke for window position:** `GetWindowRect(hWnd, out RECT)` — same API we already use in TransparentWindowController

**Additional behavior (new, not in Mate-Engine):**
- Slightly raise arms and tilt head back using bone rotation offsets (procedural "grabbed" pose)
- Blend this over idle animation using existing `AnimationBlendController`

**Integration — TransparentWindowController must expose:**
```csharp
public bool IsDragging => _dragging;
public event System.Action OnDragStart;
public event System.Action OnDragEnd;
```

**~80 lines.**

---

### Phase 5B: TouchSoundHandler.cs (Interaction/TouchSoundHandler.cs)

**What it does:** Play audio clips when touching/dragging the avatar.

**Ported nearly directly from `AvatarDragSoundHandler.cs`** — only ~25 lines of core logic:
```csharp
// Track isDragging state changes
// On drag start → play dragStartSound with random pitch variation
// On drag end → play dragStopSound with random pitch variation
// Pitch: Random.Range(1 - lowPercent/100, 1 + highPercent/100)
```

**We extend to also handle touch sounds (from TouchReactionController):**
- `AudioClip[] touchSounds` — random clip on each touch reaction
- `AudioClip dragStartSound, dragEndSound` — play on drag state changes
- Volume reads from `SettingsManager.Instance.data.sfxVolume`
- Feature toggle from `SettingsManager.Instance.data.enableTouchSounds`

**Integration:**
- `TouchReactionController.cs`: After triggering a reaction, call `TouchSoundHandler.PlayTouchSound(zone)`
- `TransparentWindowController.cs`: On drag start/end events

**~50 lines.**

---

### Phase 5C: ParticleEffectHandler.cs (Interaction/ParticleEffectHandler.cs)

**What it does:** Spawn particle effects (hearts, sparkles) at touch positions on the avatar.

**What we take from AvatarParticleHandler.cs:**
Mate-Engine's version is complex (theme system, state-based activation). We take only the core concept and simplify massively:

1. **Bone-attached particle systems:** Pre-made ParticleSystem GameObjects. Normally disabled
2. **On touch:** Enable particle system at touched position for a burst
3. **Feature toggle:** `SettingsManager.Instance.data.enableParticles`

**Simplified implementation:**
```csharp
public class ParticleEffectHandler : MonoBehaviour
{
    [SerializeField] private ParticleSystem heartParticles;
    [SerializeField] private ParticleSystem sparkleParticles;
    
    public void PlayAtPosition(Vector3 worldPos, bool isHeadZone)
    {
        if (!SettingsManager.Instance.data.enableParticles) return;
        var ps = isHeadZone ? heartParticles : sparkleParticles;
        ps.transform.position = worldPos;
        ps.Play();
    }
}
```

**Unity setup needed:** Create 2 ParticleSystem prefabs (heart shapes, sparkle shapes)

**~40 lines of code + 2 ParticleSystem prefabs.**

---

### Phase 5 Files Summary

| File | Lines (est.) | Based on | Changes to existing |
|------|:------------:|----------|-------------------|
| `Avatar/DragAnimationController.cs` | ~80 | `AvatarGravityController.cs` | `TransparentWindowController.cs`: expose IsDragging + events |
| `Interaction/TouchSoundHandler.cs` | ~50 | `AvatarDragSoundHandler.cs` | `TouchReactionController.cs`: call PlayTouchSound |
| `Interaction/ParticleEffectHandler.cs` | ~40 | `AvatarParticleHandler.cs` (simplified) | `TouchReactionController.cs`: call PlayAtPosition |

---

## Phase 6: System Tray, Sleep Mode, Start with Windows

> **Goal:** Proper background app behavior — tray icon, sleep when idle, auto-start.
> **Mate-Engine reference:** `AvatarSleepController.cs`, `SystemStartHandler.cs`

### Phase 6A: SystemTrayController.cs (Core/SystemTrayController.cs)

**What it does:** Add a system tray (notification area) icon with right-click context menu: Show/Hide, Settings, Sleep, Quit.

**Mate-Engine's System Tray folder is a private asset (404 on GitHub).** We build our own:

**Our approach — WinForms NotifyIcon (simplest that works):**
Unity can reference `System.Windows.Forms.dll`. Create a hidden Form on a background thread with a `NotifyIcon` + `ContextMenuStrip`. Menu items invoke Unity actions via `UnitySynchronizationContext`:

```csharp
// Key pieces:
var notifyIcon = new System.Windows.Forms.NotifyIcon();
notifyIcon.Icon = LoadIconFromResources();
notifyIcon.Visible = true;
notifyIcon.ContextMenuStrip = new ContextMenuStrip();
notifyIcon.ContextMenuStrip.Items.Add("Show/Hide", null, (s, e) => ToggleVisibility());
notifyIcon.ContextMenuStrip.Items.Add("Settings", null, (s, e) => OpenSettings());
notifyIcon.ContextMenuStrip.Items.Add("Quit", null, (s, e) => QuitApp());
```

**Menu items:**
- Show / Hide → toggle avatar root visibility
- Settings → `RadialMenu.OpenSettings()`  
- Sleep / Wake → toggle `SleepController`
- Quit → `Application.Quit()`

**Integration:**
- When user clicks Alt+F4, minimize to tray instead (if `minimizeToTray` is true) via `Application.wantsToQuit` callback
- `OnApplicationQuit` → Remove tray icon (`notifyIcon.Dispose()`)

**~120 lines.**

---

### Phase 6B: SleepController.cs (Core/SleepController.cs)

**What it does:** After the PC is idle for configurable time, reduce activity (lower FPS, stop tracking, play sleep animation).

**Ported from `AvatarSleepController.cs`** — clean ~70-line script. We take:
1. **Idle timer:** Each Update, if avatar is idle, increment `idleTime`. If `idleTime >= sleepTimer`, trigger sleep
2. **Wake-up conditions:** If `isDragging` or user interacts → instant wake
3. **Sleep state flag:** Other controllers check `IsSleeping`

**What we add (not in Mate-Engine):**
- **Reduce FPS when sleeping:** `Application.targetFrameRate = 10`
- **Disable mouse tracking** when sleeping
- **Windows idle detection:** Use `GetLastInputInfo()` Win32 API to detect system-wide idle:

```csharp
[DllImport("user32.dll")] static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);
struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }

float GetSystemIdleSeconds()
{
    var info = new LASTINPUTINFO { cbSize = (uint)Marshal.SizeOf<LASTINPUTINFO>() };
    GetLastInputInfo(ref info);
    return (Environment.TickCount - info.dwTime) / 1000f;
}
```

**Integration:**
- `CompanionManager.cs`: Check `SleepController.IsSleeping` before processing audio analysis
- `EyeTrackingController`: Disable when sleeping
- `IdleAnimationController`: Reduce amplitude when sleeping
- `FPSController`: Override target FPS when sleeping

**~80 lines.**

---

### Phase 6C: Start with Windows (in SettingsPanel)

**Ported from `SystemStartHandler.cs`** — Writes to Windows Registry `HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Run`:
```csharp
using (var key = Microsoft.Win32.Registry.CurrentUser.OpenSubKey(
    @"SOFTWARE\Microsoft\Windows\CurrentVersion\Run", writable: true))
{
    if (enable) key.SetValue("Annabeth", $"\"{exePath}\"");
    else key.DeleteValue("Annabeth", false);
}
```

Simple enough to put inside SettingsPanel as a toggle handler. ~30 lines added to SettingsPanel.cs.

---

### Phase 6 Files Summary

| File | Lines (est.) | Based on | Changes to existing |
|------|:------------:|----------|-------------------|
| `Core/SystemTrayController.cs` | ~120 | Win32 NotifyIcon / WinForms | `CompanionManager.cs`: wantsToQuit hook |
| `Core/SleepController.cs` | ~80 | `AvatarSleepController.cs` + GetLastInputInfo | `CompanionManager.cs`: sleep state checks |
| (Start with Windows) | ~30 | `SystemStartHandler.cs` | Added into `UI/SettingsPanel.cs` |

---

## Complete Implementation Order (Dependency Chain)

```
Phase 1 ─ Settings Infrastructure (FOUNDATION — everything depends on this)
  ├── 1A. SettingsManager.cs          [no dependency]
  ├── 1B. FPSController.cs            [depends on: SettingsManager]
  └── 1C. MemoryOptimizer.cs          [depends on: SettingsManager]

Phase 2 ─ Settings UI (MAKES APP CONFIGURABLE)
  ├── 2A. RadialMenu.cs               [depends on: Phase 1]
  └── 2B. SettingsPanel.cs            [depends on: Phase 1 + RadialMenu]

Phase 3 ─ Speech Bubble
  └── 3A. SpeechBubble.cs             [depends on: Phase 1 for toggle]

Phase 4 ─ VRM Management (CHARACTER SWAPPING)
  ├── 4A. VrmFilePicker.cs            [depends on: Phase 1 + StandaloneFileBrowser]
  └── 4B. VrmModelLibrary.cs          [depends on: 4A + Phase 2 (opens from menu)]

Phase 5 ─ Interaction Polish (DRAGGING + EFFECTS)
  ├── 5A. DragAnimationController.cs  [depends on: Phase 1]
  ├── 5B. TouchSoundHandler.cs        [depends on: Phase 1]
  └── 5C. ParticleEffectHandler.cs    [depends on: Phase 1]

Phase 6 ─ System Integration (BACKGROUND APP BEHAVIOR)
  ├── 6A. SystemTrayController.cs     [depends on: Phase 1 + Phase 2]
  └── 6B. SleepController.cs          [depends on: Phase 1]
```

---

## Total Scope

| Phase | New Files | Lines (est.) | Existing File Changes |
|-------|:---------:|:------------:|:--------------------:|
| Phase 1 | 3 | ~240 | 1 (CompanionManager) |
| Phase 2 | 2 | ~300 | 2 (HotkeyManager, TransparentWindowController) |
| Phase 3 | 1 | ~100 | 1 (CompanionManager) |
| Phase 4 | 2 | ~280 | 2 (AvatarController, CompanionManager) |
| Phase 5 | 3 | ~170 | 2 (TransparentWindowController, TouchReactionController) |
| Phase 6 | 2 | ~230 | 2 (CompanionManager, SettingsPanel) |
| **TOTAL** | **13 new** | **~1,320** | **6 existing modified** |

Current Annabeth Unity: **18 scripts** → After all phases: **31 scripts**

---

## External Dependencies Needed

| Package | Purpose | License | How to add |
|---------|---------|---------|-----------|
| StandaloneFileBrowser | Native file open dialog | MIT | Drop source into `Plugins/StandaloneFileBrowser/` |
| TextMeshPro | Speech bubble + settings text | Unity built-in | Already in URP project |
| System.Windows.Forms | System tray icon (Phase 6) | .NET Framework | Add reference in .asmdef or csc.rsp |

---

## Unity Editor Scene Setup Required

These UI elements must be created in Unity Editor (not via code):

### Phase 2 (Radial Menu + Settings):
1. Create `Canvas` (Screen Space - Overlay) named "CompanionUI"
2. Child `Panel` "RadialMenu" with 6 `Button` children
3. Child `Panel` "SettingsPanel" with `ScrollRect` and controls

### Phase 3 (Speech Bubble):
1. Child `Panel` "SpeechBubble" with speech-bubble sprite background
2. `TextMeshProUGUI` child for text content

### Phase 5 (Particles + Sounds):
1. Two `ParticleSystem` prefabs: `HeartParticles`, `SparkleParticles`
2. AudioClip assets: `touch_soft.wav`, `touch_head.wav`, `drag_start.wav`, `drag_end.wav`, `bubble_show.wav`, `bubble_hide.wav`, `menu_open.wav`, `menu_close.wav`

---

## What We Take From Mate-Engine vs Build New

| Our Script | Mate-Engine Source | Adaptation Level |
|------------|-------------------|:---------------:|
| SettingsManager | SaveLoadHandler.cs | **Heavy** — same architecture, our fields |
| FPSController | FPSLimiter.cs | **Direct port** — near-identical |
| MemoryOptimizer | MemoryTrim.cs + GCCollect.cs | **Direct port** — combined |
| RadialMenu | MenuActions.cs concept | **New** — inspired by, not ported |
| SettingsPanel | AvatarSettingsMenu.cs pattern | **Heavy** — same pattern, our controls |
| SpeechBubble | AvatarBubbleHandler.cs | **Medium** — bone attach + spawn anim, new text logic |
| VrmFilePicker | VRMLoader.cs | **Heavy** — file dialog + loading pipeline |
| VrmModelLibrary | AvatarLibraryMenu.cs | **Heavy** — JSON persistence, UI list, simplified |
| DragAnimationController | AvatarGravityController.cs | **Medium** — spring bone force, new pose logic |
| TouchSoundHandler | AvatarDragSoundHandler.cs | **Direct port** — extended with touch sounds |
| ParticleEffectHandler | AvatarParticleHandler.cs | **Light** — concept only, much simpler impl |
| SystemTrayController | (unavailable on GitHub) | **New** — WinForms NotifyIcon |
| SleepController | AvatarSleepController.cs | **Medium** — core logic + Win32 idle detection |
