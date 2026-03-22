# Annabeth Unity Frontend

Unity 6 (URP 17.3.0) desktop companion frontend for the Annabeth AI.
Connects to the Python backend via WebSocket and renders a VRM avatar with
lip sync, emotions, eye tracking, procedural dance, and VRMA animation playback.

## Project Location
- Unity project: `C:\Users\blakd\unit`
- This folder (`unity/Scripts/`) is a synced backup of the project scripts.

## Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| UniVRM (`com.vrmc.vrm` + `com.vrmc.gltf`) | v0.128.3 | VRM/VRMA loading |
| Input System | 1.19.0 | Mouse input for eye tracking |
| URP | 17.3.0 | Rendering pipeline |
| Unity MCP (`com.ivanmurzak.unity.mcp`) | 0.57.1 | Editor automation |

## Scripts (14 files)
```
Assets/Scripts/
├── Core/
│   ├── WebSocketClient.cs           # ws://127.0.0.1:8765/ws client
│   ├── MessageHandler.cs            # Message routing + send methods
│   └── TransparentWindowController.cs # Win32 P/Invoke transparent overlay
├── Avatar/
│   ├── AvatarController.cs          # VRM loading + controller init
│   ├── LipSyncController.cs         # Vowel cycling lip sync
│   ├── EmotionController.cs         # Expression blend shapes
│   ├── BlinkController.cs           # Randomized auto-blink
│   ├── EyeTrackingController.cs     # Mouse → VRM LookAt
│   └── IdleAnimationController.cs   # Breathing + head drift
├── Dance/
│   ├── BeatDanceController.cs       # 13-bone procedural dance
│   └── VrmaAnimationController.cs   # Runtime VRMA loading + playback
├── Input/
│   └── HotkeyManager.cs            # Keyboard shortcuts
└── CompanionManager.cs              # Main coordinator
```

## Scene: SampleScene
```
Hierarchy:
├── Main Camera (FOV 30, pos 0,1,3.5, looking at avatar)
├── Directional Light
├── Global Volume (URP post-processing)
├── CompanionManager
│   └── Components: WebSocketClient, MessageHandler, CompanionManager, HotkeyManager
└── AvatarRoot
    └── Components: AvatarController, LipSyncController, EmotionController,
                    BlinkController, EyeTrackingController, IdleAnimationController,
                    BeatDanceController, VrmaAnimationController
```

## StreamingAssets
```
Assets/StreamingAssets/
├── Models/
│   └── claire_avatar.vrm
└── Animations/
    ├── shikanoko_dance.vrma
    ├── shikanoko_dance_original.vrma
    ├── shikanoko_dance.fbx
    └── rumba_dancing.fbx
```

## Hotkeys
| Key | Action |
|-----|--------|
| D | Cycle dance style (None → Procedural → Shikanoko) |
| S | Toggle silence |
| Q | Pause read-aloud |
| R | Resume read-aloud |
| 1/2/3 | Set dance style directly |
| Space | Interrupt |
| Escape | Return to idle |
| Ctrl+Shift+R | Read aloud (global) |
| Ctrl+Shift+A | Active mode (global) |
| Ctrl+Shift+D | Dance mode (global) |
| Ctrl+Shift+M | Toggle mute (global) |

## Build
1. Open Unity project at `C:\Users\blakd\unit`
2. Menu: **Annabeth → Configure Build Settings** (sets window size, D3D11, etc.)
3. Menu: **Annabeth → Build Standalone** or **File → Build And Run**

## Transparent Window (Standalone Only)
The `TransparentWindowController` uses Win32 P/Invoke to:
- Remove window borders (frameless)
- Enable DWM transparency (transparent background)
- Set always-on-top
- Right-click drag to move window
- Only active in standalone builds, not in Editor.

### 8. Player Settings
**Edit → Project Settings → Player:**
- Color Space: Linear
- Scripting Backend: Mono (faster builds) or IL2CPP (release)
- API Compatibility: .NET Standard 2.1

### 9. Camera Setup
Configure Main Camera:
- Clear Flags: Solid Color
- Background: Black with Alpha = 0 (for transparency)
- Position: (0, 1, -2) — adjust based on VRM size

---

## Script Overview

### Core/SocketClient.cs
Handles TCP socket connection to Python backend. Features:
- Auto-reconnect on disconnect
- Message queuing for thread safety
- Newline-delimited JSON messages

### Core/MessageHandler.cs
Routes incoming messages to appropriate handlers. Provides:
- Strongly-typed events (OnSpeakStart, OnEmotionChange, etc.)
- Message type constants matching Python config
- Send methods for outgoing messages

### Avatar/AvatarController.cs
Loads and manages VRM model. Features:
- Async VRM loading from StreamingAssets
- Bone access for animation
- Coordinates sub-controllers

### Avatar/LipSyncController.cs
Animates mouth during speech:
- Cycles through A, I, U, E, O vowel shapes
- Smooth transitions between shapes
- Configurable speed and intensity

### Avatar/EmotionController.cs
Controls facial expressions:
- Maps emotion strings to VRM BlendShapes
- Supports: happy, angry, sad, surprised, relaxed
- Smooth transitions between emotions

### Avatar/BlinkController.cs
Automatic eye blinking:
- Randomized blink intervals (2-6 seconds)
- Natural blink animation curve
- Can trigger manual blinks

### Avatar/EyeTrackingController.cs
Eyes follow mouse cursor:
- Uses VRM LookAt system
- Configurable angle limits
- Smooth tracking movement

### Dance/BeatDanceController.cs
Procedural beat-reactive dance:
- Hips bounce, spine sway, head bob
- Arm swing synchronized to beat
- Responds to audio analysis data

### Input/HotkeyManager.cs
Keyboard shortcuts:
- D: Cycle dance style
- S: Toggle silence
- Q/R: Pause/resume read-aloud
- 1-4: Mode selection
- Ctrl+Shift+R/A/D/M: Global hotkeys

### CompanionManager.cs
Main coordinator:
- Wires all components together
- Handles mode transitions
- Manages companion state

---

## Python Backend Changes

The Python backend needs minimal changes. The socket server already exists; just ensure it sends newline-delimited JSON.

Message format:
```json
{"type": "speak_start", "text": "Hello!"}\n
{"type": "speak_end"}\n
{"type": "emotion", "emotion": "happy"}\n
{"type": "audio_analysis", "beat_energy": 0.5, "bass_energy": 0.3, "treble_energy": 0.2}\n
```

---

## Building

1. **File → Build Settings**
2. Add current scene
3. Platform: **Windows**
4. Architecture: **x86_64**
5. Click **Build**
6. Output: `Builds/Annabeth.exe`

Update `start_annabeth.ps1` to launch the Unity build instead of PyQt6 companion.

---

## Migration Notes

### What Works Out of Box
- VRM loading (UniVRM handles all formats)
- Lip sync (vowel cycling)
- Emotions (all VRM presets)
- Eye tracking (VRMLookAt)
- Blinking (automatic)
- Transparent window (UniWindowController)
- Socket communication (TCP)
- Hotkeys (when window focused)

### TODO / Future Work
- VRMA animation playback
- System audio capture in Unity (currently stays in Python)
- Truly global hotkeys (need native plugin)
- Subtitle display (add TextMeshPro)
- Mode indicator UI

---

## Troubleshooting

### VRM not loading
- Check path: `StreamingAssets/Models/claire_avatar.vrm`
- Check console for UniVRM errors

### Window not transparent
- Ensure UniWindowController is in scene
- Check "Is Transparent" is enabled
- Camera background alpha must be 0

### Socket not connecting
- Ensure Python backend is running
- Check port matches (default: 8765)
- Check firewall settings

### Lip sync not working
- Ensure `speak_start`/`speak_end` messages are sent
- Check LipSyncController is initialized
