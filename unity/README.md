# Annabeth Unity Scripts

This folder contains pre-written C# scripts for the Unity migration. These scripts are designed to be dropped into a Unity project after setting up UniVRM and UniWindowController.

## Setup Instructions

### 1. Install Unity
1. Download [Unity Hub](https://unity.com/download)
2. Install **Unity 2022.3 LTS** (with Windows Build Support)

### 2. Create Unity Project
1. Open Unity Hub → New Project
2. Select **3D (URP)** or **3D** template
3. Name: `AnnabethUnity`
4. Create in: `E:\Unity\AnnabethUnity` (or preferred location)

### 3. Install Required Packages
Open **Window → Package Manager**, click **+** → **Add package from git URL**:

```
# UniVRM (VRM loading)
https://github.com/vrm-c/UniVRM.git?path=/Packages/com.vrmc.vrm#v0.131.0

# UniWindowController (transparent window)
https://github.com/kirurobo/UniWindowController.git#upm
```

### 4. Install Newtonsoft.Json
In Package Manager, search for **Newtonsoft Json** and install it.

### 5. Copy Scripts
Copy the entire `Scripts/` folder to your Unity project's `Assets/Scripts/`.

### 6. Project Structure
```
Assets/
├── Scripts/
│   ├── Core/
│   │   ├── SocketClient.cs      # Python communication
│   │   └── MessageHandler.cs    # Message routing
│   ├── Avatar/
│   │   ├── AvatarController.cs  # VRM loading & coordination
│   │   ├── LipSyncController.cs # Vowel-based lip sync
│   │   ├── EmotionController.cs # Expression control
│   │   ├── BlinkController.cs   # Auto-blink
│   │   └── EyeTrackingController.cs # Mouse follow
│   ├── Dance/
│   │   └── BeatDanceController.cs # Procedural dance
│   ├── Input/
│   │   └── HotkeyManager.cs     # Keyboard shortcuts
│   └── CompanionManager.cs      # Main coordinator
├── StreamingAssets/
│   └── Models/
│       └── claire_avatar.vrm    # Copy from Anabeth/models/vrm/
├── Prefabs/
│   └── UniWindowController.prefab # From package
└── Scenes/
    └── Main.unity
```

### 7. Scene Setup
1. Add an empty GameObject named `Companion`
2. Add these components to it:
   - `SocketClient`
   - `MessageHandler`
   - `CompanionManager`
   - `HotkeyManager`
3. Add a child GameObject named `Avatar`
4. Add these components to Avatar:
   - `AvatarController`
   - `LipSyncController`
   - `EmotionController`
   - `BlinkController`
   - `EyeTrackingController`
   - `BeatDanceController`
5. Add `UniWindowController` prefab to scene
6. Configure UniWindowController:
   - Is Transparent: ✓
   - Is Topmost: ✓
   - Is Hit Test Enabled: ✓

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
