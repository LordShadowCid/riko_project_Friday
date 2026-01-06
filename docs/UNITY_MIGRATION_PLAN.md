# Annabeth Unity Migration Plan

## Executive Summary

This document details the complete migration path from the current **three-vrm + PyQt6 WebView** architecture to a native **Unity** application for the Annabeth Desktop Companion.

### Migration Verdict: ✅ **FEASIBLE** - Estimated 3-4 weeks for full implementation

---

## 1. Hardware Analysis

### Your System Specifications
| Component | Specification | Unity Compatibility |
|-----------|---------------|---------------------|
| **CPU** | Intel Core i9-12900 (16 cores, 24 threads) | ✅ Excellent - far exceeds requirements |
| **RAM** | 64 GB | ✅ Excellent - Unity recommends 8GB minimum |
| **GPU 0** | NVIDIA RTX A4000 (16GB VRAM) | ✅ Excellent - professional-grade |
| **GPU 1** | NVIDIA Quadro RTX 4000 (8GB VRAM) | ✅ Excellent - can run Unity while Whisper uses other GPU |
| **OS** | Windows 10/11 | ✅ Fully supported |
| **Storage** | 224GB free (C:), 864GB free (E:) | ✅ Plenty for Unity (~10-15GB needed) |

**Hardware Verdict**: Your system significantly exceeds Unity requirements. You can run Unity, GPT-SoVITS, Ollama, and Whisper simultaneously without issues.

---

## 2. Current Architecture Analysis

### What We Have Now
```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Backend (main_chat.py)                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Whisper ASR  │  │ Ollama LLM   │  │ GPT-SoVITS TTS        │  │
│  │ (GPU 1)      │  │ (Llama3.1-8b)│  │ (GPU 0)               │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼────────────────────────────────┐ │
│  │              Avatar Server (WebSocket :8765)                │ │
│  │  • Mode management (active/idle/dance)                      │ │
│  │  • Silence toggle                                           │ │
│  │  • Read-aloud control                                       │ │
│  │  • Audio analysis broadcast                                 │ │
│  └───────────────────────────┬────────────────────────────────┘ │
└──────────────────────────────┼──────────────────────────────────┘
                               │ WebSocket
┌──────────────────────────────▼──────────────────────────────────┐
│               PyQt6 WebEngine (desktop_companion_webview.py)     │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Chromium WebView                         │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │              companion.html (THREE.js)               │  │  │
│  │  │  • three-vrm (VRM 0.x/1.0 loading)                  │  │  │
│  │  │  • three-vrm-animation (VRMA dance)                 │  │  │
│  │  │  • Procedural dance (beat-reactive)                 │  │  │
│  │  │  • Lip sync (vowel-based)                           │  │  │
│  │  │  • Eye tracking (mouse follow)                      │  │  │
│  │  │  • Blinking animation                               │  │  │
│  │  │  • Emotion blend shapes                             │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│  • Transparent window (WA_TranslucentBackground)                │
│  • Always-on-top (WindowStaysOnTopHint)                        │
│  • Hotkeys (D, S, Q, R, 1-4, Space)                            │
│  • Global hotkeys (Ctrl+Shift+R/A/D/M)                         │
│  • Window dragging (right-click + drag)                        │
└─────────────────────────────────────────────────────────────────┘
```

### Current Components to Preserve
1. **Python Backend** (stays 100% the same)
   - `main_chat.py` - Core chat loop
   - Whisper ASR
   - Ollama LLM integration
   - GPT-SoVITS TTS
   - Read-aloud manager
   - Speaker identification

2. **Communication Layer** (needs adaptation)
   - WebSocket → TCP Socket (faster, simpler)
   - Same message types, different transport

3. **Frontend Rendering** (complete replacement)
   - THREE.js → Unity
   - three-vrm → UniVRM
   - PyQt6 window → Unity with UniWindowController

---

## 3. Target Unity Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Backend (UNCHANGED)                    │
├─────────────────────────────────────────────────────────────────┤
│  main_chat.py + Whisper + Ollama + GPT-SoVITS + ReadAloud       │
│                              │                                   │
│  ┌───────────────────────────▼────────────────────────────────┐ │
│  │         TCP Socket Server (replaces WebSocket)              │ │
│  │         Port: 8765 (or keep WebSocket for compatibility)    │ │
│  └───────────────────────────┬────────────────────────────────┘ │
└──────────────────────────────┼──────────────────────────────────┘
                               │ TCP/WebSocket
┌──────────────────────────────▼──────────────────────────────────┐
│                      Unity Application                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   UniWindowController                    │    │
│  │  • Transparent window (alpha channel)                   │    │
│  │  • Always-on-top                                        │    │
│  │  • Click-through on transparent pixels                  │    │
│  │  • Window dragging                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      UniVRM                              │    │
│  │  • VRM 0.x / 1.0 loading                                │    │
│  │  • VRMA animation playback                              │    │
│  │  • BlendShape expressions (Joy, Angry, Sad, Fun)        │    │
│  │  • Lip sync (A, I, U, E, O vowels)                      │    │
│  │  • Eye tracking (VRMLookAt)                             │    │
│  │  • Auto-blink                                           │    │
│  │  • Spring bones (hair/clothing physics)                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               Custom Scripts (C#)                        │    │
│  │  • SocketClient.cs - Python communication               │    │
│  │  • AvatarController.cs - Animation state machine        │    │
│  │  • LipSyncController.cs - Audio-reactive lip sync       │    │
│  │  • DanceController.cs - Beat-reactive dance             │    │
│  │  • HotkeyManager.cs - Keyboard input handling           │    │
│  │  • AudioAnalyzer.cs - System audio capture (WASAPI)     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Required Unity Packages

### Core Dependencies
| Package | Purpose | License | Install Method |
|---------|---------|---------|----------------|
| **UniVRM** | VRM 0.x/1.0 loading & animation | MIT | UPM git URL |
| **UniWindowController** | Transparent desktop window | MIT | UPM git URL |
| **Newtonsoft.Json** | JSON parsing for socket messages | MIT | Package Manager |

### UPM Installation URLs
```
UniVRM: https://github.com/vrm-c/UniVRM.git#upm
UniWindowController: https://github.com/kirurobo/UniWindowController.git#upm
```

### Optional Enhancements
| Package | Purpose | Notes |
|---------|---------|-------|
| **uLipSync** | Real-time lip sync from audio | Optional - better than vowel cycling |
| **VRM Spring Bone** | Physics for hair/clothes | Included with UniVRM |
| **DOTween** | Smooth animation tweening | Free on Asset Store |

---

## 5. Component-by-Component Migration

### 5.1 Window System (PyQt6 → UniWindowController)

**Current (PyQt6):**
```python
self.setWindowFlags(
    Qt.WindowType.FramelessWindowHint |
    Qt.WindowType.WindowStaysOnTopHint |
    Qt.WindowType.Tool
)
self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
```

**Unity Equivalent:**
```csharp
// Add UniWindowController prefab to scene
// Configure in Inspector:
// - IsTransparent: true
// - IsTopmost: true  
// - IsHitTestEnabled: true (for click-through)
```

**Migration Effort**: 🟢 Easy (1-2 hours)

---

### 5.2 VRM Loading (three-vrm → UniVRM)

**Current (JavaScript):**
```javascript
loader.register((parser) => new VRMLoaderPlugin(parser));
loader.load(vrmUrl, (gltf) => {
    vrm = gltf.userData.vrm;
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    scene.add(gltf.scene);
});
```

**Unity Equivalent:**
```csharp
using UniVRM10;

public async Task LoadVRM(string path)
{
    var vrm10 = await Vrm10.LoadPathAsync(path);
    _vrm = vrm10.Vrm;
    _vrmInstance = vrm10;
}
```

**Migration Effort**: 🟢 Easy (2-3 hours)

---

### 5.3 Lip Sync (vowel cycling → VRM BlendShapes)

**Current (JavaScript):**
```javascript
const vowels = ['aa', 'ih', 'ou', 'ee', 'oh'];
let currentVowelIndex = 0;

function updateLipSync(delta) {
    vowelTimer += delta;
    if (vowelTimer > 0.1) {
        currentVowelIndex = (currentVowelIndex + 1) % vowels.length;
        setMouth(vowels[currentVowelIndex], 0.6);
    }
}
```

**Unity Equivalent:**
```csharp
public class LipSyncController : MonoBehaviour
{
    private Vrm10Instance _vrm;
    private ExpressionKey[] _vowels = {
        ExpressionKey.Aa, ExpressionKey.Ih, ExpressionKey.Ou, 
        ExpressionKey.Ee, ExpressionKey.Oh
    };
    
    public void UpdateLipSync(bool isSpeaking)
    {
        if (!isSpeaking) {
            _vrm.Runtime.Expression.SetWeight(ExpressionKey.Aa, 0);
            return;
        }
        
        // Cycle vowels or use audio amplitude
        var vowel = _vowels[_currentIndex];
        _vrm.Runtime.Expression.SetWeight(vowel, 0.6f);
    }
}
```

**Migration Effort**: 🟡 Medium (3-4 hours)

---

### 5.4 Beat-Reactive Dance (procedural → C# implementation)

**Current (JavaScript):**
```javascript
function applyBeatDance(delta) {
    dancePhase += delta * (2 + beatEnergy * 3);
    const bounceAmount = beatEnergy * 0.02;
    const hips = vrm.humanoid.getNormalizedBoneNode('hips');
    hips.position.y = originalHipsY + Math.sin(dancePhase * 4) * bounceAmount;
    // ... arm sway, head bob, etc.
}
```

**Unity Equivalent:**
```csharp
public class DanceController : MonoBehaviour
{
    private Animator _animator;
    private float _dancePhase;
    
    public void UpdateBeatDance(float beatEnergy, float delta)
    {
        _dancePhase += delta * (2f + beatEnergy * 3f);
        
        var hips = _vrm.Humanoid.GetBoneTransform(HumanBodyBones.Hips);
        float bounce = beatEnergy * 0.02f;
        hips.localPosition = new Vector3(0, Mathf.Sin(_dancePhase * 4f) * bounce, 0);
    }
}
```

**Migration Effort**: 🟡 Medium (4-6 hours - need to port all bone manipulations)

---

### 5.5 VRMA Animation (three-vrm-animation → UniVRM)

**Current (JavaScript):**
```javascript
import { VRMAnimationLoaderPlugin, createVRMAnimationClip } from '@pixiv/three-vrm-animation';

async function playVRMAAnimation(url) {
    const animation = await loadVRMA(url);
    const clip = createVRMAnimationClip(animation, vrm);
    mixer.clipAction(clip).play();
}
```

**Unity Equivalent:**
```csharp
using UniVRM10.VRMAnimation;

public async Task PlayVRMAAnimation(string path)
{
    var animation = await VrmAnimation.LoadAsync(path);
    var clip = animation.CreateAnimationClip(_vrm.Humanoid);
    _animator.Play(clip.name);
}
```

**Migration Effort**: 🟢 Easy (2-3 hours)

---

### 5.6 Socket Communication (WebSocket → TCP/UDP)

**Current Python Server (avatar_server.py):**
```python
from aiohttp import web

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    async for msg in ws:
        data = json.loads(msg.data)
        # Handle messages...
```

**Modified Python Server (for Unity):**
```python
import socket
import json

class TcpServer:
    def __init__(self, host='127.0.0.1', port=8765):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen(1)
    
    def send(self, data: dict):
        msg = json.dumps(data) + '\n'
        self.client.send(msg.encode())
```

**Unity Client (SocketClient.cs):**
```csharp
using System.Net.Sockets;
using System.Text;
using Newtonsoft.Json;

public class SocketClient : MonoBehaviour
{
    private TcpClient _client;
    private NetworkStream _stream;
    private Queue<string> _messageQueue = new Queue<string>();
    
    void Start()
    {
        _client = new TcpClient("127.0.0.1", 8765);
        _stream = _client.GetStream();
        StartCoroutine(ReceiveLoop());
    }
    
    IEnumerator ReceiveLoop()
    {
        byte[] buffer = new byte[4096];
        while (true)
        {
            if (_stream.DataAvailable)
            {
                int bytes = _stream.Read(buffer, 0, buffer.Length);
                string message = Encoding.UTF8.GetString(buffer, 0, bytes);
                ProcessMessage(message);
            }
            yield return null;
        }
    }
    
    void ProcessMessage(string json)
    {
        var data = JsonConvert.DeserializeObject<Dictionary<string, object>>(json);
        string type = data["type"].ToString();
        
        switch (type)
        {
            case "speak_start":
                OnSpeakStart(data["text"]?.ToString());
                break;
            case "speak_end":
                OnSpeakEnd();
                break;
            case "emotion":
                OnEmotionChange(data["emotion"].ToString());
                break;
            case "audio_analysis":
                OnAudioData(data);
                break;
        }
    }
}
```

**Migration Effort**: 🟡 Medium (4-6 hours)

---

### 5.7 System Audio Capture (PyAudioWPatch → Unity WASAPI)

**Current (audio_analyzer.py):**
```python
import pyaudiowpatch as pyaudio

class SystemAudioAnalyzer:
    def _find_loopback_device(self):
        self.p = pyaudio.PyAudio()
        # Find WASAPI loopback device...
```

**Unity Options:**

1. **Keep in Python** (Recommended for simplicity)
   - Continue using PyAudioWPatch
   - Send audio analysis data over socket
   - No Unity changes needed

2. **Move to Unity** (Better performance)
   - Use CSCore or NAudio for WASAPI loopback
   - Native C# audio analysis

**Recommendation**: Keep audio capture in Python initially, migrate later if needed.

**Migration Effort**: 🟢 None (keep in Python) or 🔴 Complex (8+ hours for Unity)

---

### 5.8 Hotkey System (keyboard library → Unity Input)

**Current (desktop_companion_webview.py):**
```python
import keyboard

keyboard.add_hotkey('ctrl+shift+r', self._trigger_read_aloud)
keyboard.add_hotkey('ctrl+shift+a', self._toggle_active)
```

**Unity Equivalent:**
```csharp
void Update()
{
    // Global hotkeys with Unity's new Input System
    if (Input.GetKey(KeyCode.LeftControl) && Input.GetKey(KeyCode.LeftShift))
    {
        if (Input.GetKeyDown(KeyCode.R)) TriggerReadAloud();
        if (Input.GetKeyDown(KeyCode.A)) ToggleActive();
    }
    
    // Regular hotkeys
    if (Input.GetKeyDown(KeyCode.S)) ToggleSilence();
    if (Input.GetKeyDown(KeyCode.D)) CycleDance();
}
```

**Note**: Unity handles global hotkeys automatically when window is focused. For truly global hotkeys (works even when Unity isn't focused), you'd need a native plugin or run the Python keyboard listener separately.

**Migration Effort**: 🟢 Easy (2-3 hours)

---

## 6. Detailed Implementation Timeline

### Phase 1: Foundation (Week 1)
| Day | Task | Hours |
|-----|------|-------|
| 1 | Install Unity Hub, Unity 2022.3 LTS | 2h |
| 1 | Create new Unity project, import UniVRM & UniWindowController | 2h |
| 1 | Configure project settings (Linear color, D3D11) | 1h |
| 2 | Set up transparent window with UniWindowController | 3h |
| 2 | Test VRM loading with existing claire_avatar.vrm | 2h |
| 3 | Implement basic scene (lighting, camera, VRM display) | 4h |
| 4 | Create TCP socket client for Python communication | 4h |
| 5 | Test two-way communication with Python backend | 4h |

### Phase 2: Core Features (Week 2)
| Day | Task | Hours |
|-----|------|-------|
| 1 | Implement speak_start/speak_end handlers | 3h |
| 1 | Implement lip sync (vowel cycling) | 3h |
| 2 | Implement emotion blend shapes | 3h |
| 2 | Implement eye tracking (mouse follow) | 2h |
| 3 | Implement auto-blink | 2h |
| 3 | Implement idle breathing animation | 2h |
| 4 | Implement mode switching (active/idle/dance) | 3h |
| 4 | Implement procedural beat dance | 4h |
| 5 | Integrate audio analysis data from Python | 4h |

### Phase 3: Polish & Integration (Week 3)
| Day | Task | Hours |
|-----|------|-------|
| 1 | Implement VRMA animation playback | 4h |
| 1 | Implement hotkey system | 2h |
| 2 | Implement window dragging | 2h |
| 2 | Implement subtitle display (TextMeshPro) | 3h |
| 3 | Implement mode indicator UI | 2h |
| 3 | Test all features end-to-end | 4h |
| 4 | Fix bugs and edge cases | 6h |
| 5 | Build standalone executable | 2h |
| 5 | Update start_annabeth.ps1 to launch Unity build | 2h |

### Phase 4: Testing & Refinement (Week 4)
| Day | Task | Hours |
|-----|------|-------|
| 1-2 | Extended testing with real usage | 8h |
| 3-4 | Performance optimization | 8h |
| 5 | Documentation and cleanup | 4h |

**Total Estimated Hours**: ~100 hours
**Calendar Time**: 3-4 weeks (working part-time, ~20-25h/week)

---

## 7. File Structure for Unity Project

```
AnnabethUnity/
├── Assets/
│   ├── Animations/
│   │   └── shikanoko_dance.vrma
│   ├── Models/
│   │   └── claire_avatar.vrm
│   ├── Prefabs/
│   │   ├── UniWindowController.prefab
│   │   └── AnnabethAvatar.prefab
│   ├── Scenes/
│   │   └── MainScene.unity
│   ├── Scripts/
│   │   ├── Core/
│   │   │   ├── SocketClient.cs
│   │   │   ├── MessageHandler.cs
│   │   │   └── CompanionMode.cs
│   │   ├── Avatar/
│   │   │   ├── AvatarController.cs
│   │   │   ├── VRMLoader.cs
│   │   │   ├── LipSyncController.cs
│   │   │   ├── BlinkController.cs
│   │   │   └── EmotionController.cs
│   │   ├── Dance/
│   │   │   ├── DanceController.cs
│   │   │   ├── BeatDance.cs
│   │   │   └── VRMAPlayer.cs
│   │   ├── Input/
│   │   │   └── HotkeyManager.cs
│   │   └── UI/
│   │       ├── SubtitleDisplay.cs
│   │       └── ModeIndicator.cs
│   ├── Settings/
│   │   └── MessageTypes.cs  (mirrors shared/config.py)
│   └── StreamingAssets/
│       └── config.json  (optional runtime config)
├── Packages/
│   └── manifest.json
├── ProjectSettings/
└── Builds/
    └── Annabeth.exe
```

---

## 8. Python Backend Modifications

### Minimal changes required:

1. **Add TCP socket option to avatar_server.py**
   - Keep WebSocket for backward compatibility
   - Add TCP socket server as alternative

2. **Modify start_annabeth.ps1**
   - Launch Unity build instead of PyQt6 companion
   - Keep everything else the same

### New file: `server/unity_bridge.py`
```python
"""
Unity Bridge - TCP socket server for Unity communication.
Runs alongside avatar_server.py as an alternative transport.
"""
import socket
import json
import threading
from queue import Queue

class UnityBridge:
    def __init__(self, host='127.0.0.1', port=8765):
        self.host = host
        self.port = port
        self.client = None
        self.send_queue = Queue()
        self.running = False
    
    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        self.running = True
        
        threading.Thread(target=self._accept_loop, daemon=True).start()
        threading.Thread(target=self._send_loop, daemon=True).start()
    
    def _accept_loop(self):
        while self.running:
            self.client, addr = self.sock.accept()
            print(f"[UnityBridge] Unity connected from {addr}")
    
    def _send_loop(self):
        while self.running:
            if self.client and not self.send_queue.empty():
                msg = self.send_queue.get()
                try:
                    self.client.send((json.dumps(msg) + '\n').encode())
                except:
                    self.client = None
    
    def send(self, message: dict):
        self.send_queue.put(message)
```

---

## 9. Pros and Cons Summary

### Advantages of Unity Migration
| Pro | Impact |
|-----|--------|
| **Native performance** | 60+ FPS consistently, no WebGL overhead |
| **Better physics** | Spring bones for hair/clothes (built into UniVRM) |
| **True transparency** | More reliable than Chromium alpha |
| **Faster startup** | No browser initialization (~2s vs ~5s) |
| **Smaller memory** | ~200MB vs ~400MB (no Chromium) |
| **Shader flexibility** | Custom effects, post-processing |
| **Future features** | Face tracking, motion capture, AR |

### Disadvantages of Unity Migration
| Con | Mitigation |
|-----|------------|
| **Learning curve** | UniVRM docs are good, lots of examples |
| **Build size** | ~80-100MB (vs ~30MB current) - acceptable |
| **Two codebases** | Clear separation Python/C# |
| **Development time** | 3-4 weeks initial investment |
| **Global hotkeys** | Need native plugin or keep Python keyboard |

---

## 10. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| UniVRM compatibility issues | Low | Medium | Well-tested library, active community |
| Transparent window bugs | Medium | Low | UniWindowController is mature (633 stars) |
| Socket performance | Low | Low | TCP is fast enough for this use case |
| Unity learning curve | Medium | Medium | Focus on essentials, iterate |
| Spring bone issues | Low | Low | Works out of box with UniVRM |

---

## 11. Getting Started Checklist

### Prerequisites
- [ ] Download [Unity Hub](https://unity.com/download)
- [ ] Install Unity 2022.3 LTS (with Windows Build Support)
- [ ] Clone Annabeth repo to new branch: `git checkout -b unity-migration`

### Day 1 Tasks
1. [ ] Create Unity project: `AnnabethUnity`
2. [ ] Import UniVRM via Package Manager → Add from Git URL:
   ```
   https://github.com/vrm-c/UniVRM.git?path=/Packages/com.vrmc.vrm#v0.131.0
   ```
3. [ ] Import UniWindowController:
   ```
   https://github.com/kirurobo/UniWindowController.git#upm
   ```
4. [ ] Configure Player Settings:
   - Color Space: Linear
   - Scripting Backend: Mono (for faster iteration)
   - API Compatibility: .NET Standard 2.1
5. [ ] Add UniWindowController prefab to scene
6. [ ] Copy `claire_avatar.vrm` to `Assets/Models/`
7. [ ] Test VRM loading

---

## 12. Reference Links

- [UniVRM Documentation](https://vrm.dev/en/univrm/)
- [UniVRM GitHub](https://github.com/vrm-c/UniVRM)
- [UniWindowController GitHub](https://github.com/kirurobo/UniWindowController)
- [UniWinC VRM Viewer Example](https://github.com/kirurobo/UniWinC_VRM)
- [Python-Unity Socket Communication](https://github.com/Siliconifier/Python-Unity-Socket-Communication)
- [Unity 2022.3 LTS](https://unity.com/releases/editor/qa/lts-releases)

---

## 13. Decision Matrix

| Factor | Current (three-vrm) | Unity | Winner |
|--------|---------------------|-------|--------|
| Performance | WebGL, 30-60 FPS | Native, 60+ FPS | Unity |
| Physics | None | Spring bones | Unity |
| Transparency | Good | Excellent | Unity |
| Development speed | JavaScript, fast iteration | C#, slower builds | Current |
| Memory usage | ~400MB (Chromium) | ~200MB | Unity |
| Startup time | ~5s | ~2s | Unity |
| Cross-platform | Trivial | Possible but harder | Current |
| Community/docs | Smaller | Larger | Unity |
| Future features | Limited | Extensive | Unity |

**Recommendation**: Proceed with Unity migration as a **branch project**. Keep current system as fallback.

---

*Document created: January 6, 2026*
*Last updated: January 6, 2026*
