# Annabeth — Multi-Repo Feature Integration Plan
**Sources:** MekaHime-Pipeline-V1, MystiaTech/Mai, XargonWan/Synthetic_Heart, shinyflvre/Mate-Engine  
**Reference files:** `d:\Annabeth\temp_refs\` (all source code stored there)  
**Constraint:** All features are additive. Nothing existing is removed or rewritten.  
**Excludes:** Chibi mode (user excluded this per original request)

---

## Baseline Audit — What's Already Built

Understanding what exists avoids duplication:

| Feature | Status | Location |
|---|---|---|
| Model auto-routing (intent → fast/primary) | ✅ Done | `server/process/llm_funcs/model_router.py` |
| Emotion decay (`{happy 8.5}` tags, Plutchik wheel) | ✅ Done | `server/process/memory/emotion_state.py` |
| Basic Grillo reflection loop (diary + proactive queue) | ✅ Partial | `server/process/memory/reflection_loop.py` |
| RVC voice post-processing | ✅ Done | `server/process/tts_func/rvc_convert.py` |
| Eye + head + spine mouse tracking (Mate-Engine style) | ✅ Done | `unity/Scripts/Avatar/EyeTrackingController.cs` |
| Pet/pat circle detection | ✅ Done | `unity/Scripts/Interaction/PetDetectionController.cs` |
| Touch reaction animations | ✅ Done | `unity/Scripts/Interaction/TouchReactionController.cs` |
| Touch audio SFX | ✅ Done | `unity/Scripts/Interaction/TouchSoundHandler.cs` |
| Speech bubble UI | ✅ Done | `unity/Scripts/UI/SpeechBubble.cs` |
| VRM facial morph control | ✅ Partial | `unity/Scripts/Avatar/EmotionController.cs` |
| WebRTC VAD | ❌ Missing | Upgrade `client/audio_analyzer.py` |
| Facial expression timeline (`[em_NAME:INTENSITY]`) | ❌ Missing | Server parser + Unity WebSocket handler |
| Grillo weighted beat types | ❌ Missing | Upgrade `server/process/memory/reflection_loop.py` |
| Runtime config variables engine | ❌ Missing | New: `server/settings_registry.py` |
| Code self-improvement (AST-based) | ❌ Missing | New: `server/process/self_improvement/` |
| Idle speech bubbles (random timer) | ❌ Missing | New: `unity/Scripts/UI/IdleBubbleController.cs` |
| Idle/screensaver mode | ❌ Missing | New: `unity/Scripts/Avatar/IdleController.cs` |
| Discord Rich Presence | ❌ Missing | Unity package + new C# component |
| Model latency/memory-based switching | ❌ Missing | Upgrade `model_router.py` |

---

## Recommended Execution Order

```
Phase 5 (RVC rmvpe upgrade — 30min fix, zero risk)
  → Phase 8 (Settings registry — foundation for later phases)
    → Phase 1 (Grillo beat types — upgrades existing reflection_loop)
    → Phase 3 (Model latency switching — small upgrade to model_router)
Phase 4 (WebRTC VAD — additive to audio_analyzer)
Phase 2 (Facial expression tags — new server module + Unity handler)
Phase 6 (Idle speech bubbles — drains Grillo's proactive queue)
  → Phase 7 (Idle/screensaver mode — feeds idle bubbles and Discord)
Phase 9 (Discord Rich Presence — builds on idle mode states)
Phase 10 (Code self-improvement — low priority, background task)
```

---

## PHASE 1 — Grillo Weighted Beat Types
**Source:** `temp_refs/synth_heart_grillo_impl.py`  
**File:** `server/process/memory/reflection_loop.py`  
**Risk:** Low  **Effort:** ~2 hours

### Problem
The current `reflection_loop.py` runs a single generic reflection every 45 minutes. Synthetic_Heart's G.R.I.L.L.O. system uses 6 weighted beat types that give Annabeth a richer internal life with directed thinking.

### What to add

**Step 1.1 — Beat type constants** (add near top of file):
```python
BEAT_TYPES: dict[str, float] = {
    "tag_elaboration":     0.25,  # Riff on a topic from conversation
    "memory_consolidation":0.15,  # Surface a relevant memory
    "diary_consolidation": 0.15,  # Write a structured diary entry
    "self_reflection":     0.20,  # Reflect on emotional state
    "curiosity":           0.15,  # Form a question or new observation
    "relationship":        0.10,  # Reflect on bond with user
}
```

**Step 1.2 — Beat selection function**:
```python
import random

def _select_beat_type() -> str:
    types = list(BEAT_TYPES.keys())
    weights = list(BEAT_TYPES.values())
    return random.choices(types, weights=weights, k=1)[0]
```

**Step 1.3 — Per-beat prompt map**:
```python
_BEAT_PROMPTS: dict[str, str] = {
    "tag_elaboration": (
        "You are Annabeth, thinking quietly to yourself. Pick one topic from the recent "
        "conversation and expand it with one new, original thought. 1-2 sentences max."
    ),
    "memory_consolidation": (
        "You are Annabeth. Search your memory for something relevant to the last conversation. "
        "Surface it as a single diary thought: 'I remember that...' or 'This reminds me of...'"
    ),
    "diary_consolidation": (
        "You are Annabeth writing a private diary entry. Summarize what just happened in 1-2 "
        "sentences. Focus on how it made you feel. First person, present tense."
    ),
    "self_reflection": (
        "You are Annabeth. Pause and check in with your feelings. Describe your emotional state "
        "right now in one honest sentence. Don't censor. 'Right now I feel...'"
    ),
    "curiosity": (
        "You are Annabeth, curious and thoughtful. What's one thing you want to know more about "
        "or ask your user? Phrase it as a question or an observation. 1 sentence."
    ),
    "relationship": (
        "You are Annabeth. Reflect on your relationship with your user. What do you appreciate? "
        "What do you wish for? One sincere sentence. Don't be sappy."
    ),
}
```

**Step 1.4 — Database table** (add to DB init):
```sql
CREATE TABLE IF NOT EXISTS grillo_activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    beat_type TEXT NOT NULL,
    prompt_text TEXT,
    response_text TEXT,
    diary_entry_id INTEGER
);
```

**Step 1.5 — Modify main beat loop**:

Replace/extend the existing `_run_reflection_cycle()` or equivalent:
```python
async def _run_beat(llm_fn):
    """Run one Grillo beat. Call this from the timer loop."""
    beat_type = _select_beat_type()
    prompt = _BEAT_PROMPTS[beat_type]
    response = await llm_fn(prompt)  # low-priority LLM call
    
    # Log to DB
    _log_grillo_beat(beat_type, prompt, response)
    
    # Route output based on beat type
    if beat_type in ("diary_consolidation", "self_reflection"):
        _write_diary(response, trigger=beat_type)
    if beat_type in ("curiosity", "relationship", "tag_elaboration"):
        # Queue as proactive speech (Annabeth says it aloud when idle)
        if not _proactive_queue.full():
            _proactive_queue.put_nowait(response)
```

**Step 1.6 — Conversation gate**:
Add a module-level flag and check it before running any beat:
```python
_conversation_active: bool = False

def set_conversation_active(active: bool):
    global _conversation_active
    _conversation_active = active

# In the timer loop:
if not _conversation_active:
    await _run_beat(llm_fn)
```
Wire `set_conversation_active(True/False)` in `main_chat.py` when conversation starts/ends.

---

## PHASE 2 — Facial Expression Timeline
**Source:** `temp_refs/synth_heart_facial_expression_plugin.py`  
**Files:** New `server/process/llm_funcs/facial_expressions.py`, edits to `server/annabeth_config.py`, `server/process/llm_funcs/llm_scr.py`, `unity/Scripts/Avatar/EmotionController.cs`  
**Risk:** Medium (touches LLM output, TTS path, Unity)  **Effort:** ~3 hours

### What this adds

The LLM embeds facial morph hints inline: `[em_smile:0.8]`. The server strips them before TTS and sends timed WebSocket messages to Unity to drive VRM BlendShapes proportionally to where they appear in the sentence.

### Server: Step 2.1 — System prompt injection

In `server/annabeth_config.py` or wherever the system prompt is assembled, add:
```python
FACIAL_EXPR_INJECT = (
    " You may embed facial expression hints inline using [em_NAME:INTENSITY] where "
    "NAME is one of: smile, grin, sad, blush, surprised, angry, wink, shy, neutral. "
    "INTENSITY is 0.0 to 1.0. Place them naturally in the text. Example: "
    "'That's so exciting! [em_surprised:0.9] I can't wait to hear more.'"
    " Only use 1-3 expressions per response."
)
```

Only inject this if `settings_registry.get("FACIAL_EXPR_ENABLED")` is True.

### Server: Step 2.2 — Create `server/process/llm_funcs/facial_expressions.py`

New file (full code in `temp_refs/synth_heart_facial_expression_plugin.py`):
```python
import re, asyncio
from typing import List, Tuple, Callable, Awaitable, Optional

_TAG_RE = re.compile(r'\[em_(\w+):(\d+(?:\.\d+)?)\]')

def parse_facial_expressions(text: str) -> Tuple[str, List[Tuple[int, str, float]]]:
    """Strip [em_NAME:INTENSITY] tags, return (clean_text, [(char_pos, name, intensity)])."""
    events: list = []
    clean = _TAG_RE.sub(lambda m: _collect(m, events), text)
    return clean, events

def _collect(m, events: list) -> str:
    events.append((m.start(), m.group(1), float(m.group(2))))
    return ""

async def play_expression_timeline(
    events: list,
    total_chars: int,
    broadcast_fn: Callable[[dict], Awaitable[None]],
    audio_duration_s: float = 0.0,
    chars_per_sec: float = 12.0,
):
    """Schedule WebSocket broadcasts for each expression event."""
    if not events:
        return
    duration = audio_duration_s if audio_duration_s > 0 else total_chars / chars_per_sec
    tasks = [_delayed_broadcast(
                delay=(pos / max(total_chars, 1)) * duration,
                msg={"type": "face_expression", "name": name, "intensity": intensity},
                fn=broadcast_fn,
             ) for pos, name, intensity in events]
    # Reset all expressions after audio ends
    tasks.append(_delayed_broadcast(duration + 0.5, {"type": "face_expression", "name": None, "intensity": 0.0}, broadcast_fn))
    await asyncio.gather(*tasks)

async def _delayed_broadcast(delay: float, msg: dict, fn):
    await asyncio.sleep(max(0.0, delay))
    await fn(msg)
```

### Server: Step 2.3 — Hook into `llm_scr.py`

After the full LLM response is assembled and before TTS is called:
```python
from server.process.llm_funcs.facial_expressions import parse_facial_expressions, play_expression_timeline

# After streaming complete, have full response text:
clean_text, face_events = parse_facial_expressions(full_response)

# Start face_timeline concurrently (don't await — let it run alongside TTS)
asyncio.create_task(
    play_expression_timeline(
        face_events, len(clean_text), ws_broadcast_fn,
        audio_duration_s=estimated_audio_s  # If known from TTS length estimate
    )
)

# Send clean_text (tags stripped) to TTS
await tts_module.synthesize(clean_text)
```

### Unity: Step 2.4 — Extend `EmotionController.cs`

Add a `SetExpression(string name, float intensity)` public method:
```csharp
// Expression name → VRM BlendShapeKey mapping
private static readonly Dictionary<string, string> _exprMap = new()
{
    { "smile",     "Happy"      },
    { "grin",      "Happy"      },
    { "sad",       "Sad"        },
    { "blush",     "Blushed"    },  // Verify blend shape name in your VRM
    { "surprised", "Surprised"  },
    { "angry",     "Angry"      },
    { "wink",      "Blink_L"    },
    { "shy",       "Blushed"    },
    { "neutral",   "Neutral"    },
};

public void SetExpression(string name, float intensity)
{
    if (string.IsNullOrEmpty(name) || name == "null")
    { 
        ResetExpressions(); 
        return; 
    }
    if (!_exprMap.TryGetValue(name, out var blendKey)) return;
    // Assumes _vrm is Vrm10Instance reference already cached
    _vrm.Runtime.Expression.SetWeight(ExpressionKey.CreateCustom(blendKey), intensity);
}

private void ResetExpressions()
{
    foreach (var val in _exprMap.Values)
        _vrm.Runtime.Expression.SetWeight(ExpressionKey.CreateCustom(val), 0f);
}
```

### Unity: Step 2.5 — Add WebSocket dispatch

In whatever class handles incoming WebSocket messages (likely `MessageHandler.cs` or `AvatarServerHandler.cs`):
```csharp
case "face_expression":
    var exprName = (string)data["name"];
    var intensity = data["intensity"] != null ? (float)data["intensity"] : 0f;
    emotionController.SetExpression(exprName, intensity);
    break;
```

---

## PHASE 3 — Model Latency / Memory-Based Auto-Switching
**Source:** `temp_refs/mai_selfmod.py` (ModelManager patterns)  
**File:** `server/process/llm_funcs/model_router.py`  
**Risk:** Low  **Effort:** ~1.5 hours

### Problem
The current router selects models by intent category only. If the primary model is slow or RAM is under pressure, there's no automatic fallback.

### Step 3.1 — Add latency tracking

```python
import time, psutil

# Module-level state (thread-safe; writes gated by GIL for primitives)
_last_latency_ms: float = 0.0
_forced_fast_until: float = 0.0
_LATENCY_THRESHOLD_MS = 5000
_MEMORY_THRESHOLD_PCT = 85
_SWITCH_COOLDOWN_S = 30

def record_latency(ms: float) -> None:
    """Call this after each LLM response to track response time."""
    global _last_latency_ms, _forced_fast_until
    _last_latency_ms = ms
    if ms > _LATENCY_THRESHOLD_MS:
        _forced_fast_until = time.time() + _SWITCH_COOLDOWN_S
        logger.info(f"[ModelRouter] Slow response ({ms:.0f}ms) — using fast model for {_SWITCH_COOLDOWN_S}s")
```

### Step 3.2 — Check in `get_model_for_intent()`

Add at the top of the method:
```python
# Memory pressure check
mem_pct = psutil.virtual_memory().percent
if mem_pct > _MEMORY_THRESHOLD_PCT:
    logger.warning(f"[ModelRouter] Memory at {mem_pct}% — using fast model")
    return self._resolve(self._fast)

# Latency cooldown check
if time.time() < _forced_fast_until:
    return self._resolve(self._fast)
```

### Step 3.3 — Record latency in `llm_scr.py`

```python
import time
from server.process.llm_funcs.model_router import record_latency

t_start = time.time()
# ... await LLM stream / response ...
record_latency((time.time() - t_start) * 1000)
```

---

## PHASE 4 — WebRTC VAD Upgrade
**Source:** `temp_refs/mekahime_vad_rvc.py` — `VADProcessor` class  
**File:** `client/audio_analyzer.py`  
**Risk:** Low (falls back gracefully if not installed)  **Effort:** ~1 hour

### Problem
Current audio pipeline uses energy-based silence detection. This produces false positives with consistent ambient noise (fans, AC). WebRTC VAD uses an ML model that's much more noise-robust.

### Step 4.1 — Add dependency

In `requirements.txt`:
```
webrtcvad-wheels>=2.0.14
```
`webrtcvad-wheels` is a pre-compiled binary wheel that works on Windows without a C compiler.

### Step 4.2 — Add `VADProcessor` 

Copy the `VADProcessor` class from `temp_refs/mekahime_vad_rvc.py` into `client/audio_analyzer.py` (or as a new `client/vad.py` imported by audio_analyzer).

Key parameters:
- `sample_rate = 16000` (webrtcvad requires exactly 8k/16k/32k/48k)
- `frame_size = 480` samples (30ms at 16kHz — must be exact)
- `aggressiveness = 2` (balanced)

### Step 4.3 — Wrap existing chunk processing

The existing pipeline already does silence detection. Add VAD as a pre-filter:
```python
vad = VADProcessor(aggressiveness=2)

# In the audio chunk loop, before adding to speech buffer:
if vad.detect_speech(chunk_np_float32_16khz):
    speech_buffer.append(chunk)
    silence_frames = 0
else:
    silence_frames += 1
    if silence_frames > max_silence_frames and speech_buffer:
        await flush_and_process(speech_buffer)
        speech_buffer.clear()
```

**Note:** If audio is captured at a rate other than 16kHz, resample the chunk before passing to VAD using `librosa.resample()` or `torchaudio.functional.resample()`.

---

## PHASE 5 — RVC `rmvpe` Upgrade
**Source:** `temp_refs/mekahime_vad_rvc.py`  
**File:** `server/process/tts_func/rvc_convert.py`  
**Risk:** Minimal — parameter-only change  **Effort:** 30 minutes

### What changes

MekaHime identified `rmvpe` as the best F0 extraction method for RVC (beats `harvest` and `crepe` in naturalness). The current `rvc_convert.py` uses `harvest` as default.

### Step 5.1 — Change default f0_method

```python
# In RvcConverter.__init__:
f0_method: str = "rmvpe"   # Changed from "harvest"
```

### Step 5.2 — Add rmvpe availability check

```python
# In RvcConverter._load(), after loading the model:
rmvpe_check = Path(self.model_path).parent / "rmvpe.pt"
if "rmvpe" in self.f0_method and not rmvpe_check.exists():
    logger.warning(
        "[RVC] rmvpe.pt not found in model directory. "
        "Falling back to harvest. Download from: "
        "https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
    )
    self.f0_method = "harvest"
```

### Step 5.3 — Verify parameters match MekaHime values

In the `vc_single()` call (wherever it exists in `rvc_convert.py`), confirm these params:
```python
index_rate=0.8,
filter_radius=3,
rms_mix_rate=0.8,
protect=0.33,
```
These are MekaHime's tuned values and are well-tested for voice similarity.

---

## PHASE 6 — Idle Speech Bubbles
**Source:** `temp_refs/mate_engine_random_messages.cs`  
**Files:** New `unity/Scripts/UI/IdleBubbleController.cs`, `unity/Scripts/UI/SpeechBubble.cs` (minor extension)  
**Risk:** Low  **Effort:** ~2 hours

### What this adds

When Annabeth is idle and the Grillo beat loop produces a "curiosity" or "relationship" thought, it surfaces as a floating speech bubble above the avatar — without requiring a voice response.

### Step 6.1 — Server: drain proactive queue to WebSocket

The `_proactive_queue` in `reflection_loop.py` already holds these thoughts. Add a drainer in `main_chat.py`:
```python
from server.process.memory.reflection_loop import get_proactive_queue

async def drain_proactive_thoughts(ws_broadcast_fn):
    q = get_proactive_queue()
    while not q.empty():
        thought = q.get_nowait()
        await ws_broadcast_fn({"type": "idle_thought", "text": thought})
```

Call `drain_proactive_thoughts()` on a 30-second timer from the main server loop (only when not in active conversation).

### Step 6.2 — Create `unity/Scripts/UI/IdleBubbleController.cs`

Based on `temp_refs/mate_engine_random_messages.cs`. Differences:
- No hardcoded text pool (thoughts come from server via WebSocket)
- Uses existing `SpeechBubble.cs` component instead of instantiating a prefab
- Checks companion state: only show during "Idle"/"Listening", never during "Talking"/"Dancing"

```csharp
using UnityEngine;

namespace Annabeth.UI
{
    public class IdleBubbleController : MonoBehaviour
    {
        [SerializeField] private SpeechBubble speechBubble;
        [SerializeField] private float minDelaySeconds = 60f;   // From settings registry
        [SerializeField] private float maxDelaySeconds = 300f;

        private bool _idleMode = false;  // Set by IdleController

        public void ShowIdleThought(string text)
        {
            if (!_idleMode) return;  // Don't interrupt active conversation
            if (speechBubble == null) return;
            speechBubble.ShowText(text);  // Uses existing SpeechBubble streaming method
        }

        public void SetIdleMode(bool idle) => _idleMode = idle;
    }
}
```

### Step 6.3 — Wire WebSocket handler

In `MessageHandler.cs` or wherever WebSocket messages are dispatched:
```csharp
case "idle_thought":
    idleBubbleController.ShowIdleThought(data["text"]?.ToString());
    break;
```

---

## PHASE 7 — Idle / Screensaver Mode
**Source:** `temp_refs/mate_engine_screensaver.cs`  
**Files:** New `unity/Scripts/Avatar/IdleController.cs`, edits to `unity/Scripts/UI/SettingsPanel.cs`  
**Risk:** Low  **Effort:** ~2 hours

### What this adds

Detects global mouse/keyboard inactivity using Win32 API. After configurable timeout:
1. Annabeth enters "idle" pose (`isIdle` animator bool)
2. `EyeTrackingController` switches to `TrackingMode.Reduced`
3. After 2× timeout → "sleep" pose (`isSleeping`), `TrackingMode.Disabled`
4. Any user input → immediate wake + play wake animation

### Step 7.1 — Create `unity/Scripts/Avatar/IdleController.cs`

Based on `temp_refs/mate_engine_screensaver.cs`. Key adaptations:
- Use `EyeTrackingController.SetTrackingMode()` instead of manually setting animator parameters
- Reference `SettingsManager` for `idleTimeoutSeconds` (replace `SaveLoadHandler`)
- Expose an `OnIdleStateChange` event that feeds into Discord presence (Phase 9) and `IdleBubbleController` (Phase 6)

```csharp
public class IdleController : MonoBehaviour
{
    public event Action<bool> OnSleepStateChanged;   // true = sleeping, false = awake
    public event Action<bool> OnIdleStateChanged;    // true = idle, false = active

    // Win32 DllImport for GetCursorPos and GetAsyncKeyState 
    // (copy from temp_refs/mate_engine_screensaver.cs)
    
    private void OnSleep(bool sleeping)
    {
        animator.SetBool("isSleeping", sleeping);
        eyeTrackingController.SetTrackingMode(
            sleeping ? TrackingMode.Disabled : TrackingMode.Normal
        );
        OnSleepStateChanged?.Invoke(sleeping);
    }
}
```

### Step 7.2 — Add setting to `SettingsPanel.cs`

Expose "Idle Timeout" dropdown with options: Never / 30s / 1 min / 5 min / 15 min.

```csharp
// In SettingsPanel, add IdleTimeout dropdown:
idleTimeoutDropdown.onValueChanged.AddListener(idx => {
    int[] options = { 0, 30, 60, 300, 900 };  // 0 = disabled
    idleController.SetTimeout(options[idx]);
    settingsManager.idleTimeoutSeconds = options[idx];
    settingsManager.Save();
});
```

### Step 7.3 — Server notification (optional)

When Annabeth sleeps, server-side Grillo beats can pause (no point generating thoughts she won't express):
```csharp
// In IdleController.OnSleep:
AvatarServerHandler.Instance.SendEvent("avatar_sleep", new { sleeping });
```

Server-side in `main_chat.py`:
```python
elif msg_type == "avatar_sleep":
    reflection_loop.set_conversation_active(data["sleeping"])  # Pause Grillo while sleeping
```

---

## PHASE 8 — Runtime Settings Registry
**Source:** `temp_refs/synth_heart_variables_engine.py`  
**Files:** New `server/settings_registry.py`, small edits to other modules to read from it  
**Risk:** Low  **Effort:** ~1.5 hours

### What this adds

A typed, runtime-editable registry of all tuneable parameters. Replaces scattered hardcoded constants. Supports WebSocket-based live updates (no restart needed).

### Step 8.1 — Create `server/settings_registry.py`

```python
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

@dataclass
class VarDef:
    key: str
    label: str
    default: Any
    value_type: type
    description: str = ""
    validator: Optional[Callable[[Any], bool]] = None

class SettingsRegistry:
    def __init__(self):
        self._vars: Dict[str, VarDef] = {}
        self._values: Dict[str, Any] = {}

    def register(self, var: VarDef):
        self._vars[var.key] = var
        self._values[var.key] = var.default

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, self._vars.get(key, VarDef(key, "", default, type(default))).default)

    def set(self, key: str, value: Any) -> bool:
        var = self._vars.get(key)
        if var is None:
            return False
        if var.validator and not var.validator(value):
            return False
        self._values[key] = var.value_type(value)
        return True

    def all_vars(self) -> Dict[str, Any]:
        return {k: self._values.get(k, v.default) for k, v in self._vars.items()}

# Singleton
registry = SettingsRegistry()

def init_registry():
    """Register all default variables. Call once at server startup."""
    registry.register(VarDef("GRILLO_BEAT_INTERVAL",     "Grillo Beat Interval (s)",         2700, int))
    registry.register(VarDef("GRILLO_DREAM_ENABLED",     "Grillo Dream Mode",                 False, bool))
    registry.register(VarDef("EMOTION_DECAY_TAU",        "Emotion Decay Time Constant (s)",   3600, int))
    registry.register(VarDef("IDLE_TIMEOUT_S",           "Idle Timeout (s, 0=disabled)",      300, int))
    registry.register(VarDef("LATENCY_THRESHOLD_MS",     "LLM Latency Alert Threshold (ms)",  5000, int))
    registry.register(VarDef("MEMORY_THRESHOLD_PCT",     "RAM Alert Threshold %",             85, int))
    registry.register(VarDef("MODEL_SWITCH_COOLDOWN_S",  "Model Switch Cooldown (s)",         30, int))
    registry.register(VarDef("IDLE_BUBBLE_MIN_DELAY",    "Idle Bubble Min Delay (s)",         60, int))
    registry.register(VarDef("IDLE_BUBBLE_MAX_DELAY",    "Idle Bubble Max Delay (s)",         300, int))
    registry.register(VarDef("VAD_AGGRESSIVENESS",       "VAD Aggressiveness (0-3)",          2, int,
                               validator=lambda v: 0 <= int(v) <= 3))
    registry.register(VarDef("RVC_ENABLED",              "Enable RVC Post-Processing",        False, bool))
    registry.register(VarDef("FACIAL_EXPR_ENABLED",      "Enable Facial Expression Tags",     True, bool))
    registry.register(VarDef("MAX_DAILY_CODE_IMPROVES",  "Max Daily Auto Code Fixes",         3, int))
```

### Step 8.2 — Wire into `main_chat.py`

```python
from server.settings_registry import init_registry
init_registry()  # Call near top of startup sequence
```

### Step 8.3 — Add WebSocket handler for live updates

```python
elif msg_type == "set_setting":
    key = data.get("key")
    value = data.get("value")
    success = registry.set(key, value) if key else False
    await ws.send_json({"type": "setting_response", "key": key, "success": success})

elif msg_type == "get_settings":
    await ws.send_json({"type": "settings", "data": registry.all_vars()})
```

### Step 8.4 — Migrate hardcoded constants (one at a time)

Priority order:
1. `reflection_loop.py` → replace `DEFAULT_INTERVAL_SECONDS` with `registry.get("GRILLO_BEAT_INTERVAL")`
2. `emotion_state.py` → replace `DECAY_TAU` with `registry.get("EMOTION_DECAY_TAU")`
3. `model_router.py` → replace `_LATENCY_THRESHOLD_MS` with `registry.get("LATENCY_THRESHOLD_MS")`

---

## PHASE 9 — Discord Rich Presence
**Source:** `temp_refs/` (lachee/discord-rpc-unity package pattern from Mate-Engine)  
**Files:** New `unity/Scripts/Core/DiscordPresence.cs`, edit to `unity/Packages/manifest.json`  
**Risk:** Low  **Effort:** ~1.5 hours

### Step 9.1 — Add package to Unity

Edit `unity/Packages/manifest.json`:
```json
{
  "dependencies": {
    "com.lachee.discordrpc": "https://github.com/Lachee/discord-rpc-unity.git#upm",
    "com.unity.nuget.newtonsoft-json": "3.0.2"
  }
}
```

### Step 9.2 — Create Discord application

1. Go to https://discord.com/developers/applications
2. Create application named "Annabeth"
3. Copy Application ID → paste into `DiscordPresence.cs` Inspector field
4. Upload Rich Presence Assets: Annabeth art as `annabeth`, state icons as `idle`, `talking`, `listening`, `dancing`

### Step 9.3 — Create `unity/Scripts/Core/DiscordPresence.cs`

```csharp
using DiscordRPC;
using UnityEngine;

namespace Annabeth.Core
{
    public class DiscordPresence : MonoBehaviour
    {
        [SerializeField] private string applicationId = "YOUR_APP_ID_HERE";
        private DiscordRpcClient _client;

        private void Start()
        {
            _client = new DiscordRpcClient(applicationId);
            _client.Initialize();
            SetState("Idle", "Just hanging out");
        }

        public void SetState(string state, string details = "Hanging out with Annabeth")
        {
            _client?.SetPresence(new RichPresence
            {
                Details = details,
                State = state,
                Timestamps = Timestamps.Now,
                Assets = new Assets
                {
                    LargeImageKey = "annabeth",
                    LargeImageText = "Annabeth AI Companion",
                    SmallImageKey = state.ToLower().Replace(" ", "_"),
                    SmallImageText = state,
                }
            });
        }

        private void Update() => _client?.Invoke();  // Required: processes Discord callbacks
        private void OnDestroy() => _client?.Dispose();
    }
}
```

### Step 9.4 — Hook state changes

In `CompanionManager.cs`, wherever the companion state changes, call:
```csharp
discordPresence.SetState(newState.ToString());
```

States to map: `Idle`, `Listening`, `Thinking`, `Talking`, `Dancing`, `Sleeping`.

---

## PHASE 10 — Code Self-Improvement (Lite)
**Source:** `temp_refs/mai_selfmod.py`  
**Files:** New `server/process/self_improvement/` directory (3 files)  
**Risk:** Medium — writes Python source files; must never run during active session  
**Effort:** ~2 hours

### What this adds

A background task that periodically scans Annabeth's `server/` Python code for `bare except:` clauses and automatically fixes them to `except Exception:`. All other improvement types (type hints, complexity) are logged but never auto-applied.

### Step 10.1 — Create directory and files

```
server/process/self_improvement/
  __init__.py          (empty)
  analyzer.py          (CodeAnalyzer, ImprovementOpportunity — from temp_refs/mai_selfmod.py)
  generator.py         (ImprovementGenerator, GeneratedImprovement)
  scheduler.py         (ImprovementScheduler, SchedulerConfig)
```

Copy the classes directly from `temp_refs/mai_selfmod.py`. 

**Customize `SchedulerConfig` defaults for Annabeth:**
```python
SchedulerConfig(
    analysis_interval_hours=168.0,  # Weekly (not daily — don't be annoying)
    auto_apply_low_risk=True,       # Only bare except fixes
    require_approval_for_medium=True,
    max_daily_improvements=3,
    src_path=Path("server"),        # Only scan server/ directory
)
```

### Step 10.2 — Wire into `main_chat.py`

```python
from server.process.self_improvement.scheduler import ImprovementScheduler, SchedulerConfig
from pathlib import Path

_improvement_scheduler = ImprovementScheduler(
    config=SchedulerConfig(src_path=Path("server")),
    on_improvement_ready=lambda imp: logger.info(
        f"[SelfImprovement] Opportunity: {imp.opportunity.description} "
        f"in {imp.opportunity.file_path}:{imp.opportunity.line_number}"
    )
)
_improvement_scheduler.start()
```

### Step 10.3 — Safety gates

```python
# When conversation starts:
_improvement_scheduler.conversation_active = True

# When conversation ends / user goes idle:
_improvement_scheduler.conversation_active = False
```

Improvements are **never applied** when `conversation_active = True`, and file backups (`.bak`) are written before any change.

---

## Full File Checklist

### New Python files
- [ ] `server/process/llm_funcs/facial_expressions.py` — Phase 2
- [ ] `server/settings_registry.py` — Phase 8
- [ ] `server/process/self_improvement/__init__.py` — Phase 10
- [ ] `server/process/self_improvement/analyzer.py` — Phase 10
- [ ] `server/process/self_improvement/generator.py` — Phase 10
- [ ] `server/process/self_improvement/scheduler.py` — Phase 10

### Modified Python files
- [ ] `server/process/memory/reflection_loop.py` — Phase 1
- [ ] `server/annabeth_config.py` — Phase 2 (facial expression prompt)
- [ ] `server/process/llm_funcs/llm_scr.py` — Phases 2 + 3
- [ ] `server/process/llm_funcs/model_router.py` — Phase 3
- [ ] `client/audio_analyzer.py` — Phase 4
- [ ] `server/process/tts_func/rvc_convert.py` — Phase 5
- [ ] `server/main_chat.py` — Phases 6, 8, 10 (registry init, proactive drain, scheduler)

### New Unity C# files
- [ ] `unity/Scripts/UI/IdleBubbleController.cs` — Phase 6
- [ ] `unity/Scripts/Avatar/IdleController.cs` — Phase 7
- [ ] `unity/Scripts/Core/DiscordPresence.cs` — Phase 9

### Modified Unity C# files
- [ ] `unity/Scripts/Avatar/EmotionController.cs` — Phase 2 (add SetExpression)
- [ ] `unity/Scripts/UI/SettingsPanel.cs` — Phase 7 (idle timeout setting)
- [ ] `unity/Scripts/Core/CompanionManager.cs` — Phase 9 (Discord state hook)

### Unity packages (manifest.json)
- [ ] `com.lachee.discordrpc` — Phase 9
- [ ] `com.unity.nuget.newtonsoft-json` (if not already present) — Phase 9

### Requirements files
- [ ] `requirements.txt` — Phase 4: add `webrtcvad-wheels>=2.0.14`
- [ ] `requirements.txt` — Phase 3: add `psutil>=5.9` (if not present)

---

## Testing After Each Phase

Run existing test suite:
```powershell
cd d:\Annabeth
.\.venv\Scripts\python -m pytest test_deep_annabeth.py -v --tb=short
```

Phase-specific smoke tests:

| Phase | Test |
|---|---|
| 1 (Grillo beats) | Check `grillo_activity_log` table has new rows after one beat interval |
| 2 (Face tags) | Feed `"Hello [em_smile:0.9] world"` through LLM output path; verify clean text `"Hello  world"` sent to TTS and `face_expression` WebSocket fires |
| 3 (Latency switch) | Simulate `record_latency(6000)`; call `get_model_for_intent("story")`; assert returns `_fast` model |
| 4 (VAD) | Play silence then speech through system; verify transcription still triggers correctly |
| 5 (RVC rmvpe) | Generate TTS and run RVC; compare to previous harvest output |
| 6 (Idle bubbles) | Queue a thought via `_proactive_queue.put_nowait("test")`; verify WebSocket fires within 30s |
| 7 (Idle mode) | Leave mouse still 65s with timeout=60; verify `isSleeping` animator bool set |
| 8 (Registry) | Call `registry.set("GRILLO_BEAT_INTERVAL", 1800)`; verify `registry.get()` returns 1800 |
| 9 (Discord) | Launch with Discord open; verify "Playing Annabeth" appears in Discord profile |
| 10 (Self-improve) | Create a test .py with `except:` in a sandbox dir; run `analyzer.analyze_all()`; verify opportunity found |

---

## Constraints Summary

- **No breaking changes** — every phase is additive
- **Graceful degradation** — `webrtcvad`, `rvc-infer`, Discord package all fail silently with a warning log, never crash startup
- **Conversation-first** — Grillo beats, code improvements, and background tasks gate on `conversation_active` flag
- **Windows only** — Win32 API calls (`IdleController.cs`) are expected, Annabeth only runs on Windows
- **DO NOT touch** — `EyeTrackingController.cs` (already excellent), `PetDetectionController.cs` (already working), `rvc_convert.py` (Phase 5 is params only)
