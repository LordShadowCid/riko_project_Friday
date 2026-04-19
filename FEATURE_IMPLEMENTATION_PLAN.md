# Annabeth Feature Implementation Plan
**Reference document — updated as features are completed.**  
Last updated: 2026-04-13

---

## How to Use This Document
- Each Phase is self-contained and can be implemented independently.
- Each feature has:
  - **Source** — the exact GitHub file(s) to study/copy logic from
  - **Target** — which Annabeth file(s) to edit or create
  - **What to copy** — specific class/function names from the source repo
  - **What to change** — how to adapt it to Annabeth's architecture
  - **Test** — how to verify the feature works
- Check the box `[x]` when a phase or step is done.
- "Source reference" links point to the raw file in the source repo.

---

## Status Legend
- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete

---

## Phase Overview

| # | Feature | Source Repo | Priority | Status |
|---|---------|-------------|----------|--------|
| 1 | Persistent emotion state + decay | Synthetic_Heart | HIGH | `[ ]` |
| 2 | LLM emotion tags in prompt contract | Synthetic_Heart | HIGH | `[ ]` |
| 3 | GRILLO autonomous reflection loop | Synthetic_Heart | HIGH | `[ ]` |
| 4 | LLM model auto-switching by intent | Mai | MEDIUM | `[ ]` |
| 5 | Bio manager (structured user facts) | Mai + SyntH | MEDIUM | `[ ]` |
| 6 | Memory self-compression | Mai | MEDIUM | `[ ]` |
| 7 | RVC voice conversion (voice fidelity) | MekaHime | LOW | `[ ]` |
| 8 | Self-improvement system | Mai | LOW | `[ ]` |

---

## Phase 1 — Persistent Emotion State + Decay

### What this does
Annabeth currently sends a single `emotion` keyword per response (e.g. `"happy"`).
After this phase, all 11 canonical emotions have a persistent intensity (0–10) stored
in SQLite. Intensities decay exponentially while the app is running (like real
feelings fading). The animation system broadcasts the **highest-intensity** current
emotion instead of a one-shot guess.

### Source Reference
**Repo:** `XargonWan/Synthetic_Heart`  
**Branch:** `develop`  
**File:** [`plugins/emotion_manager.py`](https://raw.githubusercontent.com/XargonWan/Synthetic_Heart/develop/plugins/emotion_manager.py)

Key items to study in that file:
- `CANONICAL_EMOTIONS` set — the 11 emotion names we'll adopt
- `PLUTCHIK_OPPOSITES` dict — which emotion dampens which when triggered
- `EMOTION_BASELINES` dict — minimum resting values per emotion
- `EmotionState.get_decayed_intensity()` — the exponential decay math
- `EmotionManager.set_emotion()` — DB upsert + diary log pattern
- `EmotionManager.decay_emotions()` — background decay loop logic
- `EmotionManager._decay_loop()` — `asyncio.sleep(60)` heartbeat
- `strip_emotion_tags()` — regex that removes `{happy 8.5}` from text before TTS

### Target Files in Annabeth

| Action | File |
|--------|------|
| **CREATE** new module | `server/process/memory/emotion_state.py` |
| **EDIT** — add DB table init | `server/process/memory/feedback.py` |
| **EDIT** — parse tags, call emotion store | `server/process/llm_funcs/llm_scr.py` |
| **EDIT** — broadcast decayed emotion | `client/avatar_server.py` |
| **EDIT** — add canonical emotions | `shared/config.py` (expand `Emotion` enum) |
| **EDIT** — strip tags before TTS | `server/main_chat.py` (response pipeline) |

### Implementation Steps

#### Step 1.1 — Create `emotion_state.py`
- `[ ]` Copy the `CANONICAL_EMOTIONS`, `PLUTCHIK_OPPOSITES`, `EMOTION_BASELINES`, `DEFAULT_BASELINE` constants verbatim
- `[ ]` Copy the `EmotionState` dataclass (just `emotion_name`, `intensity`, `timestamp`, `get_decayed_intensity()`)
- `[ ]` Write `_ensure_table(conn)` — same `emotion_state` SQLite table as SyntH (but SQLite, not MySQL)
  - Table: `emotion_state(id, emotion_name TEXT, intensity REAL, updated_at REAL)`
  - Connect to the existing `C:\annabeth_data\self_eval\feedback.db` from `feedback.py`
- `[ ]` Write `set_emotion(emotion_name, intensity)` — SQLite upsert
- `[ ]` Write `get_all_emotions()` → `dict[str, float]` with decay applied
- `[ ]` Write `get_dominant_emotion()` → `str` single highest-intensity emotion name
- `[ ]` Write `decay_loop()` — `threading.Timer` fires every 60 s, calls `_apply_decay()`
- `[ ]` Strip tags function: `strip_emotion_tags(text: str) → str` — copy SyntH's `_EMOTION_TAG_RE` regex
- `[ ]` Parse tags function: `extract_emotion_tags(text: str) → dict[str, float]` — copy SyntH's `_extract_emotion_tags` logic

#### Step 1.2 — Wire into `feedback.py`
- `[ ]` In `_init_tables()`, call `emotion_state.py`'s `_ensure_table(conn)` so the table is created on first run

#### Step 1.3 — Wire into `llm_scr.py`
- `[ ]` After the LLM returns a response chunk/full response, call `extract_emotion_tags(response_text)`
- `[ ]` For each tag found, call `set_emotion(name, intensity)` with Plutchik opposite dampening
- `[ ]` Before returning the response text for TTS, call `strip_emotion_tags(response_text)` — so the spoken text never has `{happy 8.5}` in it

#### Step 1.4 — Wire into `avatar_server.py`
- `[ ]` Start `decay_loop()` when the server initializes (alongside the audio loop)
- `[ ]` Replace the one-shot `emotion` broadcast: Instead of `{"type": "emotion", "emotion": "happy"}` from `main_chat.py`, add a background task that broadcasts `get_dominant_emotion()` every 5 seconds
  - This means Unity always has the current emotional state even between turns

#### Step 1.5 — Update `shared/config.py`
- `[ ]` Replace the `Emotion` enum's 6 members with all 11 `CANONICAL_EMOTIONS` names
- `[ ]` Add `Emotion.from_str(s)` classmethod for safe parsing

#### Step 1.6 — Update system prompt in `character_config.yaml`
- `[ ]` Add instructions to the `system_prompt` block telling the LLM to embed emotion tags:
  ```
  EMOTION TAGS: Express your current feelings by including tags in responses.
  Format: {emotion_name intensity} e.g. {happy 8.5, curious 4.0}
  Available emotions: happy, sad, angry, fear, disgust, surprised, neutral, relaxed, love, arousal, devotion
  Intensity 0-10. Tags are stripped before being spoken aloud.
  ```

### Test
```powershell
# Start backend in self-check-only mode
$env:ANNABETH_SELF_CHECK_ONLY=1; python -m server.main_chat --self-check-only
# Then manually:
python -c "from server.process.memory.emotion_state import set_emotion, get_all_emotions, strip_emotion_tags; set_emotion('happy',8.0); print(get_all_emotions()); print(strip_emotion_tags('Hello! {happy 8.5, love 3.0}'))"
```
Expected: `{'happy': 8.0, 'neutral': 5.0, ...}` and `'Hello!'`

---

## Phase 2 — LLM Emotion Tags in Prompt Contract

> **Depends on:** Phase 1 complete (emotion tag parsing must exist first)

### What this does
Updates Annabeth's system prompt so the LLM knows it can and should express emotions
inline. Also updates the response pipeline to parse those tags in real-time during
streaming so the emotion state updates even before the full response is done.

### Source Reference
**Repo:** `XargonWan/Synthetic_Heart`  
**File:** [`plugins/emotion_manager.py`](https://raw.githubusercontent.com/XargonWan/Synthetic_Heart/develop/plugins/emotion_manager.py)  
**Key methods:**
- `get_prompt_instructions()` — the exact text injected into LLM context
- `get_static_injection()` — how current emotion state is injected as context

### Target Files in Annabeth

| Action | File |
|--------|------|
| **EDIT** system prompt | `character_config.yaml` — `presets.default.system_prompt` |
| **EDIT** streaming response handler | `server/process/llm_funcs/llm_scr.py` — `llm_response_streaming()` |
| **CREATE** context injector | `server/process/memory/emotion_state.py` — `get_injection_context()` |

### Implementation Steps

#### Step 2.1 — Streaming tag parsing
- `[ ]` In `llm_response_streaming()`, after accumulating each sentence chunk, call `extract_emotion_tags(chunk)` and update emotion state immediately (don't wait for full response)
- `[ ]` Call `strip_emotion_tags(chunk)` before passing to TTS — tags must NOT be spoken

#### Step 2.2 — Emotion context injection
- `[ ]` Add `get_injection_context() → str` to `emotion_state.py`:
  - Returns a 1–2 line string like: `"Current Emotional State: happy (7.2), neutral (5.0)\nExpress emotions with: {happy 8.5, love 3.0}"`
- `[ ]` In `llm_scr.py`, inject this into the system message before LLM call (append to or replace the end of the system prompt for that turn)

#### Step 2.3 — Plutchik dampening
- `[ ]` In `set_emotion()`, after updating the target emotion, look up `PLUTCHIK_OPPOSITES` and reduce the opposite emotion by 50% of the new intensity increase
  - E.g., setting `happy = 8.0` should reduce `sad` by `8.0 * 0.5 = 4.0`

### Test
- Chat with Annabeth, ask her to be excited about something
- Check `C:\annabeth_data\self_eval\feedback.db` `emotion_state` table
- Verify `happy` has high intensity and `sad` has reduced intensity
- Verify TTS speaks clean text with no `{...}` tags

---

## Phase 3 — GRILLO Autonomous Reflection Loop

### What this does
A background thread fires every 30–60 minutes when Annabeth is idle. It prompts the
LLM to reflect on recent conversations, write a diary entry about the day, and
optionally send Annabeth a message she can deliver unprompted (like "Hey, I was just
thinking about what you said earlier..."). Results are stored in ChromaDB
`self_notes` and SQLite.

### Source Reference
**Repo:** `XargonWan/Synthetic_Heart`  
**Branch:** `develop`  
**File:** [`plugins/grillo/` directory](https://github.com/XargonWan/Synthetic_Heart/tree/develop/plugins/grillo)  
**Key concepts from grillo_plugin.py wrapper + grillo_impl.py:**
- The "beat" timer: fires on a configurable interval (default 30 min)
- Reflection prompt categories: `memory_consolidation`, `self_reflection`, `curiosity`, `relationship_check`
- Only fires when no active conversation is happening (checks speaking/listening flags)
- Results saved to a `diary` table in the DB

**Also reference:**  
**File:** [`plugins/ai_diary.py`](https://raw.githubusercontent.com/XargonWan/Synthetic_Heart/develop/plugins/ai_diary.py)

### Target Files in Annabeth

| Action | File |
|--------|------|
| **CREATE** new module | `server/process/memory/reflection_loop.py` |
| **EDIT** start the loop | `server/main_chat.py` — after `_start_avatar_server()` |
| **EDIT** add diary table | `server/process/memory/feedback.py` — `_init_tables()` |
| **EDIT** inject diary into context | `server/process/memory/conversation_summarizer.py` |

### Implementation Steps

#### Step 3.1 — Create `reflection_loop.py`
- `[ ]` Class `ReflectionLoop` with:
  - `__init__(interval_minutes=45)` — configurable interval
  - `start()` — start background `threading.Timer` chain
  - `stop()` — cancel current timer
  - `_is_idle() → bool` — returns True when not speaking AND not transcribing (check shared state flags from `shared/state.py`)
  - `_fire()` — the actual reflection callback:
    1. Skip if not idle (reschedule for 5 min later)
    2. Build a `reflection_prompt` using recent ChromaDB `conversations` recall
    3. Call `_llm_extract()` (same pattern as `conversation_summarizer.py`)
    4. Parse the result into `{diary_entry, next_thought, emotion_update}`
    5. Store diary entry in SQLite `diary` table
    6. Store `next_thought` in a thread-safe queue that `main_chat.py` can drain
    7. Reschedule for next interval

#### Step 3.2 — Add SQLite diary table
- `[ ]` In `feedback.py` `_init_tables()`, add:
  ```sql
  CREATE TABLE IF NOT EXISTS diary (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp REAL NOT NULL,
      entry_text TEXT NOT NULL,
      themes TEXT DEFAULT '',
      dominant_emotion TEXT DEFAULT 'neutral'
  );
  ```

#### Step 3.3 — Plug the proactive message queue into `main_chat.py`
- `[ ]` Add a `_proactive_queue: queue.Queue` singleton in `main_chat.py`
- `[ ]` In the main VAD loop, before waiting for mic input, drain the queue:
  - If a proactive thought is waiting AND it's been > 15 min since last interaction, speak it as Annabeth (no user input needed)
- `[ ]` Start `ReflectionLoop` after avatar server starts
- `[ ]` Include `ANNABETH_REFLECTION_INTERVAL` env var override for the interval

#### Step 3.4 — Inject recent diary into LLM context
- `[ ]` In `conversation_summarizer.py` (or a new `context_builder.py`), add `get_diary_context(n=3) → str`:
  - Pulls last N diary entries from SQLite
  - Returns a short block like: "Recent reflections: [entry 1] / [entry 2]"
- `[ ]` Inject into the system message in `llm_scr.py` alongside emotion context

### Reflection Prompt Template (add to `character_config.yaml`)
```yaml
reflection_prompt: |
  Based on recent conversations, reflect on:
  1. What interesting things did the user share today?
  2. How are you feeling about your recent interactions?
  3. Is there anything you want to remember or follow up on?
  
  Respond as Annabeth in first person. Keep it concise (2-4 sentences).
  Then provide a short thought to share with the user next time (1 sentence).
  
  Format your response as JSON:
  {"diary_entry": "...", "next_thought": "...", "dominant_emotion": "..."}
```

### Test
```powershell
# Run reflection manually
python -c "
from server.process.memory.reflection_loop import ReflectionLoop
r = ReflectionLoop(interval_minutes=0)  # 0 = fire immediately
r._fire()
import sqlite3; conn = sqlite3.connect(r'C:\annabeth_data\self_eval\feedback.db')
print(conn.execute('SELECT * FROM diary ORDER BY id DESC LIMIT 1').fetchone())
"
```

---

## Phase 4 — LLM Model Auto-Switching by Intent

### What this does
Annabeth already classifies intent in `intent_classifier.py` (greeting, question_short,
story, command, general). After this phase, short/simple intents use a lightweight fast
model (e.g. `gemma3:4b`) while complex requests keep the current capable model
(`mannix/llama3.1-8b-abliterated`). This makes greetings and short answers much faster.

### Source Reference
**Repo:** `MystiaTech/Mai`  
**File:** [`src/models/`](https://github.com/MystiaTech/Mai/tree/main/src/models)  
**Key concepts:**
- `ModelManager` — catalog of available models fetched from Ollama `/api/tags`
- Task-based model selection: `select_model_for_task(task_type)` returns best available
- Fallback: always falls back to primary model if preferred isn't available

### Target Files in Annabeth

| Action | File |
|--------|------|
| **CREATE** | `server/process/llm_funcs/model_router.py` |
| **EDIT** | `server/annabeth_config.py` — add model routing config section |
| **EDIT** | `server/process/llm_funcs/llm_scr.py` — use routed model per call |
| **EDIT** | `character_config.yaml` — add `model_routing` block |

### Implementation Steps

#### Step 4.1 — Create `model_router.py`
- `[ ]` `fetch_available_models(host: str) → list[str]` — `GET /api/tags`, returns model name list
- `[ ]` `ModelRouter` class:
  - `__init__(config)` — loads routing config, calls `fetch_available_models()` at startup
  - `get_model_for_intent(intent_category: str) → str` — returns model name
  - Routing rules (from `character_config.yaml`):
    - `greeting` → `fast_model` (e.g. `gemma3:4b`) if available, else fallback
    - `question_short` → `fast_model`
    - `story`, `general` → `primary_model`
    - `command` → `primary_model`
- `[ ]` Singleton: `get_model_router() → ModelRouter`

#### Step 4.2 — Add config block to `character_config.yaml`
```yaml
model_routing:
  enabled: true
  primary_model: mannix/llama3.1-8b-abliterated   # complex/story/command
  fast_model: gemma3:4b                            # greeting/short questions
  # If fast_model is not available in Ollama, falls back to primary_model
```

#### Step 4.3 — Wire into `llm_scr.py`
- `[ ]` Import `get_model_router` and `classify_intent` (already imported in `main_chat.py`)
- `[ ]` In `llm_response()` and `llm_response_streaming()`, determine the model:
  ```python
  intent = classify_intent(user_message)
  model = get_model_router().get_model_for_intent(intent.category)
  ```
- `[ ]` Pass `model` to the Ollama payload instead of the hardcoded config value

### Test
```powershell
python -c "
from server.process.agents.intent_classifier import classify_intent
from server.process.llm_funcs.model_router import get_model_router
router = get_model_router()
print(router.get_model_for_intent(classify_intent('hey').category))   # should be gemma3:4b
print(router.get_model_for_intent(classify_intent('tell me a story').category))  # primary model
"
```

---

## Phase 5 — Bio Manager (Structured User Facts)

### What this does
Annabeth already stores facts in ChromaDB's `facts` collection via
`conversation_summarizer.py`, but they are unstructured text blobs. This phase adds
a structured bio for each known speaker: known name, timezone, relationship to user,
remembered preferences. This bio is injected into every LLM context turn so Annabeth
always "knows" the person she's talking to.

### Source Reference
**Repo:** `XargonWan/Synthetic_Heart`  
**File:** [`plugins/bio_manager.py`](https://raw.githubusercontent.com/XargonWan/Synthetic_Heart/develop/plugins/bio_manager.py)  
**Key concepts:**
- `BioManager.get_bio(speaker_id)` → formatted string injected into prompt
- `BioManager.update_bio(speaker_id, field, value)` — LLM extracts fielded updates
- SQLite-backed, one row per known speaker
- Schema: `speaker_id`, `real_name`, `timezone`, `relationship`, `known_facts` (JSON), `last_seen`

**Also reference:**  
**Repo:** `MystiaTech/Mai`  
**File:** [`src/memory/`](https://github.com/MystiaTech/Mai/tree/main/src/memory) — for pattern of memory/personality layer interaction

### Target Files in Annabeth

| Action | File |
|--------|------|
| **CREATE** | `server/process/memory/bio_manager.py` |
| **EDIT** | `server/process/memory/feedback.py` — add `speaker_bio` table |
| **EDIT** | `server/process/memory/conversation_summarizer.py` — write bio updates |
| **EDIT** | `server/process/llm_funcs/llm_scr.py` — inject bio context |

### Implementation Steps

#### Step 5.1 — Create `bio_manager.py`
- `[ ]` Add `speaker_bio` table in `_init_tables()`:
  ```sql
  CREATE TABLE IF NOT EXISTS speaker_bio (
      speaker_id TEXT PRIMARY KEY,
      real_name TEXT DEFAULT '',
      relationship TEXT DEFAULT '',
      timezone TEXT DEFAULT '',
      known_facts TEXT DEFAULT '[]',  -- JSON array of strings
      last_seen REAL DEFAULT 0
  );
  ```
- `[ ]` `get_bio(speaker_id: str) → str` — builds a 2-3 line context string:
  ```
  Speaker "[Dad]": Real name: unknown. Relationship: primary user.
  Known facts: likes coffee, works from home, has a daughter named Riley.
  ```
- `[ ]` `update_bio(speaker_id, field, value)` — UPDATE or INSERT the named field
- `[ ]` `update_last_seen(speaker_id)` — stamps current time
- `[ ]` `add_fact(speaker_id, fact_text)` — appends to `known_facts` JSON array (deduplicate)

#### Step 5.2 — Wire into `conversation_summarizer.py`
- `[ ]` After extracting user facts via LLM, also call:
  ```python
  bio_manager.update_last_seen(speaker)
  # If any fact mentions "real name", "lives in", "timezone", update those fields
  ```
- The existing fact-extraction prompt already pulls these — just route the output to both ChromaDB and the bio table

#### Step 5.3 — Wire into `llm_scr.py`
- `[ ]` Before building the messages payload, call `bio_manager.get_bio(speaker)` and prepend the result to the system message for that call

### Test
```powershell
python -c "
from server.process.memory.bio_manager import get_bio, add_fact, update_bio
update_bio('Dad', 'relationship', 'primary user, father')
add_fact('Dad', 'likes coffee in the morning')
print(get_bio('Dad'))
"
```
Expected:
```
Speaker [Dad]: Relationship: primary user, father.
Known facts: likes coffee in the morning.
```

---

## Phase 6 — Memory Self-Compression

### What this does
ChromaDB's `conversations` collection grows unboundedly. When it exceeds 500 entries,
an LLM-powered compression pass summarizes the oldest 100 entries into a single
meta-summary entry and deletes the originals. This keeps recall fast.

### Source Reference
**Repo:** `MystiaTech/Mai`  
**File:** [`src/memory/`](https://github.com/MystiaTech/Mai/tree/main/src/memory)  
**Key concept from README:**
- `memory.auto_compress_at: 100000` — token budget triggers compression
- LLM called: "Summarize these conversations into 3-5 sentences preserving key facts"

### Target Files in Annabeth

| Action | File |
|--------|------|
| **EDIT** | `server/process/memory/memory_store.py` — add `compress_if_needed()` |
| **EDIT** | `server/process/memory/conversation_summarizer.py` — call compression check |

### Implementation Steps

#### Step 6.1 — Add `compress_if_needed()` to `MemoryStore`
- `[ ]` `get_conversation_count() → int` — `self.conversations.count()`
- `[ ]` `compress_conversations(keep_n_newest=400, batch_size=100)`:
  1. If `count <= keep_n_newest + batch_size`, return early (nothing to do)
  2. Fetch the oldest `batch_size` entries from `conversations` (by `timestamp` metadata)
  3. Concatenate their text content
  4. Call LLM with: `"Summarize these {n} conversation excerpts into 3-5 sentences preserving key facts about the user:"`
  5. Delete the old `batch_size` entries by ID
  6. Add the compressed summary as a single new entry with `metadata.type = "compressed_summary"`
  7. Log: `[Memory] Compressed {n} entries into 1 summary`
- `[ ]` `compress_if_needed(threshold=500)` — calls `compress_conversations()` if count > threshold

#### Step 6.2 — Hook into `conversation_summarizer.py`
- `[ ]` After every `extract_and_store()` call, call:
  ```python
  store = get_memory_store()
  store.compress_if_needed()
  ```
  This runs in the existing background thread so it doesn't block chat.

### Test
```powershell
python -c "
from server.process.memory.memory_store import get_memory_store
store = get_memory_store()
print('Count:', store.conversations.count())
store.compress_if_needed(threshold=0)  # force compress
print('Count after:', store.conversations.count())
"
```

---

## Phase 7 — RVC Voice Conversion (Optional Voice Fidelity Upgrade)

### What this does
Stack Retrieval-based Voice Conversion on top of GPT-SoVITS output. Running TTS
→ RVC takes an existing voice and reshapes it to match a trained character `.pth`
model. GPT-SoVITS already sounds character-like; RVC makes the voice **exactly match**
a specific character at the cost of ~0.5–1 second extra latency.

**Only implement this if GPT-SoVITS alone isn't giving enough character voice fidelity.**

### Source Reference
**Repo:** `zeekk0/MekaHime-Pipeline-V1`  
**File:** [`MKHM_Pipeline_V1.py`](https://raw.githubusercontent.com/zeekk0/MekaHime-Pipeline-V1/main/MKHM_Pipeline_V1.py)  
**Key methods to study:**
- `initialize_rvc()` — how `VC(config)` is initialized from the RVC repo
- `convert_with_rvc(input_path) → output_path` — the full conversion call
  - `self.vc.vc_single(sid=0, input_audio_path=..., f0_up_key=pitch, f0_method="rmvpe", ...)`
- `select_rvc_model()` — how `.pth` and `.index` files are located
- `select_pitch_tuning()` — pitch shift parameter (for feminine voices: +6 to +12)
- The `torch.load` patch for PyTorch 2.6+ compatibility

**Required external repo:** [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)  
Models go in: `d:\Annabeth\models\rvc\`

### Target Files in Annabeth

| Action | File |
|--------|------|
| **CREATE** | `server/process/tts_func/rvc_convert.py` |
| **EDIT** | `server/process/tts_func/sovits_ping.py` — optional post-process step |
| **EDIT** | `character_config.yaml` — add `rvc:` block |

### Implementation Steps

#### Step 7.1 — Create `rvc_convert.py`
- `[ ]` Apply the `torch.load` compatibility patch (copy verbatim from MekaHime)
- `[ ]` `RvcConverter.__init__(model_path, index_path, pitch_shift=6)` — init `VC(config)`, call `vc.get_vc(model_name)`
- `[ ]` `RvcConverter.convert(input_wav_path) → output_wav_path` — wrap MekaHime's `convert_with_rvc()` logic
- `[ ]` `get_rvc_converter() → Optional[RvcConverter]` — singleton, returns `None` if RVC disabled or model not found

#### Step 7.2 — Plug into `sovits_ping.py`
- `[ ]` After GPT-SoVITS produces a `.wav` at its output path, check `get_rvc_converter()`
- `[ ]` If not `None`, pass the wav through `converter.convert()` and use the result path instead
- `[ ]` If `None`, pass through unchanged (graceful degradation — works without RVC)

#### Step 7.3 — Config block in `character_config.yaml`
```yaml
rvc:
  enabled: false            # Set true to activate
  model_dir: models/rvc     # Relative to repo root
  model_name: annabeth      # Loads models/rvc/annabeth.pth + models/rvc/annabeth.index
  pitch_shift: 8            # Semitones up (+) or down (-)
  f0_method: rmvpe          # rmvpe (best quality) or harvest (fallback)
```

### Test
```powershell
# Only if RVC models are installed
python -c "
from server.process.tts_func.rvc_convert import get_rvc_converter
c = get_rvc_converter()
if c:
    out = c.convert('test.wav')
    print('Output:', out)
else:
    print('RVC disabled or model not found')
"
```

---

## Phase 8 — Self-Improvement System

> **NOTE:** Implement LAST. High complexity, high risk. Requires all other phases stable first.  
> **DO NOT auto-apply changes without user approval.** Propose-only until trust is established.

### What this does
On a weekly schedule (or on-demand), scan Annabeth's own source code using AST
analysis, generate improvement suggestions via LLM, and present them for review.
User approves/rejects each change. Approved changes are applied and committed.

### Source Reference
**Repo:** `MystiaTech/Mai`  
**Files:**  
- [`src/selfmod/analyzer.py`](https://raw.githubusercontent.com/MystiaTech/Mai/main/src/selfmod/analyzer.py) — AST-based code analysis (copy verbatim as starting point)
  - `CodeAnalyzer.analyze_all()` — finds bare excepts, missing type hints, complex functions, unused imports
  - `ImprovementOpportunity` dataclass — the unit of work
- [`src/safety/`](https://github.com/MystiaTech/Mai/tree/main/src/safety) — second-agent safety review pattern
- Mai README: Risk tiers — LOW/MEDIUM/HIGH/BLOCKED

### Target Files in Annabeth

| Action | File |
|--------|------|
| **CREATE** | `server/process/self_improve/code_analyzer.py` |
| **CREATE** | `server/process/self_improve/proposal_generator.py` |
| **CREATE** | `server/process/self_improve/approval_ui.py` |
| **EDIT** | `run_runtime_checks.ps1` — add weekly analysis trigger |

### Implementation Steps

#### Step 8.1 — Copy `CodeAnalyzer` from Mai
- `[ ]` Copy `src/selfmod/analyzer.py` verbatim to `server/process/self_improve/code_analyzer.py`
- `[ ]` Change `src_path` default to point at `d:\Annabeth\server`

#### Step 8.2 — Create `proposal_generator.py`
- `[ ]` Takes a list of `ImprovementOpportunity` objects
- `[ ]` For each, calls the LLM with the surrounding code context and asks for a concrete patch
- `[ ]` Returns a list of `{file, line, old_code, new_code, risk_level, description}` dicts
- `[ ]` Risk classification: any change to `main_chat.py`, `avatar_server.py`, or `feedback.py` = HIGH (requires approval); all others = LOW (can auto-apply after review)

#### Step 8.3 — Create `approval_ui.py`
- `[ ]` Prints each proposal to the terminal with diff highlighting
- `[ ]` Prompts `[A]pply / [S]kip / [Q]uit`
- `[ ]` Applied changes are written to file and committed via `git commit -m "self-improve: <description>"`
- `[ ]` All proposals (approved and rejected) are logged to SQLite `improvement_log` table

#### Step 8.4 — Add to `character_config.yaml`
```yaml
self_improvement:
  enabled: false          # Manual opt-in
  scan_paths:
    - server/process
    - shared
  auto_apply_risk: none   # none | low | medium  (start with 'none' = propose only)
```

### Test
```powershell
python -c "
from server.process.self_improve.code_analyzer import CodeAnalyzer
from pathlib import Path
results = CodeAnalyzer(Path('server/process')).analyze_all()
print(f'Found {len(results)} improvement opportunities')
for r in results[:3]:
    print(f'  {r.file_path.name}:{r.line_number} [{r.improvement_type.value}] {r.description}')
"
```

---

## File Map — Where Everything Lives

### Annabeth Current Key Files
| File | Purpose |
|------|---------|
| `server/main_chat.py` | Main loop: Whisper → LLM → TTS → Avatar |
| `server/process/llm_funcs/llm_scr.py` | LLM calls, streaming, cache, intent |
| `server/process/agents/intent_classifier.py` | 1ms intent classification |
| `server/process/memory/memory_store.py` | ChromaDB conversations/facts/self_notes |
| `server/process/memory/feedback.py` | SQLite feedback.db, tables |
| `server/process/memory/conversation_summarizer.py` | Post-turn LLM extraction |
| `server/process/memory/self_eval.py` | Post-turn response scoring |
| `server/process/tts_func/sovits_ping.py` | GPT-SoVITS TTS + audio playback |
| `server/process/read_aloud/text_capture.py` | System text capture + word timing |
| `client/avatar_server.py` | aiohttp WebSocket to Unity |
| `shared/config.py` | Enums: CompanionMode, Emotion, MessageType |
| `shared/state.py` | Thread-safe state: mode, silenced, speaking, emotion |
| `character_config.yaml` | All tunable config: models, VAD, TTS, system prompt |

### New Files This Plan Creates
| File | Created By Phase |
|------|-----------------|
| `server/process/memory/emotion_state.py` | Phase 1 |
| `server/process/llm_funcs/model_router.py` | Phase 4 |
| `server/process/memory/bio_manager.py` | Phase 5 |
| `server/process/memory/reflection_loop.py` | Phase 3 |
| `server/process/tts_func/rvc_convert.py` | Phase 7 |
| `server/process/self_improve/code_analyzer.py` | Phase 8 |
| `server/process/self_improve/proposal_generator.py` | Phase 8 |
| `server/process/self_improve/approval_ui.py` | Phase 8 |

### Source Repo Files to Reference
| What you're implementing | Source file URL |
|--------------------------|----------------|
| Emotion decay math | `https://raw.githubusercontent.com/XargonWan/Synthetic_Heart/develop/plugins/emotion_manager.py` |
| Grillo reflection loop | `https://github.com/XargonWan/Synthetic_Heart/tree/develop/plugins/grillo` |
| Diary/ai_diary pattern | `https://raw.githubusercontent.com/XargonWan/Synthetic_Heart/develop/plugins/ai_diary.py` |
| Bio manager pattern | `https://raw.githubusercontent.com/XargonWan/Synthetic_Heart/develop/plugins/bio_manager.py` |
| RVC convert + torch patch | `https://raw.githubusercontent.com/zeekk0/MekaHime-Pipeline-V1/main/MKHM_Pipeline_V1.py` |
| AST code analyzer | `https://raw.githubusercontent.com/MystiaTech/Mai/main/src/selfmod/analyzer.py` |
| Mai model manager pattern | `https://github.com/MystiaTech/Mai/tree/main/src/models` |
| Mai memory pattern | `https://github.com/MystiaTech/Mai/tree/main/src/memory` |
| Mai self-improvement safety | `https://github.com/MystiaTech/Mai/tree/main/src/safety` |

---

## Dependency Order

```
Phase 1 (Emotion State)
    └── Phase 2 (Emotion Tags in Prompt)   ← depends on Phase 1
            └── Phase 3 (Reflection Loop)  ← benefits from Phase 1+2 emotions
Phase 4 (Model Router)                     ← independent
Phase 5 (Bio Manager)                      ← independent
    └── Phase 6 (Memory Compression)       ← benefits from Phase 5 existing
Phase 7 (RVC)                              ← independent (optional)
Phase 8 (Self-Improvement)                 ← all others should be stable first
```

---

## Pre-Implementation Checks (Run Before Starting Any Phase)

```powershell
# 1. Verify ports are clean
Get-NetTCPConnection -LocalPort 8765,8766,9880 -State Listen -ErrorAction SilentlyContinue

# 2. Verify venv and imports work
cd D:\Annabeth
.\.venv\Scripts\Activate.ps1
python -c "from server.process.memory.memory_store import get_memory_store; print('ChromaDB OK')"
python -c "from server.process.memory.feedback import log_feedback; print('SQLite OK')"
python -c "from server.utils import get_ollama_settings; print(get_ollama_settings())"

# 3. Run the full test harness to confirm baseline is green before any changes
powershell -File .\run_runtime_checks.ps1 -StopOnFailure
```

Expected baseline: **5 passed, 0 failed**

---

## Change Log

| Date | Phase | What was done |
|------|-------|--------------|
| 2026-04-13 | — | Plan created |
