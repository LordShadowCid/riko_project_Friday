# Annabeth Development — Temp Reference Files

Auto-pulled source code from 4 reference repos for implementation planning.
These files are READ-ONLY references. Do NOT modify them — they are the originals.

---

## File Index

### Synthetic_Heart (XargonWan/Synthetic_Heart)
- `synth_heart_grillo_impl.py` — Full GrilloPlugin: asyncio beat scheduler, 6 beat types, activity log DB tables
- `synth_heart_variables_engine.py` — ExposedVarDefinition, ExposedVariableRegistry, all config var registrations
- `synth_heart_facial_expression_plugin.py` — FacialExpressionPlugin: [em_name:intensity] tag parsing, timeline driving

### Mai (MystiaTech/Mai)
- `mai_selfmod_analyzer.py` — CodeAnalyzer: AST-based improvement detection (bare excepts, type hints, complexity)
- `mai_selfmod_scheduler.py` — ImprovementScheduler: schedules analysis runs, applies LOW-risk improvements, git commits
- `mai_selfmod_generator.py` — ImprovementGenerator: generates code improvements from opportunities, validates with AST

### MekaHime (zeekk0/MekaHime-Pipeline-V1)
- `mekahime_pipeline_rvc_vad.py` — Full MKHM pipeline: WebRTC VAD, Whisper STT, Ollama LLM, XTTS TTS, RVC voice conversion

### Mate-Engine (shinyflvre/Mate-Engine)
- `mate_engine_avatar_random_messages.cs` — AvatarRandomMessages: event-driven contextual messages, localization, streaming text
- `mate_engine_screensaver.cs` — AvatarBigScreenScreenSaver: idle timer, mouse/keyboard detection, animator triggers
- `mate_engine_pet_voice_reaction.cs` — PetVoiceReactionHandler: touch regions on bones, SFX playback, pat detection
- `mate_engine_mouse_tracking.cs` — AvatarMouseTracking: head/spine/eye tracking toward mouse cursor, VRM10 support

---

## Key Architecture Patterns

### Grillo → Annabeth's Reflection Loop upgrade
The reflection_loop.py in Annabeth is the hook point. GrilloPlugin wraps an asyncio beat loop
that fires every GRILLO_BEAT_INTERVAL seconds. Each beat picks a type (tag_elaboration 25%,
self_reflection 20%, memory_consolidation 15%, diary_consolidation 15%, curiosity 15%, 
relationship 10%) and generates a journal-style prompt that writes to the diary.

### Facial Expressions → Unity VRM blendshapes
LLM outputs [em_smile:0.8] inline tags. Python server parses them, strips from display text, 
then schedules a timeline of WebSocket pushes timed to TTS audio duration. Unity side receives
expression events and applies VRM blendshape weights.

### Touch Regions → Annabeth Unity
PetVoiceReactionHandler uses bone world positions + hover radius detection. When mouse enters
hover radius of a bone region, it triggers animator parameters and plays audio clips.

### Screensaver → Annabeth Unity
AvatarBigScreenScreenSaver uses Win32 GetCursorPos + GetAsyncKeyState for global mouse/keyboard.
Idle timer triggers isBigScreen + isBigScreenSaver animator booleans after configurable timeout.

### Random Messages → Annabeth Unity
AvatarRandomMessages fires on: (1) random interval timer, (2) animator state transitions (onActive).
Messages streamed character by character with configurable speed.

### RVC Voice → Annabeth
MekaHime uses: Whisper STT + WebRTC VAD → Ollama LLM → XTTS TTS → RVC voice conversion.
The RVC portion (vc_single) takes a TTS-generated WAV and pitch-shifts + voice-converts it.
Annabeth hook point: after GPT-SoVITS TTS generates audio, apply RVC as a post-processing step.

### Code Self-Improvement → Annabeth
Mai's system has 3 parts: CodeAnalyzer (AST analysis), ImprovementGenerator (code rewrites),
ImprovementScheduler (schedules runs, applies LOW-risk automatically). For Annabeth, we adapt
this to audit Annabeth's own server-side Python files periodically.

---
_Generated during implementation planning session._
