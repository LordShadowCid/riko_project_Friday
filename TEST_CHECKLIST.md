# Annabeth — Live Test Checklist

**How to use this:**
1. Start the server and Unity client
2. Work through each section
3. Mark each item ✅ Pass / ❌ Fail / ⚠️ Partial
4. After testing, share `d:\Annabeth\test_session.log` for review

> **Debug log auto-writes to:** `d:\Annabeth\test_session.log`  
> It captures all print output + structured events. Share it after the session.

---

## 1. Server Startup

| # | Test | What to observe | Expected |
|---|------|-----------------|----------|
| 1.1 | Launch server (`start_annabeth.ps1` or `python -m server.main_chat`) | Console output | `[DebugLogger] Logging to …test_session.log` appears |
| 1.2 | Unity client connects | WebSocket handshake log | `[Main] Unity client connected` |
| 1.3 | Settings registry init | No errors in console | No `[Registry]` warnings at startup |
| 1.4 | Model router probe | Console shows available models | `[ModelRouter] Available models: [...]` within ~5s |

---

## 2. Basic Conversation (LLM)

| # | Test | What to observe | Expected |
|---|------|-----------------|----------|
| 2.1 | Short greeting: "Hey!" | Response time, content | Reply in < 3s, natural greeting |
| 2.2 | Longer question (10+ words) | Streaming chunks arrive to Unity | Text appears progressively in speech bubble |
| 2.3 | Two rapid messages back-to-back | No duplicate/echo responses | Second message gets its own unique reply |
| 2.4 | Repeat the exact same message 3 times | Repetition guard triggers | Each response is different (fallback fires on 3rd match) |
| 2.5 | Very long response prompt ("tell me a story") | TTS sentence chunking | Audio plays sentence-by-sentence, no 30s gaps |

---

## 3. Facial Expressions

| # | Test | What to observe | Expected |
|---|------|-----------------|----------|
| 3.1 | Say something surprising | VRM blendshape changes on avatar | Eyes widen / surprised expression visible |
| 3.2 | Check log for `[FACE]` events | `test_session.log` | Lines like `face_expression {"name":"surprised","intensity":0.9}` |
| 3.3 | Expression tags in LLM output | Ask Annabeth to react dramatically | `[em_smile:0.8]` tags appear in raw LLM text, stripped from TTS audio |
| 3.4 | Bare `[em]` reset tag | Monitor log after expression fires | After audio ends, a `{"name": null, "intensity": 0}` reset message is sent |
| 3.5 | "thinking" expression | Prompt: "hmm, let me think…" | `[em_thinking:...]` tag triggers Neutral blendshape on Unity side |
| 3.6 | "happy" expression | Prompt: "I'm so happy!" | `[em_happy:...]` tag triggers Happy blendshape |

---

## 4. Voice Activity Detection (VAD)

| # | Test | What to observe | Expected |
|---|------|-----------------|----------|
| 4.1 | Speak normally | Response triggers after speech ends | Annabeth responds within ~1s of stopping |
| 4.2 | Background music playing | System audio doesn't trigger VAD | Silence from Annabeth unless you speak |
| 4.3 | Push-to-talk (if configured) | Hold key, speak, release | Only processes audio during hold |
| 4.4 | VAD aggressiveness 0 vs 3 | Change in `character_config.yaml` | Lower = more sensitive to soft speech; 3 = strict |

---

## 5. Emotion System

| # | Test | What to observe | Expected |
|---|------|-----------------|----------|
| 5.1 | Say something sad | Emotion state updates | Log shows `emotion_state` update; avatar expression shifts |
| 5.2 | Say something happy/excited | Positive emotion registered | `get_dominant_emotion()` returns "joy" or similar |
| 5.3 | Long pause after emotion | Decay over time | After ~1h (configured by `EMOTION_DECAY_TAU`) emotion fades back toward neutral |
| 5.4 | Emotion in reflection beat | Check diary after beat fires | Beat diary entry contains `mood=` field matching last dominant emotion |

---

## 6. Idle / Screensaver Mode

| # | Test | What to observe | Expected |
|---|------|-----------------|----------|
| 6.1 | Leave PC idle for `IDLE_TIMEOUT_S` (default 5 min) | Avatar enters idle pose | Animator `isIdle = true`, eye tracking reduces |
| 6.2 | Leave idle for double the timeout (10 min) | Avatar enters sleep state | Animator `isSleeping = true`; `disableOnSleep` components (if set) disable |
| 6.3 | Move the mouse / press a key | Avatar wakes immediately | Returns to normal pose within 1 Update cycle |
| 6.4 | Check log for `[IdleController]` | Unity Player log | `Idle: True`, `Sleeping: True`, `Sleeping: False` in sequence |
| 6.5 | Assign post-processing effect to `disableOnSleep` in Inspector | Sleep transition | Effect component `.enabled` goes False on sleep, True on wake |

---

## 7. Idle Speech Bubbles

| # | Test | What to observe | Expected |
|---|------|-----------------|----------|
| 7.1 | Wait in idle mode for `IDLE_BUBBLE_MIN_DELAY` (default 60s) | Speech bubble appears above avatar | Bubble shows a proactive thought from the reflection queue |
| 7.2 | Click bubble or respond | Bubble dismisses | Animation plays out; next bubble respects min delay |
| 7.3 | Active conversation | No idle bubble during active chat | `_conversation_active = True` blocks bubble dispatch |

---

## 8. Grillo Reflection Loop

| # | Test | What to observe | Expected |
|---|------|-----------------|----------|
| 8.1 | Server running for first beat interval (default 45 min; set `GRILLO_BEAT_INTERVAL=120` for testing) | Console shows beat fire | `[Reflection] Firing beat: <type>` then `[Reflection] [<type>] <text>` |
| 8.2 | Trigger two beats rapidly (set interval to 10s and verify `_beat_in_flight` guard) | Only one beat runs at a time | Second timer tick logs `Beat already in flight — skipping` |
| 8.3 | Check SQLite DB after beat | Open `server/reflection.db` | Both `grillo_activity_log` AND `grillo_action_execs` tables exist |
| 8.4 | Diary entry written | `docs/diary.md` or diary storage | Beat creates a timestamped reflection entry |
| 8.5 | Proactive thought queued | Long idle (>2h or lower `PROACTIVE_IDLE_SECONDS`) | `[Reflection] Proactive thought queued: ...` in log |

---

## 9. Model Router / Latency Switching

| # | Test | What to observe | Expected |
|---|------|-----------------|----------|
| 9.1 | Short greeting triggers fast model | Log shows model selection | `[ModelRouter]` (or nothing = fast by intent category) |
| 9.2 | Slow LLM response > 5000ms | Model forced to fast for next turn | `[ModelRouter] High latency Xms > 5000ms — fast model forced for 30s` |
| 9.3 | High RAM usage (simulate or observe) | Memory threshold check | `[ModelRouter] High memory — using fast model` |
| 9.4 | 30s cooldown expires | Returns to primary model | No forced-fast flag after cooldown |

---

## 10. Self-Improvement Scheduler

| # | Test | What to observe | Expected |
|---|------|-----------------|----------|
| 10.1 | Server runs for a week interval period (or call `run_manual_analysis()`) | Analysis log | `[SelfImprovement] Scheduler started.` at startup |
| 10.2 | Bare `except:` clause exists in any server file | Analyzer finds it | `ImprovementOpportunity` logged with `error_handling` type |
| 10.3 | Analyzer runs during active conversation | Should be blocked | No file writes while `conversation_active = True` |
| 10.4 | `.bak` backup created before any auto-fix | Check filesystem | `*.py.bak` file appears alongside modified file |

---

## 11. Read Aloud Feature

| # | Test | What to observe | Expected |
|---|------|-----------------|----------|
| 11.1 | Select text in any window → trigger read-aloud hotkey | Annabeth reads selected text | TTS audio plays the selected content |
| 11.2 | Long text selection | Sentence chunking applied | Audio flows sentence-by-sentence |
| 11.3 | No text selected | Graceful fallback | Either reads clipboard or notifies no text found |

---

## 12. Discord Rich Presence

| # | Test | What to observe | Expected |
|---|------|-----------------|----------|
| 12.1 | Discord running and Annabeth Unity client starts | Discord status updates | Status shows "Hanging out with Annabeth" (or similar) |
| 12.2 | Idle state engages | Discord status may change | Optional: status shows idle state |
| 12.3 | Discord not running | Client starts without crash | No exception; `[DiscordPresence]` logs a warning and disables gracefully |

---

## 13. Audio Beat Sync (WASAPI Loopback)

| # | Test | What to observe | Expected |
|---|------|-----------------|----------|
| 13.1 | Play music from any app | Avatar reacts to beats | Avatar animates / pulses with bass |
| 13.2 | Stop music | Avatar calms | Beat intensity returns to 0 |
| 13.3 | No WASAPI device | Loopback capture fails gracefully | `[Audio] No loopback devices found` — no crash |
| 13.4 | Different preferred device in config | Correct device selected | Log confirms selected device name matches config |

---

## 14. Touch / Pet Reactions

| # | Test | What to observe | Expected |
|---|------|-----------------|----------|
| 14.1 | Click/tap avatar head region | Pet reaction fires | Animation plays; TTS reaction audio |
| 14.2 | Click avatar body region | Different reaction than head | Different audio/animation from head pat |
| 14.3 | Rapid repeated clicks | Cooldown applies | Won't spam reactions; min delay between triggers |

---

## 15. WebSocket & Settings API

| # | Test | What to observe | Expected |
|---|------|-----------------|----------|
| 15.1 | Unity connects after server starts | Connection established | `[WS] Client connected` in log |
| 15.2 | Send `{"type":"get_settings"}` command | Settings JSON returned | All `VarDef` keys and current values in response |
| 15.3 | Send `{"type":"set_setting","key":"IDLE_TIMEOUT_S","value":60}` | Setting updates live | `[Registry] IDLE_TIMEOUT_S = 60` in log |
| 15.4 | Send invalid value `{"key":"VAD_AGGRESSIVENESS","value":99}` | Validation rejects it | `[Registry] Validation failed` logged; old value kept |

---

## 16. GPT-SoVITS TTS

| # | Test | What to observe | Expected |
|---|------|-----------------|----------|
| 16.1 | Server up, GPT-SoVITS running | Voice synthesis works | Audio plays within 1-2s of text ready |
| 16.2 | GPT-SoVITS down / fallback | Error handling | Logs `[TTS]` warning; no server crash |
| 16.3 | RVC enabled in config | Voice passes through RVC | Post-processed audio with different vocal timbre |

---

## Pass/Fail Summary

After testing, fill this in:

| Section | Status | Notes |
|---------|--------|-------|
| 1. Startup | | |
| 2. Basic Conversation | | |
| 3. Facial Expressions | | |
| 4. VAD | | |
| 5. Emotion System | | |
| 6. Idle / Screensaver | | |
| 7. Idle Speech Bubbles | | |
| 8. Grillo Reflection | | |
| 9. Model Router | | |
| 10. Self-Improvement | | |
| 11. Read Aloud | | |
| 12. Discord Presence | | |
| 13. Audio Beat Sync | | |
| 14. Touch Reactions | | |
| 15. WebSocket API | | |
| 16. GPT-SoVITS TTS | | |

---

## Quick Test Config Tips

To speed up testing of slow-trigger features, temporarily set these in `character_config.yaml` or via the settings API:

```yaml
# Accelerated test values (revert after testing)
grillo_beat_interval: 120        # 2 min instead of 45 min
idle_timeout_seconds: 30         # 30s instead of 5 min
idle_bubble_min_delay: 15        # 15s instead of 60s
proactive_idle_seconds: 60       # 1 min instead of 2h
```

After testing is done:
1. Share `d:\Annabeth\test_session.log` for review
2. Revert any accelerated test config values
3. Remove `debug_logger` hook from `main_chat.py` (or set `ANNABETH_DEBUG_LOG=0`)
