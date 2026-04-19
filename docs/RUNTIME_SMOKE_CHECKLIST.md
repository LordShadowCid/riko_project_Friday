# Runtime Smoke Checklist

Use this checklist when you want to verify a local Annabeth build after startup, launcher, Unity, or read-aloud changes.

## 1. Bring Up The Stack

Preferred launcher:

```powershell
.\start_annabeth.ps1
```

Legacy frontend:

```powershell
.\start_annabeth.ps1 -Legacy
```

Recovery / cleanup:

```powershell
.\stop_annabeth.ps1
```

Manual debug path:

```powershell
./start-gpt-sovits-native.ps1 -Gpu 1
python -m server.main_chat
```

Expected startup signs:

- GPT-SoVITS reachable on `http://127.0.0.1:9880/docs`
- backend reachable on `http://127.0.0.1:8765/`
- Unity launches `Annabeth.exe` or Legacy launches the PyQt6 companion
- backend logs `[Avatar] Client connected. Total: 1`

## 2. Voice Interaction Phrases

Use complete sentences, not one or two words, so VAD does not drop them as too short.

Say these in order:

1. `Annabeth, can you hear me?`
2. `Tell me in one sentence what mode you are in.`
3. `Switch to idle mode.`
4. `Switch back to active mode.`

Expected results:

- ASR transcription appears in the backend log
- Annabeth responds through TTS
- Unity or Legacy avatar stays connected while speaking

## 3. Read-Aloud Checks

### Voice-triggered read-aloud

1. Highlight text in another application.
2. Leave that source application focused.
3. Say: `Read this for me.`

Expected results:

- backend logs `[ReadAloud] Detected read intent - capturing text...`
- if capture succeeds, backend logs `[ReadAloud] Captured ... characters`
- TTS acknowledgment plays: `Sure, let me read that for you.`
- read-aloud begins sentence playback
- in Unity, the current sentence appears in the avatar speech bubble while playback is active

If capture fails, inspect the new backend diagnostic line:

```text
[ReadAloud] Capture debug: hwnd=... title='...' target_was_companion=... clipboard_had_text_before=...
```

Interpretation:

- `target_was_companion=True`: the companion window was foreground, so the copy target was wrong
- `clipboard_had_text_before=False`: there was no useful fallback clipboard text
- a source-app title with no capture usually means the selection did not actually copy

### Legacy global hotkey path

Only for the Legacy PyQt6 frontend:

1. Highlight text in another app.
2. Keep that app focused.
3. Press `Ctrl+Shift+R`.

Expected result: backend starts reading selected text without a voice trigger.

## 4. Pause / Resume / Stop Read-Aloud

While read-aloud is active:

1. Press `Q` in the companion window, or send the read pause command from the client.
2. Confirm backend logs the pause request.
3. Press `R` to resume.
4. Say `Stop reading.` to cancel entirely.

Expected results:

- read-aloud enters paused state
- resume continues from the next sentence
- stop returns the manager to idle

## 5. Focused Regression Commands

Run these after touching the relevant slice:

```powershell
.\run_runtime_checks.ps1
```

Optional fail-fast mode:

```powershell
.\run_runtime_checks.ps1 -StopOnFailure
```

Individual commands:

```powershell
.\check_backend_startup.ps1
python test_read_aloud.py
python test_avatar_state_sync.py
python test_avatar_message_broadcast.py
python test_system_integrity.py
```

`test_avatar_message_broadcast.py` now covers `speak_start`, `speak_end`, `emotion`, `debug_status`, `read_highlight`, and `read_clear`.

## 6. Common Failure Patterns

- `Too short, ignoring...`
  Use a full sentence for voice tests.

- `I don't see any text selected.`
  Re-highlight the text and keep the source app focused instead of the companion.

- `target_was_companion=True`
  The companion window was foreground during copy.

- `Client connected. Total: 0` never appears
  The frontend launched but did not attach to the avatar server.

- TTS port does not open
  Recheck GPT-SoVITS startup and the shared `gpt_sovits_models` junction path.

- `Port 8765 is already in use by PID ...`
  A stale backend is still running. Use `./stop_annabeth.ps1` and retry startup.