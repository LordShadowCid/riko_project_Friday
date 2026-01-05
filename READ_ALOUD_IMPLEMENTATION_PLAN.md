# Read Aloud Feature - Implementation Plan

**Date:** January 4, 2026  
**Status:** ✅ Phase 1-3 COMPLETE - Ready for Full Testing

---

## 🎯 Feature Overview

Add a "Read Aloud" capability that allows Annabeth to read selected text from any application (browser, PDF reader, Word, etc.) using her existing GPT-SoVITS voice, with text highlighting so you can follow along, and the ability to pause, ask questions about the content, and resume reading.

---

## ✅ What We Already Have (Won't Touch)

| Component | Status | Notes |
|-----------|--------|-------|
| GPT-SoVITS TTS | ✅ Working | `sovits_gen()` in [sovits_ping.py](server/process/tts_func/sovits_ping.py) - Same voice |
| Audio playback | ✅ Working | `play_audio()` with sounddevice - supports interruption |
| Hotkey system | ✅ Working | `HotkeyManager` in [desktop_companion_webview.py](client/desktop_companion_webview.py) |
| LLM chat | ✅ Working | `llm_response()` in [llm_scr.py](server/process/llm_funcs/llm_scr.py) |
| Thread-safe state | ✅ Working | `CompanionState` in [shared/state.py](shared/state.py) |
| S key silence | ✅ Working | Toggles `_state.silenced` |
| keyboard library | ✅ Installed | Used for global hotkeys |

---

## 🆕 New Features to Add

### 1. **Smart Pause Behavior (S Key)**
When you press **S** while reading:
- Finish speaking the current sentence (don't cut off mid-word)
- Enter "paused reading" mode
- Allow voice questions about what was read (normal voice chat)
- Wait for **R** to resume reading

### 2. **Text Highlighting (Follow Along)**
Highlight words on the source page as Annabeth reads them, so you can follow along visually.

### 3. **Context-Aware Q&A**
When paused, you can ask questions about the text being read. Annabeth uses the read content as context for her answers.

---

## 🔍 Research Findings

### Text Highlighting Challenge

| Approach | Feasibility | Complexity | Notes |
|----------|-------------|------------|-------|
| **Browser Extension** | ⭐⭐⭐⭐⭐ | Medium | Best for web pages - Chrome/Edge extension with native messaging |
| **UI Automation (UIA)** | ⭐⭐ | Very High | Windows only, unreliable for browsers, complex |
| **Inject JS via DevTools** | ⭐⭐⭐ | High | Only Chrome/Edge in debug mode, fragile |
| **Overlay Window** | ⭐⭐⭐⭐ | Medium | Works everywhere, but doesn't highlight IN the page |

**Recommendation:** Start with **Browser Extension** for web pages (most common use case), then consider **Overlay Window** for PDFs/Word later.

### Pause/Resume with Q&A

This is straightforward to implement:
- Track reading state: `idle`, `reading`, `paused`
- Store sentences queue and current position
- On pause: finish current sentence, stop queue, enable voice input
- Keep read text as context for LLM questions
- On resume: continue from where we left off

### Word-Level Timing

To highlight words as they're spoken, we need to know when each word is spoken:
1. **Estimate from TTS audio duration** (simple, approximate)
2. **Force alignment with audio** (complex, accurate) - tools like Gentle, Montreal Forced Aligner
3. **Word count / duration ratio** (simplest, good enough for following along)

**Recommendation:** Start with option 3 (duration estimation), upgrade to forced alignment later if needed.

---

## 📦 Dependencies Required

### For Core Read-Aloud (Phase 1)
```
pyperclip          # Clipboard access
pyautogui          # Send Ctrl+C to copy selection
```

### For Browser Extension (Phase 2)
- Chrome Extension (manifest v3) - communicates via WebSocket
- No additional Python packages needed

---

## 🏗️ Implementation Architecture

### New Files to Create

```
server/
└── process/
    └── read_aloud/
        ├── __init__.py
        ├── manager.py           # ReadAloudManager class (~200 lines)
        ├── text_capture.py      # Clipboard + Ctrl+C capture (~50 lines)
        └── word_timing.py       # Word timing estimation (~80 lines)

browser_extension/                # NEW - Chrome extension
├── manifest.json
├── background.js
├── content.js                   # Injects highlighting
└── styles.css
```

### Files to Modify (Carefully)

| File | Changes | Risk |
|------|---------|------|
| [shared/state.py](shared/state.py) | Add `ReadAloudState` class | Low - additive only |
| [desktop_companion_webview.py](client/desktop_companion_webview.py) | Add `Ctrl+Shift+R` hotkey, modify S/R behavior | Medium - must preserve existing |
| [main_chat.py](server/main_chat.py) | Check for read-aloud queue in main loop | Medium - integrate with existing flow |

---

## 📋 Phased Implementation Plan

### Phase 1: Basic Read-Aloud (No Highlighting) ⏱️ ~2-3 hours

**Goal:** Press Ctrl+Shift+R to read selected text using Annabeth's voice

1. **Install dependencies**
   ```bash
   pip install pyperclip pyautogui
   ```

2. **Create `server/process/read_aloud/text_capture.py`**
   - `capture_selected_text()` - sends Ctrl+C, reads clipboard
   - `split_into_sentences()` - regex-based sentence splitting

3. **Create `server/process/read_aloud/manager.py`**
   - `ReadAloudManager` class
   - States: `idle`, `reading`, `paused`
   - Queue sentences for TTS
   - Track position for resume

4. **Add `Ctrl+Shift+R` hotkey**
   - Capture text → queue for reading

5. **Test with browser, notepad, PDF reader**

### Phase 2: Smart Pause + Q&A ⏱️ ~2 hours

**Goal:** S pauses at sentence end, allows questions, R resumes

1. **Add `ReadAloudState` to [shared/state.py](shared/state.py)**
   ```python
   @dataclass
   class ReadAloudState:
       status: str = "idle"  # idle, reading, paused
       sentences: List[str] = field(default_factory=list)
       current_index: int = 0
       read_context: str = ""  # Full text for Q&A context
   ```

2. **Modify S key behavior**
   - If reading: set `pause_requested = True`
   - Manager finishes current sentence, then pauses
   - Enable voice input for questions

3. **Add R key for resume**
   - If paused: resume from `current_index`

4. **Context-aware Q&A**
   - When paused, prepend read text to LLM context
   - "Here's what I was reading: [text]. User's question: [question]"

### Phase 3: Browser Extension for Highlighting ⏱️ ~4-5 hours

**Goal:** Words highlight in browser as Annabeth reads

1. **Create Chrome Extension**
   - `manifest.json` with permissions for active tab
   - `content.js` - injects into pages, listens for highlight commands
   - WebSocket connection to Annabeth

2. **Add WebSocket messages for highlighting**
   ```javascript
   // Annabeth → Extension
   { type: "highlight_word", word: "hello", index: 5 }
   { type: "clear_highlights" }
   { type: "highlight_sentence", sentenceIndex: 2 }
   ```

3. **Word timing estimation**
   - Calculate words per second from audio duration
   - Send highlight commands at estimated times

4. **Install extension**
   - Load unpacked in Chrome
   - Register native messaging host (for non-WebSocket option)

### Phase 4: Overlay Window for Non-Browser Apps ⏱️ ~3-4 hours (Future)

**Goal:** Show highlighted text in overlay for PDFs, Word, etc.

1. **Create transparent overlay window**
   - Shows current sentence with word highlighting
   - Positioned near mouse or fixed location

2. **Alternative to in-page highlighting**
   - Works for any application
   - Less seamless but universal

---

## ⚠️ Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing S key | High | Add `reading` state check before changing behavior |
| Breaking dance modes | High | Test thoroughly - don't touch dance logic |
| Clipboard conflicts | Medium | Brief delay before reading clipboard, restore original |
| Browser extension security | Medium | Only inject into active tab when triggered |
| Word timing inaccuracy | Low | Acceptable for following along, not karaoke |

---

## 🧪 Testing Plan

### Phase 1 Tests
- [ ] Ctrl+Shift+R captures selected text from browser
- [ ] Ctrl+Shift+R captures from Notepad
- [ ] Ctrl+Shift+R captures from PDF reader
- [ ] Text is read with correct voice
- [ ] Original clipboard content preserved

### Phase 2 Tests
- [ ] S key pauses at end of sentence (not mid-word)
- [ ] Voice input works while paused
- [ ] Questions about read content get context-aware answers
- [ ] R key resumes from correct position
- [ ] Finishing all sentences returns to idle

### Phase 3 Tests
- [ ] Extension loads in Chrome
- [ ] Words highlight in sync with speech (approximately)
- [ ] Highlighting clears on new text
- [ ] Extension doesn't break regular browsing

---

## 📊 Estimated Total Time

| Phase | Time | Dependencies |
|-------|------|--------------|
| Phase 1: Basic Read-Aloud | 2-3 hours | pyperclip, pyautogui |
| Phase 2: Smart Pause + Q&A | 2 hours | Phase 1 complete |
| Phase 3: Browser Extension | 4-5 hours | Phase 1 complete |
| Phase 4: Overlay Window | 3-4 hours | Optional, future |

**Total for Phases 1-3:** ~8-10 hours

---

## 🚀 Ready to Proceed?

Please review this plan and let me know:

1. **Do you want to proceed with Phase 1 first?** (Basic read-aloud, no highlighting)
2. **Any concerns about the approach?**
3. **Should I prioritize browser highlighting (Phase 3) earlier?**

Once approved, I'll start with Phase 1 and we can test before moving to Phase 2.
