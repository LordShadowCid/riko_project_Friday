# Annabeth — Removed / Archived Code Log

Entries are added in reverse-chronological order.  
Each entry records **what** was removed, **why**, and the **verbatim code** in case it needs to be restored.

---

## 2026-04-13 — Code cleanup pass (full audit of all 28 server Python files)

### 1. `server/process/asr_func/asr_push_to_talk.py` — `_resolve_device()`

**Why removed:** Exact byte-for-byte duplicate of `utils.resolve_device(device, kind)`.
Both functions share identical logic: `None` check → `int` passthrough → substring search
against `sd.query_devices()` with capability filtering.  Keeping two copies meant any bug
fix or device-matching improvement would silently diverge between the two code paths.

**Replacement:** `from server.utils import resolve_device` (added to existing import line).
Call site unchanged: `resolve_device(input_device, kind='input')`.

**Removed code:**
```python
def _resolve_device(device, kind='input'):
    """Resolve a sounddevice input/output device selector.

    - None: use default
    - int: treated as device index
    - str: case-insensitive substring match against device names
    - kind: 'input' or 'output' - filters to only match devices with channels for that direction
    """
    if device is None or device == "":
        return None
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        devices = sd.query_devices()
        needle = device.lower().strip()
        for idx, d in enumerate(devices):
            name = str(d.get("name", "")).lower()
            if needle and needle in name:
                # Filter by device capability
                if kind == 'output' and d.get('max_output_channels', 0) > 0:
                    return idx
                elif kind == 'input' and d.get('max_input_channels', 0) > 0:
                    return idx
                # If kind doesn't match, keep searching
    return None
```

---

### 2. `server/process/read_aloud/manager.py` — `ReadAloudState.advance()`

**Why removed:** Dead code — never called from any other file.  `advance_index()` (which
does the same index increment without the return value) is what `main_chat.py` calls at
lines 471, 497, and 554.  `advance()` silently duplicated `advance_index()` but also
returned the next sentence text, which was never used.  Having two near-identical methods
invited confusion about which should be the canonical call.

**What it was supposed to do:** Advance the reading cursor AND return the next sentence
string in a single call (convenience method for a streaming-read pattern that was never
wired up in the main pipeline).

**Removed code:**
```python
def advance(self) -> Optional[str]:
    """Move to next sentence and return it, or None if done."""
    with self._lock:
        self._current_index += 1
        if self._current_index >= len(self._sentences):
            self._status = ReadAloudStatus.IDLE
            return None
        return self._sentences[self._current_index]
```

---

### 3. `server/process/read_aloud/manager.py` — `if __name__ == "__main__"` test block

**Why removed:** Library modules should not contain runnable test harnesses in their
`__main__` block — this mixes concerns and adds import-time noise.  The functionality
is covered by the integrity test suite.

**What it was supposed to do:** Quick smoke-test of `ReadAloudManager` from the terminal
(`python -m server.process.read_aloud.manager`) without running the full server.

**Removed code:**
```python
if __name__ == "__main__":
    # Test the manager
    manager = get_read_aloud_manager()

    # Simulate reading
    test_text = "Hello there! This is a test. How are you doing today? I hope you're well."
    manager.state.start_reading(test_text)

    print(f"Status: {manager.state.status}")
    print(f"Sentences: {manager.state.sentences}")

    # Simulate reading through sentences
    while manager.state.is_reading:
        sentence = manager.get_next_sentence()
        if sentence:
            print(f"Reading: {sentence}")
            # Simulate TTS delay
            time.sleep(0.5)
            # Note: get_next_sentence handles index advancement internally
        else:
            break

    print(f"Final status: {manager.state.status}")
```

---

### 4. `server/process/memory/bio_manager.py` — wasteful throwaway connection in `_ensure_table()`

**Why changed (not removed):** `_ensure_table()` was opening a brand-new `sqlite3.connect()`
call just to `CREATE TABLE IF NOT EXISTS`, committing, then immediately closing it.  This
left a redundant transient connection alongside the module-level `_conn` opened by
`_get_conn()`, wasting a file handle and producing two WAL writers at startup.

**What it does now:** Uses the shared `_get_conn()` connection that the rest of the module
already uses.  The table schema is identical — `feedback.py` also creates `speaker_bio` via
`_init_tables()` (both use `CREATE TABLE IF NOT EXISTS` so there is no conflict).

**Old code:**
```python
def _ensure_table() -> None:
    with _db_lock:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS speaker_bio (
                ...
            );
            CREATE INDEX IF NOT EXISTS idx_bio_speaker ON speaker_bio(speaker_id);
        """)
        conn.commit()
        conn.close()   # ← throwaway — caused a second open/close cycle
```

---

### 5. `server/process/read_aloud/text_capture.py` — inline `abbreviations` list in `split_into_sentences()`

**Why changed (not removed):** The list was re-allocated on every call to
`split_into_sentences()`.  A function that splits a sentence for TTS is called many times
per reading session.  Allocating the same 13-element list on every call is wasteful.

**What changed:** Moved to module-level constant `_SENTENCE_ABBREVIATIONS` (defined once at
import time).  The function now reads `abbreviations = _SENTENCE_ABBREVIATIONS`.

**Old inline code:**
```python
abbreviations = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.',
                 'vs.', 'etc.', 'i.e.', 'e.g.', 'Inc.', 'Ltd.', 'Co.']
```

---

### 6. `server/process/tools/self_modify.py` — magic numbers 0.3 / 3.5 / 3.5 / 3.5

**Why changed (not removed):** Hard-coded threshold values scattered across 4 `if` branches
with no explanation of their origin.  If the thresholds need tuning (e.g. after gathering
more feedback data), a maintainer would need to hunt down every occurrence rather than
changing one constant.

**What changed:** Two named module-level constants added:
- `_INTERRUPT_RATE_HIGH = 0.30` — interrupt fraction above which verbosity is reduced
- `_SCORE_LOW_THRESHOLD = 3.5` — average eval score below which a personality trait adjusts

All four `if` branches now reference these constants instead of raw literals.

---

## Items audited but intentionally left unchanged

The following were flagged during audit but are **not** dead code and were left alone:

| File | Item | Reason kept |
|------|------|-------------|
| `llm_scr.py` | `_response_cache`/`ResponseCache` | Cache logic is active; disabled only when `_cache_enabled=False` which is a config toggle, not dead code |
| `llm_scr.py` | `TOOL_SCHEMAS` list | All 5 tool executors are implemented and called via `_match_and_run_tools()` |
| `main_chat.py` | `estimate_word_timings` import | Actively used at line 485 in the read-aloud prefetch loop |
| `asr_vad.py` | `BatchedInferencePipeline` lazy import | Intentional lazy-load for audio > 10 s; safe pattern |
| `code_analyzer.py` | `pass` in `visit_Constant` | Documented placeholder — magic number detection intentionally conservative |
| `manager.py` | `set_tts_callback`, `set_sentence_callback`, `set_complete_callback` | `_on_sentence_start` IS invoked at runtime; `_on_reading_complete` IS invoked at runtime; setters are extension points for future callers |
| `speaker_id.py` | Simple FIFO eviction in `_embedding_cache` | Correct for insertion-ordered dict (Python 3.7+); not a true LRU but sufficient for <10 enrolled speakers |
| `feedback.py` + `bio_manager.py` | Dual `CREATE TABLE IF NOT EXISTS speaker_bio` | Both use `IF NOT EXISTS`; no conflict; `feedback.py` owns the canonical schema |
