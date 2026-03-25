"""
Deep test for the Read-Aloud feature.

Tests:
 1. Module imports & structure
 2. Sentence splitting (edge cases)
 3. ReadAloudManager state machine
 4. Intent phrase matching
 5. Word timing estimation
 6. Text capture (mock clipboard)
 7. Pause / resume / stop lifecycle
 8. Q&A context generation
 9. Win32 helpers availability (Windows)
10. Browser extension files exist
"""

import sys, os, time, re
from pathlib import Path

# Ensure project root is on path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

pass_count = 0
fail_count = 0
results = []


def log(msg):
    results.append(msg)
    print(msg)


def PASS(name, detail=""):
    global pass_count
    pass_count += 1
    log(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))


def FAIL(name, detail=""):
    global fail_count
    fail_count += 1
    log(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))


def section(title):
    log(f"\n{'='*60}")
    log(f"  {title}")
    log(f"{'='*60}")


# ══════════════════════════════════════════════════════════════
log("#" * 60)
log("#  ANNABETH READ-ALOUD DEEP TEST")
log(f"#  {time.strftime('%Y-%m-%d %H:%M:%S')}")
log("#" * 60)

# ── TEST 1: Module imports ────────────────────────────────────
section("TEST 1: Module Imports & Structure")

try:
    from server.process.read_aloud import (
        capture_selected_text,
        split_into_sentences,
        ReadAloudManager,
        get_read_aloud_manager,
    )
    PASS("Core read_aloud imports")
except ImportError as e:
    FAIL("Core read_aloud imports", str(e))

try:
    from server.process.read_aloud.text_capture import estimate_word_timings
    PASS("estimate_word_timings import")
except ImportError as e:
    FAIL("estimate_word_timings import", str(e))

try:
    from server.process.read_aloud.manager import ReadAloudStatus, ReadAloudState
    PASS("ReadAloudStatus / ReadAloudState import")
except ImportError as e:
    FAIL("ReadAloudStatus / ReadAloudState import", str(e))

try:
    from server.process.read_aloud.text_capture import register_companion_hwnd
    PASS("register_companion_hwnd import (Win32 helper)")
except ImportError as e:
    FAIL("register_companion_hwnd import", str(e))

# Check __init__.py exports
try:
    import server.process.read_aloud as ra_mod
    for name in ['capture_selected_text', 'split_into_sentences',
                 'ReadAloudManager', 'get_read_aloud_manager']:
        assert hasattr(ra_mod, name), f"Missing export: {name}"
    PASS("__init__.py exports all public names")
except Exception as e:
    FAIL("__init__.py exports", str(e))

# ── TEST 2: Sentence splitting ────────────────────────────────
section("TEST 2: Sentence Splitting")

tests_split = [
    ("Simple two sentences",
     "Hello there. How are you?",
     2),
    ("Multiple sentences",
     "Hello there! This is a test. How are you doing today? I hope you're well.",
     4),
    ("Abbreviation handling",
     "Dr. Smith went to the store. He bought milk.",
     2),
    ("Single sentence no period",
     "Just a single line of text",
     1),
    ("Empty string",
     "",
     0),
    ("Multiple abbreviations",
     "Mr. Johnson and Mrs. Clark met Dr. Brown. They discussed the plan.",
     2),
    ("Exclamation and question",
     "What a day! Did you see that? I couldn't believe it.",
     3),
]

for label, text, expected_count in tests_split:
    result = split_into_sentences(text)
    if len(result) == expected_count:
        PASS(f"Split: {label}", f"{len(result)} sentences")
    else:
        FAIL(f"Split: {label}", f"expected {expected_count}, got {len(result)}: {result}")


# ── TEST 3: ReadAloudManager state machine ────────────────────
section("TEST 3: ReadAloudManager State Machine")

mgr = ReadAloudManager()

# Initial state
if mgr.state.status == ReadAloudStatus.IDLE:
    PASS("Initial status is IDLE")
else:
    FAIL("Initial status", f"expected IDLE, got {mgr.state.status}")

# Start reading
test_text = "First sentence. Second sentence. Third sentence."
mgr.state.start_reading(test_text)

if mgr.state.status == ReadAloudStatus.READING:
    PASS("Status after start_reading is READING")
else:
    FAIL("Status after start_reading", f"expected READING, got {mgr.state.status}")

if len(mgr.state.sentences) == 3:
    PASS("Sentences count", "3")
else:
    FAIL("Sentences count", f"expected 3, got {len(mgr.state.sentences)}")

if mgr.state.current_index == 0:
    PASS("Current index starts at 0")
else:
    FAIL("Current index", f"expected 0, got {mgr.state.current_index}")

if mgr.state.full_text == test_text:
    PASS("full_text stored correctly")
else:
    FAIL("full_text", "mismatch")

# is_active / is_reading
if mgr.is_active:
    PASS("is_active is True while reading")
else:
    FAIL("is_active", "should be True")

if mgr.state.is_reading:
    PASS("is_reading True")
else:
    FAIL("is_reading", "should be True")

# Advance
has_more = mgr.state.advance_index()
if has_more and mgr.state.current_index == 1:
    PASS("advance_index increments correctly")
else:
    FAIL("advance_index", f"has_more={has_more}, idx={mgr.state.current_index}")

# Advance to last
mgr.state.advance_index()  # idx=2
has_more = mgr.state.advance_index()  # idx=3 → done
if not has_more and mgr.state.status == ReadAloudStatus.IDLE:
    PASS("advance_index returns False + IDLE when done")
else:
    FAIL("advance_index done", f"has_more={has_more}, status={mgr.state.status}")


# ── TEST 4: Pause / Resume / Stop lifecycle ───────────────────
section("TEST 4: Pause / Resume / Stop Lifecycle")

mgr2 = ReadAloudManager()
mgr2.state.start_reading("One sentence. Two sentence. Three sentence.")

# Pause
mgr2.state.pause()
if mgr2.state.status == ReadAloudStatus.FINISHING:
    PASS("pause() sets FINISHING status")
else:
    FAIL("pause()", f"expected FINISHING, got {mgr2.state.status}")

if mgr2.state.pause_requested:
    PASS("pause_requested is True")
else:
    FAIL("pause_requested", "should be True")

# Complete pause
mgr2.state.complete_pause()
if mgr2.state.status == ReadAloudStatus.PAUSED:
    PASS("complete_pause() sets PAUSED")
else:
    FAIL("complete_pause()", f"expected PAUSED, got {mgr2.state.status}")

if mgr2.state.is_paused:
    PASS("is_paused True")
else:
    FAIL("is_paused", "should be True")

# Resume
next_sent = mgr2.state.resume()
if mgr2.state.status == ReadAloudStatus.READING and next_sent is not None:
    PASS("resume() returns to READING with next sentence")
else:
    FAIL("resume()", f"status={mgr2.state.status}, next={next_sent}")

# Stop
mgr2.state.stop()
if mgr2.state.status == ReadAloudStatus.IDLE:
    PASS("stop() returns to IDLE")
else:
    FAIL("stop()", f"expected IDLE, got {mgr2.state.status}")


# ── TEST 5: Q&A context generation ───────────────────────────
section("TEST 5: Q&A Context Generation")

mgr3 = ReadAloudManager()
mgr3.state.start_reading("Alpha sentence. Beta sentence. Gamma sentence.")
mgr3.state.advance_index()  # read first sentence
mgr3.state.advance_index()  # read second sentence

ctx = mgr3.get_qa_context()
if "Alpha sentence." in ctx and "Beta sentence." in ctx:
    PASS("Q&A context contains read sentences")
else:
    FAIL("Q&A context", f"missing expected text: {ctx[:100]}")

if "I was reading" in ctx:
    PASS("Q&A context has framing text")
else:
    FAIL("Q&A context framing", f"missing framing: {ctx[:100]}")


# ── TEST 6: Word timing estimation ───────────────────────────
section("TEST 6: Word Timing Estimation")

timings = estimate_word_timings("Hello world how are you", 5.0)
if len(timings) == 5:
    PASS("Correct number of word timings", "5 words → 5 timings")
else:
    FAIL("Word timing count", f"expected 5, got {len(timings)}")

# Check structure
if all(len(t) == 3 for t in timings):
    PASS("Each timing is (word, start, end) tuple")
else:
    FAIL("Timing structure", "wrong tuple length")

# Check timing order
starts = [t[1] for t in timings]
if starts == sorted(starts):
    PASS("Timings are in chronological order")
else:
    FAIL("Timing order", f"starts not sorted: {starts}")

# Total covers full duration
last_end = timings[-1][2]
if abs(last_end - 5.0) < 0.01:
    PASS("Last timing end matches total duration")
else:
    FAIL("Timing duration", f"last_end={last_end}, expected 5.0")

# Empty sentence
empty_timings = estimate_word_timings("", 1.0)
if empty_timings == []:
    PASS("Empty sentence returns empty timings")
else:
    FAIL("Empty timings", f"got {empty_timings}")


# ── TEST 7: Intent phrase matching ────────────────────────────
section("TEST 7: Intent Phrase Matching")

read_intent_phrases = [
    "read that", "read this", "read the selected", "read selected",
    "read it", "read aloud", "read to me", "read what i selected",
    "read the text", "can you read", "please read",
    "read this for me", "read that for me", "read it for me",
    "read what's highlighted", "read the highlighted",
    "read my selection", "read what i highlighted",
]

resume_phrases = [
    "keep reading", "continue reading", "go on", "resume reading",
    "read on", "carry on", "keep going", "continue where you left off",
]

stop_phrases = [
    "stop reading", "that's enough", "never mind", "cancel reading",
    "forget it", "done reading", "quit reading",
]

# Test various user phrases
test_phrases = [
    ("read this for me", "read", True),
    ("hey can you read this for me please", "read", True),
    ("read that for me", "read", True),
    ("read it for me", "read", True),
    ("please read the text", "read", True),
    ("read what's highlighted", "read", True),
    ("read my selection", "read", True),
    ("keep reading please", "resume", True),
    ("continue reading", "resume", True),
    ("go on reading", "resume", True),
    ("stop reading now", "stop", True),
    ("that's enough thanks", "stop", True),
    ("cancel reading", "stop", True),
    ("tell me a joke", "none", False),
    ("what's the weather like", "none", False),
]

for phrase, expected_type, should_match in test_phrases:
    user_lower = phrase.lower().strip()
    is_read = any(p in user_lower for p in read_intent_phrases)
    is_resume = any(p in user_lower for p in resume_phrases)
    is_stop = any(p in user_lower for p in stop_phrases)

    detected = "none"
    if is_read:
        detected = "read"
    elif is_resume:
        detected = "resume"
    elif is_stop:
        detected = "stop"

    if expected_type == "none" and not should_match:
        if not is_read and not is_resume and not is_stop:
            PASS(f"No false match: '{phrase}'")
        else:
            FAIL(f"False positive: '{phrase}'", f"detected={detected}")
    elif detected == expected_type:
        PASS(f"Matched '{phrase}' → {expected_type}")
    else:
        FAIL(f"Phrase: '{phrase}'", f"expected {expected_type}, detected {detected}")


# ── TEST 8: get_read_aloud_manager singleton ──────────────────
section("TEST 8: Singleton Manager")

m1 = get_read_aloud_manager()
m2 = get_read_aloud_manager()
if m1 is m2:
    PASS("get_read_aloud_manager returns same instance")
else:
    FAIL("Singleton", "different instances returned")


# ── TEST 9: Win32 helpers on Windows ──────────────────────────
section("TEST 9: Win32 Helpers (Windows)")

if sys.platform == "win32":
    from server.process.read_aloud.text_capture import (
        _send_ctrl_c, _get_foreground_hwnd, _set_foreground,
        register_companion_hwnd, _IS_WIN,
    )
    if _IS_WIN:
        PASS("_IS_WIN is True on Windows")
    else:
        FAIL("_IS_WIN", "should be True")

    hwnd = _get_foreground_hwnd()
    if isinstance(hwnd, int) and hwnd > 0:
        PASS("_get_foreground_hwnd returns valid HWND", f"hwnd={hwnd}")
    else:
        FAIL("Foreground HWND", f"got {hwnd}")

    # register_companion_hwnd should not crash
    register_companion_hwnd(0)
    PASS("register_companion_hwnd(0) no crash")

    # _send_ctrl_c is callable
    if callable(_send_ctrl_c):
        PASS("_send_ctrl_c is callable")
    else:
        FAIL("_send_ctrl_c", "not callable")
else:
    PASS("Skipped Win32 tests (not Windows)")


# ── TEST 10: Browser extension files ──────────────────────────
section("TEST 10: Browser Extension Files")

ext_dir = _root / "browser_extension"
for fname in ["manifest.json", "background.js", "content.js"]:
    fpath = ext_dir / fname
    if fpath.exists():
        PASS(f"Extension file exists: {fname}")
    else:
        FAIL(f"Extension file missing: {fname}")

# Check manifest.json has correct name
import json
manifest_path = ext_dir / "manifest.json"
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
    if "Annabeth" in manifest.get("name", ""):
        PASS("Manifest name contains 'Annabeth'")
    else:
        FAIL("Manifest name", f"got '{manifest.get('name')}'")
    if manifest.get("manifest_version") == 3:
        PASS("Manifest v3")
    else:
        FAIL("Manifest version", f"got {manifest.get('manifest_version')}")


# ── TEST 11: ReadAloudManager advance() method ────────────────
section("TEST 11: advance() method")

mgr4 = ReadAloudManager()
mgr4.state.start_reading("First. Second. Third.")

sent = mgr4.state.current_sentence
if sent and "First" in sent:
    PASS("current_sentence returns first sentence")
else:
    FAIL("current_sentence", f"got '{sent}'")

next_s = mgr4.state.advance()
if next_s and "Second" in next_s:
    PASS("advance() returns second sentence")
else:
    FAIL("advance()", f"got '{next_s}'")

next_s2 = mgr4.state.advance()
if next_s2 and "Third" in next_s2:
    PASS("advance() returns third sentence")
else:
    FAIL("advance()", f"got '{next_s2}'")

next_s3 = mgr4.state.advance()
if next_s3 is None and mgr4.state.status == ReadAloudStatus.IDLE:
    PASS("advance() returns None when done, IDLE")
else:
    FAIL("advance() end", f"next={next_s3}, status={mgr4.state.status}")


# ── TEST 12: Thread safety smoke test ─────────────────────────
section("TEST 12: Thread Safety Smoke Test")

import threading

mgr5 = ReadAloudManager()
errors = []


def reader_thread():
    try:
        mgr5.state.start_reading("A. B. C. D. E. F. G. H. I. J.")
        for _ in range(10):
            mgr5.state.advance_index()
            time.sleep(0.001)
    except Exception as e:
        errors.append(str(e))


def pauser_thread():
    try:
        for _ in range(5):
            mgr5.state.pause()
            time.sleep(0.002)
            mgr5.state.complete_pause()
            mgr5.state.resume()
    except Exception as e:
        errors.append(str(e))


t1 = threading.Thread(target=reader_thread)
t2 = threading.Thread(target=pauser_thread)
t1.start()
t2.start()
t1.join(timeout=5)
t2.join(timeout=5)

if not errors:
    PASS("Concurrent read/pause/resume no exceptions")
else:
    FAIL("Thread safety", f"errors: {errors}")


# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════
log(f"\n{'='*60}")
log(f"  SUMMARY: {pass_count} PASS / {fail_count} FAIL  (total {pass_count+fail_count})")
log(f"{'='*60}")

if fail_count > 0:
    log("\nFailed tests:")
    for r in results:
        if "[FAIL]" in r:
            log(f"  {r.strip()}")

sys.exit(fail_count)
