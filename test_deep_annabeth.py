"""
Deep automated test suite for Annabeth LLM pipeline.

Tests LLM uniqueness, tool calling, memory, TTS, gibberish detection,
history integrity, and streaming behavior WITHOUT audio hardware.
Results are saved to test_results_deep.txt.

Run from the Anabeth root with the venv activated:
  python test_deep_annabeth.py
"""

import json
import os
import sys
import time
import copy
import traceback
from pathlib import Path
from io import StringIO
from difflib import SequenceMatcher

# ── Fix Windows console encoding for Unicode output ─────────────────
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Ensure project root is on path ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Set CUDA PATH for Whisper GPU (needed by some imports)
venv_site = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
for sub in ("nvidia/cudnn/bin", "nvidia/cublas/bin"):
    p = str(venv_site / sub)
    if p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + ";" + os.environ.get("PATH", "")

REPORT_FILE = PROJECT_ROOT / "test_results_deep.txt"
HISTORY_BACKUP = PROJECT_ROOT / "chat_history_backup_test.json"
HISTORY_FILE = PROJECT_ROOT / "chat_history.json"

# Global results collector
results: list[str] = []
pass_count = 0
fail_count = 0
warn_count = 0


def log(msg: str):
    results.append(msg)
    print(msg)


def PASS(name: str, detail: str = ""):
    global pass_count
    pass_count += 1
    log(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))


def FAIL(name: str, detail: str = ""):
    global fail_count
    fail_count += 1
    log(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def WARN(name: str, detail: str = ""):
    global warn_count
    warn_count += 1
    log(f"  [WARN] {name}" + (f" — {detail}" if detail else ""))


def section(title: str):
    log(f"\n{'='*60}")
    log(f"  {title}")
    log(f"{'='*60}")


# ════════════════════════════════════════════════════════════════════
# TEST 1: Config & Environment
# ════════════════════════════════════════════════════════════════════
def test_config():
    section("TEST 1: Configuration & Environment")
    try:
        from server.annabeth_config import load_config
        cfg = load_config()
        if cfg:
            PASS("character_config.yaml loads")
        else:
            FAIL("character_config.yaml loads", "Empty config")

        # Check tools enabled
        tools = cfg.get("tools", {})
        if tools.get("enabled"):
            PASS("Tools enabled in config")
        else:
            WARN("Tools NOT enabled in config")

        # Check system prompt exists
        sp = cfg.get("presets", {}).get("default", {}).get("system_prompt", "")
        if len(sp) > 50:
            PASS("System prompt present", f"{len(sp)} chars")
        else:
            FAIL("System prompt present", f"Only {len(sp)} chars")

        # Check Ollama settings
        try:
            from server.utils import get_ollama_settings
            settings = get_ollama_settings(cfg)
            PASS("Ollama settings loaded", f"model={settings.get('model')}, num_ctx={settings.get('num_ctx')}")
        except Exception as e:
            FAIL("Ollama settings", str(e))

    except Exception as e:
        FAIL("Config loading", str(e))


# ════════════════════════════════════════════════════════════════════
# TEST 2: Ollama connectivity
# ════════════════════════════════════════════════════════════════════
def test_ollama():
    section("TEST 2: Ollama Connectivity")
    try:
        import requests as req
        from server.utils import get_ollama_settings
        settings = get_ollama_settings()
        host = settings["host"]

        # Check Ollama is running
        r = req.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json()
        models = [m["name"] for m in data.get("models", [])]
        PASS("Ollama reachable", f"{len(models)} models loaded")

        target = settings["model"]
        found = any(target in m for m in models)
        if found:
            PASS(f"Model '{target}' available")
        else:
            FAIL(f"Model '{target}' available", f"Found: {models}")

        # Quick generation test (low tokens to be fast)
        payload = {
            "model": target,
            "messages": [
                {"role": "system", "content": "Say exactly: TEST_OK"},
                {"role": "user", "content": "Say TEST_OK"},
            ],
            "stream": False,
            "options": {"num_predict": 20, "temperature": 0.1},
        }
        t0 = time.time()
        r = req.post(f"{host}/api/chat", json=payload, timeout=180)
        elapsed = time.time() - t0
        r.raise_for_status()
        resp = (r.json().get("message") or {}).get("content", "")
        if resp.strip():
            PASS("Quick generation works", f"{elapsed:.1f}s, got {len(resp)} chars: '{resp.strip()[:60]}'")
        else:
            FAIL("Quick generation works", "Empty response")

    except Exception as e:
        FAIL("Ollama connectivity", str(e))


# ════════════════════════════════════════════════════════════════════
# TEST 3: TTS (GPT-SoVITS) connectivity
# ════════════════════════════════════════════════════════════════════
def test_tts():
    section("TEST 3: TTS (GPT-SoVITS) Connectivity")
    try:
        import requests as req

        # Check if TTS server is reachable
        try:
            r = req.get("http://127.0.0.1:9880/", timeout=5)
            PASS("TTS server reachable", f"status={r.status_code}")
        except Exception:
            WARN("TTS server not reachable at 127.0.0.1:9880 (may not be running)")
            return

        # Try to synthesize using the same payload format as sovits_ping.py
        from server.annabeth_config import load_config, resolve_repo_path
        cfg = load_config()
        sovits_cfg = cfg.get("sovits_ping_config", {})
        ref_audio_path = sovits_cfg.get("ref_audio_path", "")
        if isinstance(ref_audio_path, str) and ref_audio_path.strip().startswith("/"):
            ref_audio_path = ref_audio_path.strip()
        else:
            ref_audio_path = resolve_repo_path(ref_audio_path)

        payload = {
            "text": "Hello senpai, testing one two three.",
            "text_lang": sovits_cfg.get("text_lang", "en"),
            "ref_audio_path": ref_audio_path,
            "prompt_text": sovits_cfg.get("prompt_text", ""),
            "prompt_lang": sovits_cfg.get("prompt_lang", "en"),
        }
        t0 = time.time()
        r = req.post("http://127.0.0.1:9880/tts", json=payload, timeout=30)
        elapsed = time.time() - t0
        if r.status_code == 200 and len(r.content) > 1000:
            PASS("TTS synthesis works", f"{elapsed:.1f}s, {len(r.content)} bytes audio")
        else:
            FAIL("TTS synthesis works", f"status={r.status_code}, {len(r.content)} bytes")

    except Exception as e:
        FAIL("TTS connectivity", str(e))


# ════════════════════════════════════════════════════════════════════
# TEST 4: Gibberish detector
# ════════════════════════════════════════════════════════════════════
def test_gibberish_detector():
    section("TEST 4: Gibberish Detector")
    try:
        from server.process.llm_funcs.llm_scr import _is_gibberish

        # Good text should NOT be flagged
        good_texts = [
            "Hey senpai, what's up? I'm doing great today!",
            "The time is currently 3:45 PM Mountain Time.",
            "Oh great, you want me to say pineapple again? Fine... PINEAPPLE!",
            "My favorite GPU would probably be the RTX 4090 for obvious reasons.",
        ]
        for t in good_texts:
            if not _is_gibberish(t):
                PASS(f"Good text not flagged", t[:50])
            else:
                FAIL(f"Good text wrongly flagged as gibberish", t[:50])

        # Bad text SHOULD be flagged
        bad_texts = [
            "th/at si/nk i/n wh/iLl e/whil-e I ann*w'er thee/me",
            "&amp;#39;hello &amp; world &amp;quot;test&amp;#x27;",
            "tHeLeTs KeEp ThIe gOiNg FoReVeR UnTiL wE",
            "superlongwordwithnospacesanditjustkeepsgoingandgoingforever anothersuperlongwordthatmakesnosenseatall",
        ]
        for t in bad_texts:
            if _is_gibberish(t):
                PASS(f"Bad text flagged", t[:50])
            else:
                FAIL(f"Bad text NOT flagged", t[:50])

    except Exception as e:
        FAIL("Gibberish detector", str(e))


# ════════════════════════════════════════════════════════════════════
# TEST 5: Repetition detector
# ════════════════════════════════════════════════════════════════════
def test_repetition_detector():
    section("TEST 5: Repetition Detector")
    try:
        from server.process.llm_funcs.llm_scr import _is_repetition

        base = "oh great, senpai finally decides to talk again! so much excitement in this room."

        # Exact match should be caught
        if _is_repetition(base, [base], threshold=0.90):
            PASS("Exact duplicate detected")
        else:
            FAIL("Exact duplicate NOT detected")

        # Near-match (minor word change)
        variant = "oh great, senpai finally decides to talk again! so much excitement in this room. not!"
        if _is_repetition(variant, [base], threshold=0.85):
            PASS("Near-duplicate detected")
        else:
            FAIL("Near-duplicate NOT detected")

        # Completely different text should NOT match
        different = "I love anime and video games, they're the best thing ever!"
        if not _is_repetition(different, [base], threshold=0.90):
            PASS("Different text not flagged")
        else:
            FAIL("Different text wrongly flagged as repeat")

        # Accumulated prefix test: old response appears as prefix of new
        accumulated = base + " Anyway, here's the new content about GPUs."
        if _is_repetition(accumulated, [base], threshold=0.85):
            PASS("Accumulated-prefix repeat detected")
        else:
            WARN("Accumulated-prefix NOT detected by repetition check (prefix strip should handle)")

    except Exception as e:
        FAIL("Repetition detector", str(e))


# ════════════════════════════════════════════════════════════════════
# TEST 6: Sanitizer
# ════════════════════════════════════════════════════════════════════
def test_sanitizer():
    section("TEST 6: Response Sanitizer")
    try:
        from server.process.llm_funcs.llm_scr import _sanitize_response

        # HTML entities should be decoded
        s = _sanitize_response("Hello &amp; world &lt;3 it&#39;s great")
        if "&amp;" not in s and "&lt;" not in s and "&#39;" not in s:
            PASS("HTML entities decoded", s[:60])
        else:
            FAIL("HTML entities NOT decoded", s[:60])

        # Asterisk actions should be stripped
        s = _sanitize_response("*laughs nervously* Hey there *waves*")
        if "*" not in s:
            PASS("Asterisk actions stripped", s[:60])
        else:
            FAIL("Asterisk actions NOT stripped", s[:60])

        # Long text should be truncated
        long_text = "word " * 200
        s = _sanitize_response(long_text)
        if len(s) <= 650:
            PASS("Long text truncated", f"{len(s)} chars")
        else:
            FAIL("Long text NOT truncated", f"{len(s)} chars")

    except Exception as e:
        FAIL("Sanitizer", str(e))


# ════════════════════════════════════════════════════════════════════
# TEST 7: Dedup history
# ════════════════════════════════════════════════════════════════════
def test_dedup():
    section("TEST 7: History Deduplication")
    try:
        from server.process.llm_funcs.llm_scr import _dedup_history

        response_a = "oh great senpai finally decides to talk again so much excitement in this room huh"
        response_b = "oh great senpai finally decides to talk again so much excitement in this room huh not!"

        messages = [
            {"role": "system", "content": [{"type": "input_text", "text": "System prompt"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": response_a}]},
            {"role": "user", "content": [{"type": "input_text", "text": "Say pineapple"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": response_b}]},
            {"role": "user", "content": [{"type": "input_text", "text": "What time?"}]},
        ]

        deduped = _dedup_history(messages)
        assistant_msgs = [m for m in deduped if isinstance(m, dict) and m.get("role") == "assistant"]

        if len(assistant_msgs) < 2:
            PASS("Dedup removed duplicate", f"{len(assistant_msgs)} assistant msgs remain")
        else:
            FAIL("Dedup did NOT remove duplicate", f"{len(assistant_msgs)} assistant msgs remain")

        # Non-duplicate should be kept
        msgs2 = [
            {"role": "system", "content": [{"type": "input_text", "text": "System prompt"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Totally unique response one!"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "Question"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Completely different answer!"}]},
        ]
        deduped2 = _dedup_history(msgs2)
        a2 = [m for m in deduped2 if isinstance(m, dict) and m.get("role") == "assistant"]
        if len(a2) == 2:
            PASS("Non-duplicates preserved", f"{len(a2)} assistant msgs kept")
        else:
            FAIL("Non-duplicates lost", f"{len(a2)} assistant msgs remain (expected 2)")

    except Exception as e:
        FAIL("Dedup history", str(e))


# ════════════════════════════════════════════════════════════════════
# TEST 8: Prefix Stripper
# ════════════════════════════════════════════════════════════════════
def test_prefix_stripper():
    section("TEST 8: Accumulated Prefix Stripper")
    try:
        from server.process.llm_funcs.llm_scr import _strip_accumulated_prefix

        old_resp = "Oh great, senpai finally decides to talk again! So much excitement in this room."
        new_resp = old_resp + " And haha, still thinking about pineapples? Yeah, sure thing. PINEAPPLE!"

        stripped = _strip_accumulated_prefix(new_resp, [old_resp.strip().lower()])

        if "pineapple" in stripped.lower() and len(stripped) < len(new_resp):
            PASS("Prefix stripped successfully", f"'{stripped[:60]}...'")
        else:
            FAIL("Prefix NOT stripped", f"Got {len(stripped)} chars, original={len(new_resp)}")

        # Short unique text should not be stripped
        unique = "I love pineapples! They're the best tropical fruit."
        kept = _strip_accumulated_prefix(unique, [old_resp.strip().lower()])
        if kept == unique:
            PASS("Unique text not stripped")
        else:
            FAIL("Unique text wrongly stripped", kept[:60])

    except Exception as e:
        FAIL("Prefix stripper", str(e))


# ════════════════════════════════════════════════════════════════════
# TEST 9: Tool Calling
# ════════════════════════════════════════════════════════════════════
def test_tools():
    section("TEST 9: Tool Calling")
    try:
        from server.process.llm_funcs.llm_scr import _match_and_run_tools

        # Time tool
        result = _match_and_run_tools("What time is it right now?", speaker_name="Dad")
        if result and ("AM" in result or "PM" in result or ":" in result):
            PASS("Time tool works", result[:80])
        else:
            FAIL("Time tool", f"Got: {result}")

        # Memory tool — remember
        result = _match_and_run_tools("Remember that my favorite color is blue", speaker_name="Dad")
        if result:
            PASS("Remember tool works", result[:80])
        else:
            WARN("Remember tool returned nothing")

        # Memory tool — recall
        result = _match_and_run_tools("What do you recall about my favorite color?", speaker_name="Dad")
        if result:
            PASS("Recall tool works", result[:80])
        else:
            WARN("Recall tool returned nothing")

    except Exception as e:
        FAIL("Tool calling", str(e))


# ════════════════════════════════════════════════════════════════════
# TEST 10: Memory Store
# ════════════════════════════════════════════════════════════════════
def test_memory():
    section("TEST 10: Memory Store (ChromaDB)")
    try:
        from server.process.memory.memory_store import get_memory_store
        store = get_memory_store()

        # Count existing
        conv_count = store.conversations.count()
        fact_count = store.facts.count()
        PASS("ChromaDB connected", f"conversations={conv_count}, facts={fact_count}")

        # Add and recall a test conversation
        test_id = store.add_conversation("Test: user asked about GPU recommendations", speaker="TestBot")
        if test_id:
            PASS("add_conversation works", test_id)
        else:
            FAIL("add_conversation returned None")

        # Recall it
        results_ = store.recall_conversations("GPU recommendations", n_results=1)
        if results_ and "GPU" in results_[0].get("text", ""):
            PASS("recall_conversations works", results_[0]["text"][:60])
        else:
            FAIL("recall_conversations", f"Got {results_}")

        # recall_all
        all_mem = store.recall_all("GPU", n_results=3)
        if all_mem:
            PASS("recall_all works", f"{len(all_mem)} results")
        else:
            WARN("recall_all returned empty")

    except Exception as e:
        FAIL("Memory store", str(e))


# ════════════════════════════════════════════════════════════════════
# TEST 11: Multi-Turn LLM Uniqueness (THE BIG ONE)
# ════════════════════════════════════════════════════════════════════
def test_llm_multi_turn():
    section("TEST 11: Multi-Turn LLM Response Uniqueness")
    log("  Sending 8 diverse questions to the LLM pipeline...")
    log("  This is the critical test for the repeat bug.")

    questions = [
        "Hello Annabeth, how are you doing today?",
        "What is your favorite kind of food?",
        "Can you say the word watermelon for me?",
        "What do you think about space exploration?",
        "If you could have any superpower, what would it be?",
        "Tell me a fun fact about dolphins.",
        "What would you name a pet cat?",
        "What is two plus two?",
    ]

    # Backup and reset history
    if HISTORY_FILE.exists():
        import shutil
        shutil.copy2(HISTORY_FILE, HISTORY_BACKUP)

    try:
        from server.process.llm_funcs.llm_scr import (
            llm_response_streaming, load_history, save_history,
            _is_repetition, _is_gibberish, SYSTEM_PROMPT,
        )

        # Start with clean history
        save_history(SYSTEM_PROMPT)

        responses: list[str] = []
        timings: list[float] = []
        repeat_flags: list[bool] = []
        gibberish_flags: list[bool] = []

        for i, q in enumerate(questions):
            log(f"\n  --- Turn {i+1}: \"{q}\"")
            sentences_received: list[str] = []

            def on_sentence(s):
                sentences_received.append(s)

            t0 = time.time()
            try:
                full = llm_response_streaming(q, on_sentence=on_sentence, speaker_name="Dad")
            except Exception as e:
                FAIL(f"Turn {i+1} raised exception", str(e))
                full = ""
            elapsed = time.time() - t0

            responses.append(full)
            timings.append(elapsed)

            # Check against all prior responses
            prior_lower = [r.strip().lower() for r in responses[:-1]]
            is_rep = _is_repetition(full, prior_lower, threshold=0.85)
            is_gib = _is_gibberish(full)
            repeat_flags.append(is_rep)
            gibberish_flags.append(is_gib)

            # Log the response (truncated)
            display = full[:120].replace('\n', ' ')
            log(f"  Response ({elapsed:.1f}s): {display}...")
            if is_rep:
                log(f"  [!] REPEAT detected against prior responses")
            if is_gib:
                log(f"  [!] GIBBERISH detected")

        # Summarize
        log(f"\n  --- Uniqueness Summary ---")
        total_repeats = sum(repeat_flags)
        total_gibberish = sum(gibberish_flags)
        avg_time = sum(timings) / len(timings) if timings else 0

        if total_repeats == 0:
            PASS("All 8 responses unique", "No repeats detected!")
        elif total_repeats <= 2:
            WARN(f"{total_repeats}/8 turns were repeats", "Improved but not perfect")
        else:
            FAIL(f"{total_repeats}/8 turns were repeats", "Repeat problem persists")

        if total_gibberish == 0:
            PASS("No gibberish detected in any response")
        elif total_gibberish <= 1:
            WARN(f"{total_gibberish}/8 responses had gibberish")
        else:
            FAIL(f"{total_gibberish}/8 responses had gibberish")

        PASS(f"Average LLM response time: {avg_time:.1f}s")

        # Check for response variety — all first 40 chars should differ
        prefixes = [r[:40].strip().lower() for r in responses if r]
        unique_prefixes = set(prefixes)
        if len(unique_prefixes) >= len(prefixes) * 0.7:
            PASS(f"Response opener variety: {len(unique_prefixes)}/{len(prefixes)} unique prefixes")
        else:
            FAIL(f"Response opener variety: {len(unique_prefixes)}/{len(prefixes)} unique prefixes",
                 "Model stuck on same opener")

    except Exception as e:
        FAIL("Multi-turn test", f"{e}\n{traceback.format_exc()}")


# ════════════════════════════════════════════════════════════════════
# TEST 12: History File Integrity
# ════════════════════════════════════════════════════════════════════
def test_history_integrity():
    section("TEST 12: History File Integrity")
    try:
        if not HISTORY_FILE.exists():
            FAIL("History file exists", "chat_history.json not found")
            return

        with open(HISTORY_FILE, "r", encoding="utf-8-sig") as f:
            history = json.load(f)

        if not isinstance(history, list):
            FAIL("History is a list", f"Got {type(history)}")
            return

        PASS(f"History loaded", f"{len(history)} messages")

        # Check system prompt is first
        if history and history[0].get("role") == "system":
            PASS("System prompt is first message")
        else:
            FAIL("System prompt is first message")

        # Check for consecutive user messages (broken alternation)
        prev_role = None
        consecutive_user = 0
        max_consecutive_user = 0
        for m in history:
            role = m.get("role")
            if role == "system":
                prev_role = role
                continue
            if role == "user" and prev_role == "user":
                consecutive_user += 1
                max_consecutive_user = max(max_consecutive_user, consecutive_user)
            else:
                consecutive_user = 0
            prev_role = role

        if max_consecutive_user == 0:
            PASS("No consecutive user messages (proper alternation)")
        elif max_consecutive_user <= 2:
            WARN(f"Some consecutive user messages: max {max_consecutive_user} in a row",
                 "Some repeat-skips happened")
        else:
            FAIL(f"Many consecutive user messages: max {max_consecutive_user} in a row",
                 "History structure is broken")

        # Check for duplicate assistant responses
        assistant_texts = []
        for m in history:
            if m.get("role") == "assistant":
                content = m.get("content", "")
                if isinstance(content, list):
                    text = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                elif isinstance(content, str):
                    text = content
                else:
                    text = str(content)
                assistant_texts.append(text.strip().lower())

        if len(assistant_texts) > 1:
            dup_pairs = 0
            for i in range(len(assistant_texts)):
                for j in range(i + 1, len(assistant_texts)):
                    ratio = SequenceMatcher(None, assistant_texts[i][:120], assistant_texts[j][:120]).ratio()
                    if ratio >= 0.85:
                        dup_pairs += 1
            if dup_pairs == 0:
                PASS("No duplicate assistant responses in history")
            else:
                FAIL(f"{dup_pairs} duplicate pairs found in history")

    except Exception as e:
        FAIL("History integrity", str(e))


# ════════════════════════════════════════════════════════════════════
# TEST 13: Self-Eval System
# ════════════════════════════════════════════════════════════════════
def test_self_eval():
    section("TEST 13: Self-Evaluation System")
    try:
        from server.process.memory.self_eval import self_evaluate
        PASS("self_eval module imports")

        from server.process.memory.feedback import _get_conn
        conn = _get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM self_eval")
        count = cursor.fetchone()[0]
        PASS(f"self_eval DB accessible", f"{count} evaluations stored")

        if count > 0:
            cursor = conn.execute(
                "SELECT AVG(helpfulness), AVG(in_character), AVG(appropriate_length) "
                "FROM self_eval ORDER BY timestamp DESC LIMIT 20"
            )
            row = cursor.fetchone()
            if row:
                PASS(f"Recent eval averages: help={row[0]:.1f} char={row[1]:.1f} len={row[2]:.1f}")

    except Exception as e:
        FAIL("Self-eval system", str(e))


# ════════════════════════════════════════════════════════════════════
# TEST 14: Personality System
# ════════════════════════════════════════════════════════════════════
def test_personality():
    section("TEST 14: Personality System")
    try:
        personality_path = Path(r"C:\annabeth_data\personality.json")
        if personality_path.exists():
            with open(personality_path, "r") as f:
                p = json.load(f)
            PASS(f"Personality loaded", str(p))
            for key in ("verbosity", "snarkiness", "formality"):
                val = p.get(key)
                if isinstance(val, (int, float)) and 1 <= val <= 5:
                    PASS(f"  {key}={val} (valid range)")
                else:
                    FAIL(f"  {key}={val} (out of range 1-5)")
        else:
            WARN("Personality file not found at C:\\annabeth_data\\personality.json")

    except Exception as e:
        FAIL("Personality system", str(e))


# ════════════════════════════════════════════════════════════════════
# TEST 15: Streaming infrastructure
# ════════════════════════════════════════════════════════════════════
def test_streaming():
    section("TEST 15: Streaming Infrastructure")
    try:
        from server.process.llm_funcs.llm_scr import stream_ollama_response

        messages = [
            {"role": "system", "content": [{"type": "input_text", "text": "You are a helpful assistant. Keep it short."}]},
            {"role": "user", "content": [{"type": "input_text", "text": "Say hello in exactly five words."}]},
        ]

        sentences = []
        t0 = time.time()
        for sentence in stream_ollama_response(messages, temp_boost=0.0):
            sentences.append(sentence)
        elapsed = time.time() - t0

        full = " ".join(sentences)
        if sentences:
            PASS(f"Streaming yielded {len(sentences)} sentences in {elapsed:.1f}s")
            PASS(f"First sentence: '{sentences[0][:60]}'")
        else:
            FAIL("Streaming yielded no sentences")

    except Exception as e:
        FAIL("Streaming infrastructure", str(e))


# ════════════════════════════════════════════════════════════════════
# TEST 16: WebSocket server reachability
# ════════════════════════════════════════════════════════════════════
def test_websocket():
    section("TEST 16: WebSocket Server")
    try:
        import requests as req
        r = req.get("http://localhost:8765", timeout=3)
        if r.status_code == 200:
            PASS("WebSocket/HTTP server reachable at :8765")
        else:
            WARN(f"Server returned status {r.status_code}")
    except Exception:
        WARN("WebSocket server not reachable (may not be running during test)")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    log(f"{'#'*60}")
    log(f"#  ANNABETH DEEP TEST SUITE")
    log(f"#  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'#'*60}")

    test_config()
    test_ollama()
    test_tts()
    test_gibberish_detector()
    test_repetition_detector()
    test_sanitizer()
    test_dedup()
    test_prefix_stripper()
    test_tools()
    test_memory()
    test_self_eval()
    test_personality()
    test_streaming()
    test_websocket()

    # The big one — multi-turn uniqueness. Run last because it takes longest.
    test_llm_multi_turn()
    test_history_integrity()

    # Final summary
    section("FINAL SUMMARY")
    log(f"  PASS: {pass_count}")
    log(f"  FAIL: {fail_count}")
    log(f"  WARN: {warn_count}")
    log(f"  Total: {pass_count + fail_count + warn_count}")
    if fail_count == 0:
        log(f"\n  [OK] ALL TESTS PASSED (with {warn_count} warnings)")
    else:
        log(f"\n  [FAIL] {fail_count} FAILURES detected - review above")

    # Save report
    report = "\n".join(results)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Report saved to {REPORT_FILE}]")

    # Restore history backup if it existed
    if HISTORY_BACKUP.exists():
        import shutil
        shutil.copy2(HISTORY_BACKUP, HISTORY_FILE)
        HISTORY_BACKUP.unlink()
        print("[History restored from backup]")


if __name__ == "__main__":
    main()
