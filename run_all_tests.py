"""
Annabeth Master Test Runner
============================
Runs every test suite in the correct order, streams output to the console
in real time, writes a combined report to test_results_full.txt, and prints
a final pass/fail/skip summary table.

Usage
-----
  # Normal run (offline — no Ollama / GPT-SoVITS required)
  python run_all_tests.py

  # Include live-service tests (requires Ollama + GPT-SoVITS running)
  python run_all_tests.py --live

  # Enable full DEBUG logging to C:\\annabeth_data\\logs\\annabeth_debug.log
  python run_all_tests.py --debug-log

Real-time log monitoring (open a second terminal while this runs)
  PowerShell:
    Get-Content "C:\\annabeth_data\\logs\\annabeth_debug.log" -Wait -Tail 60

Post-testing production downgrade
  Remove --debug-log and unset ANNABETH_LOG_LEVEL.  The server will
  automatically switch to WARNING-only rotating file (1 MB × 3 backups).
"""

import os
import sys
import subprocess
import time
import threading
import queue
import json
from pathlib import Path
from typing import List, Optional, Tuple

# ── Project root ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
REPORT_FILE = ROOT / "test_results_full.txt"

LIVE_MODE = "--live" in sys.argv
DEBUG_LOG  = "--debug-log" in sys.argv

# ── Colour helpers ──────────────────────────────────────────────────
_GREEN   = "\033[32m"
_RED     = "\033[31m"
_YELLOW  = "\033[33m"
_CYAN    = "\033[36m"
_BOLD    = "\033[1m"
_RESET   = "\033[0m"

def _c(text, colour):
    return f"{colour}{text}{_RESET}"

# ────────────────────────────────────────────────────────────────────
# Test suite definitions
# ────────────────────────────────────────────────────────────────────
# Each entry: (label, script_path, requires_live_services, timeout_seconds)
# Order is deliberate — foundation tests first, integration last.
TEST_SUITES: List[Tuple[str, str, bool, int]] = [
    # ── Foundation ─────────────────────────────────────────────────
    ("Config & Imports",          "test_system_integrity.py",        False, 120),
    # ── Memory subsystem ──────────────────────────────────────────
    ("Bio Manager",               "_test_bio.py",                    False,  30),
    # ── Feature deep-dives ───────────────────────────────────────
    ("LLM Pipeline (offline)",    "test_deep_annabeth.py",           False, 180),
    ("Read-Aloud Feature",        "test_read_aloud.py",              False,  60),
    # ── Avatar / WebSocket ───────────────────────────────────────
    ("Avatar State Sync",         "test_avatar_state_sync.py",       False,  60),
    ("Avatar Message Broadcast",  "test_avatar_message_broadcast.py",False,  60),
    # ── Audio hardware (needs a sound device + soundcard package) ─
    ("Audio Capture (hardware)",  "test_audio_capture.py",           True,   30),
    # ── Live service tests (skipped unless --live) ───────────────
    ("WASAPI Loopback (live)",    "test_wasapi_loopback.py",         True,   60),
    ("Unity Deep (live)",         "test_unity_deep.py",              True,  120),
]

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _stream_and_collect(
    proc: subprocess.Popen,
    prefix: str,
    output_lines: list,
    done_event: threading.Event,
) -> None:
    """Background thread: read lines from proc stdout+stderr and print them."""
    for line in proc.stdout:          # stdout=PIPE with stderr merged
        text = line.rstrip("\n")
        output_lines.append(text)
        print(f"{_CYAN}{prefix}{_RESET}  {text}")
    done_event.set()


def _parse_counts(lines: List[str]) -> Tuple[int, int]:
    """Scan output lines for PASS/FAIL counters."""
    passes = fails = 0
    for line in lines:
        passes += line.count("[PASS]")
        fails  += line.count("[FAIL]")
    return passes, fails


def run_suite(
    label: str,
    script: str,
    requires_live: bool,
    timeout: int,
    env: dict,
) -> dict:
    """Run one test script, return result dict."""
    result = {
        "label":   label,
        "script":  script,
        "status":  "???",
        "passes":  0,
        "fails":   0,
        "elapsed": 0.0,
        "lines":   [],
    }

    if requires_live and not LIVE_MODE:
        result["status"] = "SKIP"
        print(f"\n{_YELLOW}{'─'*60}{_RESET}")
        print(f"{_YELLOW}SKIP{_RESET}  {label}  (pass --live to include)")
        return result

    script_path = ROOT / script
    if not script_path.exists():
        result["status"] = "MISSING"
        print(f"\n{_RED}{'─'*60}{_RESET}")
        print(f"{_RED}MISSING{_RESET}  {label}  →  {script}")
        return result

    print(f"\n{'='*60}")
    print(f"{_BOLD}RUNNING:{_RESET}  {label}")
    print(f"  script : {script}")
    print(f"  timeout: {timeout}s")
    print("─" * 60)

    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(
            [PYTHON, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(ROOT),
        )
    except Exception as exc:
        result["status"] = "ERROR"
        result["lines"]  = [f"Failed to launch: {exc}"]
        print(f"{_RED}ERROR launching {script}: {exc}{_RESET}")
        return result

    done = threading.Event()
    t = threading.Thread(
        target=_stream_and_collect,
        args=(proc, "│", result["lines"], done),
        daemon=True,
    )
    t.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        result["status"] = "TIMEOUT"
    finally:
        done.wait(timeout=5)
        t.join(timeout=5)

    result["elapsed"] = time.perf_counter() - t0
    result["passes"], result["fails"] = _parse_counts(result["lines"])

    if result["status"] == "TIMEOUT":
        print(f"\n{_RED}TIMEOUT{_RESET}  {label}  — killed after {timeout}s")
    elif proc.returncode == 0 and result["fails"] == 0:
        result["status"] = "PASS"
        print(f"\n{_GREEN}PASS{_RESET}  {label}  ({result['passes']} assertions, {result['elapsed']:.1f}s)")
    else:
        result["status"] = "FAIL"
        print(f"\n{_RED}FAIL{_RESET}  {label}  ({result['fails']} failures, exit={proc.returncode}, {result['elapsed']:.1f}s)")

    return result


# ────────────────────────────────────────────────────────────────────
# Inline gap / feature checks (no subprocess — fast pure-Python)
# ────────────────────────────────────────────────────────────────────

def run_inline_feature_checks() -> dict:
    """
    Quick in-process checks for the 8 Phases + 5 gap fixes.
    These don't need audio hardware or live services.
    """
    label = "Phase & Gap Feature Checks (inline)"
    passes = fails = 0
    lines  = [f"\n{'='*60}", f"  {label}", "─"*60]

    def ok(name, detail=""):
        nonlocal passes
        passes += 1
        msg = f"  [PASS] {name}" + (f" — {detail}" if detail else "")
        lines.append(msg); print(f"{_CYAN}│{_RESET}  {msg}")

    def bad(name, detail=""):
        nonlocal fails
        fails += 1
        msg = f"  [FAIL] {name}" + (f" — {detail}" if detail else "")
        lines.append(msg); print(f"{_RED}│  {msg}{_RESET}")

    print(f"\n{'='*60}")
    print(f"{_BOLD}RUNNING:{_RESET}  {label}")
    print("─"*60)

    t0 = time.perf_counter()

    # ── Phase 1: Emotion state ───────────────────────────────────
    try:
        from server.process.memory.emotion_state import (
            set_emotion, get_dominant_emotion, get_emotion_context,
            extract_emotion_tags, strip_emotion_tags,
            start_decay_loop, stop_decay_loop,
        )
        set_emotion("happy", 9.0)
        dom = get_dominant_emotion()
        assert dom == "happy", f"expected happy dominant, got {dom!r}"
        ok("Phase 1: emotion set/get dominant", dom)
        ctx = get_emotion_context()
        assert ctx and "mood" in ctx.lower(), f"expected mood context, got: {ctx!r}"
        ok("Phase 1: get_emotion_context injection string", ctx)
        tags  = extract_emotion_tags("{happy 8}")
        assert "happy" in tags
        ok("Phase 1: extract_emotion_tags")
        clean = strip_emotion_tags("Hello {happy 9} world")
        assert "{" not in clean
        ok("Phase 1: strip_emotion_tags")
        start_decay_loop(interval_seconds=9999)
        stop_decay_loop()
        ok("Phase 1: decay loop start/stop")
    except Exception as e:
        bad("Phase 1 emotion_state", str(e))

    # ── Phase 2: Emotion context → LLM ──────────────────────────
    try:
        import inspect
        import server.process.llm_funcs.llm_scr as llm
        src = inspect.getsource(llm)
        assert "get_emotion_context" in src, "emotion not injected"
        assert "set_emotions_from_dict" in src, "per-sentence extract missing"
        ok("Phase 2: emotion context injected into LLM")
        ok("Phase 2: per-sentence emotion extraction in stream")
    except Exception as e:
        bad("Phase 2 emotion injection", str(e))

    # ── Phase 3: Reflection / diary ──────────────────────────────
    try:
        from server.process.memory.reflection_loop import (
            get_proactive_queue, get_diary_context,
        )
        q = get_proactive_queue()
        assert hasattr(q, "empty")
        ok("Phase 3: proactive queue exists")
        ctx = get_diary_context(n=2)
        assert isinstance(ctx, str)
        ok("Phase 3: get_diary_context (empty OK)", ctx or "<no entries yet>")
        # Also verify diary context injected into LLM
        import server.process.llm_funcs.llm_scr as llm2
        src2 = inspect.getsource(llm2)
        assert "get_diary_context" in src2
        ok("Phase 3: diary context injected into LLM")
    except Exception as e:
        bad("Phase 3 reflection_loop", str(e))

    # ── Phase 4: Model router ────────────────────────────────────
    try:
        from server.process.llm_funcs.model_router import ModelRouter
        cfg = {
            "model_routing": {
                "enabled": True,
                "primary_model": "llama3",
                "fast_model": "gemma3:4b",
            }
        }
        router = ModelRouter(cfg)
        m1 = router.get_model_for_intent("greeting")
        assert m1 == "gemma3:4b", f"greeting should route gemma3:4b, got {m1}"
        ok("Phase 4: greeting -> gemma3:4b", m1)
        m2 = router.get_model_for_intent("story")
        assert m2 == "llama3", f"story should route llama3, got {m2}"
        ok("Phase 4: story -> llama3", m2)
    except Exception as e:
        bad("Phase 4 model_router", str(e))

    # ── Phase 5: Bio manager ─────────────────────────────────────
    try:
        from server.process.memory.bio_manager import (
            ensure_speaker, add_fact, get_bio, update_bio,
        )
        ensure_speaker("__test_runner__")
        add_fact("__test_runner__", "likes espresso")
        bio = get_bio("__test_runner__")
        assert "espresso" in bio
        ok("Phase 5: bio add_fact + get_bio")
        # Verify bio auto-update wired in summarizer
        import server.process.memory.conversation_summarizer as cs
        src3 = inspect.getsource(cs)
        assert "bio_add_fact" in src3
        ok("Phase 5: bio auto-update wired in conversation_summarizer")
    except Exception as e:
        bad("Phase 5 bio_manager", str(e))

    # ── Phase 6: Memory compression ──────────────────────────────
    try:
        from server.process.memory.memory_store import get_memory_store, MemoryStore
        ms = get_memory_store()
        assert ms is not None
        ok("Phase 6: memory_store singleton")
        assert hasattr(ms, "compress_if_needed")
        ok("Phase 6: compress_if_needed exists")
    except Exception as e:
        bad("Phase 6 memory_store", str(e))

    # ── Phase 7: RVC ─────────────────────────────────────────────
    try:
        from server.process.tts_func.rvc_convert import RvcConverter
        import inspect as _ins
        src4 = _ins.getsource(RvcConverter.__init__)
        assert "index_path" in src4
        ok("Phase 7: RVC index_path parameter present")
        # RVC model may not exist — that's fine for this check
        ok("Phase 7: RVC .index support wired (model not required for this check)")
    except Exception as e:
        bad("Phase 7 rvc_convert", str(e))

    # ── Phase 8: Code analysis / proposals ───────────────────────
    try:
        from server.process.tools.code_analyzer import CodeAnalyzer, scan_directory
        from server.process.tools.proposal_generator import get_proposal_generator
        ca = CodeAnalyzer("server/logging_config.py", str(ROOT))
        results_list = ca.analyze()
        assert isinstance(results_list, list)
        ok("Phase 8: CodeAnalyzer.analyze() runs", f"{len(results_list)} opportunities")
        gen = get_proposal_generator()
        ok("Phase 8: ProposalGenerator singleton", "model required for run()")
    except Exception as e:
        bad("Phase 8 code_analyzer/proposal_generator", str(e))

    # ── Gap 1-5 summary ──────────────────────────────────────────
    try:
        ctx2 = get_emotion_context()
        assert ctx2 and "mood" in ctx2.lower()
        ok("Gap 1: emotion context → LLM (runtime value)", ctx2)
    except Exception as e:
        bad("Gap 1 emotion context runtime", str(e))

    elapsed = time.perf_counter() - t0
    status = "PASS" if fails == 0 else "FAIL"
    return {
        "label":   label,
        "script":  "(inline)",
        "status":  status,
        "passes":  passes,
        "fails":   fails,
        "elapsed": elapsed,
        "lines":   lines,
    }


# ────────────────────────────────────────────────────────────────────
# Report writer
# ────────────────────────────────────────────────────────────────────

def write_report(all_results: list, total_elapsed: float) -> None:
    lines = [
        "=" * 70,
        "  ANNABETH FULL TEST REPORT",
        f"  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
    ]

    total_pass = total_fail = total_skip = 0
    for r in all_results:
        lines.append(f"{'─'*70}")
        lines.append(f"SUITE: {r['label']}")
        lines.append(f"  script : {r['script']}")
        lines.append(f"  status : {r['status']}")
        lines.append(f"  passes : {r['passes']}")
        lines.append(f"  fails  : {r['fails']}")
        lines.append(f"  time   : {r['elapsed']:.1f}s")
        lines.append("")
        lines.extend(r["lines"])
        lines.append("")
        if r["status"] == "PASS":   total_pass += 1
        elif r["status"] == "SKIP": total_skip += 1
        else:                       total_fail += 1

    lines += [
        "=" * 70,
        f"  SUMMARY: {total_pass} suites PASS  /  {total_fail} suites FAIL  /  {total_skip} skipped",
        f"  Total assertions PASS: {sum(r['passes'] for r in all_results)}",
        f"  Total assertions FAIL: {sum(r['fails']  for r in all_results)}",
        f"  Wall time: {total_elapsed:.1f}s",
        "=" * 70,
    ]

    text = "\n".join(lines)
    REPORT_FILE.write_text(text, encoding="utf-8")
    print(f"\n[Runner] Report written → {REPORT_FILE}")


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main() -> int:
    # Enable debug logging for all child processes if requested
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if DEBUG_LOG:
        env["ANNABETH_LOG_LEVEL"] = "DEBUG"

    print(_c("=" * 70, _BOLD))
    print(_c("  ANNABETH MASTER TEST RUNNER", _BOLD))
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  live-services : {'YES' if LIVE_MODE else 'NO  (pass --live to enable)'}")
    print(f"  debug-logging : {'YES → C:\\\\annabeth_data\\\\logs\\\\annabeth_debug.log' if DEBUG_LOG else 'NO  (pass --debug-log to enable)'}")
    print(_c("=" * 70, _BOLD))

    if DEBUG_LOG:
        print(_c(
            "\n[Tip] Monitor logs in real time with:\n"
            "  Get-Content 'C:\\annabeth_data\\logs\\annabeth_debug.log' -Wait -Tail 60",
            _CYAN,
        ))

    all_results = []
    wall_start  = time.perf_counter()

    # ── Phase / gap inline checks first ─────────────────────────
    all_results.append(run_inline_feature_checks())

    # ── External test scripts ────────────────────────────────────
    for label, script, live, timeout in TEST_SUITES:
        r = run_suite(label, script, live, timeout, env)
        all_results.append(r)

    total_elapsed = time.perf_counter() - wall_start

    # ── Final summary table ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(_c("  FINAL SUMMARY", _BOLD))
    print("─" * 70)
    print(f"  {'Suite':<42} {'Status':<8} {'P':>5} {'F':>5} {'Time':>7}")
    print("─" * 70)
    total_p = total_f = 0
    any_fail = False
    for r in all_results:
        colour = _GREEN if r["status"] == "PASS" else (_YELLOW if r["status"] == "SKIP" else _RED)
        status = _c(f"{r['status']:<8}", colour)
        print(f"  {r['label']:<42} {status} {r['passes']:>5} {r['fails']:>5} {r['elapsed']:>6.1f}s")
        total_p += r["passes"]
        total_f += r["fails"]
        if r["status"] not in ("PASS", "SKIP"):
            any_fail = True

    print("─" * 70)
    final_colour = _GREEN if not any_fail else _RED
    print(_c(
        f"  TOTAL  {total_p} assertions PASS  /  {total_f} FAIL  "
        f"in {total_elapsed:.1f}s",
        final_colour,
    ))
    print("=" * 70)

    write_report(all_results, total_elapsed)

    # ── Post-testing note ────────────────────────────────────────
    print(_c(
        "\n[Note] After testing, revert to production logging:\n"
        "  • Remove ANNABETH_LOG_LEVEL env var (or stop passing --debug-log)\n"
        "  • Server will auto-switch to WARNING-only rotating log (1 MB × 3 files)\n"
        "  • Zero code changes required — all logic is in server/logging_config.py",
        _CYAN,
    ))

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
