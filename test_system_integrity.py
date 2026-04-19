"""
Full System Integrity Test for Annabeth Desktop Companion.

Validates:
 - All Python module imports
 - Server functions (TTS, ASR, LLM, read-aloud)
 - Shared state & config (enums, classes, singletons)
 - Client file syntax & API surface
 - Config files (YAML, JSON)
 - Browser extension files
 - Win32 helpers
 - Unity C# scripts
 - Read-aloud lifecycle
"""
import sys, os, importlib, inspect, json, re, traceback, threading, time
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
if "." not in sys.path:
    sys.path.insert(0, ".")

pass_c = 0
fail_c = 0
results = []
SEP = "=" * 60

def PASS(n, d=""):
    global pass_c; pass_c += 1
    msg = f"  [PASS] {n}" + (f" -- {d}" if d else "")
    results.append(msg); print(msg)

def FAIL(n, d=""):
    global fail_c; fail_c += 1
    msg = f"  [FAIL] {n}" + (f" -- {d}" if d else "")
    results.append(msg); print(msg)

def section(t):
    print(f"\n{SEP}\n  {t}\n{SEP}")


print("#" * 60)
print("#  ANNABETH FULL SYSTEM INTEGRITY TEST")
print(f"#  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("#" * 60)

# ═══ 1. SERVER: annabeth_config ═══════════════════════════
section("1. SERVER: annabeth_config")
from server.annabeth_config import load_config, repo_root, resolve_repo_path
cfg = load_config()
if isinstance(cfg, dict) and "sovits_ping_config" in cfg:
    PASS("load_config returns valid dict with sovits_ping_config")
else:
    FAIL("load_config", f"keys={list(cfg.keys()) if isinstance(cfg, dict) else type(cfg)}")
_rr = repo_root() if callable(repo_root) else repo_root
if Path(_rr).exists():
    PASS("repo_root exists", str(_rr))
else:
    FAIL("repo_root", str(_rr))

# ═══ 2. SERVER: TTS function ═════════════════════════════
section("2. SERVER: TTS function")
from server.process.tts_func.sovits_ping import sovits_gen, play_audio
for fn_name, fn_obj in [("sovits_gen", sovits_gen), ("play_audio", play_audio)]:
    if callable(fn_obj):
        PASS(f"{fn_name} callable")
    else:
        FAIL(f"{fn_name} not callable")

sig = inspect.signature(play_audio)
params = list(sig.parameters.keys())
for p in ["path", "output_device", "interrupt_flag"]:
    if p in params:
        PASS(f"play_audio param: {p}")
    else:
        FAIL(f"play_audio param: {p}", f"params={params}")

# ═══ 3. SERVER: ASR functions ════════════════════════════
section("3. SERVER: ASR functions")
from server.process.asr_func.asr_vad import (
    record_vad_and_transcribe,
    get_interrupt_flag,
    get_speaking_flag,
)
if callable(record_vad_and_transcribe):
    PASS("record_vad_and_transcribe callable")
else:
    FAIL("record_vad_and_transcribe")

iflag = get_interrupt_flag()
sflag = get_speaking_flag()
if isinstance(iflag, threading.Event):
    PASS("get_interrupt_flag returns Event")
else:
    FAIL("get_interrupt_flag", str(type(iflag)))
if isinstance(sflag, threading.Event):
    PASS("get_speaking_flag returns Event")
else:
    FAIL("get_speaking_flag", str(type(sflag)))

# ═══ 4. SERVER: LLM functions ═══════════════════════════
section("4. SERVER: LLM functions")
from server.process.llm_funcs.llm_scr import llm_response, llm_response_streaming
for fn_name, fn_obj in [("llm_response", llm_response), ("llm_response_streaming", llm_response_streaming)]:
    if callable(fn_obj):
        PASS(f"{fn_name} callable")
    else:
        FAIL(f"{fn_name} not callable")

# ═══ 5. SHARED: config module ═══════════════════════════
section("5. SHARED: config module")
from shared.config import (
    CompanionMode, MessageType, Emotion,
    ServerConfig, AudioConfig, AnimationConfig, PathConfig, AnnabeConfig,
    get_config, reset_config, config_to_js,
)
modes = [m.name for m in CompanionMode]
for expected in ["ACTIVE", "IDLE", "DANCE_BEAT", "DANCE_FULL"]:
    if expected in modes:
        PASS(f"CompanionMode.{expected}")
    else:
        FAIL(f"CompanionMode.{expected}", f"modes={modes}")

msg_types = [m.name for m in MessageType]
for expected in ["SPEAK_START", "SPEAK_END", "EMOTION", "READ_HIGHLIGHT", "READ_CLEAR",
                 "MODE_CHANGE", "TOGGLE_SILENCE", "AUDIO_ANALYSIS", "READ_PAUSE", "READ_RESUME"]:
    if expected in msg_types:
        PASS(f"MessageType.{expected}")
    else:
        FAIL(f"MessageType.{expected}", f"types={msg_types}")

scfg = get_config()
if isinstance(scfg, AnnabeConfig):
    PASS("get_config returns AnnabeConfig")
else:
    FAIL("get_config", str(type(scfg)))

# ═══ 6. SHARED: state module ════════════════════════════
section("6. SHARED: state module")
from shared.state import (
    CompanionState, AudioState,
    get_companion_state, get_audio_state, reset_state,
    get_read_aloud_manager,
)
state = get_companion_state()
if isinstance(state, CompanionState):
    PASS("get_companion_state returns CompanionState")
else:
    FAIL("get_companion_state", str(type(state)))

astate = get_audio_state()
if isinstance(astate, AudioState):
    PASS("get_audio_state returns AudioState")
else:
    FAIL("get_audio_state", str(type(astate)))

from server.process.read_aloud.manager import ReadAloudManager as _RAM
ram = get_read_aloud_manager()
if isinstance(ram, _RAM):
    PASS("get_read_aloud_manager via shared.state")
else:
    FAIL("get_read_aloud_manager", str(type(ram)))

# ═══ 7. SHARED: __init__.py exports ═════════════════════
section("7. SHARED: __init__.py exports")
import shared
expected_exports = [
    "CompanionMode", "MessageType", "Emotion",
    "ServerConfig", "AudioConfig", "AnimationConfig", "PathConfig", "AnnabeConfig",
    "CompanionState", "AudioState",
    "get_config", "reset_config", "config_to_js",
    "get_companion_state", "get_audio_state", "reset_state",
    "get_read_aloud_manager",
]
for name in expected_exports:
    if hasattr(shared, name):
        PASS(f"shared.{name} exported")
    else:
        FAIL(f"shared.{name} missing")

# ═══ 8. CONFIG: character_config.yaml ═══════════════════
section("8. CONFIG: character_config.yaml")
import yaml
with open("character_config.yaml") as f:
    cc = yaml.safe_load(f)
for k in ["sovits_ping_config", "ollama"]:
    if k in cc:
        PASS(f"character_config has key: {k}")
    else:
        FAIL(f"character_config missing: {k}")

spc = cc.get("sovits_ping_config", {})
for field in ["ref_audio_path", "text_lang", "prompt_text", "prompt_lang"]:
    if field in spc:
        PASS(f"sovits_ping_config.{field}")
    else:
        FAIL(f"sovits_ping_config.{field} missing")

# ═══ 9. CONFIG: chat_history.json ═══════════════════════
section("9. CONFIG: chat_history.json")
ch_path = Path("chat_history.json")
if ch_path.exists():
    with open(ch_path) as f:
        ch = json.load(f)
    if isinstance(ch, list):
        PASS("chat_history.json is valid JSON list", f"{len(ch)} entries")
    else:
        FAIL("chat_history.json", f"expected list, got {type(ch)}")
else:
    PASS("chat_history.json not present (fresh start)")

# ═══ 10. CLIENT: file syntax checks ═════════════════════
section("10. CLIENT: Python file syntax")
client_files = [
    "client/avatar_server.py",
    "client/desktop_companion_webview.py",
    "client/desktop_companion.py",
    "client/audio_analyzer.py",
]
for fp in client_files:
    p = Path(fp)
    if p.exists():
        code = p.read_text(encoding="utf-8")
        try:
            compile(code, fp, "exec")
            PASS(f"Syntax OK: {fp}")
        except SyntaxError as e:
            FAIL(f"Syntax error: {fp}", str(e))
    else:
        FAIL(f"File missing: {fp}")

# ═══ 11. CLIENT: avatar_server API ══════════════════════
section("11. CLIENT: avatar_server API")
with open("client/avatar_server.py", encoding="utf-8") as f:
    av_src = f.read()
expected_funcs = [
    "speak_start", "speak_end", "set_emotion",
    "send_audio_data", "send_read_highlight", "send_read_clear",
    "send_debug_status", "broadcast",
]
for fn in expected_funcs:
    pat = rf"(async\s+)?def\s+{fn}\s*\("
    if re.search(pat, av_src):
        PASS(f"avatar_server func: {fn}")
    else:
        FAIL(f"avatar_server func missing: {fn}")

# ═══ 12. MAIN_CHAT: key functions ═══════════════════════
section("12. MAIN_CHAT: key functions")
with open("server/main_chat.py", encoding="utf-8") as f:
    mc_src = f.read()
for fn in ["process_read_aloud_queue", "clean_text_for_tts", "get_wav_duration",
           "_start_avatar_server", "avatar_speak_start", "avatar_speak_end",
           "avatar_debug_status", "is_listening_paused"]:
    if re.search(rf"def\s+{fn}\s*\(", mc_src):
        PASS(f"main_chat func: {fn}")
    else:
        FAIL(f"main_chat func missing: {fn}")

# ═══ 13. MAIN_CHAT: intent phrases ═════════════════════
section("13. MAIN_CHAT: intent phrases")
for phrase in ["read this for me", "keep reading", "stop reading",
               "read that for me", "read it for me", "continue reading",
               "cancel reading"]:
    if phrase in mc_src:
        PASS(f"Intent phrase: '{phrase}'")
    else:
        FAIL(f"Intent phrase missing: '{phrase}'")

# Check voice feedback
if "Sure, let me read that for you" in mc_src:
    PASS("TTS acknowledgment on read start")
else:
    FAIL("TTS acknowledgment missing")

if "I don't see any text selected" in mc_src:
    PASS("TTS failure feedback present")
else:
    FAIL("TTS failure feedback missing")

# ═══ 14. BROWSER EXTENSION ══════════════════════════════
section("14. BROWSER EXTENSION")
ext_dir = Path("browser_extension")
for fn in ["manifest.json", "background.js", "content.js"]:
    if (ext_dir / fn).exists():
        PASS(f"Extension: {fn}")
    else:
        FAIL(f"Extension missing: {fn}")

mf = json.loads((ext_dir / "manifest.json").read_text())
if "Annabeth" in mf.get("name", ""):
    PASS("Manifest name contains Annabeth")
else:
    FAIL("Manifest name", mf.get("name"))
if mf.get("manifest_version") == 3:
    PASS("Manifest v3")
else:
    FAIL("Manifest version", mf.get("manifest_version"))

# ═══ 15. READ-ALOUD: Win32 helpers ══════════════════════
section("15. READ-ALOUD: Win32 helpers")
from server.process.read_aloud.text_capture import (
    _IS_WIN, _send_ctrl_c, _get_foreground_hwnd, register_companion_hwnd,
)
if _IS_WIN:
    PASS("_IS_WIN=True on Windows")
else:
    FAIL("_IS_WIN should be True")

hwnd = _get_foreground_hwnd()
if isinstance(hwnd, int) and hwnd > 0:
    PASS(f"Foreground HWND: {hwnd}")
else:
    FAIL(f"Foreground HWND invalid: {hwnd}")

register_companion_hwnd(0)
PASS("register_companion_hwnd(0) no crash")

if callable(_send_ctrl_c):
    PASS("_send_ctrl_c callable")
else:
    FAIL("_send_ctrl_c not callable")

# ═══ 16. READ-ALOUD: full lifecycle ════════════════════
section("16. READ-ALOUD: full lifecycle")
from server.process.read_aloud.manager import ReadAloudManager, ReadAloudStatus
from server.process.read_aloud.text_capture import split_into_sentences, estimate_word_timings

mgr = ReadAloudManager()
mgr.state.start_reading("Alpha. Beta. Gamma.")
assert mgr.state.status == ReadAloudStatus.READING, "Should be READING"
PASS("start_reading -> READING")

mgr.state.pause()
assert mgr.state.status == ReadAloudStatus.FINISHING
PASS("pause -> FINISHING")

mgr.state.complete_pause()
assert mgr.state.status == ReadAloudStatus.PAUSED
PASS("complete_pause -> PAUSED")

ctx = mgr.get_qa_context()
assert "Alpha." in ctx or "reading" in ctx.lower()
PASS("Q&A context available while paused")

sent = mgr.state.resume()
assert mgr.state.status == ReadAloudStatus.READING
assert sent is not None
PASS("resume -> READING with sentence")

mgr.state.stop()
assert mgr.state.status == ReadAloudStatus.IDLE
PASS("stop -> IDLE")

# Sentence splitting
sents = split_into_sentences("Mr. Smith went to the store. He bought milk. Did he forget anything?")
assert len(sents) == 3, f"Expected 3 sentences, got {len(sents)}"
PASS("Sentence split: 3 sentences from abbreviation text")

# Word timings
timings = estimate_word_timings("Hello world foo bar", 4.0)
assert len(timings) == 4
assert abs(timings[-1][2] - 4.0) < 0.01
PASS("Word timings: 4 words, correct duration")

# ═══ 17. UNITY SCRIPTS: file count & declarations ══════
section("17. UNITY SCRIPTS: file inventory")
unity_dir = Path("unity/Scripts")
cs_files = list(unity_dir.rglob("*.cs"))
PASS(f"C# script files found: {len(cs_files)}")
if len(cs_files) >= 40:
    PASS(f"Enough C# files (>= 40): {len(cs_files)}")
else:
    FAIL(f"Only {len(cs_files)} C# files (expected >= 40)")

bad_cs = []
for cs in cs_files:
    src = cs.read_text(encoding="utf-8-sig", errors="replace")
    if not re.search(r"\b(class|struct|enum|interface)\s+\w+", src):
        bad_cs.append(cs.name)
if not bad_cs:
    PASS("All C# files contain class/struct/enum/interface")
else:
    FAIL("C# files without declarations", ", ".join(bad_cs))

# Check key types exist in C# files
key_types = [
    "CompanionManager", "WebSocketClient", "MessageHandler",
    "TransparentWindowController", "AvatarController",
    "HotkeyManager", "WindowSnapper", "SettingsManager",
    "WalkAnimationController", "DragPoseController",
    "PetDetectionController", "OccluderQuadManager",
    "DesktopAmbientProbe", "AlarmTimerManager",
    "CustomDanceLoader", "DancePlayerPanel",
    "DanceBlendshapeForwarder", "VmdPlayer",
    "DesktopLocomotionController",
]
all_cs_src = ""
for cs in cs_files:
    all_cs_src += cs.read_text(encoding="utf-8-sig", errors="replace") + "\n"

for t in key_types:
    if re.search(rf"\bclass\s+{t}\b", all_cs_src):
        PASS(f"C# type: {t}")
    else:
        FAIL(f"C# type missing: {t}")

# ═══ 18. DEPENDENCIES: key packages ═════════════════════
section("18. DEPENDENCIES: key packages")
dep_modules = [
    ("pyperclip", "clipboard access"),
    ("yaml", "YAML config"),
    ("sounddevice", "audio I/O"),
    ("soundfile", "WAV read/write"),
    ("requests", "HTTP for TTS"),
    ("keyboard", "global hotkeys"),
    ("aiohttp", "WebSocket server"),
    ("ollama", "LLM backend"),
]
for mod, desc in dep_modules:
    try:
        importlib.import_module(mod)
        PASS(f"Package: {mod} ({desc})")
    except ImportError:
        FAIL(f"Package missing: {mod} ({desc})")

# ═══ 19. MAIN_CHAT: syntax check ═══════════════════════
section("19. MAIN_CHAT: syntax check")
mc_path = Path("server/main_chat.py")
mc_code = mc_path.read_text(encoding="utf-8")
try:
    compile(mc_code, str(mc_path), "exec")
    PASS("server/main_chat.py syntax OK")
except SyntaxError as e:
    FAIL("server/main_chat.py syntax error", str(e))

# Also check read_aloud modules
for fp in ["server/process/read_aloud/__init__.py",
           "server/process/read_aloud/manager.py",
           "server/process/read_aloud/text_capture.py"]:
    p = Path(fp)
    code = p.read_text(encoding="utf-8")
    try:
        compile(code, fp, "exec")
        PASS(f"Syntax OK: {fp}")
    except SyntaxError as e:
        FAIL(f"Syntax error: {fp}", str(e))

# ═══ 20. MODEL FILES ═══════════════════════════════════
section("20. MODEL & REFERENCE FILES")
model_paths = [
    ("models/vrm/claire_avatar.vrm", "VRM avatar model"),
    ("gpt_sovits_models/G2PWModel/g2pW.onnx", "G2PW ONNX model"),
]
for fp, desc in model_paths:
    p = Path(fp)
    if p.exists():
        size_mb = p.stat().st_size / (1024 * 1024)
        PASS(f"{desc}: {fp}", f"{size_mb:.1f} MB")
    else:
        FAIL(f"{desc} missing: {fp}")

# Speaker profiles
sp_dir = Path("speaker_profiles")
if sp_dir.exists():
    profiles = list(sp_dir.glob("*.npy"))
    PASS(f"Speaker profiles: {len(profiles)} found", ", ".join(p.stem for p in profiles))
else:
    FAIL("speaker_profiles directory missing")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print(f"\n{SEP}")
print(f"  FULL SYSTEM INTEGRITY: {pass_c} PASS / {fail_c} FAIL  (total {pass_c + fail_c})")
print(SEP)

if fail_c:
    print("\nFailed tests:")
    for r in results:
        if "[FAIL]" in r:
            print(f"  {r.strip()}")

sys.exit(fail_c)
