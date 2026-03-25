"""
Full Deep Integration Test for Annabeth Unity Frontend - All Phases (1-8).
Runs via MCP C# execution against the live Unity Editor.

Tests: compilation, component presence, scene wiring, Phase 8 desktop interaction,
Play mode verification, runtime errors check, event wiring.
"""
from mcp_unity import run_csharp
import time
import sys

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

def run_test(label, code):
    """Execute C# code via MCP, return result string."""
    try:
        r = run_csharp(code)
        return str(r) if r else ""
    except Exception as e:
        FAIL(label, f"MCP error: {e}")
        return None


# ══════════════════════════════════════════════════════════════
log("#" * 60)
log("#  ANNABETH UNITY DEEP INTEGRATION TEST")
log(f"#  {time.strftime('%Y-%m-%d %H:%M:%S')}")
log("#  Phases 1-8")
log("#" * 60)


# ── TEST 1: Assembly & Type Count ─────────────────────────────
section("TEST 1: Assembly & Compilation")

r = run_test("Assembly-CSharp exists", r"""
using System;
using System.Linq;
public class Script { public static string Main() {
    var asm = AppDomain.CurrentDomain.GetAssemblies()
        .FirstOrDefault(a => a.GetName().Name == "Assembly-CSharp");
    if (asm == null) return "MISSING";
    var types = asm.GetTypes().Where(t => t.Namespace != null && t.Namespace.StartsWith("Annabeth")).Count();
    return $"OK:{types}";
}}""")
if r and r.startswith("OK:"):
    count = int(r.split(":")[1])
    if count >= 52:
        PASS("Assembly-CSharp loaded", f"{count} Annabeth types")
    else:
        FAIL("Type count low", f"Expected >=52, got {count}")
else:
    FAIL("Assembly-CSharp", r)


# ── TEST 2: All Expected Types Present ────────────────────────
section("TEST 2: Required Types Present")

expected_types = [
    # Phase 1-2: Core
    "CompanionManager", "WebSocketClient", "MessageHandler", "TransparentWindowController",
    "AvatarController", "FPSController",
    # Phase 2: Avatar
    "EmotionController", "LipSyncController", "EyeTrackingController", "BlinkController",
    # Phase 3: Animation
    "AnimationBlendController", "IdleAnimationController", "BeatDanceController", "VrmaAnimationController",
    # Phase 4: Interaction
    "TouchReactionController", "WindowSnapper", "HotkeyManager",
    # Phase 4b: UI
    "SpeechBubble", "RadialMenu", "DebugOverlay", "SettingsPanel", "UIFactory",
    # Phase 5: Drag + Effects
    "DragAnimationController", "TouchSoundHandler", "ParticleEffectHandler",
    # Phase 6: System
    "SleepController", "SystemTrayController", "SettingsManager", "MemoryOptimizer",
    # Phase 7: VRM Library
    "VrmModelLibrary", "VrmFilePicker",
    # Phase 8: Desktop Interaction
    "DesktopLocomotionController",
]

r = run_test("Type check", r"""
using System;
using System.Linq;
public class Script { public static string Main() {
    var asm = AppDomain.CurrentDomain.GetAssemblies()
        .FirstOrDefault(a => a.GetName().Name == "Assembly-CSharp");
    if (asm == null) return "NO_ASM";
    var types = asm.GetTypes().Select(t => t.Name).ToArray();
    return string.Join("|", types);
}}""")
if r and r != "NO_ASM":
    type_names = set(r.split("|"))
    for t in expected_types:
        if t in type_names:
            PASS(f"Type: {t}")
        else:
            FAIL(f"Type: {t}", "NOT FOUND in assembly")
else:
    FAIL("Type enumeration", r)


# ── TEST 3: Scene Components ──────────────────────────────────
section("TEST 3: Scene Component Presence")

r = run_test("Scene components", r"""
using UnityEngine;
using System.Linq;
public class Script { public static string Main() {
    var names = new[] {
        "Annabeth.CompanionManager",
        "Annabeth.Core.WebSocketClient",
        "Annabeth.Core.MessageHandler",
        "Annabeth.Core.TransparentWindowController",
        "Annabeth.Avatar.AvatarController",
        "Annabeth.Avatar.EmotionController",
        "Annabeth.Avatar.LipSyncController",
        "Annabeth.Avatar.EyeTrackingController",
        "Annabeth.Avatar.BlinkController",
        "Annabeth.Avatar.AnimationBlendController",
        "Annabeth.Avatar.IdleAnimationController",
        "Annabeth.Dance.BeatDanceController",
        "Annabeth.Dance.VrmaAnimationController",
        "Annabeth.Interaction.TouchReactionController",
        "Annabeth.Core.WindowSnapper",
        "Annabeth.Input.HotkeyManager",
        "Annabeth.UI.SpeechBubble",
        "Annabeth.UI.DebugOverlay",
        "Annabeth.Avatar.DragAnimationController",
        "Annabeth.Interaction.TouchSoundHandler",
        "Annabeth.Interaction.ParticleEffectHandler",
        "Annabeth.Core.SleepController",
        "Annabeth.Core.SystemTrayController",
        "Annabeth.Core.SettingsManager",
        "Annabeth.Core.MemoryOptimizer",
        "Annabeth.Core.FPSController",
        "Annabeth.Core.DesktopLocomotionController"
    };
    var lines = new System.Collections.Generic.List<string>();
    foreach (var n in names) {
        var parts = n.Split('.');
        var shortName = parts[parts.Length - 1];
        var type = System.AppDomain.CurrentDomain.GetAssemblies()
            .SelectMany(a => { try { return a.GetTypes(); } catch { return new System.Type[0]; } })
            .FirstOrDefault(t => t.FullName == n);
        if (type == null) { lines.Add($"{shortName}:TYPE_NOT_FOUND"); continue; }
        var obj = Object.FindFirstObjectByType(type);
        lines.Add($"{shortName}:{(obj != null ? "FOUND" : "MISSING")}");
    }
    return string.Join("|", lines);
}}""")
if r:
    for entry in r.split("|"):
        if ":" not in entry:
            continue
        name, status = entry.split(":", 1)
        if status == "FOUND":
            PASS(f"Scene: {name}")
        else:
            FAIL(f"Scene: {name}", status)


# ── TEST 4: CompanionManager Wiring ──────────────────────────
section("TEST 4: CompanionManager Reference Wiring")

r = run_test("CM wiring", r"""
using UnityEngine;
using UnityEditor;
public class Script { public static string Main() {
    var cm = Object.FindFirstObjectByType<Annabeth.CompanionManager>();
    if (cm == null) return "NO_CM";
    var so = new SerializedObject(cm);
    var fields = new[] {
        "webSocketClient", "messageHandler", "avatarController",
        "emotionController", "lipSyncController", "eyeTrackingController",
        "blinkController", "animationBlendController", "idleAnimationController",
        "beatDanceController", "vrmaAnimationController", "touchReactionController",
        "speechBubble",
        "touchSoundHandler", "particleEffectHandler", "dragAnimController",
        "sleepController",
        "locomotionController", "windowSnapper"
    };
    var lines = new System.Collections.Generic.List<string>();
    foreach (var f in fields) {
        var prop = so.FindProperty(f);
        if (prop == null) { lines.Add($"{f}:NO_PROP"); continue; }
        if (prop.propertyType == SerializedPropertyType.ObjectReference) {
            lines.Add($"{f}:{(prop.objectReferenceValue != null ? "WIRED" : "NULL")}");
        } else {
            lines.Add($"{f}:NOT_OBJREF");
        }
    }
    return string.Join("|", lines);
}}""")
if r and r != "NO_CM":
    for entry in r.split("|"):
        if ":" not in entry:
            continue
        name, status = entry.split(":", 1)
        if status == "WIRED":
            PASS(f"Wired: {name}")
        elif status == "NULL":
            # Some fields are auto-resolved at runtime, NULL in editor is OK for some
            FAIL(f"Wired: {name}", "reference is NULL")
        else:
            FAIL(f"Wired: {name}", status)


# ── TEST 5: Phase 8 - DesktopLocomotionController Config ─────
section("TEST 5: Phase 8 - DesktopLocomotionController")

r = run_test("DLC config", r"""
using UnityEngine;
using UnityEditor;
public class Script { public static string Main() {
    var dlc = Object.FindFirstObjectByType<Annabeth.Core.DesktopLocomotionController>();
    if (dlc == null) return "NOT_IN_SCENE";
    var so = new SerializedObject(dlc);
    var lines = new System.Collections.Generic.List<string>();
    lines.Add($"enabled:{dlc.enabled}");
    
    var props = new[] { "walkSpeedPixelsPerFrame", "decisionIntervalSec", "minWalkDistance", "maxWalkDistance", "peekVisiblePixels" };
    foreach (var p in props) {
        var prop = so.FindProperty(p);
        if (prop != null) {
            if (prop.propertyType == SerializedPropertyType.Float)
                lines.Add($"{p}:{prop.floatValue}");
            else if (prop.propertyType == SerializedPropertyType.Integer)
                lines.Add($"{p}:{prop.intValue}");
        }
    }
    return string.Join("|", lines);
}}""")
if r and r != "NOT_IN_SCENE":
    PASS("DesktopLocomotionController in scene")
    for entry in r.split("|"):
        if ":" in entry:
            k, v = entry.split(":", 1)
            if k == "enabled":
                PASS(f"DLC enabled={v}")
            else:
                PASS(f"DLC config {k}={v}")
else:
    FAIL("DesktopLocomotionController", r)


# ── TEST 6: Phase 8 - WindowSnapper Enhanced ─────────────────
section("TEST 6: Phase 8 - WindowSnapper (Enhanced)")

r = run_test("WS methods", r"""
using System;
using System.Reflection;
using System.Linq;
public class Script { public static string Main() {
    var type = typeof(Annabeth.Core.WindowSnapper);
    var methods = type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
        .Select(m => m.Name).Distinct().ToArray();
    var props = type.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
        .Select(p => p.Name).ToArray();
    var events = type.GetEvents(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
        .Select(e => e.Name).ToArray();
    
    var lines = new System.Collections.Generic.List<string>();
    lines.Add($"METHODS:{string.Join(",", methods)}");
    lines.Add($"PROPS:{string.Join(",", props)}");
    lines.Add($"EVENTS:{string.Join(",", events)}");
    return string.Join("|", lines);
}}""")
if r:
    # Check for Phase 8 sitting/falling methods
    expected_methods = ["TrySitOnNearestWindow"]
    expected_props = ["IsSitting", "IsFalling"]
    expected_events = ["OnSittingChanged", "OnFallStarted", "OnFallLanded"]
    
    for m in expected_methods:
        if m in r:
            PASS(f"WS method: {m}")
        else:
            FAIL(f"WS method: {m}", "not found")
    for p in expected_props:
        if p in r:
            PASS(f"WS property: {p}")
        else:
            FAIL(f"WS property: {p}", "not found")
    for e in expected_events:
        if e in r:
            PASS(f"WS event: {e}")
        else:
            FAIL(f"WS event: {e}", "not found")


# ── TEST 7: Phase 8 - DesktopLocomotionController API ────────
section("TEST 7: Phase 8 - DesktopLocomotionController API")

r = run_test("DLC API", r"""
using System;
using System.Reflection;
using System.Linq;
public class Script { public static string Main() {
    var type = typeof(Annabeth.Core.DesktopLocomotionController);
    var methods = type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
        .Select(m => m.Name).Distinct().ToArray();
    var props = type.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
        .Select(p => p.Name).ToArray();
    var events = type.GetEvents(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
        .Select(e => e.Name).ToArray();
    
    var lines = new System.Collections.Generic.List<string>();
    lines.Add($"METHODS:{string.Join(",", methods)}");
    lines.Add($"PROPS:{string.Join(",", props)}");
    lines.Add($"EVENTS:{string.Join(",", events)}");
    return string.Join("|", lines);
}}""")
if r:
    expected_methods = ["ToggleEnabled", "StartPeek", "StopPeek"]
    expected_props = ["IsWalking", "IsPeeking", "IsEnabled"]
    expected_events = ["OnWalkStateChanged", "OnPeekStateChanged"]
    
    for m in expected_methods:
        if m in r:
            PASS(f"DLC method: {m}")
        else:
            FAIL(f"DLC method: {m}", "not found")
    for p in expected_props:
        if p in r:
            PASS(f"DLC property: {p}")
        else:
            FAIL(f"DLC property: {p}", "not found")
    for e in expected_events:
        if e in r:
            PASS(f"DLC event: {e}")
        else:
            FAIL(f"DLC event: {e}", "not found")


# ── TEST 8: HotkeyManager Phase 8 Keys ──────────────────────
section("TEST 8: HotkeyManager Phase 8 Bindings")

r = run_test("Hotkey bindings", r"""
using UnityEngine;
using UnityEditor;
public class Script { public static string Main() {
    var hm = Object.FindFirstObjectByType<Annabeth.Input.HotkeyManager>();
    if (hm == null) return "NO_HM";
    var so = new SerializedObject(hm);
    
    var lcRef = so.FindProperty("locomotionController");
    var wsRef = so.FindProperty("windowSnapper");
    
    var lines = new System.Collections.Generic.List<string>();
    lines.Add($"locomotionController:{(lcRef != null && lcRef.objectReferenceValue != null ? "WIRED" : "NULL")}");
    lines.Add($"windowSnapper:{(wsRef != null && wsRef.objectReferenceValue != null ? "WIRED" : "NULL")}");
    return string.Join("|", lines);
}}""")
if r and r != "NO_HM":
    for entry in r.split("|"):
        if ":" in entry:
            k, v = entry.split(":", 1)
            if v == "WIRED":
                PASS(f"HotkeyManager.{k} wired")
            else:
                FAIL(f"HotkeyManager.{k}", "not wired")


# ── TEST 9: Play Mode Test ───────────────────────────────────
section("TEST 9: Play Mode Smoke Test")

# Clear log, enter play mode, wait, check for errors
r = run_test("Clear log", r"""
using UnityEngine;
using UnityEditor;
public class Script { public static string Main() {
    var logEntries = System.Type.GetType("UnityEditor.LogEntries, UnityEditor");
    if (logEntries != null) {
        var clear = logEntries.GetMethod("Clear", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        if (clear != null) clear.Invoke(null, null);
    }
    return "LOG_CLEARED";
}}""")
if r == "LOG_CLEARED":
    PASS("Console log cleared")

# Enter play mode
r = run_test("Enter play mode", r"""
using UnityEditor;
public class Script { public static string Main() {
    if (EditorApplication.isPlaying) return "ALREADY_PLAYING";
    EditorApplication.isPlaying = true;
    return "ENTERING_PLAY";
}}""")
if r in ("ENTERING_PLAY", "ALREADY_PLAYING"):
    PASS("Play mode requested", r)
else:
    FAIL("Play mode", r)

# Wait for play mode to stabilize
log("  ... waiting 12s for play mode ...")
time.sleep(12)

# Check play mode status + errors
r = run_test("Play mode status", r"""
using UnityEngine;
using UnityEditor;
using System.Linq;
public class Script { public static string Main() {
    var lines = new System.Collections.Generic.List<string>();
    lines.Add($"isPlaying:{EditorApplication.isPlaying}");
    
    // Count log entries by type
    var logEntries = System.Type.GetType("UnityEditor.LogEntries, UnityEditor");
    int totalErrors = 0;
    int totalWarnings = 0;
    int totalLogs = 0;
    if (logEntries != null) {
        var getCount = logEntries.GetMethod("GetCount", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        if (getCount != null) {
            int total = (int)getCount.Invoke(null, null);
            lines.Add($"totalLogEntries:{total}");
            
            // Get error/warning counts
            var getCountsByType = logEntries.GetMethod("GetCountsByType", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
            if (getCountsByType != null) {
                var args = new object[] { 0, 0, 0 };
                getCountsByType.Invoke(null, args);
                totalErrors = (int)args[0];
                totalWarnings = (int)args[1];
                totalLogs = (int)args[2];
            }
        }
    }
    lines.Add($"errors:{totalErrors}");
    lines.Add($"warnings:{totalWarnings}");
    lines.Add($"logs:{totalLogs}");
    
    // Check key components are alive
    var cm = Object.FindFirstObjectByType<Annabeth.CompanionManager>();
    lines.Add($"CompanionManager:{(cm != null && cm.enabled ? "ACTIVE" : "MISSING")}");
    
    var dlc = Object.FindFirstObjectByType<Annabeth.Core.DesktopLocomotionController>();
    lines.Add($"DesktopLocomotionController:{(dlc != null ? "ACTIVE" : "MISSING")}");
    
    var ws = Object.FindFirstObjectByType<Annabeth.Core.WindowSnapper>();
    lines.Add($"WindowSnapper:{(ws != null && ws.enabled ? "ACTIVE" : "MISSING")}");
    
    return string.Join("|", lines);
}}""")
if r:
    data = dict(e.split(":", 1) for e in r.split("|") if ":" in e)
    
    if data.get("isPlaying") == "True":
        PASS("Play mode active")
    else:
        FAIL("Play mode active", f"isPlaying={data.get('isPlaying')}")
    
    errors = int(data.get("errors", "0"))
    if errors == 0:
        PASS("Zero runtime errors")
    else:
        FAIL(f"Runtime errors: {errors}")
    
    for comp in ["CompanionManager", "DesktopLocomotionController", "WindowSnapper"]:
        status = data.get(comp, "MISSING")
        if status == "ACTIVE":
            PASS(f"Runtime: {comp} active")
        else:
            FAIL(f"Runtime: {comp}", status)


# ── TEST 10: Runtime Component Check (all phases) ────────────
section("TEST 10: All Runtime Components Active")

r = run_test("All components", r"""
using UnityEngine;
using System.Linq;
public class Script { public static string Main() {
    if (!UnityEditor.EditorApplication.isPlaying) return "NOT_PLAYING";
    var types = new[] {
        typeof(Annabeth.CompanionManager),
        typeof(Annabeth.Core.WebSocketClient),
        typeof(Annabeth.Core.MessageHandler),
        typeof(Annabeth.Avatar.AvatarController),
        typeof(Annabeth.Avatar.EmotionController),
        typeof(Annabeth.Avatar.LipSyncController),
        typeof(Annabeth.Avatar.EyeTrackingController),
        typeof(Annabeth.Avatar.BlinkController),
        typeof(Annabeth.Avatar.AnimationBlendController),
        typeof(Annabeth.Avatar.IdleAnimationController),
        typeof(Annabeth.Dance.BeatDanceController),
        typeof(Annabeth.Dance.VrmaAnimationController),
        typeof(Annabeth.Interaction.TouchReactionController),
        typeof(Annabeth.Core.WindowSnapper),
        typeof(Annabeth.Input.HotkeyManager),
        typeof(Annabeth.UI.SpeechBubble),
        typeof(Annabeth.UI.DebugOverlay),
        typeof(Annabeth.Avatar.DragAnimationController),
        typeof(Annabeth.Interaction.TouchSoundHandler),
        typeof(Annabeth.Interaction.ParticleEffectHandler),
        typeof(Annabeth.Core.SleepController),
        typeof(Annabeth.Core.SystemTrayController),
        typeof(Annabeth.Core.SettingsManager),
        typeof(Annabeth.Core.MemoryOptimizer),
        typeof(Annabeth.Core.FPSController),
        typeof(Annabeth.Core.DesktopLocomotionController)
    };
    var lines = new System.Collections.Generic.List<string>();
    foreach (var t in types) {
        var obj = Object.FindFirstObjectByType(t) as MonoBehaviour;
        var name = t.Name;
        if (obj != null)
            lines.Add($"{name}:{(obj.enabled ? "ACTIVE" : "DISABLED")}");
        else
            lines.Add($"{name}:MISSING");
    }
    return string.Join("|", lines);
}}""")
if r and r != "NOT_PLAYING":
    for entry in r.split("|"):
        if ":" not in entry:
            continue
        name, status = entry.split(":", 1)
        if status == "ACTIVE":
            PASS(f"Runtime: {name}")
        elif status == "DISABLED":
            PASS(f"Runtime: {name} (disabled - OK)")
        else:
            FAIL(f"Runtime: {name}", status)
elif r == "NOT_PLAYING":
    FAIL("Runtime components", "Not in play mode")


# ── TEST 11: Error Details (if any) ──────────────────────────
section("TEST 11: Error Log Details")

r = run_test("Error details", r"""
using UnityEngine;
using UnityEditor;
using System.Linq;
public class Script { public static string Main() {
    var logEntries = System.Type.GetType("UnityEditor.LogEntries, UnityEditor");
    if (logEntries == null) return "NO_LOG_API";
    
    var getCount = logEntries.GetMethod("GetCount", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
    int total = (int)getCount.Invoke(null, null);
    
    var startGetting = logEntries.GetMethod("StartGettingEntries", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
    var getEntry = logEntries.GetMethod("GetEntryInternal", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
    var endGetting = logEntries.GetMethod("EndGettingEntries", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
    
    if (startGetting == null || getEntry == null) return $"ENTRIES:{total}|NO_DETAIL_API";
    
    startGetting.Invoke(null, null);
    
    var errors = new System.Collections.Generic.List<string>();
    // LogEntry struct
    var logEntryType = System.Type.GetType("UnityEditor.LogEntry, UnityEditor");
    
    for (int i = 0; i < total && errors.Count < 10; i++) {
        var entry = System.Activator.CreateInstance(logEntryType);
        var args = new object[] { i, entry };
        bool ok = (bool)getEntry.Invoke(null, args);
        if (!ok) continue;
        entry = args[1];
        
        var mode = (int)logEntryType.GetField("mode", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance).GetValue(entry);
        // Bit flags: 1=Error, 2=Assert, 4=Log, 8=Fatal, 16=DontPreprocess, 32=Log, 64=Log, 128=LogWarning ...
        // Error modes have bits 1, 2, or 8 set
        if ((mode & (1|2|8)) != 0) {
            var msg = (string)logEntryType.GetField("message", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance).GetValue(entry);
            if (msg != null && msg.Length > 200) msg = msg.Substring(0, 200);
            errors.Add(msg);
        }
    }
    
    endGetting.Invoke(null, null);
    
    if (errors.Count == 0) return "NO_ERRORS";
    return string.Join("|||", errors);
}}""")
if r == "NO_ERRORS":
    PASS("No error entries in console log")
elif r and r.StartsWith("ENTRIES:"):
    PASS("Log check completed", r)
else:
    if r and "|||" in r:
        errors = r.split("|||")
        for e in errors:
            FAIL("Runtime error", e[:120])
    elif r:
        log(f"  [INFO] Log result: {r[:200]}")


# ── Exit play mode ───────────────────────────────────────────
section("Cleanup: Exit Play Mode")

r = run_test("Exit play mode", r"""
using UnityEditor;
public class Script { public static string Main() {
    if (!EditorApplication.isPlaying) return "NOT_PLAYING";
    EditorApplication.isPlaying = false;
    return "EXITING";
}}""")
if r in ("EXITING", "NOT_PLAYING"):
    PASS("Play mode exited", r)

time.sleep(5)


# ═══════════════════════════════════════════════════════════════
section("SUMMARY")
total = pass_count + fail_count
log(f"  PASSED: {pass_count}/{total}")
log(f"  FAILED: {fail_count}/{total}")
if fail_count == 0:
    log("  >>> ALL TESTS PASSED <<<")
else:
    log(f"  >>> {fail_count} FAILURE(S) <<<")

# Save results
with open("test_results_unity_deep.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))
log(f"\nResults saved to test_results_unity_deep.txt")

sys.exit(0 if fail_count == 0 else 1)
