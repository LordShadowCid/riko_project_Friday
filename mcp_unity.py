"""Unity MCP helper: execute C# scripts via SSE transport."""
import urllib.request
import json
import time
import sys

BASE = "http://localhost:53002"

def run_csharp(code):
    """Open SSE session, execute C# code, return result."""
    req = urllib.request.Request(f"{BASE}/sse", headers={"Accept": "text/event-stream"})
    sse_resp = urllib.request.urlopen(req, timeout=30)

    data = b""
    endpoint = None
    while True:
        chunk = sse_resp.read(1)
        if not chunk:
            break
        data += chunk
        if b"\n\n" in data:
            text = data.decode()
            if "endpoint" in text:
                for line in text.strip().split("\n"):
                    if line.startswith("data:"):
                        endpoint = line[5:].strip()
                        break
                break
            data = b""

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "script-execute",
            "arguments": {"csharpCode": code}
        }
    }

    msg_data = json.dumps(payload).encode()
    msg_req = urllib.request.Request(f"{BASE}{endpoint}", data=msg_data,
                                     headers={"Content-Type": "application/json"})
    urllib.request.urlopen(msg_req, timeout=30)

    data = b""
    start = time.time()
    while time.time() - start < 25:
        chunk = sse_resp.read(1)
        if not chunk:
            break
        data += chunk
        if b"\n\n" in data:
            text = data.decode()
            if "result" in text or "error" in text:
                sse_resp.close()
                # Parse the JSON result
                for line in text.strip().split("\n"):
                    if line.startswith("data:"):
                        j = json.loads(line[5:])
                        if "result" in j:
                            content = j["result"].get("content", [])
                            if content:
                                inner = content[0].get("text", "")
                                try:
                                    parsed = json.loads(inner)
                                    return parsed.get("result", {}).get("value", inner)
                                except:
                                    return inner
                            sc = j["result"].get("structuredContent", {})
                            if sc:
                                return sc.get("result", {}).get("value", str(sc))
                return text
            data = b""

    sse_resp.close()
    return "TIMEOUT"

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"

    if action == "play":
        result = run_csharp(r"""
using UnityEngine;
using UnityEditor;
public class Script { public static string Main() {
    if (EditorApplication.isPlaying) return "Already in Play mode";
    EditorApplication.isPlaying = true;
    return "Entering Play mode...";
}}""")
        print(result)

    elif action == "stop":
        result = run_csharp(r"""
using UnityEngine;
using UnityEditor;
public class Script { public static string Main() {
    if (!EditorApplication.isPlaying) return "Not in Play mode";
    EditorApplication.isPlaying = false;
    return "Exiting Play mode";
}}""")
        print(result)

    elif action == "status":
        result = run_csharp(r"""
using UnityEngine;
using UnityEditor;
using Annabeth;
using Annabeth.Avatar;
using Annabeth.Interaction;

public class Script { public static string Main() {
    var sb = new System.Text.StringBuilder();
    sb.AppendLine($"IsPlaying: {EditorApplication.isPlaying}");
    sb.AppendLine($"IsCompiling: {EditorApplication.isCompiling}");

    var cm = Object.FindFirstObjectByType<CompanionManager>();
    sb.AppendLine($"CompanionManager: {(cm != null ? "FOUND" : "NOT FOUND")}");

    if (cm != null) {
        var abc = cm.GetComponent<AnimationBlendController>();
        sb.AppendLine($"AnimationBlendController: {(abc != null ? "ACTIVE" : "MISSING")}");
    }

    return sb.ToString();
}}""")
        print(result)

    elif action == "errors":
        result = run_csharp(r"""
using UnityEngine;
using UnityEditor;

public class Script { public static string Main() {
    var sb = new System.Text.StringBuilder();
    var logType = typeof(Editor).Assembly.GetType("UnityEditor.LogEntries");
    var startGet = logType.GetMethod("StartGettingEntries", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public);
    var getCount = logType.GetMethod("GetCount", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public);
    var getEntry = logType.GetMethod("GetEntryInternal", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public);
    var endGet = logType.GetMethod("EndGettingEntries", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public);
    var entryType = typeof(Editor).Assembly.GetType("UnityEditor.LogEntry");

    startGet.Invoke(null, null);
    int count = (int)getCount.Invoke(null, null);
    int errCount = 0;
    int start = System.Math.Max(0, count - 30);

    for (int i = start; i < count; i++) {
        var entry = System.Activator.CreateInstance(entryType);
        getEntry.Invoke(null, new object[] { i, entry });
        var msg = (string)entryType.GetField("message").GetValue(entry);
        var mode = (int)entryType.GetField("mode").GetValue(entry);
        bool isErr = (mode & 1) != 0 || (mode & 2) != 0 || (mode & 8) != 0;
        if (isErr) {
            errCount++;
            if (msg.Length > 200) msg = msg.Substring(0, 200) + "...";
            sb.AppendLine($"[ERR] {msg}");
        }
    }

    endGet.Invoke(null, null);
    if (errCount == 0) sb.AppendLine($"No errors found in {count} log entries");
    else sb.AppendLine($"{errCount} errors in {count} log entries");
    return sb.ToString();
}}""")
        print(result)

    elif action == "save":
        result = run_csharp(r"""
using UnityEditor;
using UnityEditor.SceneManagement;
public class Script { public static string Main() {
    EditorSceneManager.SaveOpenScenes();
    return "Scene saved";
}}""")
        print(result)
