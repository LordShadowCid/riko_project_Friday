using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

// Force recompile after UniVRM packages resolved
namespace Annabeth.Core
{
    /// <summary>
    /// WebSocket client for communicating with the Python avatar server.
    /// Replaces the old TCP SocketClient — uses ws://host:port/ws (aiohttp).
    /// Thread-safe: network I/O runs on background tasks, messages are
    /// dispatched on the Unity main thread via a ConcurrentQueue.
    /// </summary>
    public class WebSocketClient : MonoBehaviour
    {
        [Header("Connection Settings")]
        [SerializeField] private string host = "127.0.0.1";
        [SerializeField] private int port = 8765;
        [SerializeField] private string path = "/ws";
        [SerializeField] private float reconnectDelay = 3f;
        [SerializeField] private bool connectOnStart = true;

        private ClientWebSocket _ws;
        private CancellationTokenSource _cts;
        private Task _connectTask;
        private readonly ConcurrentQueue<string> _incomingMessages = new();
        private bool _connected;
        private bool _shouldReconnect = true;

        /// <summary>
        /// Fired on the main thread when a message arrives.
        /// Parameters: message type, data dictionary.
        /// </summary>
        public event Action<string, Dictionary<string, object>> OnMessageReceived;

        /// <summary>
        /// Fired when connection state changes.
        /// </summary>
        public event Action<bool> OnConnectionChanged;

        public bool IsConnected => _connected;

        private string WsUrl => $"ws://{host}:{port}{path}";

        private void Start()
        {
            if (connectOnStart) Connect();
        }

        private void Update()
        {
            // Drain incoming queue on the main thread
            while (_incomingMessages.TryDequeue(out string raw))
            {
                try
                {
                    ParseAndDispatch(raw);
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[WebSocketClient] Parse error: {e.Message}");
                }
            }
        }

        private void OnDestroy()
        {
            _shouldReconnect = false;
            Disconnect();
        }

        private void OnApplicationQuit()
        {
            _shouldReconnect = false;
            Disconnect();
        }

        // ── public API ──────────────────────────────────────────────

        public void Connect()
        {
            _shouldReconnect = true;
            if (_connectTask == null || _connectTask.IsCompleted)
                _connectTask = ConnectLoop();
        }

        public void Disconnect()
        {
            _shouldReconnect = false;
            _cts?.Cancel();
            _cts?.Dispose();
            _cts = null;

            if (_ws != null)
            {
                _ = CloseAndDisposeSocketAsync(_ws);
                _ws = null;
            }
        }

        /// <summary>
        /// Send a typed message with optional data payload.
        /// Safe to call from any thread.
        /// </summary>
        public void Send(string type, Dictionary<string, object> data = null)
        {
            if (_ws == null || _ws.State != WebSocketState.Open) return;

            var dict = data != null
                ? new Dictionary<string, object>(data)
                : new Dictionary<string, object>();
            dict["type"] = type;

            string json = JsonUtility_Serialize(dict);
            byte[] bytes = Encoding.UTF8.GetBytes(json);

            try
            {
                // Fire-and-forget send (buffer is small, OK for control messages)
                _ = _ws.SendAsync(
                    new ArraySegment<byte>(bytes),
                    WebSocketMessageType.Text,
                    endOfMessage: true,
                    cancellationToken: _cts?.Token ?? CancellationToken.None);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[WebSocketClient] Send error: {e.Message}");
            }
        }

        // ── connection loop ─────────────────────────────────────────

        private async Task ConnectLoop()
        {
            while (_shouldReconnect)
            {
                try
                {
                    _cts?.Cancel();
                    _cts?.Dispose();
                    _cts = new CancellationTokenSource();
                    var connectionToken = _cts.Token;

                    if (_ws != null)
                        await CloseAndDisposeSocketAsync(_ws);
                    _ws = new ClientWebSocket();

                    Debug.Log($"[WebSocketClient] Connecting to {WsUrl}...");
                    await _ws.ConnectAsync(new Uri(WsUrl), connectionToken);

                    _connected = true;
                    OnConnectionChanged?.Invoke(true);
                    Debug.Log("[WebSocketClient] Connected!");

                    await ReceiveLoop(connectionToken);
                }
                catch (OperationCanceledException)
                {
                    // Normal shutdown
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[WebSocketClient] Connection error: {e.Message}");
                }
                finally
                {
                    bool wasConnected = _connected;
                    _connected = false;
                    if (wasConnected)
                        OnConnectionChanged?.Invoke(false);
                }

                if (!_shouldReconnect) break;

                Debug.Log($"[WebSocketClient] Reconnecting in {reconnectDelay}s...");
                try
                {
                    await Task.Delay(TimeSpan.FromSeconds(reconnectDelay), _cts?.Token ?? CancellationToken.None);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
            }
        }

        private static async Task CloseAndDisposeSocketAsync(ClientWebSocket socket)
        {
            try
            {
                if (socket.State == WebSocketState.Open || socket.State == WebSocketState.CloseReceived)
                {
                    await socket.CloseAsync(
                        WebSocketCloseStatus.NormalClosure,
                        "Client disconnect",
                        CancellationToken.None);
                }
            }
            catch
            {
                // Best-effort shutdown during play mode exit or reconnect.
            }
            finally
            {
                socket.Dispose();
            }
        }

        private async Task ReceiveLoop(CancellationToken ct)
        {
            var buffer = new byte[8192];

            while (_ws.State == WebSocketState.Open && !ct.IsCancellationRequested)
            {
                var sb = new StringBuilder();
                WebSocketReceiveResult result;

                do
                {
                    result = await _ws.ReceiveAsync(new ArraySegment<byte>(buffer), ct);

                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        Debug.Log("[WebSocketClient] Server closed connection.");
                        return;
                    }

                    sb.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
                }
                while (!result.EndOfMessage);

                _incomingMessages.Enqueue(sb.ToString());
            }
        }

        // ── message parsing ─────────────────────────────────────────

        private void ParseAndDispatch(string json)
        {
            // Lightweight JSON parsing without Newtonsoft dependency.
            // Unity's JsonUtility doesn't handle Dictionary, so we use
            // a minimal recursive parser for the flat structures the
            // Python server actually sends.
            var dict = MiniJson.Deserialize(json) as Dictionary<string, object>;
            if (dict == null) return;

            string type = dict.TryGetValue("type", out object t) ? t?.ToString() : null;
            if (string.IsNullOrEmpty(type)) return;

            OnMessageReceived?.Invoke(type, dict);
        }

        /// <summary>
        /// Serialize a string→object dictionary to JSON.
        /// Handles string, bool, int, long, float, double.
        /// </summary>
        private static string JsonUtility_Serialize(Dictionary<string, object> dict)
        {
            return MiniJson.Serialize(dict);
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Minimal JSON serializer/deserializer (no external dependency).
    // Handles the flat JSON objects the Python avatar server sends.
    // ─────────────────────────────────────────────────────────────────
    public static class MiniJson
    {
        public static object Deserialize(string json)
        {
            if (string.IsNullOrEmpty(json)) return null;
            int index = 0;
            return ParseValue(json, ref index);
        }

        public static string Serialize(object obj)
        {
            var sb = new StringBuilder();
            SerializeValue(obj, sb);
            return sb.ToString();
        }

        // ── parse ───────────────────────────────────────────────────

        private static object ParseValue(string json, ref int i)
        {
            SkipWhitespace(json, ref i);
            if (i >= json.Length) return null;

            char c = json[i];
            if (c == '{') return ParseObject(json, ref i);
            if (c == '[') return ParseArray(json, ref i);
            if (c == '"') return ParseString(json, ref i);
            if (c == 't' || c == 'f') return ParseBool(json, ref i);
            if (c == 'n') { i += 4; return null; }
            return ParseNumber(json, ref i);
        }

        private static Dictionary<string, object> ParseObject(string json, ref int i)
        {
            var dict = new Dictionary<string, object>();
            i++; // skip '{'
            SkipWhitespace(json, ref i);
            if (json[i] == '}') { i++; return dict; }

            while (i < json.Length)
            {
                SkipWhitespace(json, ref i);
                string key = ParseString(json, ref i);
                SkipWhitespace(json, ref i);
                i++; // skip ':'
                object val = ParseValue(json, ref i);
                dict[key] = val;
                SkipWhitespace(json, ref i);
                if (i < json.Length && json[i] == ',') { i++; continue; }
                break;
            }
            if (i < json.Length && json[i] == '}') i++;
            return dict;
        }

        private static List<object> ParseArray(string json, ref int i)
        {
            var list = new List<object>();
            i++; // skip '['
            SkipWhitespace(json, ref i);
            if (json[i] == ']') { i++; return list; }

            while (i < json.Length)
            {
                list.Add(ParseValue(json, ref i));
                SkipWhitespace(json, ref i);
                if (i < json.Length && json[i] == ',') { i++; continue; }
                break;
            }
            if (i < json.Length && json[i] == ']') i++;
            return list;
        }

        private static string ParseString(string json, ref int i)
        {
            i++; // skip opening quote
            var sb = new StringBuilder();
            while (i < json.Length)
            {
                char c = json[i++];
                if (c == '"') return sb.ToString();
                if (c == '\\' && i < json.Length)
                {
                    char next = json[i++];
                    switch (next)
                    {
                        case '"': sb.Append('"'); break;
                        case '\\': sb.Append('\\'); break;
                        case '/': sb.Append('/'); break;
                        case 'n': sb.Append('\n'); break;
                        case 'r': sb.Append('\r'); break;
                        case 't': sb.Append('\t'); break;
                        case 'b': sb.Append('\b'); break;
                        case 'f': sb.Append('\f'); break;
                        case 'u':
                            if (i + 4 <= json.Length)
                            {
                                string hex = json.Substring(i, 4);
                                sb.Append((char)Convert.ToInt32(hex, 16));
                                i += 4;
                            }
                            break;
                        default: sb.Append(next); break;
                    }
                }
                else
                {
                    sb.Append(c);
                }
            }
            return sb.ToString();
        }

        private static object ParseNumber(string json, ref int i)
        {
            int start = i;
            bool isFloat = false;
            if (i < json.Length && json[i] == '-') i++;
            while (i < json.Length && char.IsDigit(json[i])) i++;
            if (i < json.Length && json[i] == '.') { isFloat = true; i++; while (i < json.Length && char.IsDigit(json[i])) i++; }
            if (i < json.Length && (json[i] == 'e' || json[i] == 'E')) { isFloat = true; i++; if (i < json.Length && (json[i] == '+' || json[i] == '-')) i++; while (i < json.Length && char.IsDigit(json[i])) i++; }

            string numStr = json.Substring(start, i - start);
            if (isFloat)
                return double.TryParse(numStr, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double d) ? (object)d : 0.0;
            return long.TryParse(numStr, out long l) ? (object)l : 0L;
        }

        private static bool ParseBool(string json, ref int i)
        {
            if (json[i] == 't') { i += 4; return true; }
            i += 5; return false;
        }

        private static void SkipWhitespace(string json, ref int i)
        {
            while (i < json.Length && char.IsWhiteSpace(json[i])) i++;
        }

        // ── serialize ───────────────────────────────────────────────

        private static void SerializeValue(object obj, StringBuilder sb)
        {
            if (obj == null) { sb.Append("null"); return; }
            if (obj is string s) { SerializeString(s, sb); return; }
            if (obj is bool b) { sb.Append(b ? "true" : "false"); return; }
            if (obj is int i32) { sb.Append(i32); return; }
            if (obj is long i64) { sb.Append(i64); return; }
            if (obj is float f) { sb.Append(f.ToString(System.Globalization.CultureInfo.InvariantCulture)); return; }
            if (obj is double d) { sb.Append(d.ToString(System.Globalization.CultureInfo.InvariantCulture)); return; }
            if (obj is Dictionary<string, object> dict) { SerializeObject(dict, sb); return; }
            if (obj is System.Collections.IList list) { SerializeArray(list, sb); return; }
            SerializeString(obj.ToString(), sb);
        }

        private static void SerializeObject(Dictionary<string, object> dict, StringBuilder sb)
        {
            sb.Append('{');
            bool first = true;
            foreach (var kv in dict)
            {
                if (!first) sb.Append(',');
                first = false;
                SerializeString(kv.Key, sb);
                sb.Append(':');
                SerializeValue(kv.Value, sb);
            }
            sb.Append('}');
        }

        private static void SerializeArray(System.Collections.IList list, StringBuilder sb)
        {
            sb.Append('[');
            for (int idx = 0; idx < list.Count; idx++)
            {
                if (idx > 0) sb.Append(',');
                SerializeValue(list[idx], sb);
            }
            sb.Append(']');
        }

        private static void SerializeString(string s, StringBuilder sb)
        {
            sb.Append('"');
            foreach (char c in s)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        if (c < 0x20)
                            sb.AppendFormat("\\u{0:x4}", (int)c);
                        else
                            sb.Append(c);
                        break;
                }
            }
            sb.Append('"');
        }
    }
}
