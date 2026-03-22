using UnityEngine;
using Annabeth.Core;

namespace Annabeth.UI
{
    /// <summary>
    /// On-screen debug overlay showing pipeline status:
    /// WebSocket connection, current mode, ASR input, LLM response, speaking state.
    /// Toggle with F1 key. Uses OnGUI (no Canvas needed).
    /// </summary>
    public class DebugOverlay : MonoBehaviour
    {
        [SerializeField] private WebSocketClient webSocketClient;
        [SerializeField] private MessageHandler messageHandler;

        private bool _visible = true;
        private string _status = "Initializing...";
        private string _userText = "";
        private string _responseText = "";
        private string _lastSpeakText = "";
        private bool _isSpeaking;
        private CompanionMode _currentMode = CompanionMode.Idle;

        // Cached styles
        private GUIStyle _boxStyle;
        private GUIStyle _labelStyle;
        private GUIStyle _headerStyle;
        private bool _stylesInit;

        private void Start()
        {
            if (messageHandler == null)
                messageHandler = FindFirstObjectByType<MessageHandler>();
            if (webSocketClient == null)
                webSocketClient = FindFirstObjectByType<WebSocketClient>();

            if (messageHandler != null)
            {
                messageHandler.OnDebugStatus += HandleDebugStatus;
                messageHandler.OnSpeakStart += HandleSpeakStart;
                messageHandler.OnSpeakEnd += HandleSpeakEnd;
                messageHandler.OnModeChange += HandleModeChange;
            }
        }

        private void OnDestroy()
        {
            if (messageHandler != null)
            {
                messageHandler.OnDebugStatus -= HandleDebugStatus;
                messageHandler.OnSpeakStart -= HandleSpeakStart;
                messageHandler.OnSpeakEnd -= HandleSpeakEnd;
                messageHandler.OnModeChange -= HandleModeChange;
            }
        }

        private void Update()
        {
            if (UnityEngine.Input.GetKeyDown(KeyCode.F1))
                _visible = !_visible;
        }

        private void HandleDebugStatus(string status, string userText, string responseText)
        {
            _status = status;
            if (!string.IsNullOrEmpty(userText))
                _userText = userText;
            if (!string.IsNullOrEmpty(responseText))
                _responseText = responseText;
        }

        private void HandleSpeakStart(string text)
        {
            _isSpeaking = true;
            if (!string.IsNullOrEmpty(text))
                _lastSpeakText = text;
        }

        private void HandleSpeakEnd()
        {
            _isSpeaking = false;
        }

        private void HandleModeChange(CompanionMode mode)
        {
            _currentMode = mode;
        }

        private void InitStyles()
        {
            _boxStyle = new GUIStyle(GUI.skin.box)
            {
                normal = { background = MakeTex(2, 2, new Color(0f, 0f, 0f, 0.75f)) }
            };

            _labelStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 14,
                normal = { textColor = Color.white },
                wordWrap = true
            };

            _headerStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 16,
                fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.3f, 0.9f, 1f) }
            };

            _stylesInit = true;
        }

        private void OnGUI()
        {
            if (!_visible) return;
            if (!_stylesInit) InitStyles();

            float w = 420f;
            float x = Screen.width - w - 10f;
            float y = 10f;

            GUILayout.BeginArea(new Rect(x, y, w, Screen.height - 20f), _boxStyle);
            GUILayout.Space(4);

            GUILayout.Label("DEBUG OVERLAY (F1 to hide)", _headerStyle);
            GUILayout.Space(4);

            // Connection
            bool connected = webSocketClient != null && webSocketClient.IsConnected;
            string connColor = connected ? "<color=#00ff00>CONNECTED</color>" : "<color=#ff0000>DISCONNECTED</color>";
            DrawRichLabel($"WebSocket: {connColor}");

            // Mode
            DrawLabel($"Mode: {_currentMode}");

            // Speaking
            string speakColor = _isSpeaking ? "<color=#ffaa00>SPEAKING</color>" : "<color=#888888>Silent</color>";
            DrawRichLabel($"TTS: {speakColor}");

            GUILayout.Space(6);

            // Status from Python
            DrawLabel($"Status: {_status}");

            GUILayout.Space(6);

            // User input
            DrawLabel("YOU:");
            string userDisplay = string.IsNullOrEmpty(_userText) ? "(waiting for speech...)" : _userText;
            DrawLabel($"  {Truncate(userDisplay, 200)}");

            GUILayout.Space(4);

            // Response
            DrawLabel("ANNABETH:");
            string respDisplay = string.IsNullOrEmpty(_responseText) ? "(no response yet)" : _responseText;
            DrawLabel($"  {Truncate(respDisplay, 300)}");

            GUILayout.Space(4);
            GUILayout.EndArea();
        }

        private void DrawLabel(string text)
        {
            GUILayout.Label(text, _labelStyle);
        }

        private void DrawRichLabel(string text)
        {
            var style = new GUIStyle(_labelStyle) { richText = true };
            GUILayout.Label(text, style);
        }

        private static string Truncate(string s, int maxLen)
        {
            if (s == null) return "";
            return s.Length <= maxLen ? s : s.Substring(0, maxLen) + "...";
        }

        private static Texture2D MakeTex(int w, int h, Color col)
        {
            var pix = new Color[w * h];
            for (int i = 0; i < pix.Length; i++) pix[i] = col;
            var tex = new Texture2D(w, h);
            tex.SetPixels(pix);
            tex.Apply();
            return tex;
        }
    }
}
