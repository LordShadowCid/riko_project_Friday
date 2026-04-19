using System.Collections.Generic;
using UnityEngine;

namespace Annabeth.Core
{
    /// <summary>
    /// Message types matching Python shared/config.py MessageType enum.
    /// Keep in sync with the Python backend.
    /// </summary>
    public static class MessageTypes
    {
        // Client → Server
        public const string MODE_CHANGE = "mode_change";
        public const string TOGGLE_SILENCE = "toggle_silence";
        public const string SET_SILENCE = "set_silence";
        public const string READ_PAUSE = "read_pause";
        public const string READ_RESUME = "read_resume";
        public const string RANDOM_PROMPT = "random_prompt";
        public const string SHUTDOWN = "shutdown";

        // Server → Client
        public const string SPEAK_START = "speak_start";
        public const string SPEAK_END = "speak_end";
        public const string EMOTION = "emotion";
        public const string AUDIO_ANALYSIS = "audio_analysis";
        public const string READ_HIGHLIGHT = "read_highlight";
        public const string READ_CLEAR = "read_clear";
        public const string DEBUG_STATUS = "debug_status";
        public const string FACE_EXPRESSION = "face_expression";
        public const string IDLE_THOUGHT = "idle_thought";
    }

    /// <summary>
    /// Companion operating modes.
    /// </summary>
    public enum CompanionMode
    {
        Active,
        Idle,
        Dance
    }

    /// <summary>
    /// Dance style options.
    /// </summary>
    public enum DanceStyle
    {
        None = 0,
        Procedural = 1,
        ShikanokoDance = 2
    }

    /// <summary>
    /// Routes messages from WebSocketClient to typed events.
    /// </summary>
    public class MessageHandler : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private WebSocketClient webSocketClient;

        // Events for specific message types
        public event System.Action<string> OnSpeakStart;
        public event System.Action OnSpeakEnd;
        public event System.Action<string> OnEmotionChange;
        public event System.Action<CompanionMode> OnModeChange;
        public event System.Action<bool> OnSilenceToggle;
        public event System.Action<float, float, float, bool> OnAudioAnalysis;
        public event System.Action<DanceStyle> OnDanceStyleChange;
        public event System.Action<string> OnReadHighlight;
        public event System.Action OnReadClear;
        public event System.Action<string> OnPlayAnimation;
        public event System.Action OnReadStart;
        public event System.Action OnReadPause;
        public event System.Action OnReadResume;
        public event System.Action OnReadEnd;
        /// <summary>
        /// Fired when debug_status message arrives: (status, userText, responseText)
        /// </summary>
        public event System.Action<string, string, string> OnDebugStatus;
        /// <summary>Fired on face_expression: (name, intensity)</summary>
        public event System.Action<string, float> OnFaceExpression;
        /// <summary>Fired on idle_thought: (text)</summary>
        public event System.Action<string> OnIdleThought;

        private void Start()
        {
            if (webSocketClient == null)
                webSocketClient = FindFirstObjectByType<WebSocketClient>();

            if (webSocketClient != null)
                webSocketClient.OnMessageReceived += HandleMessage;
            else
                Debug.LogError("[MessageHandler] WebSocketClient not found!");
        }

        private void OnDestroy()
        {
            if (webSocketClient != null)
                webSocketClient.OnMessageReceived -= HandleMessage;
        }

        private void HandleMessage(string type, Dictionary<string, object> data)
        {
            switch (type)
            {
                case MessageTypes.SPEAK_START:
                    string text = data.TryGetValue("text", out object textObj) ? textObj?.ToString() : "";
                    OnSpeakStart?.Invoke(text);
                    break;

                case MessageTypes.SPEAK_END:
                    OnSpeakEnd?.Invoke();
                    break;

                case MessageTypes.EMOTION:
                    string emotion = data.TryGetValue("emotion", out object emotionObj) ? emotionObj.ToString() : "neutral";
                    OnEmotionChange?.Invoke(emotion);
                    break;

                case MessageTypes.MODE_CHANGE:
                    if (data.TryGetValue("mode", out object modeObj))
                    {
                        string modeName = modeObj.ToString().ToLower();
                        switch (modeName)
                        {
                            case "active":
                                OnModeChange?.Invoke(CompanionMode.Active);
                                break;
                            case "idle":
                                OnModeChange?.Invoke(CompanionMode.Idle);
                                break;
                            case "dance":
                                OnModeChange?.Invoke(CompanionMode.Dance);
                                break;
                            case "dance_beat":
                                OnModeChange?.Invoke(CompanionMode.Dance);
                                OnDanceStyleChange?.Invoke(DanceStyle.Procedural);
                                break;
                            case "dance_full":
                                OnModeChange?.Invoke(CompanionMode.Dance);
                                OnDanceStyleChange?.Invoke(DanceStyle.ShikanokoDance);
                                break;
                            default:
                                OnModeChange?.Invoke(CompanionMode.Idle);
                                break;
                        }
                    }
                    break;

                case MessageTypes.SET_SILENCE:
                    bool silenced = data.TryGetValue("silenced", out object silencedObj) &&
                                   (silencedObj is bool b ? b : bool.Parse(silencedObj.ToString()));
                    OnSilenceToggle?.Invoke(silenced);
                    break;

                case MessageTypes.AUDIO_ANALYSIS:
                    float bass = GetFloat(data, "bass", 0f);
                    float mid = GetFloat(data, "mid", 0f);
                    float high = GetFloat(data, "high", 0f);
                    bool isBeat = GetBool(data, "beat", false);
                    OnAudioAnalysis?.Invoke(bass, mid, high, isBeat);
                    break;

                case MessageTypes.READ_HIGHLIGHT:
                    string sentence = data.TryGetValue("sentence", out object sentenceObj) ? sentenceObj?.ToString() ?? "" : "";
                    if (!string.IsNullOrWhiteSpace(sentence))
                    {
                        OnReadStart?.Invoke();
                        OnReadHighlight?.Invoke(sentence);
                    }
                    break;

                case MessageTypes.READ_CLEAR:
                    OnReadClear?.Invoke();
                    OnReadEnd?.Invoke();
                    break;

                case MessageTypes.DEBUG_STATUS:
                    string dbgStatus = data.TryGetValue("status", out object sObj) ? sObj?.ToString() ?? "" : "";
                    string dbgUser = data.TryGetValue("user_text", out object uObj) ? uObj?.ToString() ?? "" : "";
                    string dbgResp = data.TryGetValue("response_text", out object rObj) ? rObj?.ToString() ?? "" : "";
                    OnDebugStatus?.Invoke(dbgStatus, dbgUser, dbgResp);
                    break;

                case MessageTypes.FACE_EXPRESSION:
                    string exprName = data.TryGetValue("name", out object enObj) ? enObj?.ToString() : null;
                    float exprIntensity = GetFloat(data, "intensity", 0f);
                    OnFaceExpression?.Invoke(exprName ?? string.Empty, exprIntensity);
                    break;

                case MessageTypes.IDLE_THOUGHT:
                    string idleText = data.TryGetValue("text", out object itObj) ? itObj?.ToString() ?? "" : "";
                    OnIdleThought?.Invoke(idleText);
                    break;

                default:
                    Debug.Log($"[MessageHandler] Unhandled message type: {type}");
                    break;
            }
        }

        // Helper methods
        private float GetFloat(Dictionary<string, object> data, string key, float defaultValue)
        {
            if (data.TryGetValue(key, out object val))
            {
                if (val is double d) return (float)d;
                if (val is float f) return f;
                if (val is long l) return l;
                if (float.TryParse(val.ToString(), out float parsed)) return parsed;
            }
            return defaultValue;
        }

        private bool GetBool(Dictionary<string, object> data, string key, bool defaultValue)
        {
            if (data.TryGetValue(key, out object val))
            {
                if (val is bool b) return b;
                if (bool.TryParse(val.ToString(), out bool parsed)) return parsed;
            }
            return defaultValue;
        }

        // ── Methods to send messages to Python ──────────────────────

        public void SendModeChange(CompanionMode mode)
        {
            // Map Unity modes to Python CompanionMode enum values
            string modeStr = mode switch
            {
                CompanionMode.Active => "active",
                CompanionMode.Idle => "idle",
                CompanionMode.Dance => "dance_beat",  // default dance mode for Python
                _ => "idle"
            };
            webSocketClient?.Send(MessageTypes.MODE_CHANGE, new Dictionary<string, object>
            {
                ["mode"] = modeStr
            });
        }

        public void SendSilenceToggle()
        {
            webSocketClient?.Send(MessageTypes.TOGGLE_SILENCE);
        }

        public void SendDanceStyle(DanceStyle style)
        {
            // Send the appropriate Python CompanionMode for the dance style
            string modeStr = style switch
            {
                DanceStyle.Procedural => "dance_beat",
                DanceStyle.ShikanokoDance => "dance_full",
                _ => "active"  // DanceStyle.None = exit dance
            };
            webSocketClient?.Send(MessageTypes.MODE_CHANGE, new Dictionary<string, object>
            {
                ["mode"] = modeStr
            });
        }

        public void SendHotkey(string key)
        {
            // Map hotkey actions to actual WebSocket messages
            if (key == "read_pause")
                webSocketClient?.Send(MessageTypes.READ_PAUSE);
            else if (key == "read_resume")
                webSocketClient?.Send(MessageTypes.READ_RESUME);
        }

        public void SendReadPause()
        {
            webSocketClient?.Send(MessageTypes.READ_PAUSE);
        }

        public void SendReadResume()
        {
            webSocketClient?.Send(MessageTypes.READ_RESUME);
        }

        /// <summary>Feature #12: Send a random conversation prompt to the AI backend.</summary>
        public void SendRandomPrompt(string context)
        {
            webSocketClient?.Send(MessageTypes.RANDOM_PROMPT, new Dictionary<string, object>
            {
                ["context"] = context
            });
        }
    }
}
