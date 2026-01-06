using System.Collections.Generic;
using UnityEngine;

namespace Annabeth.Core
{
    /// <summary>
    /// Message types matching Python shared/config.py.
    /// Keep in sync with the Python backend.
    /// </summary>
    public static class MessageTypes
    {
        // Mode messages
        public const string MODE_CHANGE = "mode_change";
        public const string SET_MODE = "set_mode";
        
        // Speech messages
        public const string SPEAK_START = "speak_start";
        public const string SPEAK_END = "speak_end";
        public const string EMOTION = "emotion";
        
        // Silence control
        public const string SILENCE_TOGGLE = "silence_toggle";
        public const string SILENCED = "silenced";
        
        // Read-aloud control
        public const string READ_START = "read_start";
        public const string READ_PAUSE = "read_pause";
        public const string READ_RESUME = "read_resume";
        public const string READ_END = "read_end";
        
        // Audio analysis
        public const string AUDIO_ANALYSIS = "audio_analysis";
        
        // Dance control
        public const string DANCE_STYLE = "dance_style";
        public const string PLAY_ANIMATION = "play_animation";
        
        // Hotkey requests
        public const string HOTKEY = "hotkey";
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
    /// Handles routing of messages from SocketClient to appropriate handlers.
    /// </summary>
    public class MessageHandler : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private SocketClient socketClient;

        // Events for specific message types
        public event System.Action<string> OnSpeakStart;
        public event System.Action OnSpeakEnd;
        public event System.Action<string> OnEmotionChange;
        public event System.Action<CompanionMode> OnModeChange;
        public event System.Action<bool> OnSilenceToggle;
        public event System.Action<float, float, float> OnAudioAnalysis; // beatEnergy, bassEnergy, trebleEnergy
        public event System.Action<DanceStyle> OnDanceStyleChange;
        public event System.Action<string> OnPlayAnimation;
        public event System.Action OnReadStart;
        public event System.Action OnReadPause;
        public event System.Action OnReadResume;
        public event System.Action OnReadEnd;

        private void Start()
        {
            if (socketClient == null)
            {
                socketClient = FindObjectOfType<SocketClient>();
            }

            if (socketClient != null)
            {
                socketClient.OnMessageReceived += HandleMessage;
            }
            else
            {
                Debug.LogError("[MessageHandler] SocketClient not found!");
            }
        }

        private void OnDestroy()
        {
            if (socketClient != null)
            {
                socketClient.OnMessageReceived -= HandleMessage;
            }
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
                        CompanionMode mode = modeName switch
                        {
                            "active" => CompanionMode.Active,
                            "idle" => CompanionMode.Idle,
                            "dance" => CompanionMode.Dance,
                            _ => CompanionMode.Idle
                        };
                        OnModeChange?.Invoke(mode);
                    }
                    break;

                case MessageTypes.SILENCED:
                    bool silenced = data.TryGetValue("silenced", out object silencedObj) && 
                                   (silencedObj is bool b ? b : bool.Parse(silencedObj.ToString()));
                    OnSilenceToggle?.Invoke(silenced);
                    break;

                case MessageTypes.AUDIO_ANALYSIS:
                    float beatEnergy = GetFloat(data, "beat_energy", 0f);
                    float bassEnergy = GetFloat(data, "bass_energy", 0f);
                    float trebleEnergy = GetFloat(data, "treble_energy", 0f);
                    OnAudioAnalysis?.Invoke(beatEnergy, bassEnergy, trebleEnergy);
                    break;

                case MessageTypes.DANCE_STYLE:
                    int style = GetInt(data, "style", 0);
                    OnDanceStyleChange?.Invoke((DanceStyle)style);
                    break;

                case MessageTypes.PLAY_ANIMATION:
                    string animName = data.TryGetValue("name", out object nameObj) ? nameObj.ToString() : "";
                    OnPlayAnimation?.Invoke(animName);
                    break;

                case MessageTypes.READ_START:
                    OnReadStart?.Invoke();
                    break;

                case MessageTypes.READ_PAUSE:
                    OnReadPause?.Invoke();
                    break;

                case MessageTypes.READ_RESUME:
                    OnReadResume?.Invoke();
                    break;

                case MessageTypes.READ_END:
                    OnReadEnd?.Invoke();
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

        private int GetInt(Dictionary<string, object> data, string key, int defaultValue)
        {
            if (data.TryGetValue(key, out object val))
            {
                if (val is long l) return (int)l;
                if (val is int i) return i;
                if (int.TryParse(val.ToString(), out int parsed)) return parsed;
            }
            return defaultValue;
        }

        // Methods to send messages to Python
        public void SendModeChange(CompanionMode mode)
        {
            socketClient?.Send(MessageTypes.SET_MODE, new Dictionary<string, object>
            {
                ["mode"] = mode.ToString().ToLower()
            });
        }

        public void SendSilenceToggle()
        {
            socketClient?.Send(MessageTypes.SILENCE_TOGGLE);
        }

        public void SendDanceStyle(DanceStyle style)
        {
            socketClient?.Send(MessageTypes.DANCE_STYLE, new Dictionary<string, object>
            {
                ["style"] = (int)style
            });
        }

        public void SendHotkey(string key)
        {
            socketClient?.Send(MessageTypes.HOTKEY, new Dictionary<string, object>
            {
                ["key"] = key
            });
        }

        public void SendReadPause()
        {
            socketClient?.Send(MessageTypes.READ_PAUSE);
        }

        public void SendReadResume()
        {
            socketClient?.Send(MessageTypes.READ_RESUME);
        }
    }
}
