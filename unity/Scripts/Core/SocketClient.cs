using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using Newtonsoft.Json;

namespace Annabeth.Core
{
    /// <summary>
    /// TCP Socket client for communication with Python backend.
    /// Handles connection, message sending/receiving, and reconnection.
    /// </summary>
    public class SocketClient : MonoBehaviour
    {
        [Header("Connection Settings")]
        [SerializeField] private string host = "127.0.0.1";
        [SerializeField] private int port = 8765;
        [SerializeField] private float reconnectDelay = 2f;
        [SerializeField] private bool autoConnect = true;

        // Events for message handling
        public event Action<string, Dictionary<string, object>> OnMessageReceived;
        public event Action OnConnected;
        public event Action OnDisconnected;

        private TcpClient _client;
        private NetworkStream _stream;
        private Thread _receiveThread;
        private Queue<string> _messageQueue = new Queue<string>();
        private Queue<string> _sendQueue = new Queue<string>();
        private bool _isConnected;
        private bool _shouldReconnect = true;
        private readonly object _queueLock = new object();
        private readonly object _sendLock = new object();

        public bool IsConnected => _isConnected;

        private void Start()
        {
            if (autoConnect)
            {
                StartCoroutine(ConnectWithRetry());
            }
        }

        private void Update()
        {
            // Process received messages on main thread
            lock (_queueLock)
            {
                while (_messageQueue.Count > 0)
                {
                    string message = _messageQueue.Dequeue();
                    ProcessMessage(message);
                }
            }

            // Send queued messages
            lock (_sendLock)
            {
                while (_sendQueue.Count > 0 && _isConnected)
                {
                    string msg = _sendQueue.Dequeue();
                    SendImmediate(msg);
                }
            }
        }

        private void OnDestroy()
        {
            Disconnect();
        }

        private void OnApplicationQuit()
        {
            _shouldReconnect = false;
            Disconnect();
        }

        /// <summary>
        /// Connect to the Python backend with automatic retry.
        /// </summary>
        public IEnumerator ConnectWithRetry()
        {
            while (_shouldReconnect && !_isConnected)
            {
                Debug.Log($"[SocketClient] Connecting to {host}:{port}...");
                
                try
                {
                    _client = new TcpClient();
                    var result = _client.BeginConnect(host, port, null, null);
                    bool success = result.AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(3));
                    
                    if (success && _client.Connected)
                    {
                        _client.EndConnect(result);
                        _stream = _client.GetStream();
                        _isConnected = true;
                        
                        // Start receive thread
                        _receiveThread = new Thread(ReceiveLoop);
                        _receiveThread.IsBackground = true;
                        _receiveThread.Start();
                        
                        Debug.Log("[SocketClient] Connected!");
                        OnConnected?.Invoke();
                        yield break;
                    }
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[SocketClient] Connection failed: {e.Message}");
                }
                
                yield return new WaitForSeconds(reconnectDelay);
            }
        }

        /// <summary>
        /// Disconnect from the server.
        /// </summary>
        public void Disconnect()
        {
            _isConnected = false;
            
            try
            {
                _stream?.Close();
                _client?.Close();
            }
            catch { }
            
            OnDisconnected?.Invoke();
        }

        /// <summary>
        /// Send a message to the Python backend.
        /// </summary>
        public void Send(string messageType, Dictionary<string, object> data = null)
        {
            var message = new Dictionary<string, object>
            {
                ["type"] = messageType
            };
            
            if (data != null)
            {
                foreach (var kvp in data)
                {
                    message[kvp.Key] = kvp.Value;
                }
            }
            
            string json = JsonConvert.SerializeObject(message);
            
            lock (_sendLock)
            {
                _sendQueue.Enqueue(json);
            }
        }

        private void SendImmediate(string json)
        {
            if (!_isConnected || _stream == null) return;
            
            try
            {
                byte[] data = Encoding.UTF8.GetBytes(json + "\n");
                _stream.Write(data, 0, data.Length);
            }
            catch (Exception e)
            {
                Debug.LogError($"[SocketClient] Send error: {e.Message}");
                HandleDisconnect();
            }
        }

        private void ReceiveLoop()
        {
            byte[] buffer = new byte[8192];
            StringBuilder messageBuilder = new StringBuilder();
            
            while (_isConnected)
            {
                try
                {
                    if (_stream.DataAvailable)
                    {
                        int bytesRead = _stream.Read(buffer, 0, buffer.Length);
                        if (bytesRead > 0)
                        {
                            string chunk = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                            messageBuilder.Append(chunk);
                            
                            // Process complete messages (newline-delimited)
                            string content = messageBuilder.ToString();
                            int newlineIndex;
                            while ((newlineIndex = content.IndexOf('\n')) >= 0)
                            {
                                string message = content.Substring(0, newlineIndex);
                                content = content.Substring(newlineIndex + 1);
                                
                                if (!string.IsNullOrWhiteSpace(message))
                                {
                                    lock (_queueLock)
                                    {
                                        _messageQueue.Enqueue(message);
                                    }
                                }
                            }
                            messageBuilder.Clear();
                            messageBuilder.Append(content);
                        }
                    }
                    
                    Thread.Sleep(10); // Small delay to prevent CPU spinning
                }
                catch (Exception e)
                {
                    if (_isConnected)
                    {
                        Debug.LogError($"[SocketClient] Receive error: {e.Message}");
                        HandleDisconnect();
                    }
                    break;
                }
            }
        }

        private void ProcessMessage(string json)
        {
            try
            {
                var data = JsonConvert.DeserializeObject<Dictionary<string, object>>(json);
                if (data != null && data.TryGetValue("type", out object typeObj))
                {
                    string messageType = typeObj.ToString();
                    OnMessageReceived?.Invoke(messageType, data);
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[SocketClient] Failed to parse message: {e.Message}");
            }
        }

        private void HandleDisconnect()
        {
            if (!_isConnected) return;
            
            _isConnected = false;
            OnDisconnected?.Invoke();
            
            if (_shouldReconnect)
            {
                StartCoroutine(ConnectWithRetry());
            }
        }
    }
}
