// Background service worker for Annabeth read-aloud highlighting
// Connects to the local avatar WebSocket and relays highlight events to tabs.

const WS_URL = "ws://127.0.0.1:8765/ws"; // Uses existing avatar server
let ws = null;
let reconnectTimer = null;

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  try {
    ws = new WebSocket(WS_URL);
  } catch (err) {
    scheduleReconnect();
    return;
  }

  ws.onopen = () => {
    console.log("[Annabeth] WebSocket connected");
  };

  ws.onclose = () => {
    console.log("[Annabeth] WebSocket disconnected");
    scheduleReconnect();
  };

  ws.onerror = () => {
    console.log("[Annabeth] WebSocket error");
    ws.close();
  };

  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === "read_highlight" || msg.type === "read_clear") {
        relayToTabs(msg);
      }
    } catch (err) {
      console.log("[Annabeth] Bad message", err);
    }
  };
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, 2000);
}

function relayToTabs(message) {
  chrome.tabs.query({}, (tabs) => {
    for (const tab of tabs) {
      if (!tab.id) continue;
      chrome.tabs.sendMessage(tab.id, message, () => {
        if (chrome.runtime.lastError) {
          // Content script may not be injected yet; ignore
        }
      });
    }
  });
}

connect();
