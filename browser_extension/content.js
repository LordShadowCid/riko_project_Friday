// Content script that displays a small overlay with read-aloud highlights.

(() => {
  const overlay = document.createElement('div');
  overlay.id = 'annabeth-read-overlay';
  overlay.innerHTML = `
    <div id="annabeth-read-header">Annabeth is reading...</div>
    <div id="annabeth-read-body">
      <span id="annabeth-read-text"></span>
    </div>
  `;
  document.documentElement.appendChild(overlay);

  const style = document.createElement('style');
  style.textContent = `
    #annabeth-read-overlay {
      position: fixed;
      bottom: 16px;
      right: 16px;
      max-width: 420px;
      background: rgba(20, 20, 30, 0.92);
      color: #f4f4f8;
      font-family: 'Segoe UI', sans-serif;
      font-size: 15px;
      border-radius: 10px;
      padding: 10px 12px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.35);
      z-index: 2147483647;
      border: 1px solid rgba(255,255,255,0.08);
      display: none;
    }
    #annabeth-read-header {
      font-weight: 600;
      margin-bottom: 6px;
      font-size: 13px;
      color: #b9d4ff;
      letter-spacing: 0.2px;
    }
    #annabeth-read-body {
      line-height: 1.35;
      word-wrap: break-word;
    }
    #annabeth-read-body .word {
      padding: 1px 2px;
      border-radius: 4px;
      transition: background 120ms ease, color 120ms ease;
    }
    #annabeth-read-body .word.active {
      background: #3a6ff7;
      color: #fff;
    }
  `;
  document.documentElement.appendChild(style);

  let timers = [];

  function clearTimers() {
    timers.forEach((t) => clearTimeout(t));
    timers = [];
  }

  function renderSentence(sentence, wordTimings) {
    const body = document.getElementById('annabeth-read-text');
    if (!body) return;

    // Build word spans
    body.innerHTML = '';
    const words = sentence.split(/\s+/);
    words.forEach((w, idx) => {
      const span = document.createElement('span');
      span.textContent = w + (idx === words.length - 1 ? '' : ' ');
      span.className = 'word';
      span.dataset.index = String(idx);
      body.appendChild(span);
    });

    // Schedule highlights
    clearTimers();
    wordTimings.forEach((timing, timingIdx) => {
      const { word, start, end } = timing;
      // Normalize for matching: lowercase, strip punctuation
      const normalize = (s) => s.toLowerCase().replace(/[^\w']/g, '');
      let idx = words.findIndex((w) => normalize(w) === normalize(word));
      // Fallback: use timing index if word match fails but index is valid
      if (idx === -1 && timingIdx < words.length) idx = timingIdx;
      if (idx === -1) return;
      const startMs = Math.max(0, start * 1000);
      const endMs = Math.max(startMs + 50, end * 1000);
      timers.push(
        setTimeout(() => {
          const span = body.querySelector(`.word[data-index="${idx}"]`);
          if (span) span.classList.add('active');
        }, startMs)
      );
      timers.push(
        setTimeout(() => {
          const span = body.querySelector(`.word[data-index="${idx}"]`);
          if (span) span.classList.remove('active');
        }, endMs)
      );
    });

    overlay.style.display = 'block';
  }

  function handleHighlight(msg) {
    const sentence = msg.sentence || '';
    const timings = msg.word_timings || [];
    renderSentence(sentence, timings);
  }

  function handleClear() {
    clearTimers();
    overlay.style.display = 'none';
  }

  chrome.runtime.onMessage.addListener((msg) => {
    if (!msg || !msg.type) return;
    if (msg.type === 'read_highlight') {
      handleHighlight(msg);
    } else if (msg.type === 'read_clear') {
      handleClear();
    }
  });
})();
