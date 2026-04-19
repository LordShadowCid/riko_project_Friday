# Annabeth Multi-Agent Background Architecture

## Hardware Inventory
| Resource | Specs | Current Use | Free |
|----------|-------|-------------|------|
| **GPU 0** | RTX A4000 16GB | Ollama LLM (4.9GB) | **11.1GB VRAM** |
| **GPU 1** | RTX A4000 16GB | Whisper ASR (282MB) + GPT-SoVITS TTS | **~15GB VRAM** |
| **CPU** | i9-12900 16C/24T | Python servers, Unity | Lightly loaded |
| **RAM** | 64 GB | ~20GB used | **~44GB free** |

## Current Pipeline (Sequential Bottleneck)
```
User speaks → [Whisper ASR ~1s] → [Build messages ~0.1s] → [Memory recall ~0.1s]
→ [Tool match ~0.3s] → [LLM TTFT ~3-5s] → [LLM first sentence ~1-2s more]
→ [TTS synthesis ~1-2s] → First audio plays
                                                    Total: ~7-12s
```

## Target Pipeline (Parallel Agents)
```
User speaks → ASR Agent (GPU 1) ─────────────────┐
              ├─ Intent Classifier (CPU, 50ms) ──│─→ adjusts LLM params
              └─ Memory Prefetch (CPU, 100ms) ───│─→ ready when LLM needs it
                                                  ↓
              LLM Agent (GPU 0) ── sentences ───→ TTS Agent (GPU 1)
                                                  ↓
                                           Audio Playback
                                                    Target: ~4-6s
```

## Agent Definitions

### 1. ASR Agent (Exists — GPU 1)
- **Model**: Whisper base.en on GPU 1
- **Status**: Already implemented in `asr_vad.py`
- **Optimization**: No changes needed. base.en is fast enough (~0.5-1s).

### 2. Intent Classifier Agent (NEW — CPU)
- **Purpose**: Quick (~50ms) classification of user input BEFORE LLM
- **Why**: Adjusts LLM temperature, context length, and response style
- **Implementation**: Rule-based (no model needed), runs on CPU thread
- **Input classification**:
  - `greeting` → short response, low num_predict
  - `question` → informational, moderate length
  - `story_request` → creative, high num_predict, higher temperature
  - `command` → tool-oriented, precise
  - `followup` → uses context, moderate
- **File**: `server/process/agents/intent_classifier.py`

### 3. Memory Prefetch Agent (NEW — CPU thread)
- **Purpose**: Start memory recall the moment ASR finishes, BEFORE LLM message building
- **Why**: Currently memory recall blocks LLM start. Running async saves ~100ms.
- **Implementation**: `threading.Thread` that calls `memory_store.recall_all()` and stores result
- **File**: Integrated into `main_chat.py`

### 4. LLM Agent (Exists — GPU 0)
- **Model**: mannix/llama3.1-8b-abliterated (Q4_0, 4.7GB)
- **Status**: Already implemented in `llm_scr.py`
- **Optimizations**:
  - Remove num_predict cap (DONE) — variable-length natural responses
  - Keep EAGER_FLUSH_LEN=40 for faster first sentence delivery
  - Pre-warm context at startup (sends dummy request to allocate memory)
  - Trim system prompt tokens where possible

### 5. TTS Agent (Exists — GPU 1)
- **Model**: GPT-SoVITS on GPU 1
- **Status**: Already implemented with 2-worker thread pool in `main_chat.py`
- **Optimization**: Pre-warm HTTP session at startup (already uses session pooling)

### 6. Background Memory Agent (Exists — GPU 0, async)
- **Components**: `conversation_summarizer.py`, `self_eval.py`
- **Runs**: After conversation turns complete (non-blocking)
- **Uses**: Same Ollama model with reduced context (512-1024)

## Phase 1 Optimizations (Implement Now)

### 1.1 Ollama Context Pre-warm
On startup, send a short dummy chat request to force Ollama to allocate the full 3072-token
context window. Currently `ollama ps` shows 512 context because it auto-sizes lazily.
This eliminates a ~1-2s reallocation penalty on the first real request.

### 1.2 Intent-Aware Response Parameters
Simple keyword/pattern classifier adjusts LLM behavior per turn:
- Short greetings → `num_predict=100` for fast response
- Story/explain requests → `num_predict=-1` (unlimited) with slightly higher temp
- Normal conversation → `num_predict=512` (balanced)

### 1.3 Parallel Memory Prefetch
Start memory recall as a background thread immediately after ASR finishes,
while the LLM message array is still being constructed. Results are joined
before the LLM request is sent (typically ready by then).

### 1.4 Trim System Prompt
The system prompt is ~250 tokens. Reducing to ~150 tokens saves ~1s of prefill
on an 8B model. Consolidate redundant rules.

## Phase 2 Optimizations (Future)

### 2.1 Model Upgrade: Q4_K_M Quantization
Q4_0 (current) is the worst quality-per-bit quantization. Upgrading to Q4_K_M
gives ~5-10% better output quality at identical speed and size.
Requires creating a custom GGUF or finding a Q4_K_M variant.

### 2.2 Speculative Decoding (Draft Model on GPU 1)
Load gemma3:4b (3.3GB) on GPU 1 as a draft model. Generate candidate tokens
fast on the small model, verify on the main 8B model. Can give 2-3x speedup
on token generation. Requires custom orchestration (Ollama doesn't support this natively).

### 2.3 Multi-GPU Model Split
The 8B model fits on one GPU easily, but a future 14B model (e.g., qwen3:14b)
could be split across both GPUs. Ollama supports CUDA_VISIBLE_DEVICES for this.

### 2.4 Dedicated TTS Pre-generation
During idle time, pre-generate audio for common responses ("Sure!", "Got it!",
"Hmm, let me think...") and cache them. Play pre-generated audio instantly
while the full response generates.

## Architecture Diagram
```
┌─────────────────────────────────────────────────────────┐
│                    Unity Frontend                         │
│           (Avatar, UI, WebSocket client)                  │
└──────────────────────┬──────────────────────────────────┘
                       │ WebSocket
┌──────────────────────▼──────────────────────────────────┐
│                  Python Backend                           │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │  ASR Agent   │  │ Intent      │  │ Memory Prefetch │  │
│  │  (GPU 1)     │──│ Classifier  │──│ Agent (CPU)     │  │
│  │  Whisper     │  │ (CPU, 50ms) │  │ ChromaDB recall │  │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
│         │                │                    │           │
│         ▼                ▼                    ▼           │
│  ┌──────────────────────────────────────────────────┐    │
│  │              LLM Agent (GPU 0)                    │    │
│  │   Ollama • mannix/llama3.1-8b-abliterated         │    │
│  │   Streaming sentences → on_sentence callback      │    │
│  └──────────────────────┬───────────────────────────┘    │
│                         │ sentences                       │
│  ┌──────────────────────▼───────────────────────────┐    │
│  │              TTS Agent (GPU 1)                    │    │
│  │   GPT-SoVITS • 2-worker thread pool               │    │
│  │   Prefetch up to 3 sentences ahead                 │    │
│  └──────────────────────┬───────────────────────────┘    │
│                         │ audio                           │
│  ┌──────────────────────▼───────────────────────────┐    │
│  │           Audio Playback (BCC950)                 │    │
│  └──────────────────────────────────────────────────┘    │
│                                                           │
│  ┌──────────────────────────────────────────────────┐    │
│  │         Background Memory Agent (async)           │    │
│  │   Conversation summarizer + Self-eval              │    │
│  │   Runs after each turn, non-blocking               │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```
