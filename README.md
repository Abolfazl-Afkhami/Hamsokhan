# Persian Streaming Speech-to-Speech (STS) Pipeline - Phase 3 MVP

## Overview
Full-duplex, low-latency conversational speech-to-speech engine for Persian, targeting sub-750ms end-to-end TTFAB with natural prosody, Barge-in interruption support, and CUDA Stream isolation.

## Architecture
```
Client (Web / Mobile / SIP) 
   │ (Full-Duplex 16kHz PCM / WebSocket)
   ▼
[STS WebSocket Server (src/transport/websocket_server.py)]
   │
   ▼
[Silero VAD (3-Frame Hysteresis + 80ms Tail Silence Retention)]
   │ (Trimmed Voiced Segment)
   ▼
[Streaming Whisper ASR (Large-v3-Turbo / CTranslate2, beam_size=1, max_workers=4)]
   │ (Raw Text)
   ▼
[Deterministic TN (Arabic Glyphs Normalizer + Num2FaWords + ZWNJ Regularizer)]
   │ (Clean Persian Text)
   ▼
[Streaming LLM Client (vLLM Qwen-2.5-7B AWQ, GPU Mem 0.42, Match-Span Chunking)]
   │ (Sentence Chunks with Prosody Tags: [tone: ...], [pause: ...])
   ▼
[Prosody Tag Parser (Tone & Non-Verbal Acoustic Separation)]
   │
   ▼
[Streaming Flow Matching TTS (CosyVoice-2 / F5-TTS interface, CUDA Stream, Hanning Crossfade)]
   │ (24kHz PCM Chunks)
   ▼
Client Egress Stream
```

## Directory Structure
```
.
├── configs/
│   └── config.yaml                 # System, VAD, ASR, LLM, TTS and Network configurations
├── scripts/
│   ├── finetune_asr_lora.py        # LoRA training for Whisper on Persian conversational audio
│   └── finetune_tts_flow.py        # DDP-stabilized Flow Matching loss (normalized log-F0)
├── src/
│   ├── asr/
│   │   └── streaming_asr.py        # Multi-worker non-blocking Faster-Whisper ASR
│   ├── llm/
│   │   └── streaming_llm.py        # Atomic regex match-span streaming LLM chunker
│   ├── nlp/
│   │   ├── normalizer.py           # Deterministic TN (Glyphs, ZWNJ, Numerals, Markdown strip)
│   │   └── prosody_parser.py       # Prosody and non-verbal acoustic cue extractor
│   ├── pipeline/
│   │   └── orchestrator.py         # Asyncio coordinator with generation ID turn isolation
│   ├── transport/
│   │   ├── websocket_server.py     # Full-duplex WebSocket gateway (16kHz in -> 24kHz out)
│   │   └── client_mock.py          # Automated latency & TTFAB integration test client
│   ├── tts/
│   │   └── streaming_tts.py        # Flow Matching chunked synthesizer with CUDA Stream isolation
│   └── vad/
│       └── silero_vad.py           # Silero VAD state machine with 80ms terminal acoustic decay
├── main.py                         # Production gateway bootstrap and lifecycle supervisor
├── requirements.txt                # Core dependencies
└── README.md
```

## Running the MVP
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch LLM Backend (vLLM):**
   ```bash
   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype half --gpu-memory-utilization 0.42
   ```
3. **Start the STS Streaming Gateway:**
   ```bash
   python main.py
   ```
4. **Run Latency & Integration Test:**
   ```bash
   python src/transport/client_mock.py
   ```
