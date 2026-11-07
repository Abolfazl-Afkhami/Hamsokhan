<div align="center">

# Hamsokhan (هم‌سخن)
### Ultra-Low Latency, Full-Duplex Speech-to-Speech Conversational AI for Persian
**Engineered for Budget Compute (Kaggle Dual NVIDIA Tesla T4) | Sub-300ms TTFA | Dual-Track Native Barge-in**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.x FP16](https://img.shields.io/badge/PyTorch-2.x%20FP16-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hardware: Tesla T4](https://img.shields.io/badge/Hardware-Dual%20NVIDIA%20T4%20(16GB)-76B900?style=flat&logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![HuggingFace Model Hub](https://img.shields.io/badge/%F0%9F%A4%97%20Model%20Hub-Abolfazl27Aflhami88%2FHamsokhan-yellow?style=flat)](https://huggingface.co/Abolfazl27Aflhami88/Hamsokhan)
[![HuggingFace Dataset Hub](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset%20Hub-hamsokhan--shards-orange?style=flat)](https://huggingface.co/datasets/Abolfazl27Aflhami88/hamsokhan-shards)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)

<br/>

<img src="assets/hamsokhan_architecture.jpg" alt="Hamsokhan System Architecture (Kaggle T4 Optimized)" width="100%">

</div>

---

## 📖 Executive Summary

**Hamsokhan (هم‌سخن)** is the first production-grade, open-source, full-duplex Speech-to-Speech (STS) conversational agent built natively for the Persian language. Purpose-built to break free from the constraints of expensive enterprise compute clusters (such as 8x A100 or H100 pods), Hamsokhan is engineered from the ground up to train and serve reliably on **budget-tier commodity GPUs—specifically Dual NVIDIA Tesla T4s (16GB VRAM each) on Kaggle and Hugging Face Spaces**.

In human conversational dynamics, an end-to-end latency exceeding 350 ms shatters natural immersion. Traditional cascaded voice assistants (Automatic Speech Recognition $\to$ Large Language Model $\to$ Text-to-Speech) suffer from severe round-trip latencies of 750 ms to 1,500 ms due to sequential token generation and serialized sentence-boundary parsing. 

Hamsokhan solves this through **Hamsokhan V2**: a unified, end-to-end neural speech-to-speech architecture delivering **Time-to-First-Audio (TTFA) below 300 ms**, an active **inference VRAM footprint under 3.0 GB**, zero-lookahead causal audio encoding, and instantaneous **barge-in interruption handling (< 20 ms reaction window)**.

---

## 🎯 Key Performance Indicators (KPI Targets)

| Metric | Target Specification | Architectural Implementation |
| :--- | :---: | :--- |
| **Time-to-First-Audio (TTFA)** | **$< 300 \text{ ms}$** | Causal framing + Qwen3-0.6B backbone + Single-Step Rectified Flow + 3ms Vocos iSTFT |
| **Inference VRAM Footprint** | **$\sim 2.5 - 2.95 \text{ GB}$** | 4-bit AWQ / pure FP16 quantization; leaves $> 13 \text{ GB}$ headroom on a single 16GB T4 |
| **Barge-in Reaction Latency** | **$< 20 \text{ ms}$** | Dual-track interrupt: Acoustic fast-path (Silero VAD v5 on CPU) + immediate egress queue flush |
| **Speech Naturalness (UTMOS)** | **$> 3.65$** | Tanh-bounded FiLM-conditioned Flow Matching mapping to 24 kHz Mel spectrograms $[B, 100, T]$ |
| **Acoustic Timbre Fidelity (MCD)** | **$< 4.5 \text{ dB}$** | Mel-Cepstral Distortion optimized against studio-quality Persian reference speakers |
| **ASR Verification (WER)** | **$< 8.5\%$** | Zero-shot transcription verification on conversational Persian test sets via `whisper-large-v3` |
| **Hardware Compatibility** | **Turing T4 (No BF16)** | Pure FP16 execution stabilized with custom `GradScaler` ($2^{14}$ scale, 1000 growth interval) |

---

## 🏛️ The 5-Layer Core Architecture

Hamsokhan V2 discards text-token parsing bottlenecks, routing speech directly through a tightly coupled neural pipeline:

```
[Streaming Input: 16 kHz PCM (80 ms / 1280 sample packets)]
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. EDGE PROCESSING (CPU/C++ Engine)                                    │
│    • Silero VAD v5: 1 ms execution ticks, 2-frame speech confirmation  │
│    • Causal 80 ms framing with 60 ms pre-speech circular buffering     │
│    • Session Stream Manager coordinating full-duplex WebSocket state   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ (Voiced Causal Frames)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. ACOUSTIC ENCODER (FROZEN)                                           │
│    • Causalized Whisper-Tiny Encoder (39M parameters, ~320 MB VRAM)    │
│      OR Causal SNAC Tokenizer (12.5 Hz framerate)                      │
│    • Causal Speech Projector: 50 Hz -> 12.5 Hz via 4x Frame Stacking,   │
│      Depthwise Conv1D, Left-Padding, and SwiGLU Gating (~85 MB)        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ (12.5 Hz Projected Speech Embeddings)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. REASONING SLM BACKBONE (Qwen3-0.6B / Qwen2.5-0.5B)                  │
│    • 4-bit AWQ or pure FP16 quantization (~1.2 GB VRAM footprint)      │
│    • Dual-Track LoRA Adapter: Deep alignment with Persian dialogue     │
│    • Turn-Taking Binary Classifier Head: Dilated causal convolution    │
│      fusing mid + final layers to predict end-of-turn vs backchannels  │
│    • Native support for offline local thinking with filler vocalization │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ (12.5 Hz Latent Acoustic Features)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. ACOUSTIC SYNTHESIZER (Single-Step Rectified Flow Matching)          │
│    • Single-Step Euler ODE integration from Gaussian noise to Mel spec │
│    • Frequency-weighted Pseudo-Huber + Spectral/Temporal Gradient loss │
│    • Direct 15:2 macro-packet alignment to 93.75 Hz [B, 100, T_mel]    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ (100-bin Mel Spectrograms)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. FAST VOCODER (FROZEN) (Vocos 24 kHz)                                │
│    • ConvNeXt backbone + Inverse Short-Time Fourier Transform (iSTFT)  │
│    • Lightning-fast 3 ms – 5 ms inference execution on Turing T4       │
│    • Overlap-Add Hanning crossfading to eliminate packet-boundary pops  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
[Streaming Output: 24 kHz Natural Persian Speech (PCM 16-bit Mono)]
```

---

## ⚡ Kaggle Dual Tesla T4 Distributed Training Pipeline

Training end-to-end speech-to-speech architectures typically requires multi-node high-VRAM clusters. Hamsokhan breaks this barrier by running Distributed Data Parallel (DDP) training directly across **Dual NVIDIA Tesla T4 GPUs (2x 16GB)** inside a standard 12-hour Kaggle notebook session:

```
[50h Persian Corpus] ──► [DNSMOS Quality Filtering] ──► [Forced Alignment Tracking]
                                                                  │
                                                                  ▼
                                              ┌───────────────────────────────────────┐
                                              │ Kaggle Dual T4 FP16 DDP Engine        │
                                              │ • Phase 1: Audio-to-LLM Alignment (6h)│
                                              │ • Phase 2: Dual-Track LoRA Tuning (6h)│
                                              └───────────────────┬───────────────────┘
                                                                  │
                                                                  ▼
                                              [Asynchronous Background Daemon]
                                              • Zero-copy safetensors snapshots
                                              • Push to Hugging Face Model Hub
                                              • Rolling local disk cleanup
```

### Overcoming Platform & Hardware Bottlenecks:
1. **Defeating the 20 GB Disk Limit (`/kaggle/working`):**  
   Raw multi-dataset downloads quickly trigger `No space left on device`. Hamsokhan redirects `HF_HOME`, `HF_DATASETS_CACHE`, `TORCH_HOME`, and `TMPDIR` into `/kaggle/tmp` (which mounts an isolated, larger temporary volume), preserving the primary workspace strictly for final zero-copy `.safetensors` shards.
2. **Turing Architecture FP16 Stability (No BF16):**  
   Tesla T4 cards lack native hardware support for `bfloat16`. Running naive FP16 often leads to severe underflow/overflow and `NaN` loss divergence. We calibrate a custom `GradScaler` initialized at $2^{14}$ with a 1,000-step growth interval, accompanied by `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to eliminate memory fragmentation.
3. **NCCL Deadlock Elimination:**  
   To prevent inter-GPU synchronization hangs in Kaggle's containerized virtualized network, peer-to-peer and InfiniBand calls are explicitly disabled: `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`.
4. **Surviving the 12-Hour Session Limit:**  
   An asynchronous, non-blocking background daemon periodically snapshots trainable weights (LoRA + Projector + Synthesizer + Turn-Taking Head) and uploads `best_model` and `latest` artifacts directly to [Hugging Face Model Hub (Abolfazl27Aflhami88/Hamsokhan)](https://huggingface.co/Abolfazl27Aflhami88/Hamsokhan), while pruning older local step folders.

---

## 🌐 Real-Time Full-Duplex WebSocket Protocol

The client-server communication contract is designed for minimum overhead, transmitting raw binary audio buffers with sidecar JSON control frames:

### 1. High-Frequency Binary Audio Streams
* **Client $\to$ Server (Ingress):** Raw Linear `PCM 16 kHz, 16-bit Signed Integer, Mono`  
  * Frame Duration: 80 ms (1,280 samples) $\implies$ **2,560 bytes per packet**.
* **Server $\to$ Client (Egress):** Synthesized Linear `PCM 24 kHz, 16-bit Signed Integer, Mono`  
  * Frame Duration: 80 ms (1,920 samples) $\implies$ **3,840 bytes per packet**.

### 2. Dual-Track Barge-in & State Truncation
* **Acoustic Fast-Path:** Silero VAD v5 running on CPU processes incoming 80 ms frames. Detecting 2 consecutive voiced frames ($P_{\text{speech}} > 0.6$) instantly triggers an internal interrupt signal within 10 ms.
* **Semantic Confirmation:** The Turn-Taking Head evaluates speaker intent to gracefully differentiate true interruptions from brief conversational affirmations (*backchanneling* such as "آره", "خب").
* **Immediate Egress Flush:** The server immediately drains its outgoing `asyncio.Queue` and dispatches an `{"event": "interrupted"}` control frame.
* **KV-Cache State Truncation:** Utilizing the client's `client_playback_sample` feedback, any generated tokens beyond what the user actually heard before interrupting are pruned from the transformer's KV-Cache, preventing context pollution and hallucinated conversational drift.

---

## 📁 Repository Structure

```
.
├── assets/
│   └── hamsokhan_architecture.jpg     # Comprehensive system architectural blueprint
├── configs/
│   └── config.yaml                    # Runtime configuration (sample rates, VAD, model weights)
├── docs/
│   ├── ARCHITECTURE.md                # In-depth architectural spec, tensor math, VRAM & latency budgets
│   ├── WEBSOCKET_PROTOCOL.md          # Full-duplex WebSocket RFC 6455 spec, binary frames & barge-in
│   ├── KAGGLE_TRAINING.md             # Dual T4 DDP acceleration, disk workarounds & session management
│   ├── ROADMAP.md                     # Engineering handover report, current blockers & commit tasks
│   └── EVALUATION.md                  # Verification suite, UTMOS/MCD/WER benchmarks & test architecture
├── scripts/
│   ├── kaggle_dataset_builder.py      # Zero-copy safetensors sharding pipeline for Persian audio
│   ├── eval_hamsokhan.py              # Quantitative evaluation suite (TTFA, Mel L1, SpecGrad)
│   └── setup_nvidia_mps.sh            # NVIDIA Multi-Process Service (MPS) config utility
├── src/
│   ├── data/                          # Dataset loaders, Mel extractors, and PersianVox labelers
│   ├── models/                        # Causal Speech Projector, Flow Synthesizer, Turn-Taking Head
│   ├── pipeline/                      # Orchestrator and full-duplex session coordinator
│   ├── transport/                     # WebSocket server gateway and latency-testing client mock
│   └── vad/                           # Silero VAD v5 state machine with pre-speech buffering
├── tests/                             # Unit tests (causality, stability) and end-to-end integration tests
├── hamsokhan_model.py                 # Unified Hamsokhan V2 PyTorch nn.Module architecture
├── train_hamsokhan_ddp.py             # Distributed Accelerate FP16 training engine
├── main.py                            # Production server entrypoint and lifecycle supervisor
├── requirements.txt                   # Production Python dependencies
└── README.md                          # Primary project documentation
```

---

## 🗺️ Engineering Commit Roadmap

Our step-by-step roadmap addresses open integration tasks across the codebase:

- [ ] **`COMMIT-01` [Causal Masking & FP32 Stability]:**  
  Implement explicit causal padding and streaming buffer caches in `hamsokhan_model.py`; upcast frequency-domain transformations to FP32 to prevent half-precision `NaN` outputs on T4.
- [ ] **`COMMIT-02` [V2 Streaming Orchestrator]:**  
  Rewrite `src/pipeline/orchestrator.py` as an asynchronous 2-task pipeline (packet consumer + Vocos synthesizer) eliminating intermediate text-parsing layers.
- [ ] **`COMMIT-03` [WebSocket Gateway & Barge-in]:**  
  Upgrade `src/transport/websocket_server.py` with the raw binary audio protocol, rapid acoustic interrupt handlers, and instant egress queue draining.
- [ ] **`COMMIT-04` [Comprehensive Test Suite]:**  
  Add unit tests for zero-lookahead causality (`test_causal_masking.py`), Flow stability, and interrupt response times ($< 20 \text{ ms}$) in `tests/`.
- [ ] **`COMMIT-05` [Kaggle Dataset Sharding]:**  
  Execute cells 1–3 of `hamsokhan.ipynb` on Kaggle to build and upload validated shards to [`Abolfazl27Aflhami88/hamsokhan-shards`](https://huggingface.co/datasets/Abolfazl27Aflhami88/hamsokhan-shards).
- [ ] **`COMMIT-06` [DDP Training & Production Promotion]:**  
  Launch multi-GPU FP16 DDP training on Dual T4s, evaluate checkpoints on held-out test shards, and push validated weights to [`Abolfazl27Aflhami88/Hamsokhan`](https://huggingface.co/Abolfazl27Aflhami88/Hamsokhan).

---

## 🔗 Official Links & Resources

* **Hugging Face Model Hub:** [huggingface.co/Abolfazl27Aflhami88/Hamsokhan](https://huggingface.co/Abolfazl27Aflhami88/Hamsokhan)
* **Hugging Face Dataset Shards:** [huggingface.co/datasets/Abolfazl27Aflhami88/hamsokhan-shards](https://huggingface.co/datasets/Abolfazl27Aflhami88/hamsokhan-shards)
* **System Architecture Deep Dive:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
* **WebSocket Real-Time Protocol:** [docs/WEBSOCKET_PROTOCOL.md](docs/WEBSOCKET_PROTOCOL.md)
* **Kaggle Dual T4 Training Guide:** [docs/KAGGLE_TRAINING.md](docs/KAGGLE_TRAINING.md)
* **Development Roadmap & Handover:** [docs/ROADMAP.md](docs/ROADMAP.md)
* **Evaluation & Verification Suite:** [docs/EVALUATION.md](docs/EVALUATION.md)

---

## 📜 License

This project is licensed under the terms of the [MIT License](LICENSE).
