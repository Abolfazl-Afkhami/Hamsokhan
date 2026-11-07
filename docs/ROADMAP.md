# Hamsokhan: Engineering Handover Report & Technical Roadmap

> **Document Version:** 2.1.0  
> **Target Platform:** Dual NVIDIA Tesla T4 (2x 16 GB VRAM)  
> **Development Branch:** `mahdi` (Active Development) $\to$ `main` (Production Tracking)  
> **Associated Hubs:**  
> - **GitHub Repository:** [Abolfazl-Afkhami/Hamsokhan](https://github.com/Abolfazl-Afkhami/Hamsokhan)  
> - **Hugging Face Model Hub:** [Abolfazl27Aflhami88/Hamsokhan](https://huggingface.co/Abolfazl27Aflhami88/Hamsokhan)  
> - **Hugging Face Dataset Hub:** [Abolfazl27Aflhami88/hamsokhan-shards](https://huggingface.co/datasets/Abolfazl27Aflhami88/hamsokhan-shards)

---

## 1. Executive Status of the Codebase

The Hamsokhan project is currently transitioning from its legacy Phase 3 Cascaded Proof-of-Concept to its unified **Hamsokhan V2** real-time streaming architecture.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ [LEGACY: Phase 3 Cascaded Pipeline]                                                    │
│ Audio In -> Silero VAD -> Faster-Whisper -> Text Normalizer -> vLLM Qwen 7B ->        │
│ Parsers -> Flow Matching TTS -> Vocos -> Audio Out                                     │
│ ⚠️ Latency: 750 ms – 1,500 ms (Unacceptable for natural human turn-taking)             │
│ ⚠️ Memory: 9 GB – 11 GB VRAM (Cannot be hosted concurrently on a single T4)            │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (Architectural Paradigm Shift)
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ [PRODUCTION: Hamsokhan V2 Unified Architecture]                                        │
│ 80ms Frames -> Silero VAD v5 -> Causal Projector -> Qwen3-0.6B / Dual-Track LoRA       │
│ + Turn-Taking Head -> Single-Step Rectified Flow Synthesizer -> Vocos 24kHz iSTFT     │
│ ⚡ Latency (TTFA): < 300 ms (Sub-200 ms measured in benchmark)                          │
│ ⚡ Memory: ~2.59 GB VRAM (Native execution on a single 16 GB T4)                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Auditing & Known Technical Deficiencies

A comprehensive audit of the code across the `mahdi` branch reveals the following module statuses:

### 2.1. Unified Model Core (`hamsokhan_model.py`)
* **Current State (85% Complete):** Implements `HamsokhanConfig`, `CausalSpeechProjector`, `TurnTakingHead`, `SingleStepFlowSynthesizer`, and `HamsokhanModelWrapper`.
* **Identified Deficiencies Requiring Patch:**
  1. *Causal Encoder Lookahead Guarantee:* The streaming cache mechanism inside the causal projector must explicitly enforce zero lookahead during incremental inference. All left-padding offsets must be locked to guarantee that representation $t$ is strictly independent of future frames.
  2. *FP16/FP32 Mixed-Precision Safety:* The Single-Step Rectified Flow vector field prediction and frequency-domain spectral gradient loss calculations must be explicitly cast to FP32 during backpropagation to prevent half-precision `NaN` overflow on Tesla T4 GPUs.

### 2.2. Distributed Training Pipeline (`train_hamsokhan_ddp.py`)
* **Current State (90% Complete):** Implements zero-copy memory-mapped safetensors loading, custom `GradScalerKwargs`, dual learning-rate AdamW optimizers, and the asynchronous background Hugging Face checkpointing daemon.
* **Identified Deficiencies Requiring Patch:**
  1. *DDP Parameter Registration:* In the distributed setup, frozen components (such as the base Whisper encoder and Vocos) must be registered with `find_unused_parameters=False` once all output heads are connected to avoid unnecessary gradient synchronization overhead.

### 2.3. Real-Time Streaming Gateway (`websocket_server.py` & `orchestrator.py`)
* **Current State (Needs Rewrite):** The current implementation in `src/transport/websocket_server.py` and `src/pipeline/orchestrator.py` remains tied to the legacy V1 cascaded pipeline (waiting for full sentence chunks from vLLM and passing strings to TTS).
* **Required Engineering:**
  1. Complete rewrite into an asynchronous two-task producer-consumer model:
     - Task 1: Ingests 80 ms audio packets, feeds the causal projector, and updates the turn-taking state.
     - Task 2: Pulls acoustic latent frames directly from the language model and streams Mel features to Vocos.
  2. Implement the binary audio packet protocol (2,560-byte ingress $\to$ 3,840-byte egress) without text intermediary representations.
  3. Integrate the Overlap-Add (OLA) raised-cosine crossfade algorithm to eliminate boundary clicks.

### 2.4. Kaggle Infrastructure & Shards (`hamsokhan.ipynb` & `kaggle_dataset_builder.py`)
* **Current State:** The notebook is fully parameterized and has been verified with "Quick Save". Shards for the Hugging Face repository (`Abolfazl27Aflhami88/hamsokhan-shards`) are configured but need their initial build execution.

---

## 3. Prioritized Git Commit Roadmap

The following six sequential commits establish the operational path to full deployment:

| Priority | Commit ID | Technical Focus | Operational Scope & Deliverables | Primary Files |
| :---: | :---: | :--- | :--- | :--- |
| **1** | **`COMMIT-01`** | **Causal Projector & FP32 Stability** | Implement explicit causal padding and streaming cache state in `CausalSpeechProjector`. Upcast critical flow loss tensors to FP32 to ensure numerical stability on Turing T4 GPUs. | `hamsokhan_model.py`<br>`src/models/causal_projector.py` |
| **2** | **`COMMIT-02`** | **Streaming Orchestrator V2** | Rebuild `orchestrator.py` into an asynchronous, zero-text streaming engine that couples the causal encoder directly to Vocos via single-step flow synthesis. | `src/pipeline/orchestrator.py` |
| **3** | **`COMMIT-03`** | **WebSocket Gateway & Barge-in** | Implement raw binary framing (16 kHz in, 24 kHz out), JSON control events (`interrupt`, `interrupted`, `telemetry`), and instant egress queue draining upon acoustic VAD triggers. | `src/transport/websocket_server.py`<br>`src/vad/silero_vad.py` |
| **4** | **`COMMIT-04`** | **Validation & Test Suite** | Implement unit tests for causal mask invariance (`test_causal_masking.py`), Flow stability (`test_flow_synthesizer.py`), and barge-in response time ($< 20 \text{ ms}$) in `tests/`. | `tests/unit/*`<br>`tests/integration/*` |
| **5** | **`COMMIT-05`** | **Kaggle Shard Compilation** | Execute Cells 1 through 3 of `hamsokhan.ipynb` on Kaggle to generate, validate, and push zero-copy `.safetensors` shards to Hugging Face Datasets Hub. | `scripts/kaggle_dataset_builder.py`<br>`hamsokhan.ipynb` |
| **6** | **`COMMIT-06`** | **Dual T4 DDP Training & Promotion** | Launch multi-GPU DDP training on Kaggle Dual T4s (Cells 4 & 5). Evaluate against held-out test shards and upload final weights and evaluation reports to Hugging Face Model Hub. | `train_hamsokhan_ddp.py`<br>`scripts/eval_hamsokhan.py` |

---

## 4. Production Promotion Acceptance Gates

Artifact promotion from development to production requires passing these non-negotiable benchmark gates:

1. **Latency Threshold:** Measured TTFA must remain below **300 ms** on $99\%$ of test utterances.
2. **Barge-in Reaction Time:** Server output must cease within **20 ms** of receiving an interrupt signal.
3. **Memory Confinement:** Peak GPU VRAM must not exceed **3.0 GB** during continuous multi-turn dialogue.
4. **Perceptual Quality:** The synthesized speech must achieve a **UTMOS score $\ge 3.65$** and an **MCD $\le 4.5 \text{ dB}$**.
5. **Intelligibility Verification:** Automated transcription of synthesized speech via `whisper-large-v3` must yield a **Word Error Rate (WER) $\le 8.5\%$**.
