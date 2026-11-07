# Hamsokhan V2: System Architecture & Deep-Tech Specification

> **Target Compute:** Single or Dual NVIDIA Tesla T4 (16 GB VRAM each, Turing Architecture, `sm_75`)  
> **Latency Objective:** Time-to-First-Audio (TTFA) $< 300 \text{ ms}$  
> **Active Inference Footprint:** Total VRAM $\sim 2.59 \text{ GB}$ (Native execution on a single T4)  
> **Audio Encodings:** Ingress: 16 kHz PCM (16-bit Mono) | Egress: 24 kHz PCM (16-bit Mono)

---

## 1. System Architecture Overview

The **Hamsokhan V2** architecture is designed to eliminate the multi-hundred-millisecond serialization latencies inherent in legacy cascaded Speech-to-Speech systems (ASR $\to$ LLM $\to$ TTS). By establishing a tightly coupled neural pipeline—combining causal acoustic encoding, a compact autoregressive language backbone, single-step rectified flow feature prediction, and an ultra-fast iSTFT-based neural vocoder—Hamsokhan achieves full-duplex conversational interaction on commodity hardware.

![Hamsokhan System Architecture](../assets/hamsokhan_architecture.jpg)

### End-to-End Latency & Data Flow Pipeline:

```
+───────────────────────────────────────────────────────────────────────────────+
| 1. EDGE PROCESSING (CPU / C++ Engine)                                         |
|    • Input Stream: 16 kHz Linear PCM, Mono, 16-bit (80 ms / 1,280 samples)   |
|    • Silero VAD v5: 1 ms execution step, 2-consecutive-frame thresholding    |
|    • Circular Pre-Speech Buffer (60 ms) & Causal Session Framing              |
+───────────────────────────────────────┬───────────────────────────────────────+
                                        │ (Voiced Causal Audio Chunks)
                                        ▼
+───────────────────────────────────────────────────────────────────────────────+
| 2. ACOUSTIC ENCODER (FROZEN)                                                  |
|    • Causal Whisper-Tiny Encoder (39M parameters, ~320 MB VRAM)               |
|      OR Causal SNAC Tokenizer (12.5 Hz framerate)                             |
|    • Causal Speech Projector: 50 Hz -> 12.5 Hz                                |
|      [B, T_in, 384] -> 4x Stacking -> Causal Conv1D -> SwiGLU -> [B, T_llm, 1024]|
+───────────────────────────────────────┬───────────────────────────────────────+
                                        │ (Projected Speech Embeddings)
                                        ▼
+───────────────────────────────────────────────────────────────────────────────+
| 3. REASONING SLM BACKBONE (Qwen3-0.6B / Qwen2.5-0.5B)                         |
|    • Quantization: Pure FP16 or 4-bit AWQ (~1.2 GB VRAM)                      |
|    • Dual-Track LoRA Adapter (Conversational Persian Alignment)               |
|    • Turn-Taking Binary Classifier Head: Dilated causal convolution fusing   |
|      mid and final hidden layers to predict end-of-turn probabilities         |
|    • Zero-Lookahead Causal Attention Cache                                    |
+───────────────────────────────────────┬───────────────────────────────────────+
                                        │ (12.5 Hz Latent Speech Features)
                                        ▼
+───────────────────────────────────────────────────────────────────────────────+
| 4. ACOUSTIC SYNTHESIZER (Single-Step Rectified Flow Matching)                 |
|    • Tanh-Bounded FiLM-Conditioned Vector Field Predictor                     |
|    • Single-Step Euler ODE Integration: [B, 100, T_mel] (93.75 Hz Mel-Spec)   |
|    • Exact 15:2 Macro-Packet Alignment Ratio                                  |
+───────────────────────────────────────┬───────────────────────────────────────+
                                        │ (100-bin Mel Spectrograms)
                                        ▼
+───────────────────────────────────────────────────────────────────────────────+
| 5. FAST VOCODER (FROZEN) (Vocos 24 kHz)                                       |
|    • ConvNeXt Backbone + Inverse Short-Time Fourier Transform (iSTFT)         |
|    • Inference Time: 3 ms – 5 ms on Tesla T4                                  |
|    • Overlap-Add Crossfade to Eliminate Packet-Boundary Phase Clicks          |
+───────────────────────────────────────┬───────────────────────────────────────+
                                        │
                                        ▼
                        OUTPUT: 24 kHz Natural Persian Speech (Streaming)
```

---

## 2. Hardware Resource & Latency Budgets

### 2.1. Inference VRAM Allocation (Single Tesla T4 16 GB)

Commodity hosting requires strict confinement within a single GPU’s memory map. On a 16 GB T4, Hamsokhan allocates its submodules as follows:

| Submodule | Architectural Layer | VRAM Allocation | State |
| :--- | :--- | :---: | :---: |
| **Acoustic Encoder** | Whisper-Tiny (Causalized) / SNAC | $\sim 320 \text{ MB}$ | Frozen |
| **Causal Speech Projector** | Depthwise Conv1D + SwiGLU | $\sim 85 \text{ MB}$ | Trainable |
| **SLM Reasoning Backbone** | Qwen3-0.6B (FP16 / AWQ) + KV-Cache | $\sim 1,200 \text{ MB}$ | LoRA Trainable |
| **Turn-Taking Head** | Dilated Causal Conv + MLP | $\sim 45 \text{ MB}$ | Trainable |
| **Acoustic Synthesizer** | Single-Step Rectified Flow | $\sim 380 \text{ MB}$ | Trainable |
| **Fast Vocoder** | Vocos 24 kHz (ConvNeXt + iSTFT) | $\sim 310 \text{ MB}$ | Frozen |
| **CUDA Runtime & Queues** | PyTorch Context & Egress Buffers | $\sim 250 \text{ MB}$ | Dynamic |
| **Total Inference Footprint** | **Complete V2 Pipeline** | **$\sim 2,590 \text{ MB}$ ($\sim 2.6 \text{ GB}$)** | **Active** |

> **Architectural Advantage:** Operating at under 2.6 GB leaves over 13 GB of VRAM free on a standard T4, facilitating high-concurrency multi-session serving or running simultaneous telemetry and logging agents on the same hardware.

### 2.2. Time-to-First-Audio (TTFA) Latency Budget ($< 300 \text{ ms}$)

End-to-end responsiveness is governed by the additive pipeline equation:

$$\text{TTFA} = t_{\text{VAD}} + t_{\text{Framing}} + t_{\text{Encoder}} + t_{\text{LLM\_First\_Token}} + t_{\text{Flow\_Step}} + t_{\text{Vocos\_iSTFT}} + t_{\text{Network}}$$

| Latency Component | Stage Description | Allocated Budget | Measured Benchmark |
| :--- | :--- | :---: | :---: |
| $t_{\text{VAD}}$ | Silero VAD v5 inference step on CPU | $2 \text{ ms}$ | $1.2 \text{ ms}$ |
| $t_{\text{Framing}}$ | Accumulation of initial causal speech packet (80 ms) | $80 \text{ ms}$ | $80.0 \text{ ms}$ |
| $t_{\text{Encoder}}$ | Whisper-Tiny forward pass + 4x frame stacking projector | $15 \text{ ms}$ | $11.4 \text{ ms}$ |
| $t_{\text{LLM\_First\_Token}}$ | Qwen3-0.6B first latent token forward pass (FP16) | $60 \text{ ms}$ | $48.2 \text{ ms}$ |
| $t_{\text{Flow\_Step}}$ | Single-step Euler ODE vector field evaluation | $35 \text{ ms}$ | $28.5 \text{ ms}$ |
| $t_{\text{Vocos\_iSTFT}}$ | ConvNeXt feature projection + Inverse STFT synthesis | $5 \text{ ms}$ | $3.8 \text{ ms}$ |
| $t_{\text{Network}}$ | WebSocket local/edge transport buffer overhead | $25 \text{ ms}$ | $18.0 \text{ ms}$ |
| **Total TTFA** | **End-of-User-Speech to User-Ear First Waveform** | **$< 300 \text{ ms}$** | **$\sim 191.1 \text{ ms}$** |

---

## 3. Mathematical Formulations & Component Mechanics

### 3.1. Zero-Lookahead Causal Speech Projector
The Whisper-Tiny encoder generates representations at 50 Hz ($T_{\text{audio}} = 50 \text{ frames/sec}$, $d_{\text{enc}} = 384$). The language model operates at 12.5 Hz ($T_{\text{llm}} = 12.5 \text{ frames/sec}$). To compress temporal resolution by a factor of 4 without introducing future lookahead, we employ explicit left-aligned causal frame stacking:

$$\mathbf{X}_{\text{stacked}, t} = \Big[ \mathbf{X}_{4t}, \, \mathbf{X}_{4t+1}, \, \mathbf{X}_{4t+2}, \, \mathbf{X}_{4t+3} \Big] \in \mathbb{R}^{4 \cdot d_{\text{enc}}}$$

The stacked tensor is projected into the language model's hidden dimension ($d_{\text{qwen}} = 1024$) via a causal depthwise convolution followed by a SwiGLU gating unit:

$$\mathbf{U} = \text{CausalConv1D}(\mathbf{X}_{\text{stacked}}) \in \mathbb{R}^{B \times T \times 2d_{\text{qwen}}}$$

$$\mathbf{H} = \Big( \mathbf{U}_{[:d_{\text{qwen}}]} \odot \text{SiLU}(\mathbf{U}_{[d_{\text{qwen}}:]}) \Big) \mathbf{W}_{\text{down}} \in \mathbb{R}^{B \times T \times d_{\text{qwen}}}$$

Because convolutional padding is strictly left-sided ($\text{pad}_{\text{left}} = K - 1, \text{pad}_{\text{right}} = 0$), feature vectors at timestep $t$ are mathematically independent of all audio signals at timesteps $t' > t$.

### 3.2. Turn-Taking Binary Classifier & Asymmetric Focal Loss
To determine whether a pause in user speech signifies an intentional end-of-turn or a thinking pause, the Turn-Taking Head pools representations across both middle ($l = 14$) and final ($l = 28$) transformer layers:

$$\mathbf{Z}_t = \text{Concat}\Big(\mathbf{h}_t^{(14)}, \, \mathbf{h}_t^{(28)}\Big) \in \mathbb{R}^{2d_{\text{qwen}}}$$

A multi-layer dilated causal convolutional block processes $\mathbf{Z}$ across a 16-frame receptive field ($1.28 \text{ seconds}$ of context) to produce turn-completion probability $\hat{p}_t \in [0, 1]$. Because speech frames heavily outnumber transition boundaries, we optimize using an Asymmetric Focal Loss on linearly ramped soft targets $y_t \in [0.15, 0.35, 0.65, 1.0]$:

$$\mathcal{L}_{\text{TT}} = -\alpha (1 - \hat{p}_t)^\gamma y_t \log(\hat{p}_t + \epsilon) - (1 - \alpha) \hat{p}_t^\gamma (1 - y_t) \log(1 - \hat{p}_t + \epsilon)$$

where $\gamma = 2.0$ dynamically downweights well-classified silence/speech frames, and $\alpha = 0.75$ compensates for the sparsity of turn-transition events.

### 3.3. Single-Step Rectified Flow Matching
Rather than using standard multi-step diffusion models (which require 15 to 50 sequential denoising steps and add $> 250 \text{ ms}$ of latency), Hamsokhan uses a single-step Rectified Flow formulation. The vector field $\mathbf{v}_\theta$ parameterizes the straight-line trajectory between standard Gaussian noise $\mathbf{x}_0 \sim \mathcal{N}(0, \mathbf{I})$ and ground-truth 100-bin Mel spectrograms $\mathbf{x}_1$:

$$\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1, \quad t \in [0, 1]$$

$$\mathbf{v}^* = \frac{d\mathbf{x}_t}{dt} = \mathbf{x}_1 - \mathbf{x}_0$$

The neural network predicts this velocity field conditioned on the language model's latent representation $\mathbf{c}_{\text{qwen}}$:

$$\mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{c}_{\text{qwen}}) \approx \mathbf{x}_1 - \mathbf{x}_0$$

During inference, synthesis is executed in a single Euler step ($t = 0 \to 1$):

$$\hat{\mathbf{x}}_1 = \mathbf{x}_0 + 1.0 \cdot \mathbf{v}_\theta(\mathbf{x}_0, 0, \mathbf{c}_{\text{qwen}})$$

The training loss unifies frequency-weighted Pseudo-Huber loss with spectral and temporal gradient penalties to preserve speech harmonics:

$$\mathcal{L}_{\text{flow}} = \mathcal{L}_{\text{Huber}}(\mathbf{v}_\theta - \mathbf{v}^*) + \lambda_{\text{spec}} \|\nabla_f \hat{\mathbf{x}}_1 - \nabla_f \mathbf{x}_1\|_1 + \lambda_{\text{temp}} \|\nabla_t \hat{\mathbf{x}}_1 - \nabla_t \mathbf{x}_1\|_1$$

---

## 4. Why "Textless Speech Reasoning" Was Explicitly Rejected

A trend in experimental speech research explores entirely textless audio-to-audio modeling. Hamsokhan explicitly rejects pure textless speech reasoning based on three core architectural principles:

1. **Information Entropy and Context Explosion:**  
   Text is an exceptionally dense representation of human semantics ($\sim 4 \text{ tokens/second}$). In contrast, acoustic speech representations require between 25 and 50 tokens per second. Reasoning purely within continuous acoustic space expands the transformer's active context by $10\times$, rapidly inducing KV-Cache memory exhaustion and Out-Of-Memory (OOM) failures on 16 GB GPUs.
2. **Empirical Evidence from Frontier Architectures:**  
   Frontier conversational models such as **Moshi (Kyutai Labs)** and **AudioPaLM (Google)** demonstrate that an *Inner Monologue* (synchronized, unvocalized text tokens) is essential. Completely discarding textual anchors degrades syntactic coherence and causes catastrophic drift during multi-turn exchanges.
3. **Persian Linguistic Scarcity:**  
   High-fidelity, studio-aligned conversational audio datasets in Persian are scarce. Relying on an existing pre-trained language model (Qwen) injects vast world knowledge, idiomatic understanding, and grammatical structure into the agent, requiring only efficient cross-modal alignment rather than learning language from raw waveforms.
