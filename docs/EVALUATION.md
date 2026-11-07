# Hamsokhan: Evaluation Metrics & Verification Suite Specification

> **Quality Gate Standard:** Production-Grade Conversational Speech-to-Speech  
> **Key Target Benchmarks:** TTFA $< 300 \text{ ms}$ | UTMOS $> 3.65$ | MCD $< 4.5 \text{ dB}$ | WER $< 8.5\%$ | VRAM $< 3.0 \text{ GB}$

---

## 1. Quantitative Evaluation Metrics

To guarantee conversational fluency, natural acoustic timbre, and system stability under production workloads, Hamsokhan enforces seven quantitative performance metrics:

| Metric | Scientific Notation | Production Target | Evaluation Methodology | Engineering Significance |
| :--- | :---: | :---: | :--- | :--- |
| **Time-to-First-Audio** | **TTFA** | **$< 300 \text{ ms}$** | Wall-clock latency from the user's final speech sample to the first synthesized 24 kHz audio sample | Primary metric governing conversational immersion and fluid turn-taking |
| **Barge-in Reaction Time** | **RT$_{\text{barge}}$** | **$< 20 \text{ ms}$** | Elapsed time from interrupt packet receipt to complete server queue flush and driver silence | Prevents the assistant from talking over the user during interruptions |
| **Speech Naturalness** | **UTMOS** | **$> 3.65$** | Pre-trained neural MOS predictor assessing acoustic naturalness, prosody, and absence of robotic artifacts | Validates conversational warmth and colloquial Persian inflection |
| **Mel-Cepstral Distortion** | **MCD** | **$< 4.5 \text{ dB}$** | Dynamic Time Warping (DTW) distance between synthesized and reference Mel-frequency cepstral coefficients | Ensures fidelity to target speaker characteristics and timbre |
| **Word Error Rate** | **WER** | **$< 8.5\%$** | Automated transcription of synthesized audio via `whisper-large-v3` compared against ground-truth text | Guarantees clear phonemic articulation and absence of dropped syllables |
| **Turn-Taking Accuracy** | **TT-F1** | **$> 85\%$** | F1-score of the Turn-Taking classification head on held-out PersianVox dialogues | Distinguishes mid-sentence thinking pauses from true conversation turn completions |
| **Peak Inference VRAM** | **VRAM$_{\text{max}}$** | **$< 3.0 \text{ GB}$** | Peak active CUDA memory recorded via `torch.cuda.max_memory_allocated` | Guarantees multi-session concurrency on a single 16 GB Tesla T4 GPU |

---

## 2. Test Suite Architecture

All automated test suites reside within the `tests/` directory and are partitioned into two testing tiers:

```
tests/
├── unit/
│   ├── test_causal_masking.py       # Validates zero-lookahead independence of future audio frames
│   ├── test_flow_synthesizer.py     # Confirms numerical stability of single-step flow synthesis in FP16
│   ├── test_turn_taking_head.py     # Asserts probability calibration and batch dimension invariance
│   └── test_vocos_continuity.py     # Measures phase continuity and absence of boundary clicks in Vocos
└── integration/
    ├── test_e2e_pipeline.py         # Full pipeline execution from audio input to synthesized output (< 3GB VRAM)
    └── test_barge_in_latency.py     # Simulates mid-utterance cancellation and measures queue flush latency
```

---

## 3. Detailed Test Specifications

### 3.1. Unit Test Suite (`tests/unit/`)

#### 1. Zero-Lookahead Causal Masking (`test_causal_masking.py`)
* **Objective:** Mathematically prove that the output of `CausalSpeechProjector` at frame $t$ is strictly independent of any audio input at timesteps $t' > t$.
* **Assertion:**
  1. Generate a random audio input tensor $\mathbf{X}_1$ of length $T_1$ and record projector output $\mathbf{Y}_1$.
  2. Concatenate random audio noise $\mathbf{X}_{\text{future}}$ of length $\Delta T$ to form $\mathbf{X}_2 = [\mathbf{X}_1, \mathbf{X}_{\text{future}}]$ of length $T_2 = T_1 + \Delta T$, and record output $\mathbf{Y}_2$.
  3. Assert exact invariance over the common historical window:
     $$\max_{0 \le t \le T_1} |\mathbf{Y}_1[t] - \mathbf{Y}_2[t]| < 10^{-5}$$

#### 2. Single-Step Flow Numerical Stability (`test_flow_synthesizer.py`)
* **Objective:** Ensure that the single Euler step in Rectified Flow matching never generates `NaN` or `Inf` values under half-precision FP16 execution on Turing T4 GPUs.
* **Assertion:**
  Pass randomized extreme latent conditioning vectors $\mathbf{c} \sim \mathcal{N}(0, 5\mathbf{I})$ and verify:
  ```python
  assert torch.isfinite(mel_spectrogram).all().item() is True
  ```

#### 3. Vocos Boundary Phase Continuity (`test_vocos_continuity.py`)
* **Objective:** Verify that the 15 ms Overlap-Add (OLA) raised-cosine crossfade eliminates high-frequency spectral phase discontinuities between consecutive 80 ms audio packets.
* **Assertion:**
  Compute the Short-Time Fourier Transform (STFT) across the junction between chunk $N$ and chunk $N+1$, asserting that the spectral gradient norm at the boundary does not exceed continuous audio baselines by more than $5\%$.

### 3.2. Integration Test Suite (`tests/integration/`)

#### 1. End-to-End Latency & Memory Verification (`test_e2e_pipeline.py`)
* Feed a continuous 10-second Persian conversational speech stream into the complete pipeline.
* Benchmark wall-clock Time-to-First-Audio (TTFA) across 100 trials, asserting:
  $$\text{TTFA}_{95\%} < 300 \text{ ms}$$
* Monitor CUDA memory utilization throughout the session, asserting:
  $$\max(\text{VRAM}_{\text{allocated}}) < 3.0 \text{ GB}$$

#### 2. Barge-in Latency & KV-Cache Truncation (`test_barge_in_latency.py`)
* Initiate a lengthy speech synthesis sequence.
* Inject a synthetic `interrupt` control frame at $t = 1,200 \text{ ms}$.
* Measure the elapsed time until the server egress queue is confirmed empty, asserting:
  $$t_{\text{flush}} < 20 \text{ ms}$$
* Inspect the transformer's internal attention cache to verify that all Key-Value pairs corresponding to timesteps $t > 1,200 \text{ ms}$ were pruned.

---

## 4. Test Execution Commands

```bash
# 1. Run all unit tests with full traceback on failure
pytest tests/unit/ -v --tb=short

# 2. Run GPU-accelerated integration tests
pytest tests/integration/ -v -s

# 3. Generate a combined test coverage report
pytest --cov=src --cov=hamsokhan_model tests/ --cov-report=term-missing
```
