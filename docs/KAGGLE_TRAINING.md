# Hamsokhan: Kaggle Dual Tesla T4 Distributed Training Guide

> **Hardware Specification:** 2x NVIDIA Tesla T4 GPUs (16 GB GDDR6 VRAM each, Turing `sm_75`)  
> **Distributed Framework:** PyTorch Distributed Data Parallel (DDP) via Hugging Face `accelerate`  
> **Target Model Hub:** [huggingface.co/Abolfazl27Aflhami88/Hamsokhan](https://huggingface.co/Abolfazl27Aflhami88/Hamsokhan)  
> **Target Shard Hub:** [huggingface.co/datasets/Abolfazl27Aflhami88/hamsokhan-shards](https://huggingface.co/datasets/Abolfazl27Aflhami88/hamsokhan-shards)

---

## 1. Overview & Distributed Execution Topology

Training modern speech-to-speech architectures typically requires multi-GPU enterprise infrastructure (such as $8\times$ A100 or H100 pods). Hamsokhan circumvents this barrier by distributing the training of its causal projector, Dual-Track LoRA adapters, Turn-Taking classification head, and Single-Step Rectified Flow synthesizer across **two commodity Tesla T4 GPUs** on Kaggle.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             Kaggle Dual T4 Execution                             │
│                                                                                  │
│   [GPU 0: T4 16 GB] (Rank 0 - Master)          [GPU 1: T4 16 GB] (Rank 1)        │
│   • DDP Local Batch (FP16 Native)              • DDP Local Batch (FP16 Native)   │
│   • Forward / Backward Pass                    • Forward / Backward Pass         │
│   • Non-Blocking Background Daemon             • Worker Gradient Sync            │
│                           │                              │                       │
│                           └──────────────┬───────────────┘                       │
│                                          ▼                                       │
│                        Gradient All-Reduce via NCCL                              │
│                     (NCCL_P2P_DISABLE=1, NCCL_IB_DISABLE=1)                      │
│                                          │                                       │
│                                          ▼                                       │
│           [Asynchronous Non-blocking Thread (Daemon)]                             │
│           • Snapshot weights to CPU RAM                                          │
│           • Serialize .safetensors (Zero-Copy)                                   │
│           • Auto-Upload to Hugging Face Model Hub (Every 500 steps)              │
│           • Local Rolling Pruning (Delete step_<N-2> to save disk)               │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Platform Constraints & Systems Engineering Solutions

Operating inside the free Kaggle GPU runtime presents four critical systems challenges. Hamsokhan resolves each through targeted low-level optimizations:

### 2.1. Mitigating the 20 GB Disk Quota Limit (`/kaggle/working`)
Downloading multi-speaker Persian audio datasets and extracting raw `.wav` archives rapidly exceeds the strict 20 GB root volume, throwing `OSError: [Errno 28] No space left on device`.
* **Engineering Solution:** Kaggle provisions an unmetered, ephemeral temporary mount at `/kaggle/tmp`. By configuring global environment variables prior to importing machine learning frameworks, all large caching operations are offloaded away from the persistent workspace:
  ```python
  os.environ["HF_HOME"] = "/kaggle/tmp/hf_home"
  os.environ["HF_DATASETS_CACHE"] = "/kaggle/tmp/hf_datasets_cache"
  os.environ["TORCH_HOME"] = "/kaggle/tmp/torch_cache"
  os.environ["TMPDIR"] = "/kaggle/tmp"
  ```
  The primary `/kaggle/working` directory is reserved strictly for compiled, zero-copy `.safetensors` dataset shards.

### 2.2. FP16 Numerical Stability on Turing GPUs (No BF16 Support)
The Turing architecture (Compute Capability 7.5, Tesla T4) does not possess hardware units for Brain Floating Point (`bfloat16`). Emulating `bfloat16` in software destroys training throughput, while uncalibrated `float16` suffers catastrophic underflow in the flow matching ODE gradients or explosive overflow in the asymmetric focal loss.
* **Engineering Solution:** We enforce pure FP16 execution stabilized via a custom-calibrated `GradScaler`:
  ```python
  from accelerate.utils import GradScalerKwargs

  scaler_kwargs = GradScalerKwargs(
      init_scale=2**14,      # Initial scale factor of 16,384 prevents initial zero-grad underflow
      growth_interval=1000,  # Slow upward scale adjustment prevents sudden NaN loss spikes
  )
  ```
  Additionally, `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` is set to prevent CUDA memory pool fragmentation over long training runs.

### 2.3. Resolving NCCL Inter-GPU Deadlocks
Kaggle's virtualized container network occasionally hangs during peer-to-peer (P2P) CUDA inter-process handshakes, causing the PyTorch DDP all-reduce phase to hang indefinitely.
* **Engineering Solution:** We disable P2P and InfiniBand kernel invocations, forcing NCCL to communicate via shared system memory:
  ```python
  os.environ["NCCL_P2P_DISABLE"] = "1"
  os.environ["NCCL_IB_DISABLE"] = "1"
  ```

### 2.4. Surviving the 12-Hour Session Kill Window
Kaggle kernel sessions are terminated after 12 hours of continuous execution. Training runs that fail to save state are permanently lost.
* **Engineering Solution:** `train_hamsokhan_ddp.py` integrates an asynchronous, multi-threaded checkpointing daemon (`AsyncHFHubCheckpointManager`):
  1. Rank 0 clones trainable parameters into pinned host RAM (CPU memory).
  2. A background worker thread serializes tensors to `.safetensors` format without stalling GPU compute.
  3. The worker pushes `best_model` and `latest` checkpoints directly to the target Hugging Face repository via `huggingface_hub.HfApi`.
  4. Rolling pruning deletes historical step directories from local disk, keeping local storage under 2 GB.

---

## 3. Kaggle Notebook Walkthrough (`hamsokhan.ipynb`)

The orchestration notebook is structured into 5 deterministic execution cells:

### Cell 1: Environment & Hardware Isolation
* Verifies dual GPU availability (`torch.cuda.device_count() == 2`).
* Configures `/kaggle/tmp` isolation paths.
* Clones the active `mahdi` branch of `Abolfazl-Afkhami/Hamsokhan`.
* Installs compiled dependencies: `accelerate`, `safetensors`, `soundfile`, `librosa`.

### Cell 2 & 3: Hugging Face Authentication & Checkpoint Recovery
* Retrieves the user's `HF_TOKEN` securely via `kaggle_secrets.UserSecretsClient`.
* Logs in to the Hugging Face API and provisions repositories:
  - Model Hub: `Abolfazl27Aflhami88/Hamsokhan`
  - Dataset Hub: `Abolfazl27Aflhami88/hamsokhan-shards`
* Queries the Model Hub; if a prior checkpoint exists, it is automatically restored to resume training seamlessly.

### Cell 4 & 5: Idempotent Shard Synchronization
* Checks the Hugging Face Dataset Hub for existing pre-built shards.
* If absent or if `FORCE_REBUILD_SHARDS = True`, triggers `scripts/kaggle_dataset_builder.py`:
  - Curates audio from **Thomcles (417h)**, **ManaTTS (114h)**, **PersianVox (150h)**, and **Synthetic Diarization (60h)**.
  - Normalizes audio to $-20 \text{ dBFS}$ at 24 kHz.
  - Extracts 100-bin log-Mel spectrograms and generates zero-copy `.safetensors` binary shards for `train`, `val`, and `test` splits.
  - Pushes shards to `Abolfazl27Aflhami88/hamsokhan-shards`.

### Cell 6 & 7: Dual-GPU DDP Training Launch
* Dispatches distributed execution via `accelerate launch`:
  ```bash
  accelerate launch \
      --multi_gpu \
      --num_processes 2 \
      --mixed_precision fp16 \
      train_hamsokhan_ddp.py \
      --shard_dir /kaggle/working/cached_shards \
      --output_dir /kaggle/working/ckpts \
      --batch_size 4 \
      --gradient_accumulation_steps 4 \
      --lr_backbone 5e-5 \
      --lr_synthesizer 2e-4 \
      --max_steps 15000
  ```
* Spawns the non-blocking checkpoint sync thread.

### Cell 8 & 9: Deterministic Evaluation & Production Promotion
* Loads the best checkpoint strictly against held-out `test` shards.
* Runs `scripts/eval_hamsokhan.py` to compute:
  - Mel L1 Reconstruction Loss
  - Spectral Gradient Error (`SpecGrad`)
  - Turn-Taking Binary Classification Accuracy & F1-score
  - Time-to-First-Audio (TTFA) benchmark
* Synthesizes and plays a 24 kHz Persian audio sample within the notebook output.
* Tags and promotes verified model weights as production-ready artifacts on the Hugging Face Model Hub.
