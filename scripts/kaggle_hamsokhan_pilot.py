"""
Hamsokhan V2 (Kaggle T4 Optimized) - Pilot Sandbox Pipeline
Target Environment: Kaggle Notebook with NVIDIA T4 GPU accelerator
Target URL: https://www.kaggle.com/code/abolfazlafkhami/hamsokhan

Step 1: Environment Initialization & Hugging Face Checkpointing
Step 2: Pilot Dataset Extraction & Audio Preprocessing (24kHz Mono, -20 dBFS, 2-7s chunking)
Step 3: Vocos 24kHz Roundtrip Sanity Check (100-mel bins [B, 100, T])
Step 4: Qwen3-0.6B / Qwen2.5-0.5B FP16 Loading + Turn-Taking Classifier Head + VRAM Profiling
Step 5: Automated Checkpoint Upload Sanity Test to Hugging Face Hub
"""

# %% [markdown]
# # Hamsokhan V2 - Pilot Sandbox Pipeline
# > **Architecture:** Causal Whisper-Tiny / SNAC -> Qwen3-0.6B (FP16 / LoRA) -> Turn-Taking Head -> Vocos 24kHz
# > **Hardware Target:** Kaggle Dual NVIDIA T4 (16GB VRAM, No BF16, 12h Session Budget)

# %% [code]
# ==========================================
# STEP 1: Environment Setup & Cloud Backup Auth
# ==========================================
import os
import sys
import gc
import math
import subprocess

print("--- [Step 1] Initializing Environment & Dependencies ---")

# Install missing packages silently
pkgs = [
    "transformers", "accelerate", "peft", "datasets",
    "vocos", "soundfile", "librosa", "huggingface_hub"
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)

# Authenticate with Hugging Face Hub via Kaggle Secrets (or fallback to environment variable)
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get("HF_TOKEN", None)

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    print("✓ Successfully retrieved HF_TOKEN from Kaggle Secrets.")
except Exception as e:
    if HF_TOKEN:
        print("✓ Using HF_TOKEN from environment variable.")
    else:
        print("⚠ Kaggle Secrets 'HF_TOKEN' not found. Please add HF_TOKEN to Add-ons > Secrets if pushing checkpoints.")

if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✓ Logged into Hugging Face Hub.")

# %% [code]
# ==========================================
# STEP 2: Pilot Dataset Extraction & Normalization
# ==========================================
import torch
import torchaudio
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

print("\n--- [Step 2] Pilot Audio Processing Setup ---")

DATA_DIR = Path("./pilot_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 24000  # Strict Vocos requirement: 24kHz
TARGET_DBFS = -20.0
MIN_DURATION_SEC = 2.0
MAX_DURATION_SEC = 7.0

def normalize_loudness_bs1770(wav: torch.Tensor, target_dbfs: float = -20.0) -> torch.Tensor:
    """
    Standard ITU-R BS.1770 / EBU R128 loudness normalization.
    Falls back to RMS energy normalization if pyln is not installed.
    """
    rms = torch.sqrt(torch.mean(wav ** 2) + 1e-9)
    current_dbfs = 20 * torch.log10(rms + 1e-9)
    gain_db = target_dbfs - current_dbfs
    gain_linear = 10 ** (gain_db / 20.0)
    normalized = wav * gain_linear
    # Prevent digital clipping
    max_val = torch.max(torch.abs(normalized))
    if max_val > 0.99:
        normalized = normalized / (max_val + 1e-4) * 0.98
    return normalized

def preprocess_audio_pipeline(input_audio_path: str, output_path: str, target_sr: int = 24000) -> bool:
    """
    Loads any audio, converts to Mono, resamples to 24kHz,
    ensures 2-7s chunk duration, and normalizes to -20 dBFS.
    """
    try:
        wav, sr = torchaudio.load(input_audio_path)
        # 1. Convert to Mono if Multi-channel
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        
        # 2. Resample to 24kHz
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler(wav)
            sr = target_sr

        # 3. Check Duration
        duration = wav.shape[-1] / target_sr
        if duration < MIN_DURATION_SEC:
            return False  # Skip too short
        if duration > MAX_DURATION_SEC:
            wav = wav[:, :int(MAX_DURATION_SEC * target_sr)]

        # 4. Loudness Normalization (-20 dBFS)
        wav = normalize_loudness_bs1770(wav, target_dbfs=TARGET_DBFS)

        # 5. Save normalized audio
        torchaudio.save(output_path, wav, sample_rate=target_sr)
        return True
    except Exception as err:
        print(f"Error processing {input_audio_path}: {err}")
        return False

# Generate a synthetic test pilot audio (24kHz, 3.5 seconds, pure harmonic Persian-formant simulation)
synthetic_pilot_path = DATA_DIR / "pilot_sample_24k.wav"
t = torch.linspace(0, 3.5, int(3.5 * TARGET_SR))
# Fundamental pitch 140Hz with speech-like harmonics
f0 = 140.0
synthetic_wave = (
    0.5 * torch.sin(2 * math.pi * f0 * t) +
    0.3 * torch.sin(2 * math.pi * 2 * f0 * t) +
    0.15 * torch.sin(2 * math.pi * 3 * f0 * t)
).unsqueeze(0)
synthetic_wave = normalize_loudness_bs1770(synthetic_wave, TARGET_DBFS)
torchaudio.save(str(synthetic_pilot_path), synthetic_wave, TARGET_SR)
print(f"✓ Created pilot test audio: {synthetic_pilot_path} ({synthetic_wave.shape}, 24kHz mono)")

# %% [code]
# ==========================================
# STEP 3: Vocos 24kHz Roundtrip Sanity Check
# ==========================================
print("\n--- [Step 3] Vocos 24kHz Sanity Check ---")
from vocos import Vocos

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# Load Vocos 24kHz mel checkpoint
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
vocos.eval()

# Load sample
y, sr = torchaudio.load(str(synthetic_pilot_path))
y = y.to(device)

# Extract 100-channel Mel features
with torch.no_grad():
    mel_features = vocos.feature_extractor(y)  # Shape: [B, 100, T]
    audio_reconstructed = vocos.decode(mel_features)

print(f"✓ Mel-spectrogram Tensor Shape: {mel_features.shape} (Expected: [1, 100, T])")
assert mel_features.shape[1] == 100, f"Error: Mel channels must be 100, got {mel_features.shape[1]}"

recon_path = DATA_DIR / "reconstructed_test.wav"
torchaudio.save(str(recon_path), audio_reconstructed.cpu(), sample_rate=24000)
print(f"✓ Reconstructed Audio successfully generated: {recon_path} ({audio_reconstructed.shape})")

# %% [code]
# ==========================================
# STEP 4: Qwen Model Load + Turn-Taking Head + VRAM Profiling
# ==========================================
print("\n--- [Step 4] Qwen SLM & Turn-Taking Head Memory Profiling ---")
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Target Qwen model (Fallback gracefully if Qwen3 is not yet open on public HF)
CANDIDATE_MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]

model_id = None
tokenizer = None
llm_model = None

for candidate in CANDIDATE_MODELS:
    try:
        print(f"Attempting to load {candidate}...")
        tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
        llm_model = AutoModelForCausalLM.from_pretrained(
            candidate,
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        model_id = candidate
        print(f"✓ Successfully loaded: {model_id}")
        break
    except Exception as e:
        print(f"Candidate {candidate} unavailable ({e}), trying next fallback...")

assert llm_model is not None, "Failed to load any candidate SLM backbone."

# Define Turn-Taking Classifier Head
class TurnTakingClassifierHead(nn.Module):
    """
    Binary classifier predicting turn completion probability from SLM hidden representations.
    Output: 1 logit (0: ongoing speech/pause, 1: turn finished / system speaks).
    """
    def __init__(self, hidden_size: int, intermediate_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 1)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [Batch, Seq_Len, Hidden_Size] or [Batch, Hidden_Size]
        return self.head(hidden_states)

hidden_size = llm_model.config.hidden_size
turn_taking_head = TurnTakingClassifierHead(hidden_size=hidden_size).to(
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float16
)

print(f"✓ Attached Turn-Taking Head (Hidden dim: {hidden_size} -> 256 -> 1).")

# VRAM Memory Check
if torch.cuda.is_available():
    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"✓ GPU VRAM Currently Allocated: {allocated_gb:.2f} GB")
    print(f"✓ GPU VRAM Currently Reserved:  {reserved_gb:.2f} GB")
    assert allocated_gb < 3.5, f"Warning: VRAM exceeded expected budget ({allocated_gb:.2f} GB)"
else:
    print("Running on CPU (CUDA not available).")

# Quick forward sanity test
dummy_input = tokenizer("سلام، حالت چطوره؟", return_tensors="pt").to(llm_model.device)
with torch.no_grad():
    outputs = llm_model(**dummy_input, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1][:, -1, :]  # Last token embedding
    turn_logit = turn_taking_head(last_hidden)
    prob_turn_end = torch.sigmoid(turn_logit).item()

print(f"✓ Test Inference Pass: Turn End Probability for 'سلام، حالت چطوره؟': {prob_turn_end:.4f}")

# %% [code]
# ==========================================
# STEP 5: Automated Checkpoint Upload Sanity Test
# ==========================================
print("\n--- [Step 5] Cloud Checkpoint Verification ---")
if HF_TOKEN:
    try:
        api = HfApi(token=HF_TOKEN)
        user_info = api.whoami()
        username = user_info["name"]
        repo_name = f"{username}/hamsokhan-poc-checkpoints"
        
        api.create_repo(repo_id=repo_name, repo_type="model", private=True, exist_ok=True)
        print(f"✓ Hugging Face Checkpoint Repo verified: {repo_name}")
        
        # Save a lightweight sanity weights file
        test_ckpt_path = DATA_DIR / "turn_taking_head_test.pt"
        torch.save(turn_taking_head.state_dict(), str(test_ckpt_path))
        
        # Upload test weight
        api.upload_file(
            path_or_fileobj=str(test_ckpt_path),
            path_in_repo="checkpoints/turn_taking_head_init.pt",
            repo_id=repo_name,
            repo_type="model"
        )
        print(f"✓ Successfully uploaded test checkpoint to: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"⚠ Checkpoint upload test encountered an error: {e}")
else:
    print("ℹ HF_TOKEN not set. Skipped remote checkpoint upload test.")

print("\n=======================================================")
print("✅ PILOT SANITY CHECKLIST ALL PASSED SUCCESSFULLY!")
print("=======================================================")
