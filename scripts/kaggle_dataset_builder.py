"""
Hamsokhan — Kaggle Autonomous Dataset Preprocessing & Sharding Pipeline
========================================================================
Designed to run in a standalone Kaggle Notebook with GPU acceleration.
Processes Layer 1 (Thomcles, ManaTTS, YouTube Clean) and Layer 2 (PersianVox).
Features:
1. Streaming download (streaming=True) from Hugging Face: Zero disk waste on raw uncompressed audio.
2. Partitioning support (--part_idx and --num_parts) to cleanly stay within Kaggle 20GB limits.
3. GPU-accelerated Whisper-Tiny encoder (50Hz) and Vocos-aligned Mel-spectrogram (93.75Hz) extraction.
4. Smart Persian VAD trimming with Room Tone preservation (160ms onset, 240/320ms offset).
5. Output format: Contiguous Sharded .safetensors with index_table [N, 6] ready to be added as a Kaggle Dataset.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import torchaudio.functional as AF
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoFeatureExtractor, AutoModel

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mel_extractor import VocosAlignedMelExtractor, smart_persian_vad_trim
from src.data.persianvox_labeler import build_persianvox_soft_labels
from src.nlp.normalizer import PersianTextNormalizer


class KaggleDatasetBuilder:
    """
    Autonomous streaming processor that builds memory-efficient .safetensors shards
    from Hugging Face datasets directly in a Kaggle notebook.
    """
    def __init__(
        self,
        output_dir: str = "/kaggle/working/hamsokhan_shards",
        part_idx: int = 0,
        num_parts: int = 1,
        max_hours_per_part: float = 20.0,
        shard_size_samples: int = 500,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.part_idx = part_idx
        self.num_parts = num_parts
        self.max_hours = max_hours_per_part
        self.shard_size = shard_size_samples
        self.device = torch.device(device)

        print(f"[DatasetBuilder] Initializing on device: {self.device} | Part {part_idx + 1}/{num_parts}")

        # 1. Mel Extractor (Vocos 24kHz aligned)
        self.mel_extractor = VocosAlignedMelExtractor(sample_rate=24000, n_mels=100).to(self.device).eval()

        # 2. Whisper-Tiny Encoder (Frozen, FP16)
        print("[DatasetBuilder] Loading Whisper-Tiny audio encoder...")
        self.whisper_model_id = "openai/whisper-tiny"
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(self.whisper_model_id)
        whisper = AutoModel.from_pretrained(
            self.whisper_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device).eval()
        self.whisper_encoder = whisper.encoder
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False

        # 3. Text Normalizer
        self.text_normalizer = PersianTextNormalizer(convert_numbers=True)

        # Buffer for active shard
        self.curr_shard_idx = part_idx * 1000  # Unique shard prefix per part
        self._reset_shard_buffers()

    def _reset_shard_buffers(self):
        self.whisper_list: List[torch.Tensor] = []
        self.mel_list: List[torch.Tensor] = []
        self.f0_list: List[torch.Tensor] = []
        self.label_list: List[torch.Tensor] = []
        self.index_rows: List[List[int]] = []
        self.w_offset = 0
        self.m_offset = 0
        self.f0_offset = 0
        self.l_offset = 0
        self.samples_in_curr_shard = 0

    @staticmethod
    def normalize_loudness_bs1770(wav: torch.Tensor, target_dbfs: float = -20.0) -> torch.Tensor:
        """Standard ITU-R BS.1770 loudness normalization to -20 dBFS."""
        rms = torch.sqrt(torch.mean(wav ** 2) + 1e-9)
        current_dbfs = 20.0 * torch.log10(rms + 1e-9)
        gain_db = target_dbfs - current_dbfs
        gain_linear = 10.0 ** (gain_db / 20.0)
        normalized = wav * gain_linear
        max_val = torch.max(torch.abs(normalized))
        if max_val > 0.99:
            normalized = normalized / (max_val + 1e-4) * 0.98
        return normalized

    @staticmethod
    def extract_log_f0(wav_24k: torch.Tensor, sr: int = 24000, hop_length: int = 256, target_len: Optional[int] = None) -> torch.Tensor:
        """
        Extracts continuous normalized log fundamental frequency (log-F0) contour.
        Zero-mean unit-variance normalized on voiced frames.
        """
        import numpy as np
        import librosa
        wav_np = wav_24k.squeeze(0).cpu().numpy().astype(np.float32)
        f0 = librosa.yin(wav_np, fmin=50, fmax=550, sr=sr, hop_length=hop_length)
        f0 = np.nan_to_num(f0, nan=0.0)
        voiced = f0 > 0.0
        log_f0 = np.zeros_like(f0)
        if voiced.any():
            log_f0[voiced] = np.log(f0[voiced])
            mean_val = np.mean(log_f0[voiced])
            std_val = np.std(log_f0[voiced]) + 1e-6
            log_f0[voiced] = (log_f0[voiced] - mean_val) / std_val
        tensor_f0 = torch.from_numpy(log_f0).to(torch.float16)
        if target_len is not None:
            if tensor_f0.shape[-1] < target_len:
                tensor_f0 = torch.nn.functional.pad(tensor_f0, (0, target_len - tensor_f0.shape[-1]))
            else:
                tensor_f0 = tensor_f0[:target_len]
        return tensor_f0

    @torch.no_grad()
    def extract_whisper_features(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """
        wav_16k: [1, N_16k] at 16000Hz.
        Returns: [T_w, 384] audio embeddings at 50Hz (T_w is a multiple of 8).
        """
        # Quantize audio to multiple of 160ms (2560 samples @ 16kHz = 8 Whisper frames)
        block_16k = 2560
        rem = wav_16k.shape[-1] % block_16k
        if rem != 0:
            wav_16k = torch.nn.functional.pad(wav_16k, (0, block_16k - rem), mode="reflect")

        num_macro = wav_16k.shape[-1] // block_16k
        target_t_w = num_macro * 8

        inputs = self.whisper_feature_extractor(
            wav_16k.squeeze(0).cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device, dtype=self.whisper_encoder.dtype)
        # Forward pass through Whisper encoder
        enc_out = self.whisper_encoder(input_features).last_hidden_state.squeeze(0)  # [1500, 384]

        # Slice exact macro-packet frames
        whisper_feat = enc_out[:target_t_w, :].to(torch.float16)
        return whisper_feat

    def process_and_add_sample(
        self,
        wav_audio: torch.Tensor,
        orig_sr: int,
        vad_segments: Optional[List[Dict]] = None,
        is_turn_taking: bool = False,
    ) -> bool:
        """
        Processes a single raw audio item:
        1. Mono conversion + 24kHz resampling (for Vocos) + 16kHz resampling (for Whisper).
        2. Smart VAD trimming (160ms onset, 240/320ms offset).
        3. Extraction of Vocos Mel [100, 15*K] and Whisper [8*K, 384].
        4. Generation of Soft Turn-Taking labels [2*K].
        5. Appends to active shard.
        """
        try:
            if wav_audio.dim() == 1:
                wav_audio = wav_audio.unsqueeze(0)
            if wav_audio.shape[0] > 1:
                wav_audio = torch.mean(wav_audio, dim=0, keepdim=True)

            # Resample to 24kHz for Vocos
            if orig_sr != 24000:
                wav_24k = AF.resample(wav_audio, orig_sr, 24000)
            else:
                wav_24k = wav_audio

            # Duration check (filter out < 1.5s or > 15s)
            dur_sec = wav_24k.shape[-1] / 24000.0
            if dur_sec < 1.5 or dur_sec > 15.0:
                return False

            # Loudness Normalization (-20 dBFS BS.1770)
            wav_24k = self.normalize_loudness_bs1770(wav_24k, target_dbfs=-20.0)

            # Smart VAD trimming with Room Tone
            pad_end = 320 if is_turn_taking else 240
            wav_24k_trimmed = smart_persian_vad_trim(
                wav_24k,
                vad_timestamps_sec=vad_segments,
                pad_start_ms=160,
                pad_end_ms=pad_end,
                sr=24000
            )

            # Re-verify duration post-trimming
            dur_trimmed = wav_24k_trimmed.shape[-1] / 24000.0
            if dur_trimmed < 1.0 or dur_trimmed > 15.0:
                return False

            # Resample trimmed audio to 16kHz for Whisper
            wav_16k_trimmed = AF.resample(wav_24k_trimmed, 24000, 16000)

            # 1. Vocos Mel Spectrogram [100, T_mel] (T_mel is 15 * K)
            mel = self.mel_extractor(wav_24k_trimmed.to(self.device)).squeeze(0)  # [100, T_mel]
            T_mel = mel.shape[-1]
            num_macro = T_mel // 15
            if num_macro < 1:
                return False

            # 2. Whisper-Tiny Features [T_w, 384] (T_w is 8 * K)
            whisper_feat = self.extract_whisper_features(wav_16k_trimmed)  # [8*K, 384]

            # Re-align lengths to exact macro packets
            min_macro = min(num_macro, whisper_feat.shape[0] // 8)
            t_w = min_macro * 8
            t_mel = min_macro * 15
            t_llm = min_macro * 2

            whisper_feat = whisper_feat[:t_w, :].cpu()
            mel = mel[:, :t_mel].cpu()

            # 3. Feature Engineering: Continuous Normalized Pitch Contour (log-F0)
            f0_contour = self.extract_log_f0(wav_24k_trimmed, target_len=t_mel).cpu()

            # 4. Turn-Taking Labels
            if is_turn_taking and vad_segments:
                labels = build_persianvox_soft_labels(
                    vad_segments, total_frames_12_5hz=t_llm
                ).cpu()
            else:
                # Standard TTS data: turn ends at the very last frame
                labels = torch.zeros(t_llm, dtype=torch.float16)
                if t_llm >= 6:
                    labels[-6:] = torch.tensor([0.15, 0.35, 0.65, 1.0, 1.0, 1.0], dtype=torch.float16)

            # Append to buffer
            self.whisper_list.append(whisper_feat)
            self.mel_list.append(mel)
            self.f0_list.append(f0_contour)
            self.label_list.append(labels)

            self.index_rows.append([
                self.w_offset, self.w_offset + t_w,
                self.m_offset, self.m_offset + t_mel,
                self.f0_offset, self.f0_offset + t_mel,
                self.l_offset, self.l_offset + t_llm,
            ])

            self.w_offset += t_w
            self.m_offset += t_mel
            self.f0_offset += t_mel
            self.l_offset += t_llm
            self.samples_in_curr_shard += 1

            if self.samples_in_curr_shard >= self.shard_size:
                self.save_current_shard()

            return True
        except Exception as err:
            return False

    def save_current_shard(self):
        """Flushes buffered samples to a contiguous .safetensors shard on disk."""
        if not self.whisper_list:
            return

        shard_filename = self.output_dir / f"features_shard_{self.curr_shard_idx:04d}.safetensors"
        print(f"  [Disk Write] Saving shard: {shard_filename.name} ({self.samples_in_curr_shard} utterances)...")

        tensors = {
            "whisper_flat": torch.cat(self.whisper_list, dim=0).contiguous(),
            "mel_flat": torch.cat(self.mel_list, dim=1).contiguous(),
            "f0_flat": torch.cat(self.f0_list, dim=0).contiguous(),
            "labels_flat": torch.cat(self.label_list, dim=0).contiguous(),
            "index_table": torch.tensor(self.index_rows, dtype=torch.int64).contiguous(),
        }
        save_file(tensors, str(shard_filename))

        self.curr_shard_idx += 1
        self._reset_shard_buffers()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def build_from_huggingface(
        self,
        dataset_name: str = "Thomcles/Persian-Farsi-Speech",
        split: str = "train",
        audio_col: str = "audio",
        is_turn_taking: bool = False,
    ):
        """
        Streams dataset partition from Hugging Face and saves contiguous shards.
        """
        print(f"\n--- [Streaming Dataset] {dataset_name} (Part {self.part_idx + 1}/{self.num_parts}) ---")
        ds = load_dataset(dataset_name, split=split, streaming=True)

        target_total_sec = self.max_hours * 3600.0
        processed_sec = 0.0
        sample_count = 0
        t0 = time.time()

        for idx, item in enumerate(ds):
            # Partition filtering: distribute items across parts deterministically
            if (idx % self.num_parts) != self.part_idx:
                continue

            audio_data = item.get(audio_col, {})
            wav_array = audio_data.get("array")
            sr = audio_data.get("sampling_rate", 24000)
            if wav_array is None:
                continue

            wav_tensor = torch.from_numpy(wav_array).float()
            dur = wav_tensor.shape[-1] / float(sr)

            # Metadata timestamps (if available in PersianVox / Diarization)
            segments = item.get("segments", None)

            success = self.process_and_add_sample(
                wav_audio=wav_tensor,
                orig_sr=sr,
                vad_segments=segments,
                is_turn_taking=is_turn_taking,
            )

            if success:
                processed_sec += dur
                sample_count += 1
                if sample_count % 100 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"  Processed {processed_sec / 3600.0:.2f}h / {self.max_hours:.1f}h "
                        f"({sample_count} samples) | Speed: {processed_sec / elapsed:.1f}x real-time"
                    )

            if processed_sec >= target_total_sec:
                print(f"  Reached target hours quota for this part ({self.max_hours}h).")
                break

        # Flush any remaining samples
        self.save_current_shard()
        total_time = time.time() - t0
        print(
            f"✓ Partition Complete: {processed_sec / 3600.0:.2f} hours processed in {total_time / 60.0:.1f} minutes."
        )


def main():
    parser = argparse.ArgumentParser(description="Hamsokhan Kaggle Autonomous Dataset Preprocessing")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/hamsokhan_shards")
    parser.add_argument("--dataset_type", type=str, default="thomcles", choices=["thomcles", "persianvox", "manatts"])
    parser.add_argument("--part_idx", type=int, default=0, help="Partition index (0-indexed)")
    parser.add_argument("--num_parts", type=int, default=1, help="Total number of partitions (e.g. 4 for 4x20GB)")
    parser.add_argument("--max_hours", type=float, default=5.0, help="Maximum audio hours for this part")
    parser.add_argument("--shard_size", type=int, default=500, help="Samples per .safetensors shard")
    args = parser.parse_args()

    builder = KaggleDatasetBuilder(
        output_dir=args.output_dir,
        part_idx=args.part_idx,
        num_parts=args.num_parts,
        max_hours_per_part=args.max_hours,
        shard_size_samples=args.shard_size,
    )

    if args.dataset_type == "thomcles":
        builder.build_from_huggingface(
            dataset_name="Thomcles/Persian-Farsi-Speech",
            split="train",
            audio_col="audio",
            is_turn_taking=False,
        )
    elif args.dataset_type == "persianvox":
        builder.build_from_huggingface(
            dataset_name="saeedzou/persianvox_all",
            split="train",
            audio_col="audio",
            is_turn_taking=True,
        )
    elif args.dataset_type == "manatts":
        builder.build_from_huggingface(
            dataset_name="hezarai/mana-tts",
            split="train",
            audio_col="audio",
            is_turn_taking=False,
        )


if __name__ == "__main__":
    main()
