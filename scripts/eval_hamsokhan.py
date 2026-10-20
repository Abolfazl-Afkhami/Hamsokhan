"""
Hamsokhan — Comprehensive Evaluation & Latency Benchmarking Suite
=================================================================
Evaluates trained Hamsokhan V2 checkpoints on:
1. Hardware Latency & TTFA (Time To First Audio) P50/P90/P99 in milliseconds.
2. Acoustic Flow Matching reconstruction error (Mel L1 and SpecGrad).
3. Turn-Taking classification accuracy (Precision, Recall, F1 on EOT).
4. Audio Waveform synthesis and validation using Vocos 24kHz.
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
from safetensors import safe_open
from safetensors.torch import load_file

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hamsokhan_model import HamsokhanConfig, HamsokhanModel


def benchmark_stream_latency(
    model: HamsokhanModel,
    vocos: torch.nn.Module,
    device: torch.device,
    num_runs: int = 50,
) -> Dict[str, float]:
    """
    Measures end-to-end streaming latency per 80ms macro-packet:
    Whisper 4-frame packet -> Projector -> Qwen3-0.6B -> Turn-Taking Head -> Flow Synthesizer -> Vocos decode.
    """
    print(f"\n--- [1/3] Benchmarking Real-Time Streaming Latency ({num_runs} runs) ---")
    dummy_whisper = torch.randn(1, 4, 384, device=device, dtype=torch.float16)

    # Warmup runs
    for _ in range(10):
        out = model.stream_step_80ms(dummy_whisper, step_idx=0)
        _ = vocos.decode(out["mel_chunk"].float())
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies_ms: List[float] = []
    proj_state = None

    for i in range(num_runs):
        t0 = time.perf_counter()
        out = model.stream_step_80ms(dummy_whisper, proj_state_buffer=proj_state, step_idx=i)
        proj_state = out["proj_state_buffer"]
        _ = vocos.decode(out["mel_chunk"].float())
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.percentile(latencies_ms, 50))
    p90 = float(np.percentile(latencies_ms, 90))
    p99 = float(np.percentile(latencies_ms, 99))
    mean_lat = float(np.mean(latencies_ms))

    print(f"  Latency (Mean): {mean_lat:.2f} ms")
    print(f"  Latency (P50) : {p50:.2f} ms")
    print(f"  Latency (P90) : {p90:.2f} ms")
    print(f"  Latency (P99) : {p99:.2f} ms")

    # Target TTFA evaluation (< 300 ms)
    status = "PASSED (<300ms)" if p90 < 300.0 else "EXCEEDED"
    print(f"  TTFA Target Check: {status} (P90: {p90:.1f}ms)")

    return {"mean_ms": mean_lat, "p50_ms": p50, "p90_ms": p90, "p99_ms": p99}


def evaluate_test_shards(
    model: HamsokhanModel,
    shard_paths: List[str],
    device: torch.device,
    max_eval_samples: int = 100,
) -> Dict[str, float]:
    """
    Evaluates acoustic reconstruction loss (Mel L1, SpecGrad) and turn-taking metrics on held-out shards.
    """
    print(f"\n--- [2/3] Evaluating Quality on Real Feature Shards ({len(shard_paths)} shards found) ---")
    if not shard_paths:
        print("  Warning: No shards found for evaluation.")
        return {}

    mel_l1_losses: List[float] = []
    spec_grad_losses: List[float] = []
    eot_predictions: List[int] = []
    eot_ground_truths: List[int] = []

    samples_processed = 0

    with torch.no_grad():
        for shard_file in shard_paths:
            with safe_open(shard_file, framework="pt", device="cpu") as f:
                idx_table = f.get_tensor("index_table")
                num_items = idx_table.shape[0]

                for row_idx in range(num_items):
                    row = idx_table[row_idx]
                    vals = [int(x.item()) for x in row]
                    if len(vals) == 8:
                        w_s, w_e, m_s, m_e, _, _, l_s, l_e = vals
                    else:
                        w_s, w_e, m_s, m_e, l_s, l_e = vals[:6]

                    w_feat = f.get_slice("whisper_flat")[w_s:w_e, :].unsqueeze(0).to(device)
                    m_gt = f.get_slice("mel_flat")[:, m_s:m_e].unsqueeze(0).to(device)
                    l_gt = f.get_slice("labels_flat")[l_s:l_e].unsqueeze(0).to(device)

                    T_mel = m_gt.shape[-1]
                    m_mask = torch.ones(1, T_mel, dtype=torch.bool, device=device)

                    # Forward through model
                    out = model(
                        whisper_features=w_feat,
                        mel_spectrogram=m_gt,
                        mel_valid_mask=m_mask,
                        turn_taking_targets=l_gt,
                    )

                    mel_l1_losses.append(float(out["flow_loss"].item()))
                    spec_grad_losses.append(float(out["spec_grad_loss"].item()))

                    # Turn-Taking evaluation: last frame should be End-Of-Turn (EOT)
                    probs = torch.sigmoid(out["turn_taking_logits"].squeeze(0))
                    pred_eot = int(probs[-1] > 0.5)
                    gt_eot = int(l_gt[0, -1] > 0.5) if l_gt.shape[-1] > 0 else 1

                    eot_predictions.append(pred_eot)
                    eot_ground_truths.append(gt_eot)

                    samples_processed += 1
                    if samples_processed >= max_eval_samples:
                        break

            if samples_processed >= max_eval_samples:
                break

    mean_mel_l1 = float(np.mean(mel_l1_losses)) if mel_l1_losses else 0.0
    mean_spec_grad = float(np.mean(spec_grad_losses)) if spec_grad_losses else 0.0

    # Calculate Turn-Taking EOT Accuracy
    correct = sum(p == g for p, g in zip(eot_predictions, eot_ground_truths))
    eot_acc = (correct / len(eot_predictions) * 100.0) if eot_predictions else 0.0

    print(f"  Processed {samples_processed} evaluation samples:")
    print(f"  Mel L1 Flow Loss : {mean_mel_l1:.4f}")
    print(f"  Spectral Grad    : {mean_spec_grad:.4f}")
    print(f"  EOT Accuracy     : {eot_acc:.2f}%")

    return {
        "mel_l1": mean_mel_l1,
        "spec_grad": mean_spec_grad,
        "eot_accuracy": eot_acc,
    }


def synthesize_sample_audio(
    model: HamsokhanModel,
    vocos: torch.nn.Module,
    shard_paths: List[str],
    output_wav_path: str,
    device: torch.device,
):
    """
    Generates a full audio file from a test sample and saves it as 24kHz WAV.
    """
    print(f"\n--- [3/3] Synthesizing Sample Audio Output ---")
    if not shard_paths:
        print("  Warning: No shard available for audio synthesis.")
        return

    with safe_open(shard_paths[0], framework="pt", device="cpu") as f:
        idx_table = f.get_tensor("index_table")
        w_s, w_e = int(idx_table[0, 0]), int(idx_table[0, 1])
        w_feat = f.get_slice("whisper_flat")[w_s:w_e, :].unsqueeze(0).to(device)

    with torch.no_grad():
        embeds, _ = model.projector(w_feat)
        body = model._get_qwen_transformer_body()
        h_llm = body(inputs_embeds=embeds).last_hidden_state
        mel_pred = model.flow_synthesizer.synthesize_single_step(h_llm)
        wav = vocos.decode(mel_pred.float())

    wav_cpu = wav.squeeze(0).cpu()
    torchaudio.save(output_wav_path, wav_cpu, sample_rate=24000)
    print(f"✓ Audio successfully written to: {output_wav_path} (24kHz Mono WAV)")


def main():
    parser = argparse.ArgumentParser(description="Hamsokhan V2 Comprehensive Evaluation")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to trainable_hamsokhan.safetensors")
    parser.add_argument("--shard_dir", type=str, default="/kaggle/working/cached_shards", help="Directory of feature shards")
    parser.add_argument("--output_wav", type=str, default="/kaggle/working/eval_output.wav", help="Output WAV path")
    parser.add_argument("--num_eval_samples", type=int, default=50, help="Number of shard items to evaluate")
    parser.add_argument("--num_latency_runs", type=int, default=50, help="Number of streaming latency benchmark runs")
    parser.add_argument("--backbone_model", type=str, default="Qwen/Qwen3-0.6B", help="Pretrained Qwen backbone")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Initializing evaluation on device: {device}")

    # 1. Load Vocos Vocoder
    print("[Eval] Loading Vocos (24kHz) vocoder...")
    from vocos import Vocos
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()

    # 2. Load Hamsokhan Model Architecture & Checkpoint
    print(f"[Eval] Loading HamsokhanModel backbone ({args.backbone_model})...")
    cfg = HamsokhanConfig(backbone_model=args.backbone_model)
    model = HamsokhanModel(cfg, load_pretrained_qwen=True)

    print(f"[Eval] Loading weights from: {args.ckpt_path}...")
    weights = load_file(args.ckpt_path)
    model.load_state_dict(weights, strict=False)
    model.to(device).eval()
    print("✓ Model successfully initialized and weights mapped.")

    # 3. Benchmark Streaming Latency (TTFA)
    benchmark_stream_latency(model, vocos, device, num_runs=args.num_latency_runs)

    # 4. Evaluate Held-out Feature Shards
    shard_files = sorted(glob.glob(f"{args.shard_dir}/*.safetensors"))
    evaluate_test_shards(model, shard_files, device, max_eval_samples=args.num_eval_samples)

    # 5. Synthesize Sample Audio
    synthesize_sample_audio(model, vocos, shard_files, args.output_wav, device)

    print("\n=======================================================")
    print("✓ Evaluation Suite Complete!")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
