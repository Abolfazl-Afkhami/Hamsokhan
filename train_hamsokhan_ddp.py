"""
Hamsokhan — Dual NVIDIA T4 Accelerate FP16 DDP Training Script
===============================================================
Features:
1. ShardedSpeechFeatureDataset + 160ms Macro-Packet Dynamic Collate Function
   (zero-copy mmap via .safetensors, preventing Copy-on-Write RAM leaks on Kaggle).
2. Custom GradScalerKwargs (init_scale=2**14, growth_interval=1000) calibrated
   for Single-Step Rectified Flow & Asymmetric Focal Loss in FP16 without bfloat16.
3. Differential learning rates & weight decay grouping for Projector/Flow/TT vs. Qwen LoRA.
4. Non-blocking AsyncHFHubCheckpointManager (500-step snapshots to CPU RAM + background
   safetensors write & Hugging Face Hub upload + rolling local disk cleanup).
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, GradScalerKwargs, set_seed
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file, safe_open
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

from hamsokhan_model import HamsokhanConfig, HamsokhanModelWrapper


# ==============================================================================
# 1. Zero-Copy Sharded Safetensors Dataset & 160ms Macro-Packet Collate Function
# ==============================================================================
class ShardedSpeechFeatureDataset(Dataset):
    """
    Zero-copy mmap Dataset reading concatenated feature shards (.safetensors).
    Stores the global index strictly as a shared PyTorch int64 tensor to eliminate
    Python reference-counting Copy-on-Write (CoW) memory leaks across DataLoader workers.
    """

    def __init__(self, shard_paths: List[str]):
        super().__init__()
        self.shard_paths = sorted([str(p) for p in shard_paths])
        if not self.shard_paths:
            raise FileNotFoundError("No .safetensors feature shards provided.")

        index_tensors = []
        for shard_id, path in enumerate(self.shard_paths):
            with safe_open(path, framework="pt", device="cpu") as f:
                # index_table shape: [N_utterances, 6] -> (w_s, w_e, m_s, m_e, l_s, l_e)
                idx_table = f.get_tensor("index_table").to(torch.int64)
                shard_col = torch.full((idx_table.shape[0], 1), shard_id, dtype=torch.int64)
                index_tensors.append(torch.cat([shard_col, idx_table], dim=1))

        self.global_index = torch.cat(index_tensors, dim=0).share_memory_()

    def __len__(self) -> int:
        return int(self.global_index.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.global_index[idx]
        shard_id = int(row[0].item())
        w_s, w_e, m_s, m_e, l_s, l_e = (int(x.item()) for x in row[1:])

        with safe_open(self.shard_paths[shard_id], framework="pt", device="cpu") as f:
            whisper_feat = f.get_slice("whisper_flat")[w_s:w_e, :]  # [T_w, 384] FP16
            vocos_mel = f.get_slice("mel_flat")[:, m_s:m_e]         # [100, T_mel] FP16
            tt_labels = f.get_slice("labels_flat")[l_s:l_e]         # [T_llm] FP16

        return {
            "whisper_features": whisper_feat,
            "vocos_mel": vocos_mel,
            "turn_taking_labels": tt_labels,
        }


def hamsokhan_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Pads variable-length batch items to the nearest 160ms macro-packet boundary:
      - 1 macro-packet (160ms) = 8 Whisper frames (50Hz) = 2 Qwen frames (12.5Hz) = 15 Vocos Mel frames (93.75Hz)
    Creates exact validity masks for Qwen attention, Mel Flow loss, and Turn-Taking loss (-100.0).
    """
    B = len(batch)
    max_macro_packets = max(math.ceil(item["turn_taking_labels"].shape[0] / 2.0) for item in batch)
    max_macro_packets = max(1, max_macro_packets)

    max_t_w = max_macro_packets * 8
    max_t_llm = max_macro_packets * 2
    max_t_mel = max_macro_packets * 15

    d_whisper = batch[0]["whisper_features"].shape[-1]
    n_mels = batch[0]["vocos_mel"].shape[0]

    whisper_batch = torch.zeros((B, max_t_w, d_whisper), dtype=torch.float16)
    mel_batch = torch.full((B, n_mels, max_t_mel), fill_value=-11.5129, dtype=torch.float16)  # ln(1e-5) silence floor
    tt_labels_batch = torch.full((B, max_t_llm), fill_value=-100.0, dtype=torch.float16)
    attn_mask_batch = torch.zeros((B, max_t_llm), dtype=torch.long)
    mel_valid_mask = torch.zeros((B, max_t_mel), dtype=torch.bool)

    for i, item in enumerate(batch):
        w = item["whisper_features"]
        m = item["vocos_mel"]
        lbl = item["turn_taking_labels"]

        # Ensure consistent macro-packet truncation/padding per sample
        n_macro = min(max_macro_packets, w.shape[0] // 8, m.shape[1] // 15, lbl.shape[0] // 2)
        t_w = n_macro * 8
        t_llm = n_macro * 2
        t_mel = n_macro * 15

        whisper_batch[i, :t_w, :] = w[:t_w, :]
        mel_batch[i, :, :t_mel] = m[:, :t_mel]
        tt_labels_batch[i, :t_llm] = lbl[:t_llm]
        attn_mask_batch[i, :t_llm] = 1
        mel_valid_mask[i, :t_mel] = True

    return {
        "whisper_features": whisper_batch,
        "vocos_mel": mel_batch,
        "turn_taking_labels": tt_labels_batch,
        "attention_mask_12_5hz": attn_mask_batch,
        "mel_valid_mask": mel_valid_mask,
    }


# ==============================================================================
# 2. Non-Blocking Async Checkpoint Manager for Kaggle -> Hugging Face Hub
# ==============================================================================
class AsyncHFHubCheckpointManager:
    """
    Captures trainable weights (LoRA + Projector + TurnTakingHead + FlowSynthesizer)
    to CPU RAM in ~40ms on Rank 0, then writes .safetensors and uploads to HF Hub
    in a single background worker thread while GPUs immediately proceed to step N+1.
    """

    def __init__(
        self,
        repo_id: str,
        hf_token: Optional[str],
        local_dir: str = "/kaggle/working/ckpts",
        max_keep_local: int = 2,
        enable_hub_upload: bool = True,
    ):
        self.repo_id = repo_id
        self.enable_hub_upload = bool(enable_hub_upload and hf_token and repo_id)
        self.api = HfApi(token=hf_token) if self.enable_hub_upload else None
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep_local = max_keep_local
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf_async_ckpt")
        self.active_future = None
        self.saved_folders: List[Path] = []

    @staticmethod
    def _extract_trainable_cpu_snapshot(unwrapped_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        cpu_state: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in unwrapped_model.named_parameters():
                if param.requires_grad:
                    cpu_state[name] = (
                        param.detach().to(device="cpu", dtype=torch.float16, non_blocking=False).clone()
                    )
        return cpu_state

    def _save_and_upload_worker(self, cpu_state: Dict[str, torch.Tensor], step: int, loss_val: float):
        step_dir = self.local_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        weights_path = step_dir / "trainable_hamsokhan.safetensors"

        # 1. Serialize to Kaggle local SSD
        save_file(cpu_state, str(weights_path), metadata={"step": str(step), "loss": f"{loss_val:.4f}"})
        del cpu_state

        # 2. Upload to Hugging Face Hub if token & repo_id are provided
        if self.enable_hub_upload and self.api is not None:
            try:
                self.api.upload_folder(
                    repo_id=self.repo_id,
                    folder_path=str(step_dir),
                    path_in_repo=f"checkpoints/step_{step}",
                    commit_message=f"Hamsokhan async checkpoint step {step} (loss={loss_val:.4f})",
                    repo_type="model",
                )
            except Exception as exc:
                print(f"[AsyncHFHub] Warning: Hub upload at step {step} encountered an error: {exc}")

        # 3. Rolling cleanup to protect Kaggle's 20GB /kaggle/working quota
        self.saved_folders.append(step_dir)
        while len(self.saved_folders) > self.max_keep_local:
            old_dir = self.saved_folders.pop(0)
            shutil.rmtree(old_dir, ignore_errors=True)

    def maybe_save_async(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        step: int,
        loss_val: float,
        save_every: int = 500,
    ):
        if step <= 0 or (step % save_every != 0):
            return

        # Rank 0 takes a fast RAM snapshot without blocking Rank 1 over NCCL
        if accelerator.is_main_process:
            if self.active_future is not None and not self.active_future.done():
                self.active_future.result()

            unwrapped = accelerator.unwrap_model(model)
            cpu_snapshot = self._extract_trainable_cpu_snapshot(unwrapped)
            self.active_future = self.executor.submit(
                self._save_and_upload_worker, cpu_snapshot, step, loss_val
            )

    def finalize(self, accelerator: Accelerator):
        if accelerator.is_main_process:
            if self.active_future is not None:
                self.active_future.result()
            self.executor.shutdown(wait=True)
        accelerator.wait_for_everyone()


# ==============================================================================
# 3. Optimizer Grouping & Dual-T4 Accelerate Training Loop
# ==============================================================================
def build_optimizer(
    model: HamsokhanModelWrapper,
    lr_new_modules: float = 2e-4,
    lr_lora: float = 1e-4,
    weight_decay: float = 0.01,
) -> AdamW:
    """
    Separates newly initialized modules (Projector, TurnTakingHead, FlowSynthesizer)
    from pretrained Qwen LoRA adapters, disabling weight decay on 1D norms/biases/FiLM.
    """
    new_decay, new_no_decay = [], []
    lora_Params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "qwen" in name or "lora_" in name:
            lora_Params.append(param)
        elif param.ndim <= 1 or "norm" in name or "film" in name or "bias" in name:
            new_no_decay.append(param)
        else:
            new_decay.append(param)

    param_groups = [
        {"params": new_decay, "lr": lr_new_modules, "weight_decay": weight_decay},
        {"params": new_no_decay, "lr": lr_new_modules, "weight_decay": 0.0},
        {"params": lora_Params, "lr": lr_lora, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, betas=(0.9, 0.98), eps=1e-6)


def generate_dummy_shard_if_needed(shard_dir: Path, num_samples: int = 16) -> List[str]:
    """Creates a synthetic .safetensors shard matching the exact offline preprocessing schema for dry-run verification."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / "shard_0000.safetensors"
    if shard_path.exists():
        return [str(shard_path)]

    whisper_list, mel_list, label_list, index_rows = [], [], [], []
    w_offset, m_offset, l_offset = 0, 0, 0

    for _ in range(num_samples):
        n_macro = int(torch.randint(4, 10, (1,)).item())  # 0.64s to 1.60s per sample
        t_w = n_macro * 8
        t_llm = n_macro * 2
        t_mel = n_macro * 15

        w_feat = torch.randn(t_w, 384, dtype=torch.float16) * 0.5
        m_feat = torch.randn(100, t_mel, dtype=torch.float16) * 1.5 - 4.0
        lbl = torch.zeros(t_llm, dtype=torch.float16)
        if t_llm >= 6:
            lbl[-6:] = torch.tensor([0.15, 0.35, 0.65, 1.0, 1.0, 1.0], dtype=torch.float16)

        whisper_list.append(w_feat)
        mel_list.append(m_feat)
        label_list.append(lbl)
        index_rows.append([w_offset, w_offset + t_w, m_offset, m_offset + t_mel, l_offset, l_offset + t_llm])

        w_offset += t_w
        m_offset += t_mel
        l_offset += t_llm

    tensors = {
        "whisper_flat": torch.cat(whisper_list, dim=0),
        "mel_flat": torch.cat(mel_list, dim=1),
        "labels_flat": torch.cat(label_list, dim=0),
        "index_table": torch.tensor(index_rows, dtype=torch.int64),
    }
    save_file(tensors, str(shard_path))
    return [str(shard_path)]


def train(args: argparse.Namespace):
    set_seed(args.seed)

    # Calibrated FP16 GradScaler for NVIDIA T4 (Turing lacks bfloat16):
    # init_scale=2**14 (16384.0) prevents step-0 FP16 overflow in Flow velocity gradients
    # while preserving ample dynamic range above FP16 subnormal underflow (~6e-8).
    scaler_kwargs = GradScalerKwargs(
        init_scale=16384.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=1000,
    )
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )

    accelerator = Accelerator(
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        gradient_accumulation_steps=args.grad_accum_steps,
        kwargs_handlers=[scaler_kwargs, ddp_kwargs],
    )

    # 1. Discover or synthesize feature shards
    shard_dir = Path(args.shard_dir)
    shard_paths = sorted([str(p) for p in shard_dir.glob("*.safetensors")]) if shard_dir.exists() else []
    if not shard_paths:
        if accelerator.is_main_process:
            generate_dummy_shard_if_needed(shard_dir, num_samples=24)
        accelerator.wait_for_everyone()
        shard_paths = sorted([str(p) for p in shard_dir.glob("*.safetensors")])

    dataset = ShardedSpeechFeatureDataset(shard_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=hamsokhan_collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )

    # 2. Initialize Model, Optimizer & Scheduler
    cfg = HamsokhanConfig(
        qwen_model_name_or_path=args.qwen_model,
        use_4bit_quantization=args.use_4bit,
    )
    model = HamsokhanModelWrapper(cfg, load_pretrained_qwen=not args.offline_mock_qwen)

    optimizer = build_optimizer(
        model,
        lr_new_modules=args.lr_new,
        lr_lora=args.lr_lora,
        weight_decay=args.weight_decay,
    )

    warmup_steps = min(200, max(1, args.max_steps // 10))
    warmup_sched = LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, args.max_steps - warmup_steps), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # 3. Initialize Non-Blocking Async Checkpoint Manager
    ckpt_manager = AsyncHFHubCheckpointManager(
        repo_id=args.hf_repo_id,
        hf_token=os.environ.get("HF_TOKEN", args.hf_token),
        local_dir=args.ckpt_dir,
        max_keep_local=2,
        enable_hub_upload=bool(args.hf_repo_id and (os.environ.get("HF_TOKEN") or args.hf_token)),
    )

    if accelerator.is_main_process:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[Hamsokhan DDP] Starting training on {accelerator.num_processes} process(es) | "
            f"Trainable Params: {trainable_params / 1e6:.2f}M | Mixed Precision: {accelerator.mixed_precision}"
        )

    # 4. Main Training Loop
    model.train()
    global_step = 0
    start_time = time.time()

    while global_step < args.max_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                outputs = model(
                    whisper_features=batch["whisper_features"],
                    vocos_mel=batch["vocos_mel"],
                    turn_taking_labels=batch["turn_taking_labels"],
                    attention_mask_12_5hz=batch["attention_mask_12_5hz"],
                    mel_valid_mask=batch["mel_valid_mask"],
                )
                loss = outputs["loss"]
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                loss_item = float(loss.detach().item())

                if global_step % args.log_every == 0 and accelerator.is_main_process:
                    elapsed = time.time() - start_time
                    print(
                        f"Step {global_step:05d}/{args.max_steps:05d} | "
                        f"Total Loss: {loss_item:.4f} | "
                        f"Flow: {float(outputs['flow_loss'].item()):.4f} "
                        f"(SpecGrad: {float(outputs['spec_grad_loss'].item()):.4f}) | "
                        f"TT Focal: {float(outputs['tt_loss'].item()):.4f} | "
                        f"Elapsed: {elapsed:.1f}s"
                    )

                # Non-blocking async 500-step checkpoint snapshot + background upload
                ckpt_manager.maybe_save_async(
                    accelerator=accelerator,
                    model=model,
                    step=global_step,
                    loss_val=loss_item,
                    save_every=args.save_every,
                )

                if global_step >= args.max_steps:
                    break

    ckpt_manager.finalize(accelerator)
    if accelerator.is_main_process:
        print(f"[Hamsokhan DDP] Training completed cleanly at step {global_step}.")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hamsokhan Dual-T4 Accelerate DDP Trainer")
    parser.add_argument("--shard_dir", type=str, default="/kaggle/working/cached_shards")
    parser.add_argument("--ckpt_dir", type=str, default="/kaggle/working/ckpts")
    parser.add_argument("--backbone_model", "--qwen_model", dest="qwen_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--hf_repo_id", type=str, default="")
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--lr_new", type=float, default=2e-4)
    parser.add_argument("--lr_lora", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--offline_mock_qwen", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())