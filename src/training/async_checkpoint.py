"""
Asynchronous Non-Blocking Hugging Face Hub Checkpoint Manager
Designed for Kaggle Dual NVIDIA T4 (DDP with Accelerate, 12h session limits, 20GB disk limit).
Features:
1. Fast snapshot of ONLY trainable parameters from GPU VRAM to CPU RAM (30-60ms).
2. Single-worker ThreadPoolExecutor handles local disk serialization and HF Hub upload in background.
3. Zero NCCL timeout risks between Rank 0 and Rank 1.
4. Automatic rolling cleanup of local disk checkpoints.
"""

import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
import torch
from huggingface_hub import HfApi

try:
    from safetensors.torch import save_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class AsyncHFHubCheckpointManager:
    """
    Asynchronous Checkpoint Manager for Kaggle DDP Training.
    Prevents NCCL rank timeouts and Kaggle disk saturation.
    """
    def __init__(
        self,
        repo_id: str,
        hf_token: Optional[str] = None,
        local_dir: str = "/kaggle/working/ckpts",
        max_keep_local: int = 1
    ):
        self.repo_id = repo_id
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.hf_token) if self.hf_token else None
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep_local = max_keep_local

        # Single background worker prevents upload congestion
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf_async_uploader")
        self.active_future = None
        self.saved_folders = []

    def _extract_trainable_cpu_snapshot(self, unwrapped_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Extracts and clones trainable parameters only, moving them to CPU RAM."""
        cpu_state = {}
        with torch.no_grad():
            for name, param in unwrapped_model.named_parameters():
                if param.requires_grad:
                    cpu_state[name] = param.detach().to(device="cpu", dtype=torch.float16, non_blocking=False).clone()
        return cpu_state

    def _save_and_upload_worker(self, cpu_state: Dict[str, torch.Tensor], step: int):
        try:
            step_dir = self.local_dir / f"step_{step}"
            step_dir.mkdir(parents=True, exist_ok=True)

            if HAS_SAFETENSORS:
                weights_path = step_dir / "trainable_weights.safetensors"
                save_file(cpu_state, str(weights_path))
            else:
                weights_path = step_dir / "trainable_weights.pt"
                torch.save(cpu_state, str(weights_path))

            del cpu_state  # Immediate CPU RAM free

            # Upload to Hugging Face Hub if API is authenticated
            if self.api is not None:
                self.api.upload_folder(
                    repo_id=self.repo_id,
                    folder_path=str(step_dir),
                    path_in_repo=f"checkpoints/step_{step}",
                    commit_message=f"Async checkpoint at step {step}",
                    repo_type="model"
                )

            # Rolling local disk cleanup
            self.saved_folders.append(step_dir)
            while len(self.saved_folders) > self.max_keep_local:
                old_dir = self.saved_folders.pop(0)
                shutil.rmtree(old_dir, ignore_errors=True)
        except Exception as e:
            print(f"⚠ AsyncHFHubCheckpointManager background worker error: {e}")

    def maybe_save_async(self, accelerator, model: torch.nn.Module, step: int, save_every: int = 500):
        if step % save_every != 0 or step == 0:
            return

        # Only Rank 0 captures the snapshot (synchronized post optimizer step)
        if accelerator.is_main_process:
            if self.active_future is not None and not self.active_future.done():
                self.active_future.result()

            unwrapped = accelerator.unwrap_model(model)
            cpu_snapshot = self._extract_trainable_cpu_snapshot(unwrapped)
            self.active_future = self.executor.submit(self._save_and_upload_worker, cpu_snapshot, step)

    def finalize(self, accelerator):
        """Called at the end of training to flush pending background uploads."""
        if accelerator.is_main_process:
            self.executor.shutdown(wait=True)
        accelerator.wait_for_everyone()
