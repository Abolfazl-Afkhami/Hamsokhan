"""
Zero-Copy Sharded Speech Feature Dataset
Backed by Contiguous Sharded .safetensors and Tensor Index Table.
Prevents Python Copy-on-Write RAM Leaks on Kaggle T4 during multi-worker DDP training.
"""

from typing import List, Dict
from pathlib import Path
import torch
from torch.utils.data import Dataset
from safetensors.torch import safe_open


class ShardedSpeechFeatureDataset(Dataset):
    """
    Zero-Copy Dataset for precomputed speech features.
    Storage Schema per shard (.safetensors):
    - whisper_flat: [Total_T_w, 384] FP16
    - mel_flat: [100, Total_T_m] FP16
    - labels_flat: [Total_T_llm] FP16
    - index_table: [N_samples, 6] int64 -> (w_start, w_end, m_start, m_end, l_start, l_end)
    """
    def __init__(self, shard_paths: List[str]):
        super().__init__()
        self.shard_paths = [str(Path(p).resolve()) for p in shard_paths]

        # Aggregate index tables as a single contiguous tensor (not Python dicts)
        index_tensors = []
        for shard_id, path in enumerate(self.shard_paths):
            with safe_open(path, framework="pt", device="cpu") as f:
                idx_table = f.get_tensor("index_table")  # [N_i, 6]
                shard_col = torch.full((idx_table.shape[0], 1), shard_id, dtype=torch.int64)
                index_tensors.append(torch.cat([shard_col, idx_table], dim=1))

        if index_tensors:
            self.global_index = torch.cat(index_tensors, dim=0).share_memory_()
        else:
            self.global_index = torch.empty((0, 7), dtype=torch.int64)

    def __len__(self) -> int:
        return self.global_index.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.global_index[idx]
        shard_id = int(row[0].item())
        w_s, w_e, m_s, m_e, l_s, l_e = (int(x.item()) for x in row[1:])

        # Read slice directly from kernel page cache (mmap) without loading entire shard
        with safe_open(self.shard_paths[shard_id], framework="pt", device="cpu") as f:
            whisper_feat = f.get_slice("whisper_flat")[w_s:w_e, :]   # [T_w, 384] FP16
            vocos_mel = f.get_slice("mel_flat")[:, m_s:m_e]          # [100, T_mel] FP16
            tt_labels = f.get_slice("labels_flat")[l_s:l_e]          # [T_llm] FP16

        return {
            "whisper_features": whisper_feat,
            "vocos_mel": vocos_mel,
            "turn_taking_labels": tt_labels,
        }


def collate_speech_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function with causal padding for DDP batches.
    """
    w_lens = [b["whisper_features"].shape[0] for b in batch]
    m_lens = [b["vocos_mel"].shape[1] for b in batch]
    l_lens = [b["turn_taking_labels"].shape[0] for b in batch]

    max_w = max(w_lens)
    max_m = max(m_lens)
    max_l = max(l_lens)

    B = len(batch)
    whisper_padded = torch.zeros(B, max_w, 384, dtype=torch.float16)
    mel_padded = torch.full((B, 100, max_m), -11.51, dtype=torch.float16)  # Silence floor
    labels_padded = torch.full((B, max_l), -100.0, dtype=torch.float16)   # ignore_index

    for i, b in enumerate(batch):
        w = b["whisper_features"]
        m = b["vocos_mel"]
        l = b["turn_taking_labels"]
        whisper_padded[i, :w.shape[0], :] = w
        mel_padded[i, :, :m.shape[1]] = m
        labels_padded[i, :l.shape[0]] = l

    return {
        "whisper_features": whisper_padded,
        "vocos_mel": mel_padded,
        "turn_taking_labels": labels_padded,
        "whisper_lengths": torch.tensor(w_lens, dtype=torch.int64),
        "mel_lengths": torch.tensor(m_lens, dtype=torch.int64),
        "labels_lengths": torch.tensor(l_lens, dtype=torch.int64),
    }
