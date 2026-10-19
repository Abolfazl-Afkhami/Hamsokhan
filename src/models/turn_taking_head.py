"""
Turn-Taking Classifier Head & Asymmetric Focal Loss
Designed for Persian full-duplex conversations on PersianVox dataset.
Distinguishes thinking pauses (e.g. "ummm...", "well...", mid-clause hesitation) from true sentence completion (where Persian verbs terminate the clause).
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1dBlock(nn.Module):
    """Causal 1D Convolution with dilation and residual connection."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, dilation: int = 1):
        super().__init__()
        self.pad_len = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        # Pad left strictly
        x_in = x.transpose(1, 2)  # [B, C, T]
        x_padded = F.pad(x_in, (self.pad_len, 0))
        out = self.conv(x_padded).transpose(1, 2)  # [B, T, C]
        out = self.act(self.norm(out))
        return out


class TurnTakingHead(nn.Module):
    """
    Multi-Scale Turn-Taking Classifier Head.
    Combines mid-layer prosodic representations (L/2) and final-layer syntactic representations (L)
    over a 16-frame (1.28s) causal window at 12.5Hz.
    """
    def __init__(
        self,
        hidden_size: int = 896,
        intermediate_dim: int = 256,
        kernel_size: int = 4,
        beta_init: float = 0.5
    ):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init))

        # Multi-scale causal dilated convolutions (dilations 1, 2)
        self.conv1 = CausalConv1dBlock(hidden_size, intermediate_dim, kernel_size=kernel_size, dilation=1)
        self.conv2 = CausalConv1dBlock(intermediate_dim, intermediate_dim, kernel_size=kernel_size, dilation=2)

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, 1)
        )

    def forward(
        self,
        h_last: torch.Tensor,
        h_mid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            h_last: [B, T, hidden_size] final layer representations (syntax / semantic completion)
            h_mid: [B, T, hidden_size] middle layer representations (prosody / F0 / pitch drop)
        Returns:
            logits: [B, T, 1] raw turn completion logits
        """
        if h_mid is not None:
            beta = torch.sigmoid(self.beta)
            h = beta * h_last + (1.0 - beta) * h_mid
        else:
            h = h_last

        feat = self.conv1(h)
        feat = self.conv2(feat)
        logits = self.classifier(feat)
        return logits


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss with Soft Temporal Ramp targets.
    alpha = 0.75, gamma_pos = 1.0, gamma_neg = 2.5
    Heavily penalizes False Positives (interrupting mid-sentence during thinking pauses)
    while remaining numerically stable in FP16/FP32.
    """
    def __init__(
        self,
        alpha: float = 0.75,
        gamma_pos: float = 1.0,
        gamma_neg: float = 2.5,
        eps: float = 1e-7
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, T, 1] or [B, T]
            targets: [B, T, 1] or [B, T] soft ramp labels in [0, 1]
        """
        logits = logits.float().squeeze(-1)
        targets = targets.float().squeeze(-1)

        probs = torch.sigmoid(logits).clamp(min=self.eps, max=1.0 - self.eps)

        pos_weight = self.alpha * torch.pow(1.0 - probs, self.gamma_pos)
        pos_loss = pos_weight * targets * torch.log(probs)

        neg_weight = (1.0 - self.alpha) * torch.pow(probs, self.gamma_neg)
        neg_loss = neg_weight * (1.0 - targets) * torch.log(1.0 - probs)

        loss = - (pos_loss + neg_loss)
        return loss.mean()


def generate_soft_temporal_ramp(seq_len: int, eot_indices: List[int], device: torch.device) -> torch.Tensor:
    """
    Generates causal soft temporal ramp targets:
    [0.15, 0.35, 0.65, 1.0, 1.0, 1.0] across 4 frames before EOT and 2 frames after.
    """
    ramp_profile = [0.15, 0.35, 0.65, 1.0, 1.0, 1.0]
    target = torch.zeros(seq_len, device=device)
    for idx in eot_indices:
        start = max(0, idx - 3)
        for offset, val in enumerate(ramp_profile):
            target_idx = start + offset
            if 0 <= target_idx < seq_len:
                target[target_idx] = max(target[target_idx].item(), val)
    return target


if __name__ == "__main__":
    head = TurnTakingHead(hidden_size=896, intermediate_dim=256)
    h_l = torch.randn(2, 16, 896)
    h_m = torch.randn(2, 16, 896)
    logits = head(h_l, h_m)
    assert logits.shape == (2, 16, 1)

    loss_fn = AsymmetricFocalLoss()
    labels = torch.zeros(2, 16)
    labels[:, -2:] = 1.0
    loss = loss_fn(logits, labels)
    print(f"✓ TurnTakingHead & AsymmetricFocalLoss verified (Loss: {loss.item():.4f}).")
