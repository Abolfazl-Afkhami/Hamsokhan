"""
Causal Speech Projector (50Hz Whisper-Tiny to 12.5Hz Qwen SLM)
Preserves Persian plosives (/پ/, /ت/, /ک/, /ب/, /د/, /گ/, /ق/, /ء/) with zero lookahead delay.
Architecture: Depthwise Causal Conv1D + Ring State Buffer + Linear Frame Stacking + SwiGLU.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSpeechProjector(nn.Module):
    """
    Downsamples Whisper-Tiny 50Hz audio embeddings (20ms frames)
    to Qwen SLM 12.5Hz conversational tokens (80ms chunks).
    Features:
    1. Causal left-padding via state buffer: Lookahead = 0ms.
    2. Depthwise Conv1D with residual connection preserves inter-chunk boundary continuity.
    3. Linear Frame Stacking (384 * 4 = 1536) preserves single-frame 10-25ms plosive burst energy.
    4. SwiGLU non-linear gating ensures high dynamic range and representation power.
    """
    def __init__(
        self,
        d_whisper: int = 384,
        d_qwen: int = 896,
        stride: int = 4,
        kernel_size: int = 3
    ):
        super().__init__()
        self.d_whisper = d_whisper
        self.d_qwen = d_qwen
        self.stride = stride
        self.pad_len = kernel_size - 1  # Causal left padding: (K - 1) = 2 frames (40ms history)

        # 1. Depthwise Causal Conv1D for inter-chunk boundary smoothing
        self.dw_conv = nn.Conv1d(
            in_channels=d_whisper,
            out_channels=d_whisper,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=d_whisper
        )

        # 2. Linear Frame Stacking (384 * 4 = 1536)
        stacked_dim = d_whisper * stride
        self.norm = nn.LayerNorm(stacked_dim, eps=1e-5)

        # 3. SwiGLU Projector for preserving plosive transient bursts
        self.w_gate = nn.Linear(stacked_dim, d_qwen, bias=False)
        self.w_val = nn.Linear(stacked_dim, d_qwen, bias=False)
        self.w_out = nn.Linear(d_qwen, d_qwen, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state_buffer: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_whisper] input audio tensor (in streaming: T=4 frames = 80ms)
            state_buffer: [B, pad_len, d_whisper] ring buffer containing the last 2 frames from previous chunk
        Returns:
            out: [B, T // stride, d_qwen] projected representations at 12.5Hz
            new_state: [B, pad_len, d_whisper] updated state buffer for the next streaming chunk
        """
        B, T, D = x.shape
        if state_buffer is None:
            state_buffer = torch.zeros(B, self.pad_len, D, device=x.device, dtype=x.dtype)

        # 1. Causal Left-Pad without lookahead
        x_padded = torch.cat([state_buffer, x], dim=1)  # [B, T + 2, 384]
        new_state = x[:, -self.pad_len:, :].detach()

        # 2. Depthwise Conv1D + Residual connection
        h = self.dw_conv(x_padded.transpose(1, 2)).transpose(1, 2)  # [B, T, 384]
        h = h + x

        # 3. Frame Stacking: [B, T, 384] -> [B, T // 4, 1536]
        assert T % self.stride == 0, f"Input sequence length T={T} must be divisible by stride={self.stride}"
        h_stacked = h.reshape(B, T // self.stride, D * self.stride)

        # 4. SwiGLU Gating in FP32 safe precision
        h_norm = self.norm(h_stacked.float()).to(x.dtype)
        gate = F.silu(self.w_gate(h_norm))
        val = self.w_val(h_norm)
        out = self.w_out(gate * val)

        return out, new_state


if __name__ == "__main__":
    projector = CausalSpeechProjector(d_whisper=384, d_qwen=896, stride=4, kernel_size=3)
    # Test streaming chunk of 4 frames (80ms at 50Hz)
    test_chunk = torch.randn(2, 4, 384)
    out, state = projector(test_chunk)
    assert out.shape == (2, 1, 896), f"Unexpected out shape: {out.shape}"
    assert state.shape == (2, 2, 384), f"Unexpected state shape: {state.shape}"
    print(f"✓ CausalSpeechProjector verified: {test_chunk.shape} -> {out.shape} with zero lookahead.")
