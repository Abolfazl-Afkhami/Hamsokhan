"""
Hamsokhan (همسخن) — Unified Speech-to-Speech Architecture for Kaggle Dual NVIDIA T4
====================================================================================
Contains:
1. HamsokhanConfig: Complete hyperparameter & alignment specification.
2. CausalSpeechProjector: 50Hz (Whisper-Tiny) -> 12.5Hz (Qwen) zero-lookahead projector
   with Causal Depthwise Conv1D, 4x Frame Stacking, FP32 LayerNorm, and SwiGLU gating.
3. TurnTakingHead: Multi-layer (mid + final Qwen state) fusion with a 16-frame (1.28s)
   causal dilated convolution window and FP32 Asymmetric Focal Loss on soft ramp targets.
4. SingleStepFlowSynthesizer: Tanh-Bounded FiLM conditioned Single-Step Rectified Flow
   mapping 12.5Hz LLM states to [B, 100, T_mel] (93.75Hz Vocos 24kHz Mel spectrogram,
   exact 15:2 macro-packet ratio) with Frequency-Weighted Pseudo-Huber, Spectral/Temporal
   Gradient, and Contrastive Flow Matching losses.
5. HamsokhanModelWrapper: Single nn.Module wrapper for PyTorch DDP & HF Accelerate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HamsokhanConfig:
    # Backbone SLM (Qwen2.5-0.5B or Qwen3-0.6B)
    qwen_model_name_or_path: str = "Qwen/Qwen2.5-0.5B"
    d_qwen: int = 896  # Auto-updated from loaded Qwen config (896 for Qwen2.5-0.5B, 1024 for Qwen3-0.6B)
    qwen_num_layers: int = 24
    turn_taking_mid_layer_idx: Optional[int] = None  # Defaults to qwen_num_layers // 2
    use_4bit_quantization: bool = True

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # Audio Encoder (Causal Whisper-Tiny 50Hz -> 12.5Hz)
    d_whisper: int = 384
    projector_stride: int = 4  # 50Hz -> 12.5Hz (80ms packets)
    projector_kernel_size: int = 3  # 2 frames (40ms) causal history buffer

    # Turn-Taking Head (PersianVox)
    tt_hidden_dim: int = 256
    tt_kernel_size: int = 4
    tt_dilations: Tuple[int, ...] = (1, 2, 3)  # Receptive field = 1 + 3*(1+2+3) = 19 frames (>= 16 frames / 1.28s)
    tt_focal_alpha: float = 0.75
    tt_focal_gamma_pos: float = 1.0
    tt_focal_gamma_neg: float = 2.5

    # Single-Step Rectified Flow Synthesizer -> Vocos 24kHz [B, 100, T_mel]
    n_mels: int = 100
    mel_frames_per_macro_packet: int = 15  # 160ms = 2 LLM frames (12.5Hz) = 15 Mel frames (93.75Hz)
    llm_frames_per_macro_packet: int = 2
    flow_hidden_dim: int = 512
    flow_num_layers: int = 6
    flow_kernel_size: int = 5
    flow_single_step_anchor_prob: float = 0.30  # Probability of sampling t=0 during training for NFE=1 distillation

    # Loss weights
    loss_weight_flow: float = 1.0
    loss_weight_tt: float = 0.5
    flow_huber_delta: float = 0.03
    flow_low_freq_bins: int = 35  # Mel bins 0..34 covering Persian F0 and F1/F2 formants
    flow_low_freq_weight: float = 2.0
    flow_spec_grad_weight: float = 0.5
    flow_time_grad_weight: float = 0.25
    flow_contrastive_weight: float = 0.05
    flow_contrastive_margin: float = 1.5


class FP32LayerNorm(nn.Module):
    """LayerNorm executed strictly in FP32 to prevent FP16 overflow/underflow on NVIDIA T4."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        with torch.amp.autocast("cuda", enabled=False):
            y = self.norm(x.float())
        return y.to(orig_dtype)


class CausalSpeechProjector(nn.Module):
    """
    Zero-Lookahead Causal Projector: 50Hz Whisper-Tiny [B, T_w, 384] -> 12.5Hz Qwen [B, T_w // 4, d_qwen].
    Preserves Persian plosive bursts (/p/, /t/, /k/, /b/, /d/, /g/, /q/) via Frame Stacking + SwiGLU.
    """

    def __init__(self, cfg: HamsokhanConfig):
        super().__init__()
        self.stride = cfg.projector_stride
        self.pad_len = cfg.projector_kernel_size - 1  # Causal left padding: (K - 1) * d

        self.dw_conv = nn.Conv1d(
            in_channels=cfg.d_whisper,
            out_channels=cfg.d_whisper,
            kernel_size=cfg.projector_kernel_size,
            stride=1,
            padding=0,
            groups=cfg.d_whisper,
        )

        stacked_dim = cfg.d_whisper * self.stride  # 384 * 4 = 1536
        self.norm = FP32LayerNorm(stacked_dim, eps=1e-5)

        self.w_gate = nn.Linear(stacked_dim, cfg.d_qwen, bias=False)
        self.w_val = nn.Linear(stacked_dim, cfg.d_qwen, bias=False)
        self.w_out = nn.Linear(cfg.d_qwen, cfg.d_qwen, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        if T % self.stride != 0:
            raise ValueError(f"Whisper frame length T={T} must be divisible by stride={self.stride}.")

        if state_buffer is None:
            state_buffer = torch.zeros(B, self.pad_len, D, device=x.device, dtype=x.dtype)

        # Strict left causal concatenation (0ms lookahead)
        x_padded = torch.cat([state_buffer, x], dim=1)  # [B, T + pad_len, 384]
        new_state = x[:, -self.pad_len :, :].detach()

        h = self.dw_conv(x_padded.transpose(1, 2)).transpose(1, 2)  # [B, T, 384]
        h = h + x  # Residual preserves transient plosive bursts

        h_stacked = h.reshape(B, T // self.stride, D * self.stride)  # [B, T // 4, 1536]
        h_norm = self.norm(h_stacked)
        out = self.w_out(F.silu(self.w_gate(h_norm)) * self.w_val(h_norm))
        return out, new_state


class CausalDilatedConvBlock(nn.Module):
    """Strictly causal 1D dilated convolution block with Pad_left = (K - 1) * d and Pad_right = 0."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad_left = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
        )
        self.norm = FP32LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x_pad = F.pad(x, (self.pad_left, 0))
        h = self.conv(x_pad).transpose(1, 2)
        h = F.silu(self.norm(h)).transpose(1, 2)
        return x + h


class TurnTakingHead(nn.Module):
    """
    Fuses Qwen intermediate prosodic states (L/2) and final syntactic states (L)
    over a ~1.28s (16-frame @ 12.5Hz) local causal window to separate mid-utterance
    Persian hesitation pauses from true End-of-Turn (EOT).
    """

    def __init__(self, cfg: HamsokhanConfig):
        super().__init__()
        self.cfg = cfg
        self.layer_mix_logit = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5 initial balance
        self.in_proj = nn.Linear(cfg.d_qwen, cfg.tt_hidden_dim)
        self.causal_blocks = nn.ModuleList(
            [
                CausalDilatedConvBlock(cfg.tt_hidden_dim, cfg.tt_kernel_size, d)
                for d in cfg.tt_dilations
            ]
        )
        self.out_norm = FP32LayerNorm(cfg.tt_hidden_dim)
        self.logit_head = nn.Linear(cfg.tt_hidden_dim, 1)

    def forward(self, h_mid: torch.Tensor, h_last: torch.Tensor) -> torch.Tensor:
        """
        h_mid, h_last: [B, T_llm, d_qwen]
        Returns: tt_logits [B, T_llm]
        """
        beta = torch.sigmoid(self.layer_mix_logit).to(h_last.dtype)
        h_fused = beta * h_last + (1.0 - beta) * h_mid
        x = self.in_proj(h_fused).transpose(1, 2)  # [B, C, T_llm]
        for block in self.causal_blocks:
            x = block(x)
        x = self.out_norm(x.transpose(1, 2))
        logits = self.logit_head(x).squeeze(-1)  # [B, T_llm]
        return logits

    def compute_loss(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        """
        Asymmetric Focal BCE Loss computed strictly in FP32 with ignore_index = -100.0 masking.
        soft_targets: [B, T_llm] containing values in [0.0, 1.0] (e.g. [0.15, 0.35, 0.65, 1.0, 1.0, 1.0]) or -100.0.
        """
        with torch.amp.autocast("cuda", enabled=False):
            logits_fp32 = logits.float()
            targets_fp32 = soft_targets.float()
            valid_mask = targets_fp32 >= 0.0

            if not valid_mask.any():
                return (logits_fp32 * 0.0).sum()

            valid_logits = logits_fp32[valid_mask]
            y = targets_fp32[valid_mask].clamp(0.0, 1.0)

            p = torch.sigmoid(valid_logits)
            log_p = F.logsigmoid(valid_logits)
            log_not_p = F.logsigmoid(-valid_logits)

            alpha = self.cfg.tt_focal_alpha
            gamma_pos = self.cfg.tt_focal_gamma_pos
            gamma_neg = self.cfg.tt_focal_gamma_neg

            pos_term = -alpha * torch.pow(1.0 - p, gamma_pos) * y * log_p
            neg_term = -(1.0 - alpha) * torch.pow(p, gamma_neg) * (1.0 - y) * log_not_p
            loss = (pos_term + neg_term).mean()
        return loss


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] in [0, 1] -> [B, dim]"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(t.dtype)
        return self.mlp(emb)


class BoundedFiLMConvBlock(nn.Module):
    """
    Tanh-Bounded FiLM conditioned Causal ConvNeXt-style block for FP16 stability on T4:
    BoundedFiLM(h, c) = LN_fp32(h) * (1 + 0.5 * tanh(W_gamma * c)) + W_beta * c
    """

    def __init__(self, channels: int, cond_dim: int, kernel_size: int = 5):
        super().__init__()
        self.pad_left = kernel_size - 1
        self.dw_conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, stride=1, padding=0, groups=channels
        )
        self.norm = FP32LayerNorm(channels, elementwise_affine=False)
        self.film_proj = nn.Linear(cond_dim, channels * 2)
        self.pw_ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T_mel]
        cond: [B, T_mel, cond_dim]
        """
        residual = x
        x_pad = F.pad(x, (self.pad_left, 0))
        h = self.dw_conv(x_pad).transpose(1, 2)  # [B, T_mel, C]
        h = self.norm(h)

        gamma_raw, beta = self.film_proj(cond).chunk(2, dim=-1)
        scale = 1.0 + 0.5 * torch.tanh(gamma_raw)  # Bounded strictly in [0.5, 1.5]
        h = h * scale + beta

        h = self.pw_ffn(h).transpose(1, 2)  # [B, C, T_mel]
        return residual + h


class SingleStepFlowSynthesizer(nn.Module):
    """
    Single-Step Rectified Flow Synthesizer generating [B, 100, T_mel] for Vocos 24kHz.
    Aligns 2 LLM frames (160ms @ 12.5Hz) -> 15 Mel frames (160ms @ 93.75Hz).
    """

    def __init__(self, cfg: HamsokhanConfig):
        super().__init__()
        self.cfg = cfg
        self.ratio_num = cfg.mel_frames_per_macro_packet  # 15
        self.ratio_den = cfg.llm_frames_per_macro_packet  # 2

        # Macro-packet Upsampler: maps 2 LLM frames [B, T_llm/2, 2*d_qwen] -> 15 Mel conditioning frames [B, T_mel, D_flow]
        self.macro_packet_expander = nn.Sequential(
            FP32LayerNorm(cfg.d_qwen * self.ratio_den),
            nn.Linear(cfg.d_qwen * self.ratio_den, cfg.flow_hidden_dim * self.ratio_num),
        )

        self.time_embed = SinusoidalTimeEmbedding(cfg.flow_hidden_dim)
        self.in_mel_proj = nn.Conv1d(cfg.n_mels, cfg.flow_hidden_dim, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                BoundedFiLMConvBlock(
                    channels=cfg.flow_hidden_dim,
                    cond_dim=cfg.flow_hidden_dim,
                    kernel_size=cfg.flow_kernel_size,
                )
                for _ in range(cfg.flow_num_layers)
            ]
        )

        self.out_norm = FP32LayerNorm(cfg.flow_hidden_dim)
        self.out_vel_proj = nn.Linear(cfg.flow_hidden_dim, cfg.n_mels)

        # Register frequency weight vector for Persian F0 and F1/F2 preservation
        freq_weights = torch.ones(cfg.n_mels, dtype=torch.float32)
        freq_weights[: cfg.flow_low_freq_bins] = cfg.flow_low_freq_weight
        self.register_buffer("freq_weights", freq_weights.view(1, cfg.n_mels, 1), persistent=False)

    def upsample_llm_to_mel_frames(self, h_llm: torch.Tensor) -> torch.Tensor:
        """
        h_llm: [B, T_llm, d_qwen] where T_llm is divisible by 2 (160ms macro-packets).
        Returns: cond [B, T_mel, flow_hidden_dim] where T_mel = (T_llm // 2) * 15.
        """
        B, T_llm, D = h_llm.shape
        if T_llm % self.ratio_den != 0:
            raise ValueError(f"T_llm={T_llm} must be divisible by {self.ratio_den} (160ms macro-packets).")
        num_macro = T_llm // self.ratio_den
        h_macro = h_llm.reshape(B, num_macro, D * self.ratio_den)
        expanded = self.macro_packet_expander(h_macro)  # [B, num_macro, 15 * D_flow]
        cond = expanded.reshape(B, num_macro * self.ratio_num, self.cfg.flow_hidden_dim)
        return cond

    def predict_velocity(
        self, z_t: torch.Tensor, t: torch.Tensor, cond_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        z_t: [B, 100, T_mel]
        t: [B] in [0, 1]
        cond_seq: [B, T_mel, D_flow]
        Returns: v_theta [B, 100, T_mel]
        """
        t_emb = self.time_embed(t).unsqueeze(1)  # [B, 1, D_flow]
        cond = cond_seq + t_emb  # [B, T_mel, D_flow]

        h = self.in_mel_proj(z_t)  # [B, D_flow, T_mel]
        for block in self.blocks:
            h = block(h, cond)

        h = self.out_norm(h.transpose(1, 2))
        v_pred = self.out_vel_proj(h).transpose(1, 2)  # [B, 100, T_mel]
        return v_pred

    @torch.no_grad()
    def synthesize_single_step(
        self,
        h_llm: torch.Tensor,
        temperature: float = 0.85,
        step_parity: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Single-Step (NFE=1) inference: M_hat = z_0 + v_theta(z_0, t=0, c).
        Supports both 160ms macro-packets and 80ms single-frame streaming (alternating 8 and 7 mel frames).
        """
        B, T_llm, _ = h_llm.shape
        if T_llm == 1:
            # Streaming 80ms mode: duplicate frame to form a causal 2-frame pair and slice 8 or 7 frames
            h_pair = h_llm.repeat(1, 2, 1)
            cond_seq = self.upsample_llm_to_mel_frames(h_pair)
            if step_parity is None or (step_parity % 2 == 0):
                cond_seq = cond_seq[:, :8, :]
            else:
                cond_seq = cond_seq[:, 8:15, :]
        else:
            cond_seq = self.upsample_llm_to_mel_frames(h_llm)

        T_mel = cond_seq.shape[1]
        z_0 = torch.randn(B, self.cfg.n_mels, T_mel, device=h_llm.device, dtype=h_llm.dtype) * temperature
        t_zero = torch.zeros(B, device=h_llm.device, dtype=h_llm.dtype)
        v_0 = self.predict_velocity(z_0, t_zero, cond_seq)
        return z_0 + v_0

    def forward_training(
        self,
        h_llm: torch.Tensor,
        target_mel: torch.Tensor,
        mel_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes Single-Step Rectified Flow training losses in FP32 to prevent F0 Collapse & Mode Averaging.
        h_llm: [B, T_llm, d_qwen]
        target_mel: [B, 100, T_mel]
        mel_valid_mask: [B, T_mel] boolean mask for valid (non-padded) frames
        """
        B, _, T_mel = target_mel.shape
        cond_seq = self.upsample_llm_to_mel_frames(h_llm)  # [B, T_mel, D_flow]

        z_0 = torch.randn_like(target_mel)
        t = torch.rand(B, device=target_mel.device, dtype=target_mel.dtype)
        # Anchor a subset of batch samples strictly at t=0 for direct single-step NFE=1 fidelity
        zero_mask = torch.rand(B, device=target_mel.device) < self.cfg.flow_single_step_anchor_prob
        t = torch.where(zero_mask, torch.zeros_like(t), t)

        t_view = t.view(B, 1, 1)
        z_t = (1.0 - t_view) * z_0 + t_view * target_mel
        v_pred = self.predict_velocity(z_t, t, cond_seq)

        # Single-step Euler reconstruction from z_t: M_hat = z_t + (1 - t) * v_pred
        pred_mel = z_t + (1.0 - t_view) * v_pred

        with torch.amp.autocast("cuda", enabled=False):
            v_pred_f = v_pred.float()
            v_target_f = (target_mel.float() - z_0.float())
            pred_mel_f = pred_mel.float()
            target_mel_f = target_mel.float()

            if mel_valid_mask is None:
                mask_f = torch.ones((B, 1, T_mel), device=target_mel.device, dtype=torch.float32)
            else:
                mask_f = mel_valid_mask.float().unsqueeze(1)  # [B, 1, T_mel]

            denom = (mask_f.sum() * self.cfg.n_mels).clamp(min=1.0)

            # 1. Frequency-Weighted Pseudo-Huber Velocity Loss
            delta = self.cfg.flow_huber_delta
            diff = v_pred_f - v_target_f
            huber = torch.sqrt(diff * diff + delta * delta) - delta
            f_huber_loss = (huber * self.freq_weights * mask_f).sum() / denom

            # 2. Spectral (Frequency-Axis) & Temporal Derivative Losses (Prevents F0/Pitch Collapse)
            pred_df = pred_mel_f[:, 1:, :] - pred_mel_f[:, :-1, :]
            targ_df = target_mel_f[:, 1:, :] - target_mel_f[:, :-1, :]
            spec_grad_loss = (torch.abs(pred_df - targ_df) * mask_f).sum() / denom

            mask_dt = mask_f[:, :, 1:] * mask_f[:, :, :-1]
            denom_dt = (mask_dt.sum() * self.cfg.n_mels).clamp(min=1.0)
            pred_dt = pred_mel_f[:, :, 1:] - pred_mel_f[:, :, :-1]
            targ_dt = target_mel_f[:, :, 1:] - target_mel_f[:, :, :-1]
            time_grad_loss = (torch.abs(pred_dt - targ_dt) * mask_dt).sum() / denom_dt

            # 3. Bounded Contrastive Flow Matching Loss (Prevents Mode Averaging without a GAN)
            if B > 1:
                neg_mel_f = torch.roll(target_mel_f, shifts=1, dims=0)
                v_neg_f = neg_mel_f - z_0.float()
                neg_dist = (torch.abs(v_pred_f - v_neg_f) * mask_f).sum() / denom
                contrastive_loss = F.relu(self.cfg.flow_contrastive_margin - neg_dist)
            else:
                contrastive_loss = f_huber_loss * 0.0

            total_flow_loss = (
                f_huber_loss
                + self.cfg.flow_spec_grad_weight * spec_grad_loss
                + self.cfg.flow_time_grad_weight * time_grad_loss
                + self.cfg.flow_contrastive_weight * contrastive_loss
            )

        metrics = {
            "f_huber_loss": f_huber_loss.detach(),
            "spec_grad_loss": spec_grad_loss.detach(),
            "time_grad_loss": time_grad_loss.detach(),
            "contrastive_loss": contrastive_loss.detach(),
        }
        return total_flow_loss, pred_mel, metrics


def build_qwen_lora_backbone(cfg: HamsokhanConfig, load_pretrained: bool = True) -> nn.Module:
    """
    Loads Qwen2.5-0.5B / Qwen3-0.6B with 4-bit NF4/FP16 or FP16 LoRA for Kaggle T4 (< 1.5GB VRAM).
    If load_pretrained=False (e.g. offline unit testing), builds an exact-config Qwen2Model from scratch.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

    if load_pretrained:
        quant_kwargs = {}
        if cfg.use_4bit_quantization and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                pass

        base_lm = AutoModelForCausalLM.from_pretrained(
            cfg.qwen_model_name_or_path,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            **quant_kwargs,
        )
    else:
        q_cfg = Qwen2Config(
            vocab_size=151936,
            hidden_size=cfg.d_qwen,
            intermediate_size=cfg.d_qwen * 4,
            num_hidden_layers=cfg.qwen_num_layers,
            num_attention_heads=14,
            num_key_value_heads=2,
            max_position_embeddings=4096,
            torch_dtype=torch.float16,
        )
        base_lm = Qwen2ForCausalLM(q_cfg)

    # Synchronize dimensions with actual backbone config
    cfg.d_qwen = base_lm.config.hidden_size
    cfg.qwen_num_layers = base_lm.config.num_hidden_layers
    if cfg.turn_taking_mid_layer_idx is None:
        cfg.turn_taking_mid_layer_idx = cfg.qwen_num_layers // 2

    # Freeze lm_head since Hamsokhan operates directly on continuous hidden states -> Flow & Turn-Taking
    for param in base_lm.parameters():
        param.requires_grad = False

    try:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(cfg.lora_target_modules),
        )
        base_lm = get_peft_model(base_lm, lora_cfg)
    except ImportError:
        # Fallback if PEFT is not installed in a minimal test environment: unfreeze last 2 transformer layers
        layers = base_lm.model.layers[-2:]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True

    return base_lm


class HamsokhanModelWrapper(nn.Module):
    """
    Unified DDP-ready Model Wrapper combining:
      1. CausalSpeechProjector (50Hz -> 12.5Hz)
      2. Qwen SLM Backbone with LoRA
      3. TurnTakingHead (PersianVox Soft-Ramp Asymmetric Focal Loss)
      4. SingleStepFlowSynthesizer (Bounded-FiLM Rectified Flow -> [B, 100, T_mel])
    """

    def __init__(self, cfg: HamsokhanConfig, load_pretrained_qwen: bool = True):
        super().__init__()
        self.cfg = cfg
        self.qwen = build_qwen_lora_backbone(cfg, load_pretrained=load_pretrained_qwen)
        self.projector = CausalSpeechProjector(cfg)
        self.turn_taking_head = TurnTakingHead(cfg)
        self.flow_synthesizer = SingleStepFlowSynthesizer(cfg)

    def _get_qwen_transformer_body(self) -> nn.Module:
        """Returns the inner transformer body (`Qwen2Model`) regardless of PEFT wrapping."""
        if hasattr(self.qwen, "get_base_model"):
            base = self.qwen.get_base_model()
            return base.model if hasattr(base, "model") else base
        return self.qwen.model if hasattr(self.qwen, "model") else self.qwen

    def forward(
        self,
        whisper_features: torch.Tensor,
        vocos_mel: torch.Tensor,
        turn_taking_labels: torch.Tensor,
        attention_mask_12_5hz: Optional[torch.Tensor] = None,
        mel_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for DDP training.
        Args:
            whisper_features: [B, T_w, 384] @ 50Hz (T_w is a multiple of 8)
            vocos_mel: [B, 100, T_mel] @ 93.75Hz (T_mel is a multiple of 15)
            turn_taking_labels: [B, T_llm] @ 12.5Hz (T_llm = T_w // 4, multiple of 2)
            attention_mask_12_5hz: [B, T_llm] (1 for valid tokens, 0 for padded tokens)
            mel_valid_mask: [B, T_mel] (True for valid mel frames, False for padded frames)
        """
        # 1. Project 50Hz Whisper features -> 12.5Hz Qwen embeddings with 0ms lookahead
        inputs_embeds, _ = self.projector(whisper_features, state_buffer=None)  # [B, T_llm, d_qwen]

        # 2. Forward through Qwen Transformer Body (skipping unused vocab lm_head for DDP efficiency)
        transformer_body = self._get_qwen_transformer_body()
        qwen_out = transformer_body(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_12_5hz,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        all_hidden_states = qwen_out.hidden_states
        mid_idx = self.cfg.turn_taking_mid_layer_idx or (len(all_hidden_states) // 2)
        h_mid = all_hidden_states[mid_idx]
        h_last = qwen_out.last_hidden_state

        # 3. Turn-Taking Prediction & Asymmetric Focal Loss
        tt_logits = self.turn_taking_head(h_mid, h_last)  # [B, T_llm]
        tt_loss = self.turn_taking_head.compute_loss(tt_logits, turn_taking_labels)

        # 4. Single-Step Rectified Flow Mel Synthesis & Multi-Component Loss
        flow_loss, pred_mel, flow_metrics = self.flow_synthesizer.forward_training(
            h_llm=h_last,
            target_mel=vocos_mel,
            mel_valid_mask=mel_valid_mask,
        )

        total_loss = self.cfg.loss_weight_flow * flow_loss + self.cfg.loss_weight_tt * tt_loss

        return {
            "loss": total_loss,
            "flow_loss": flow_loss.detach(),
            "tt_loss": tt_loss.detach(),
            "pred_mel": pred_mel.detach(),
            "tt_logits": tt_logits.detach(),
            **flow_metrics,
        }

    @torch.no_grad()
    def stream_step_80ms(
        self,
        whisper_packet_4frames: torch.Tensor,
        proj_state_buffer: Optional[torch.Tensor] = None,
        step_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Real-time 80ms causal streaming inference step:
        Takes 4 frames of Whisper-Tiny (80ms @ 50Hz), updates causal state, predicts EOT probability
        and synthesizes the corresponding 80ms Mel chunk (8 or 7 frames @ 93.75Hz) in 1 step.
        """
        embeds, new_proj_state = self.projector(whisper_packet_4frames, state_buffer=proj_state_buffer)
        transformer_body = self._get_qwen_transformer_body()
        qwen_out = transformer_body(
            inputs_embeds=embeds,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        mid_idx = self.cfg.turn_taking_mid_layer_idx or (len(qwen_out.hidden_states) // 2)
        h_mid = qwen_out.hidden_states[mid_idx]
        h_last = qwen_out.last_hidden_state

        tt_logits = self.turn_taking_head(h_mid, h_last)
        eot_prob = torch.sigmoid(tt_logits[:, -1])
        mel_chunk = self.flow_synthesizer.synthesize_single_step(
            h_last[:, -1:, :], step_parity=step_idx
        )
        return {
            "eot_prob": eot_prob,
            "mel_chunk": mel_chunk,
            "proj_state_buffer": new_proj_state,
        }