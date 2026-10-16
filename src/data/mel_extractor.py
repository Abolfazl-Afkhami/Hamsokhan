"""
Vocos-Aligned Mel Spectrogram Extractor & Smart Persian VAD Trimmer
Features:
1. Exact native Vocos 24kHz parameters: n_fft=1024, hop_length=256, n_mels=100.
2. 160ms Macro-Packet Alignment: Enforces exact integer ratio (15 Mel frames per 160ms / 2 Qwen steps).
3. Strips the single extra frame from center=True to maintain exact 15*K frames.
4. Smart VAD trimming with Room Tone preservation (160ms onset, 240-320ms offset) + 15ms Hann cosine taper.
"""

from typing import List, Dict, Optional
import torch
import torch.nn.functional as F
import torchaudio


class VocosAlignedMelExtractor(torch.nn.Module):
    """
    Extracts 100-channel log Mel-spectrograms strictly aligned with Vocos 24kHz ISTFTHead.
    Input: wav_24k [1, N] or [B, N]
    Output: log_mel [100, 15*K] in FP16
    """
    def __init__(self, sample_rate: int = 24000, n_mels: int = 100):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_samples = 3840  # 160ms @ 24kHz = 2 LLM steps (12.5Hz) = 15 Mel frames

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=0.0,
            f_max=12000.0,
            n_mels=n_mels,
            center=True,
            pad_mode="reflect",
            power=1.0,
            norm="slaney",
            mel_scale="slaney",
        )

    @torch.no_grad()
    def forward(self, wav_24k: torch.Tensor) -> torch.Tensor:
        """
        wav_24k: [1, N] or [B, 1, N] audio at 24kHz.
        Returns: [B, 100, T_mel] where T_mel is an exact multiple of 15.
        """
        if wav_24k.dim() == 2:
            wav_24k = wav_24k.unsqueeze(1) if wav_24k.shape[0] != 1 else wav_24k
        if wav_24k.dim() == 1:
            wav_24k = wav_24k.unsqueeze(0).unsqueeze(0)

        # 160ms Macro-Packet quantizing: pad audio to multiple of 3840 samples
        orig_len = wav_24k.shape[-1]
        rem = orig_len % self.block_samples
        if rem != 0:
            pad_amt = self.block_samples - rem
            wav_24k = F.pad(wav_24k, (0, pad_amt), mode="reflect")

        # Extract Mel spectrogram
        mel = self.mel_transform(wav_24k)  # [B, 1, 100, 15*K + 1]
        mel = mel.squeeze(1)               # [B, 100, 15*K + 1]

        # Strip single extra center=True frame -> exact [B, 100, 15*K]
        mel = mel[:, :, :-1]

        # Clamp and natural log
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel.to(torch.float16)


def smart_persian_vad_trim(
    wav_24k: torch.Tensor,
    vad_timestamps_sec: Optional[List[Dict[str, float]]] = None,
    pad_start_ms: int = 160,
    pad_end_ms: int = 240,
    sr: int = 24000
) -> torch.Tensor:
    """
    Trims silence while preserving natural room tone and Persian phoneme edges:
    - 160ms onset padding preserves plosive bursts (/پ/, /ت/, /ک/).
    - 240ms (TTS) or 320ms (PersianVox) offset padding preserves sentence-final devoicing/breath.
    - 15ms Hann cosine taper (360 samples) eliminates STFT boundary clicks.
    """
    if not vad_timestamps_sec:
        return wav_24k

    if wav_24k.dim() == 1:
        wav_24k = wav_24k.unsqueeze(0)

    total_samples = wav_24k.shape[-1]
    start_sec = vad_timestamps_sec[0]["start"]
    end_sec = vad_timestamps_sec[-1]["end"]

    # Slice directly from actual audio (room tone)
    s_idx = max(0, int((start_sec - pad_start_ms / 1000.0) * sr))
    e_idx = min(total_samples, int((end_sec + pad_end_ms / 1000.0) * sr))
    trimmed = wav_24k[:, s_idx:e_idx].clone()

    # 15ms Hann window cosine taper (360 samples @ 24kHz)
    fade_len = min(int(0.015 * sr), trimmed.shape[-1] // 4)
    if fade_len > 0:
        ramp = torch.hann_window(fade_len * 2, device=trimmed.device, dtype=trimmed.dtype)
        trimmed[:, :fade_len] *= ramp[:fade_len]
        trimmed[:, -fade_len:] *= ramp[fade_len:]

    return trimmed


if __name__ == "__main__":
    extractor = VocosAlignedMelExtractor()
    # Test audio of 3840 * 2 = 7680 samples (320ms = 2 macro packets)
    test_wav = torch.randn(1, 7680)
    mel = extractor(test_wav)
    assert mel.shape == (1, 100, 30), f"Expected [1, 100, 30], got {mel.shape}"
    print(f"✓ VocosAlignedMelExtractor verified: {test_wav.shape} -> {mel.shape}")
