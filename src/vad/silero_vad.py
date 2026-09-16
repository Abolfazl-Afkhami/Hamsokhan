from collections import deque
from typing import Optional, Tuple

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None


class SileroVADDetector:
    """
    Production-grade streaming VAD:
    1. Pre-speech ring buffer to preserve plosives (/p/, /t/, /k/).
    2. Hysteresis onset confirmation (minimum consecutive speech frames).
    3. Dual-event emission: emits (speech_segment, speech_onset) to enable
       instantaneous sub-100ms acoustic barge-in without waiting for turn offset.
    4. Trailing silence retention (80ms) to preserve terminal acoustic decay.
    5. Exact sample-based trimming independent of packet jitter or variable chunk sizes.
    6. Hard maximum speech duration cutoff.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 60,
        min_speech_frames: int = 3,         # ~90ms confirmation
        min_speech_duration_ms: int = 250,  # Minimum true voiced speech required
        terminal_silence_keep_ms: int = 80, # Keep 80ms tail padding for acoustic decay
        max_speech_duration_sec: float = 8.0,
        device: str = "cpu",
    ):
        if torch is None or np is None:
            raise ImportError("PyTorch and NumPy are required for SileroVADDetector.")

        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence_samples = int(sample_rate * (min_silence_duration_ms / 1000.0))
        self.min_speech_samples = int(sample_rate * (min_speech_duration_ms / 1000.0))
        self.keep_silence_samples = int(sample_rate * (terminal_silence_keep_ms / 1000.0))
        self.max_speech_samples = int(sample_rate * max_speech_duration_sec)
        self.min_speech_frames = min_speech_frames
        self.device = torch.device(device)

        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
        )

        self.pad_capacity_samples = int(sample_rate * (speech_pad_ms / 1000.0))
        self.pre_speech_ring_buffer = deque()
        self.pre_speech_current_samples = 0

        self.speech_buffer = []
        self.is_speaking = False
        self.speech_frame_counter = 0
        self.silence_samples_count = 0
        self.current_speech_samples = 0

    def reset_states(self):
        if hasattr(self, "model") and self.model:
            self.model.reset_states()
        self.is_speaking = False
        self.speech_frame_counter = 0
        self.silence_samples_count = 0
        self.speech_buffer.clear()
        self.pre_speech_ring_buffer.clear()
        self.pre_speech_current_samples = 0
        self.current_speech_samples = 0

    def _push_pre_speech(self, chunk: "np.ndarray"):
        self.pre_speech_ring_buffer.append(chunk)
        self.pre_speech_current_samples += len(chunk)
        while self.pre_speech_ring_buffer and (
            self.pre_speech_current_samples - len(self.pre_speech_ring_buffer[0])
        ) >= self.pad_capacity_samples:
            removed = self.pre_speech_ring_buffer.popleft()
            self.pre_speech_current_samples -= len(removed)

    def process_chunk(self, audio_chunk: "np.ndarray") -> Tuple[Optional["np.ndarray"], bool]:
        """
        Processes incoming 16kHz float32 audio chunk.
        Returns:
            (complete_speech_segment, is_speech_onset)
            is_speech_onset is True strictly on the exact frame where speech is confirmed.
        """
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        tensor_chunk = torch.from_numpy(audio_chunk).to(self.device)
        if tensor_chunk.ndim == 1:
            tensor_chunk = tensor_chunk.unsqueeze(0)

        with torch.no_grad():
            speech_prob = self.model(tensor_chunk, self.sample_rate).item()

        chunk_len = len(audio_chunk)
        speech_onset_detected = False

        if speech_prob >= self.threshold:
            self.speech_frame_counter += 1

            if not self.is_speaking:
                if self.speech_frame_counter >= self.min_speech_frames:
                    self.is_speaking = True
                    speech_onset_detected = True  # Instantaneous Onset Barge-in Trigger

                    self.speech_buffer.extend(self.pre_speech_ring_buffer)
                    self.current_speech_samples += self.pre_speech_current_samples
                    self.pre_speech_ring_buffer.clear()
                    self.pre_speech_current_samples = 0

                    self.speech_buffer.append(audio_chunk)
                    self.current_speech_samples += chunk_len
                    self.silence_samples_count = 0
                else:
                    self._push_pre_speech(audio_chunk)
            else:
                self.silence_samples_count = 0
                self.speech_buffer.append(audio_chunk)
                self.current_speech_samples += chunk_len

                if self.current_speech_samples >= self.max_speech_samples:
                    result = np.concatenate(self.speech_buffer)
                    self.reset_states()
                    return result, speech_onset_detected

        else:
            self.speech_frame_counter = 0

            if self.is_speaking:
                self.silence_samples_count += chunk_len
                self.speech_buffer.append(audio_chunk)
                self.current_speech_samples += chunk_len

                if (
                    self.silence_samples_count >= self.min_silence_samples
                    or self.current_speech_samples >= self.max_speech_samples
                ):
                    voiced_samples = self.current_speech_samples - self.silence_samples_count

                    if voiced_samples >= self.min_speech_samples:
                        trim = max(0, self.silence_samples_count - self.keep_silence_samples)
                        full_audio = np.concatenate(self.speech_buffer)
                        result = full_audio[:-trim] if trim > 0 else full_audio
                        self.reset_states()
                        return result, speech_onset_detected
                    else:
                        self.reset_states()
            else:
                self._push_pre_speech(audio_chunk)

        return None, speech_onset_detected
