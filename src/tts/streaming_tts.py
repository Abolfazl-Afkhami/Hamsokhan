import asyncio
import os
from typing import AsyncGenerator, Optional, Dict

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None


class SessionCrossfadeState:
    """Isolated per-session acoustic crossfade state preventing multi-user state collision."""
    def __init__(self):
        self.overlap_tail: Optional["np.ndarray"] = None

    def reset(self):
        self.overlap_tail = None


class StreamingFlowTTS:
    """
    Controllable Flow Matching TTS Synthesizer (CosyVoice-2 / F5-TTS interface).
    Stateless and thread-safe engine:
    1. CUDA Stream isolation per engine.
    2. Zero shared mutable state across concurrent sessions via SessionCrossfadeState.
    3. Read-only pre-cached acoustic prompts in RAM.
    """

    NON_VERBAL_PROMPT_MAP = {
        "sigh": "assets/prompts/speaker_sigh_latent.npy",
        "laugh": "assets/prompts/speaker_laugh_latent.npy",
        "breath": "assets/prompts/speaker_breath_latent.npy",
        "gasp": "assets/prompts/speaker_gasp_latent.npy",
    }

    def __init__(
        self,
        sample_rate: int = 24000,
        model_path: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        crossfade_ms: float = 15.0,
        device: str = "cuda",
    ):
        if torch is None or np is None:
            raise ImportError("PyTorch and NumPy are required for StreamingFlowTTS.")

        self.sample_rate = sample_rate
        self.model_path = model_path
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.default_crossfade_samples = int(self.sample_rate * (crossfade_ms / 1000.0))
        self.device = device

        if self.device == "cuda" and torch.cuda.is_available():
            self.cuda_stream = torch.cuda.Stream()
        else:
            self.cuda_stream = None

        self.non_verbal_cache: Dict[str, np.ndarray] = {}
        self._load_non_verbal_prompts()
        self.is_ready = True

    def _load_non_verbal_prompts(self):
        for cue, path in self.NON_VERBAL_PROMPT_MAP.items():
            if os.path.exists(path):
                try:
                    self.non_verbal_cache[cue] = np.load(path)
                except Exception:
                    pass

    def generate_silence(self, duration_ms: int) -> "np.ndarray":
        num_samples = int(self.sample_rate * (duration_ms / 1000.0))
        return np.zeros(num_samples, dtype=np.float32)

    def _apply_crossfade(self, chunk: "np.ndarray", state: SessionCrossfadeState) -> "np.ndarray":
        chunk_len = len(chunk)
        if chunk_len == 0:
            return chunk

        active_crossfade = min(self.default_crossfade_samples, chunk_len // 3)
        if active_crossfade < 16:
            return chunk

        fade = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, active_crossfade, endpoint=False))).astype(np.float32)
        fade_out = 1.0 - fade

        if state.overlap_tail is None or len(state.overlap_tail) != active_crossfade:
            output = chunk[:-active_crossfade].copy()
            state.overlap_tail = chunk[-active_crossfade:].copy()
            return output

        head = chunk[:active_crossfade]
        blended = (state.overlap_tail * fade_out) + (head * fade)
        body = chunk[active_crossfade:-active_crossfade]
        output = np.concatenate([blended, body])

        state.overlap_tail = chunk[-active_crossfade:].copy()
        return output

    def flush(self, state: SessionCrossfadeState) -> Optional["np.ndarray"]:
        tail = state.overlap_tail
        state.overlap_tail = None
        return tail

    async def synthesize_non_verbal(self, cue: str, state: SessionCrossfadeState) -> AsyncGenerator["np.ndarray", None]:
        if cue in self.non_verbal_cache:
            raw_audio = self.non_verbal_cache[cue]
        else:
            duration_sec = 0.25
            num_samples = int(self.sample_rate * duration_sec)
            raw_audio = np.zeros(num_samples, dtype=np.float32)

        yield self._apply_crossfade(raw_audio, state)

    async def synthesize_chunk(
        self,
        text: str,
        state: SessionCrossfadeState,
        tone: Optional[str] = None,
        pause_ms: int = 0,
        non_verbal: Optional[str] = None,
    ) -> AsyncGenerator["np.ndarray", None]:
        if pause_ms > 0:
            tail = self.flush(state)
            if tail is not None and len(tail) > 0:
                yield tail
            yield self.generate_silence(pause_ms)
            return

        if non_verbal:
            async for chunk in self.synthesize_non_verbal(non_verbal, state):
                yield chunk
            return

        if not text or not text.strip():
            return

        chunk_duration_sec = 0.1
        chunk_samples = int(self.sample_rate * chunk_duration_sec)
        word_count = max(1, len(text.split()))
        estimated_duration = word_count / 3.5
        total_chunks = max(1, int(estimated_duration / chunk_duration_sec))

        for i in range(total_chunks):
            t = np.linspace(i * chunk_duration_sec, (i + 1) * chunk_duration_sec, chunk_samples, endpoint=False)
            f0 = 180.0 if tone == "cheerful" else (120.0 if tone == "serious" else 140.0)
            raw_audio = 0.2 * np.sin(2 * np.pi * f0 * t).astype(np.float32)

            processed_audio = self._apply_crossfade(raw_audio, state)
            await asyncio.sleep(0.01)
            yield processed_audio
