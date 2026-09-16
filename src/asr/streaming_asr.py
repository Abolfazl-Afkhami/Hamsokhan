import asyncio
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


class StreamingWhisperASR:
    """
    Sub-second Persian ASR engine using Faster-Whisper.
    Includes zlib entropy checking and repetition loop suppression tuned to preserve
    colloquial conversational affirmations (e.g. 'نه نه نه نه').
    """

    def __init__(
        self,
        model_size_or_path: str = "openai/whisper-large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "fa",
        max_workers: int = 4,
    ):
        if WhisperModel is None:
            raise ImportError("faster-whisper is required to run StreamingWhisperASR.")

        self.language = language
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        try:
            self.model = WhisperModel(
                model_size_or_path,
                device=device,
                compute_type=compute_type,
                num_workers=max_workers,
            )
        except Exception:
            self.model = WhisperModel(
                model_size_or_path,
                device="cpu",
                compute_type="int8",
                num_workers=max_workers,
            )

    def _is_hallucination_loop(self, text: str) -> bool:
        """
        Detects repetitive acoustic hallucination loops while preserving short
        conversational affirmations ('نه نه نه نه', 'باشه باشه').
        """
        if not text or len(text) < 20:
            return False

        encoded = text.encode("utf-8")
        compressed = zlib.compress(encoded)
        ratio = len(compressed) / len(encoded)

        # Extreme compression in long outputs indicates catastrophic token repetition
        if len(text) > 40 and ratio < 0.28:
            return True

        words = text.split()
        # Word-dominance check is restricted to utterances >= 8 words to prevent false rejection
        if len(words) >= 8:
            most_common_freq = max(words.count(w) for w in set(words))
            if most_common_freq / len(words) > 0.70:
                return True

        return False

    def _sync_transcribe(self, audio_data: "np.ndarray") -> str:
        if audio_data is None or len(audio_data) == 0:
            return ""

        segments, _ = self.model.transcribe(
            audio_data,
            language=self.language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=True,
        )

        full_text = " ".join([s.text.strip() for s in segments if s.text]).strip()

        if self._is_hallucination_loop(full_text):
            return ""

        return full_text

    async def transcribe_segment_async(self, audio_data: "np.ndarray") -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._sync_transcribe, audio_data)

    def shutdown(self):
        if hasattr(self, "executor") and self.executor:
            self.executor.shutdown(wait=False)
