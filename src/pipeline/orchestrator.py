import asyncio
import time
from typing import Callable, Optional, List, Dict

try:
    import numpy as np
except ImportError:
    np = None

from src.vad.silero_vad import SileroVADDetector
from src.asr.streaming_asr import StreamingWhisperASR
from src.nlp.normalizer import PersianTextNormalizer
from src.nlp.prosody_parser import ProsodyTagParser
from src.nlp.g2p_diacritizer import PersianG2PDiacritizer
from src.llm.streaming_llm import StreamingLLMClient
from src.tts.streaming_tts import StreamingFlowTTS, SessionCrossfadeState


class SpeechToSpeechOrchestrator:
    """
    Cascaded Ultra-Low Latency Speech-to-Speech (STS) pipeline orchestrator.
    Features:
    1. Isolated SessionCrossfadeState per session instance (eliminates multi-client acoustic bleed).
    2. Instant zero-delay cancellation with generation ID turn isolation.
    3. Safe task supervisor callback preventing unretrieved exception leaks.
    """

    def __init__(
        self,
        vad: SileroVADDetector,
        asr: StreamingWhisperASR,
        normalizer: PersianTextNormalizer,
        prosody_parser: ProsodyTagParser,
        llm: StreamingLLMClient,
        tts: StreamingFlowTTS,
        diacritizer: Optional[PersianG2PDiacritizer] = None,
        max_history_turns: int = 20,
    ):
        self.vad = vad
        self.asr = asr
        self.normalizer = normalizer
        self.prosody_parser = prosody_parser
        self.llm = llm
        self.tts = tts
        self.diacritizer = diacritizer or PersianG2PDiacritizer()
        self.max_history_turns = max_history_turns

        # Per-session isolated acoustic crossfade state
        self.crossfade_state = SessionCrossfadeState()

        # Multi-turn dialogue history tracking (capped at max_history_turns)
        self.conversation_history: List[Dict[str, str]] = []

        self._lock = asyncio.Lock()
        self.active_turn_id = 0
        self.current_pipeline_task: Optional[asyncio.Task] = None

    def clear_history(self):
        """Resets dialogue history for the active session."""
        self.conversation_history.clear()

    async def interrupt(self):
        """
        Instantaneous turn invalidation and task cancellation without artificial delays.
        """
        self.active_turn_id += 1
        if self.current_pipeline_task and not self.current_pipeline_task.done():
            self.current_pipeline_task.cancel()
            self.current_pipeline_task = None

        self.crossfade_state.reset()

    def _task_supervisor_callback(self, t: asyncio.Task):
        if not t.cancelled():
            exc = t.exception()
            if exc:
                pass

    async def execute_turn(
        self,
        audio_segment: "np.ndarray",
        on_audio_chunk: Optional[Callable[["np.ndarray"], None]] = None,
    ):
        async with self._lock:
            await self.interrupt()
            turn_id = self.active_turn_id

            self.current_pipeline_task = asyncio.create_task(
                self._process_turn_worker(audio_segment, on_audio_chunk, turn_id)
            )
            self.current_pipeline_task.add_done_callback(self._task_supervisor_callback)

    async def _process_turn_worker(
        self,
        audio_segment: "np.ndarray",
        on_audio_chunk: Optional[Callable[["np.ndarray"], None]],
        turn_id: int,
    ):
        start_time = time.perf_counter()

        # 1. Non-blocking ASR transcription on thread pool
        raw_text = await self.asr.transcribe_segment_async(audio_segment)
        if not raw_text.strip() or turn_id != self.active_turn_id:
            return

        normalized_query = self.normalizer.normalize(raw_text)
        active_tone: Optional[str] = None
        spoken_response_accumulator: List[str] = []

        try:
            # 2. LLM Streaming with match-span chunking and dialogue history context
            async for chunk in self.llm.generate_chunks(
                user_query=normalized_query,
                conversation_history=self.conversation_history,
            ):
                if turn_id != self.active_turn_id:
                    break

                # 3. Prosody and Tag Parsing
                segments = self.prosody_parser.parse_chunk(chunk, current_tone=active_tone)

                for seg in segments:
                    if turn_id != self.active_turn_id:
                        break

                    if seg.tone:
                        active_tone = seg.tone

                    clean_text = self.normalizer.normalize(seg.text)
                    if not clean_text and seg.pause_ms == 0 and not seg.non_verbal:
                        continue

                    # 4. Micro-G2P / Homograph Diacritization for Neural Flow TTS
                    tts_text = self.diacritizer.diacritize(clean_text) if clean_text else ""

                    # 5. Neural Flow TTS Streaming with isolated session state
                    segment_audio_emitted = False
                    async for audio_chunk in self.tts.synthesize_chunk(
                        text=tts_text,
                        state=self.crossfade_state,
                        tone=seg.tone,
                        pause_ms=seg.pause_ms,
                        non_verbal=seg.non_verbal,
                    ):
                        if turn_id != self.active_turn_id:
                            break

                        segment_audio_emitted = True
                        if on_audio_chunk:
                            on_audio_chunk(audio_chunk)

                    if segment_audio_emitted and clean_text:
                        spoken_response_accumulator.append(clean_text)

            # Flush remaining audio tail
            if turn_id == self.active_turn_id:
                final_tail = self.tts.flush(self.crossfade_state)
                if final_tail is not None and len(final_tail) > 0 and on_audio_chunk:
                    on_audio_chunk(final_tail)

        except asyncio.CancelledError:
            self.crossfade_state.reset()
            raise
        finally:
            # Acoustic truncation on barge-in: commit strictly what was synthesized and emitted
            if spoken_response_accumulator:
                spoken_text = " ".join(spoken_response_accumulator).strip()
                if spoken_text:
                    self.conversation_history.append({"role": "user", "content": normalized_query})
                    self.conversation_history.append({"role": "assistant", "content": spoken_text})
                    if len(self.conversation_history) > self.max_history_turns:
                        self.conversation_history = self.conversation_history[-self.max_history_turns:]
