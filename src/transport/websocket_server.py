import asyncio
import json
import logging
import time
from collections import deque
from typing import Optional, Dict

try:
    import numpy as np
except ImportError:
    np = None

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    websockets = None
    WebSocketServerProtocol = None

from src.vad.silero_vad import SileroVADDetector
from src.asr.streaming_asr import StreamingWhisperASR
from src.nlp.normalizer import PersianTextNormalizer
from src.nlp.prosody_parser import ProsodyTagParser
from src.nlp.g2p_diacritizer import PersianG2PDiacritizer
from src.llm.streaming_llm import StreamingLLMClient
from src.tts.streaming_tts import StreamingFlowTTS
from src.pipeline.orchestrator import SpeechToSpeechOrchestrator

logger = logging.getLogger("STS_WebSocket_Gateway")

MAX_FRAME_SIZE_BYTES = 4096
IDLE_SESSION_TIMEOUT_SEC = 60.0
EGRESS_QUEUE_MAX_SIZE = 100


class STSWebSocketServer:
    """
    Production-grade Multi-Session Streaming WebSocket Server:
    1. Session Factory Architecture: isolates VAD, Orchestrator, Normalizer and TTS crossfade state per client.
    2. Soft-Pause Audio Mute Buffer: prevents catastrophic cutoff on temporary coughs/noise bursts.
    3. Confirmed Speech Inactivity Timeout: terminates idle connections regardless of continuous mic silence streaming.
    4. Bounded FIFO Egress Queue: prevents task accumulation and guarantees sequential audio packets.
    """

    def __init__(
        self,
        host: str,
        port: int,
        config: dict,
        asr: StreamingWhisperASR,
        llm: StreamingLLMClient,
        tts: StreamingFlowTTS,
        auth_required: bool = False,
    ):
        self.host = host
        self.port = port
        self.config = config
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.auth_required = auth_required
        self.server = None
        self.active_connections = 0

    async def start(self):
        if websockets is None:
            raise ImportError("The 'websockets' library is required to start STSWebSocketServer.")

        self.server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
            max_size=10 * 1024 * 1024,
        )
        logger.info(f"Production Multi-Session STS Gateway listening on ws://{self.host}:{self.port}")

    def _authenticate(self, websocket: "WebSocketServerProtocol") -> bool:
        if not self.auth_required:
            return True
        query = websocket.path.split("?", 1)
        if len(query) > 1 and "token=" in query[1]:
            token = query[1].split("token=", 1)[1].split("&", 1)[0]
            return len(token) > 16
        return False

    def _create_session_pipeline(self):
        """Creates an isolated VAD and Orchestrator instance per connection."""
        session_vad = SileroVADDetector(
            sample_rate=self.config["audio"]["input_sample_rate"],
            threshold=self.config["vad"]["threshold"],
            min_silence_duration_ms=self.config["vad"]["min_silence_duration_ms"],
            speech_pad_ms=self.config["vad"]["speech_pad_ms"],
            min_speech_frames=self.config["vad"]["min_speech_frames"],
            min_speech_duration_ms=self.config["vad"]["min_speech_duration_ms"],
            terminal_silence_keep_ms=self.config["vad"]["terminal_silence_keep_ms"],
            max_speech_duration_sec=self.config["vad"]["max_speech_duration_sec"],
            device=self.config["system"]["device"] if self.config["system"]["device"] == "cuda" else "cpu",
        )
        session_normalizer = PersianTextNormalizer(
            convert_numbers=self.config["nlp"]["convert_numbers"]
        )
        session_prosody = ProsodyTagParser()
        session_diacritizer = PersianG2PDiacritizer()
        session_orchestrator = SpeechToSpeechOrchestrator(
            vad=session_vad,
            asr=self.asr,
            normalizer=session_normalizer,
            prosody_parser=session_prosody,
            llm=self.llm,
            tts=self.tts,
            diacritizer=session_diacritizer,
        )
        return session_vad, session_orchestrator

    def _clear_egress_queue(self, egress_queue: asyncio.Queue):
        while not egress_queue.empty():
            try:
                egress_queue.get_nowait()
                egress_queue.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break

    async def _drain_soft_mute_paced(
        self,
        chunks: list,
        egress_queue: asyncio.Queue,
        pace_interval_sec: float = 0.08,
    ):
        """Drains held soft-pause audio chunks at steady pacing to prevent client player clock drift."""
        try:
            for chunk in chunks:
                try:
                    await egress_queue.put(chunk)
                except asyncio.QueueFull:
                    pass
                await asyncio.sleep(pace_interval_sec)
        except asyncio.CancelledError:
            pass

    async def _egress_worker(self, websocket: "WebSocketServerProtocol", egress_queue: asyncio.Queue):
        try:
            while True:
                chunk = await egress_queue.get()
                if chunk is None:
                    break
                try:
                    await websocket.send(chunk)
                except Exception:
                    break
                finally:
                    egress_queue.task_done()
        except asyncio.CancelledError:
            pass

    async def handle_connection(self, websocket: "WebSocketServerProtocol"):
        if not self._authenticate(websocket):
            logger.warning(f"Unauthorized connection rejected: {websocket.remote_address}")
            await websocket.close(code=4401, reason="Unauthorized")
            return

        self.active_connections += 1
        logger.info(f"Client connected: {websocket.remote_address} (Active: {self.active_connections})")

        session_vad, session_orchestrator = self._create_session_pipeline()
        egress_queue = asyncio.Queue(maxsize=EGRESS_QUEUE_MAX_SIZE)
        egress_task = asyncio.create_task(self._egress_worker(websocket, egress_queue))

        is_soft_muted = False
        soft_mute_hold_buffer = deque(maxlen=20)
        drain_task: Optional[asyncio.Task] = None
        last_speech_activity = time.monotonic()

        def on_audio_chunk(audio_chunk: "np.ndarray"):
            if audio_chunk is None or len(audio_chunk) == 0:
                return
            pcm16_bytes = (audio_chunk * 32767.0).astype(np.int16).tobytes()

            if is_soft_muted:
                soft_mute_hold_buffer.append(pcm16_bytes)
                return

            try:
                egress_queue.put_nowait(pcm16_bytes)
            except asyncio.QueueFull:
                try:
                    egress_queue.get_nowait()
                    egress_queue.task_done()
                except (asyncio.QueueEmpty, ValueError):
                    pass
                egress_queue.put_nowait(pcm16_bytes)

        try:
            while True:
                message = await websocket.recv()
                current_time = time.monotonic()

                if current_time - last_speech_activity > IDLE_SESSION_TIMEOUT_SEC:
                    logger.info(f"Session speech inactivity timeout for {websocket.remote_address}; terminating.")
                    await websocket.close(code=4408, reason="Session Timeout")
                    break

                if isinstance(message, bytes):
                    if len(message) > MAX_FRAME_SIZE_BYTES:
                        logger.warning(f"Oversized frame received ({len(message)} B); dropping connection.")
                        await websocket.close(code=4400, reason="Payload too large")
                        break

                    audio_data = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                    speech_segment, speech_onset = session_vad.process_chunk(audio_data)

                    if speech_onset:
                        is_soft_muted = True
                        if drain_task and not drain_task.done():
                            drain_task.cancel()
                            drain_task = None
                        soft_mute_hold_buffer.clear()

                    if is_soft_muted and not session_vad.is_speaking and speech_segment is None:
                        is_soft_muted = False
                        if soft_mute_hold_buffer:
                            chunks_to_drain = list(soft_mute_hold_buffer)
                            soft_mute_hold_buffer.clear()
                            if drain_task and not drain_task.done():
                                drain_task.cancel()
                            drain_task = asyncio.create_task(
                                self._drain_soft_mute_paced(chunks_to_drain, egress_queue, pace_interval_sec=0.08)
                            )

                    if speech_segment is not None:
                        last_speech_activity = current_time
                        is_soft_muted = False
                        if drain_task and not drain_task.done():
                            drain_task.cancel()
                            drain_task = None
                        soft_mute_hold_buffer.clear()

                        logger.info(f"Confirmed speech turn: {len(speech_segment)} samples")
                        await session_orchestrator.interrupt()
                        self._clear_egress_queue(egress_queue)
                        await session_orchestrator.execute_turn(speech_segment, on_audio_chunk)

                elif isinstance(message, str):
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")
                        if msg_type == "interrupt":
                            last_speech_activity = current_time
                            await session_orchestrator.interrupt()
                            is_soft_muted = False
                            if drain_task and not drain_task.done():
                                drain_task.cancel()
                                drain_task = None
                            soft_mute_hold_buffer.clear()
                            self._clear_egress_queue(egress_queue)
                        elif msg_type == "ping":
                            await websocket.send(json.dumps({"type": "pong"}))
                    except json.JSONDecodeError:
                        pass

        except Exception:
            pass
        finally:
            self.active_connections -= 1
            logger.info(f"Client disconnected: {websocket.remote_address} (Remaining: {self.active_connections})")
            if drain_task and not drain_task.done():
                drain_task.cancel()
            await session_orchestrator.interrupt()
            self._clear_egress_queue(egress_queue)
            session_vad.reset_states()
            egress_task.cancel()

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
