import asyncio
import json
import time
import numpy as np
import websockets


async def run_client_mock(
    server_uri: str = "ws://127.0.0.1:8765",
    speech_duration_sec: float = 1.5,
):
    """
    Simulated real-time streaming client:
    1. Sends 16kHz 16-bit PCM audio chunks (30ms per frame).
    2. Sends 400ms silence to trigger VAD segmentation.
    3. Measures TTFAB (Time to First Audio Byte).
    4. Tests barge-in interruption signal.
    """
    print(f"[TEST CLIENT] Connecting to STS Gateway at {server_uri}...")

    async with websockets.connect(server_uri) as ws:
        print("[TEST CLIENT] Connected. Generating synthetic Persian speech signal...")

        sample_rate = 16000
        frame_duration_sec = 0.03  # 30ms frame = 480 samples
        frame_samples = int(sample_rate * frame_duration_sec)

        # Generate voiced speech waveform (harmonic formants @ 140Hz)
        t_speech = np.linspace(0, speech_duration_sec, int(sample_rate * speech_duration_sec), endpoint=False)
        voiced_signal = (0.4 * np.sin(2 * np.pi * 140 * t_speech) + 0.2 * np.sin(2 * np.pi * 280 * t_speech)).astype(np.float32)

        # Generate 400ms trailing silence to trigger VAD
        silence_samples = int(sample_rate * 0.4)
        silence_signal = np.zeros(silence_samples, dtype=np.float32)

        full_stream = np.concatenate([voiced_signal, silence_signal])
        total_frames = len(full_stream) // frame_samples

        print(f"[TEST CLIENT] Streaming {total_frames} frames ({len(full_stream)/sample_rate:.2f}s)...")
        speech_end_time = None
        ttfab = None
        received_bytes = 0

        async def receive_loop():
            nonlocal ttfab, received_bytes
            async for msg in ws:
                if isinstance(msg, bytes):
                    if ttfab is None and speech_end_time is not None:
                        ttfab = (time.perf_counter() - speech_end_time) * 1000.0
                        print(f"[METRIC] >> TTFAB (Time-to-First-Audio-Byte): {ttfab:.2f} ms <<")
                    received_bytes += len(msg)

        recv_task = asyncio.create_task(receive_loop())

        # Stream audio chunks at real-time pace
        for i in range(total_frames):
            frame = full_stream[i * frame_samples : (i + 1) * frame_samples]
            pcm_bytes = (frame * 32767.0).astype(np.int16).tobytes()
            await ws.send(pcm_bytes)

            # Record point where voiced speech concludes and silence starts
            if i == int(speech_duration_sec / frame_duration_sec):
                speech_end_time = time.perf_counter()

            await asyncio.sleep(frame_duration_sec)

        # Wait for audio generation response
        await asyncio.sleep(2.0)

        # Test Barge-in interruption
        print("[TEST CLIENT] Sending manual barge-in interrupt...")
        await ws.send(json.dumps({"type": "interrupt"}))
        await asyncio.sleep(0.5)

        recv_task.cancel()
        print(f"[TEST CLIENT] Test complete. Total audio bytes received: {received_bytes}")


if __name__ == "__main__":
    asyncio.run(run_client_mock())
