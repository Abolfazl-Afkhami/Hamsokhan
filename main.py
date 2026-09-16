import asyncio
import logging
import os
import signal
import sys

try:
    import yaml
except ImportError:
    yaml = None

from src.asr.streaming_asr import StreamingWhisperASR
from src.llm.streaming_llm import StreamingLLMClient
from src.tts.streaming_tts import StreamingFlowTTS
from src.transport.websocket_server import STSWebSocketServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("STS_Main")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if yaml is not None and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "system": {"device": "cuda", "precision": "float16"},
            "server": {"host": "0.0.0.0", "port": 8765},
            "audio": {"input_sample_rate": 16000, "output_sample_rate": 24000},
            "vad": {
                "threshold": 0.5,
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 60,
                "min_speech_frames": 3,
                "min_speech_duration_ms": 250,
                "terminal_silence_keep_ms": 80,
                "max_speech_duration_sec": 8.0,
            },
            "asr": {
                "model_id": "openai/whisper-large-v3-turbo",
                "language": "fa",
                "max_workers": 4,
                "precision": "float16",
            },
            "nlp": {"convert_numbers": True, "diacritize_homographs": True},
            "llm": {
                "api_base": "http://localhost:8000/v1",
                "api_key": "EMPTY",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "temperature": 0.6,
                "max_tokens": 256,
                "system_prompt": "",
            },
            "chunking": {
                "delimiters": [".", "!", "؟", "،", "\n", "؛"],
                "max_words_per_chunk": 12,
                "min_words_per_chunk": 3,
            },
            "tts": {
                "model_path": None,
                "speaker_reference": {"audio_path": None, "transcript": ""},
                "crossfade_ms": 15.0,
            },
        }

    # Environment variables override static configuration (Docker / K8s priority)
    if os.getenv("LLM_API_BASE"):
        config["llm"]["api_base"] = os.getenv("LLM_API_BASE")
    if os.getenv("SERVER_HOST"):
        config["server"]["host"] = os.getenv("SERVER_HOST")
    if os.getenv("SERVER_PORT"):
        config["server"]["port"] = int(os.getenv("SERVER_PORT"))
    if os.getenv("CUDA_DEVICE"):
        config["system"]["device"] = os.getenv("CUDA_DEVICE")

    return config


async def bootstrap_server():
    config = load_config()

    logger.info("Initializing Production Persian Cascaded Streaming STS Pipeline...")
    logger.info(f"Target LLM Backend: {config['llm']['api_base']}")

    asr = StreamingWhisperASR(
        model_size_or_path=config["asr"]["model_id"],
        device=config["system"]["device"],
        compute_type=config["system"]["precision"],
        language=config["asr"]["language"],
        max_workers=config["asr"]["max_workers"],
    )

    llm = StreamingLLMClient(
        api_base=config["llm"]["api_base"],
        api_key=config["llm"]["api_key"],
        model=config["llm"]["model"],
        system_prompt=config["llm"]["system_prompt"],
        delimiters=config["chunking"]["delimiters"],
        max_words_per_chunk=config["chunking"]["max_words_per_chunk"],
        min_words_per_chunk=config["chunking"]["min_words_per_chunk"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )

    tts = StreamingFlowTTS(
        sample_rate=config["audio"]["output_sample_rate"],
        model_path=config["tts"]["model_path"],
        ref_audio_path=config["tts"]["speaker_reference"]["audio_path"],
        ref_text=config["tts"]["speaker_reference"]["transcript"],
        crossfade_ms=config["tts"]["crossfade_ms"],
        device=config["system"]["device"],
    )

    server = STSWebSocketServer(
        host=config["server"]["host"],
        port=config["server"]["port"],
        config=config,
        asr=asr,
        llm=llm,
        tts=tts,
    )

    await server.start()
    logger.info("Production Pipeline Gateway active.")

    stop_event = asyncio.Event()

    def handle_signal():
        logger.info("Shutdown signal received.")
        stop_event.set()

    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_signal)

    try:
        await stop_event.wait()
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        logger.info("Closing STS WebSocket Gateway...")
        await server.close()
        asr.shutdown()
        logger.info("Production Gateway shutdown cleanly.")


if __name__ == "__main__":
    try:
        asyncio.run(bootstrap_server())
    except KeyboardInterrupt:
        pass
