import os
import unittest

from src.nlp.normalizer import PersianTextNormalizer
from src.nlp.prosody_parser import ProsodyTagParser, ProsodySegment
from src.nlp.g2p_diacritizer import PersianG2PDiacritizer
from src.asr.streaming_asr import StreamingWhisperASR
from src.tts.streaming_tts import SessionCrossfadeState
from src.pipeline.orchestrator import SpeechToSpeechOrchestrator
from main import load_config


class TestSTSPipelineUnits(unittest.TestCase):

    def setUp(self):
        self.normalizer = PersianTextNormalizer(convert_numbers=True)
        self.parser = ProsodyTagParser()
        self.asr_filter = StreamingWhisperASR.__new__(StreamingWhisperASR)

    def test_normalizer_arabic_and_zwnj(self):
        raw_text = "اين يك كتاب است.  كتاب‌هاي من."
        normalized = self.normalizer.normalize(raw_text)
        self.assertIn("این یک کتاب است.", normalized)
        self.assertIn("کتاب‌ها", normalized)
        self.assertNotIn("ك", normalized)
        self.assertNotIn("ي", normalized)

    def test_normalizer_homograph_preservation(self):
        text = "در میان ما مشکلی وجود ندارد."
        normalized = self.normalizer.normalize(text)
        self.assertIn("در میان ما", normalized)

    def test_normalizer_markdown_stripping(self):
        markdown_text = "**سلام**، این یک *پروژه* `تست` است."
        normalized = self.normalizer.normalize(markdown_text)
        self.assertNotIn("*", normalized)
        self.assertNotIn("`", normalized)
        self.assertIn("سلام، این یک پروژه تست است.", normalized)

    def test_prosody_tag_parsing(self):
        tagged_text = "[tone: empathetic] درکت میکنم، [pause: 200ms] [sigh] نگران نباش."
        segments = self.parser.parse_chunk(tagged_text)

        tones = [s.tone for s in segments if s.tone]
        pauses = [s.pause_ms for s in segments if s.pause_ms > 0]
        non_verbals = [s.non_verbal for s in segments if s.non_verbal]

        self.assertIn("empathetic", tones)
        self.assertIn(200, pauses)
        self.assertIn("sigh", non_verbals)

        sigh_seg = [s for s in segments if s.non_verbal == "sigh"][0]
        self.assertEqual(sigh_seg.text, "")

    def test_asr_hallucination_loop_actual_code(self):
        hallucinated_text = (
            "این یک تست است این یک تست است این یک تست است این یک تست است این یک تست است این یک تست است"
        )
        self.assertTrue(self.asr_filter._is_hallucination_loop(hallucinated_text))

        valid_short_repeated = "نه نه نه نه"
        self.assertFalse(self.asr_filter._is_hallucination_loop(valid_short_repeated))

        valid_text = "سلام خسته نباشید من میخواستم وضعیت سفارش خودم رو پیگیری کنم"
        self.assertFalse(self.asr_filter._is_hallucination_loop(valid_text))

    def test_session_crossfade_state_isolation(self):
        state1 = SessionCrossfadeState()
        state2 = SessionCrossfadeState()

        # Mutate state1
        state1.overlap_tail = "audio_tail_client_1"

        # Verify state2 remains completely unaffected (Zero State Collision)
        self.assertIsNone(state2.overlap_tail)
        self.assertEqual(state1.overlap_tail, "audio_tail_client_1")

        state1.reset()
        self.assertIsNone(state1.overlap_tail)

    def test_environment_variable_override(self):
        os.environ["LLM_API_BASE"] = "http://custom-vllm-host:8000/v1"
        os.environ["SERVER_PORT"] = "9999"

        config = load_config()
        self.assertEqual(config["llm"]["api_base"], "http://custom-vllm-host:8000/v1")
        self.assertEqual(config["server"]["port"], 9999)

        # Cleanup test environment variables
        del os.environ["LLM_API_BASE"]
        del os.environ["SERVER_PORT"]

    def test_g2p_diacritizer_homographs(self):
        diacritizer = PersianG2PDiacritizer()
        input_text = "او کشت و کار را در هوای سرد پست کرد."
        output = diacritizer.diacritize(input_text)
        self.assertIn("کُشت", output)
        self.assertIn("سَرد", output)
        self.assertIn("پُست", output)
        self.assertIn("کَرد", output)

    def test_g2p_diacritizer_ezafe_and_tags(self):
        diacritizer = PersianG2PDiacritizer()
        input_text = "[tone: empathetic] خانه‌ی من و خانهٔ تو [pause: 200ms] [sigh] زیباست."
        output = diacritizer.diacritize(input_text)
        self.assertIn("[tone: empathetic]", output)
        self.assertIn("[pause: 200ms]", output)
        self.assertIn("[sigh]", output)
        self.assertIn("خانه‌یِ من", output)
        self.assertIn("خانه‌یِ تو", output)

    def test_orchestrator_dialogue_history_and_truncation(self):
        import asyncio

        class MockASR:
            async def transcribe_segment_async(self, audio):
                return "سلام چطوری؟"

        class MockLLM:
            async def generate_chunks(self, user_query, conversation_history=None):
                yield "سلام من خوبم. "
                yield "امروز چطور می‌تونم کمکت کنم؟"

        class MockTTS:
            async def synthesize_chunk(self, text, state, tone=None, pause_ms=0, non_verbal=None):
                if text:
                    yield [0.1, 0.2]

            def flush(self, state):
                return None

        orchestrator = SpeechToSpeechOrchestrator(
            vad=None,
            asr=MockASR(),
            normalizer=PersianTextNormalizer(convert_numbers=False),
            prosody_parser=ProsodyTagParser(),
            llm=MockLLM(),
            tts=MockTTS(),
            diacritizer=PersianG2PDiacritizer(),
            max_history_turns=4,
        )

        async def run_turn():
            emitted = []
            await orchestrator.execute_turn(
                audio_segment=[1, 2, 3],
                on_audio_chunk=lambda chunk: emitted.append(chunk),
            )
            await orchestrator.current_pipeline_task

        asyncio.run(run_turn())

        self.assertEqual(len(orchestrator.conversation_history), 2)
        self.assertEqual(orchestrator.conversation_history[0]["role"], "user")
        self.assertEqual(orchestrator.conversation_history[0]["content"], "سلام چطوری؟")
        self.assertEqual(orchestrator.conversation_history[1]["role"], "assistant")
        self.assertIn("سلام من خوبم", orchestrator.conversation_history[1]["content"])

    def test_orchestrator_barge_in_acoustic_truncation(self):
        import asyncio

        class MockASR:
            async def transcribe_segment_async(self, audio):
                return "آب و هوا چطوره؟"

        class MockLLM:
            async def generate_chunks(self, user_query, conversation_history=None):
                yield "هوا ابریه. "
                await asyncio.sleep(0.5)
                yield "احتمال باران هم هست."

        class MockTTS:
            async def synthesize_chunk(self, text, state, tone=None, pause_ms=0, non_verbal=None):
                if text:
                    yield [0.1, 0.2]

            def flush(self, state):
                return None

        orchestrator = SpeechToSpeechOrchestrator(
            vad=None,
            asr=MockASR(),
            normalizer=PersianTextNormalizer(convert_numbers=False),
            prosody_parser=ProsodyTagParser(),
            llm=MockLLM(),
            tts=MockTTS(),
            diacritizer=PersianG2PDiacritizer(),
        )

        async def run_turn_with_barge_in():
            emitted = []
            await orchestrator.execute_turn(
                audio_segment=[1, 2, 3],
                on_audio_chunk=lambda chunk: emitted.append(chunk),
            )
            # Yield control for the first chunk to be synthesized and emitted
            await asyncio.sleep(0.05)
            # Simulate user barge-in interruption while the LLM is waiting for next chunk
            await orchestrator.interrupt()

        asyncio.run(run_turn_with_barge_in())

        # Assistant history should strictly contain only the first emitted chunk
        self.assertEqual(len(orchestrator.conversation_history), 2)
        assistant_content = orchestrator.conversation_history[1]["content"]
        self.assertIn("هوا ابریه", assistant_content)
        self.assertNotIn("احتمال باران", assistant_content)

    def test_paced_soft_pause_drainage(self):
        import asyncio
        from src.transport.websocket_server import STSWebSocketServer

        server = STSWebSocketServer.__new__(STSWebSocketServer)
        egress_queue = asyncio.Queue()
        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        async def run_drain():
            drain_task = asyncio.create_task(
                server._drain_soft_mute_paced(chunks, egress_queue, pace_interval_sec=0.01)
            )
            await drain_task

        asyncio.run(run_drain())

        self.assertEqual(egress_queue.qsize(), 3)
        self.assertEqual(egress_queue.get_nowait(), b"chunk1")
        self.assertEqual(egress_queue.get_nowait(), b"chunk2")
        self.assertEqual(egress_queue.get_nowait(), b"chunk3")


if __name__ == "__main__":
    unittest.main()
