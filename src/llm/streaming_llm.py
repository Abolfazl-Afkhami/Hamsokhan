import asyncio
import re
from typing import AsyncGenerator, List, Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


class StreamingLLMClient:
    """
    Streaming LLM client interfacing with vLLM / OpenAI-compatible API.
    Uses regex Match Spans on atomic tokens to extract exact chunks without substring false-matching
    or whitespace/newline flattening.
    """

    DEFAULT_DELIMITERS = [".", "!", "؟", "،", "\n", "؛"]
    PERSIAN_ENGLISH_DIGITS = "0123456789۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩"
    VALID_TAG_PREFIXES = ("tone:", "pause:", "sigh", "laugh", "gasp", "breath")
    MAX_TAG_ACCUMULATION_CHARS = 30

    ATOMIC_TOKEN_REGEX = re.compile(r"\[[^\]]*\]|\S+")

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        system_prompt: str = "",
        delimiters: Optional[List[str]] = None,
        max_words_per_chunk: int = 12,
        min_words_per_chunk: int = 3,
        temperature: float = 0.6,
        max_tokens: int = 256,
    ):
        if AsyncOpenAI is None:
            raise ImportError("The 'openai' library is required to run StreamingLLMClient.")

        self.client = AsyncOpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.delimiters = delimiters or self.DEFAULT_DELIMITERS
        self.max_words_per_chunk = max_words_per_chunk
        self.min_words_per_chunk = min_words_per_chunk
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _is_decimal_point(self, text: str, index: int) -> bool:
        if 0 < index < len(text) - 1:
            prev_char = text[index - 1]
            next_char = text[index + 1]
            if prev_char in self.PERSIAN_ENGLISH_DIGITS and next_char in self.PERSIAN_ENGLISH_DIGITS:
                return True
        return False

    def _has_unclosed_valid_tag(self, buffer: str) -> bool:
        last_open = buffer.rfind("[")
        if last_open == -1:
            return False

        last_close = buffer.rfind("]")
        if last_close > last_open:
            return False

        tag_fragment = buffer[last_open + 1:].strip().lower()
        if len(tag_fragment) > self.MAX_TAG_ACCUMULATION_CHARS:
            return False

        return any(
            p.startswith(tag_fragment) or tag_fragment.startswith(p[:4])
            for p in self.VALID_TAG_PREFIXES
        )

    async def generate_chunks(
        self,
        user_query: str,
        conversation_history: Optional[List[dict]] = None,
    ) -> AsyncGenerator[str, None]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_query})

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        buffer = ""

        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue

                buffer += delta

                if self._has_unclosed_valid_tag(buffer):
                    continue

                while True:
                    delim_found = False
                    earliest_pos = len(buffer)
                    selected_delim = None

                    for d in self.delimiters:
                        pos = 0
                        while True:
                            idx = buffer.find(d, pos)
                            if idx == -1:
                                break
                            if d == "." and self._is_decimal_point(buffer, idx):
                                pos = idx + 1
                                continue
                            if idx < earliest_pos:
                                earliest_pos = idx
                                selected_delim = d
                            break

                    if selected_delim is not None:
                        split_idx = earliest_pos + len(selected_delim)
                        candidate = buffer[:split_idx].strip()
                        remaining = buffer[split_idx:]
                        token_count = len(self.ATOMIC_TOKEN_REGEX.findall(candidate))

                        if token_count >= self.min_words_per_chunk or selected_delim in [".", "!", "؟", "\n"]:
                            yield candidate
                            buffer = remaining
                            delim_found = True
                        else:
                            break

                    if not delim_found:
                        tokens_matches = list(self.ATOMIC_TOKEN_REGEX.finditer(buffer))
                        if len(tokens_matches) >= self.max_words_per_chunk and not self._has_unclosed_valid_tag(buffer):
                            target_match = tokens_matches[self.max_words_per_chunk - 1]
                            cut_idx = target_match.end()
                            chunk_to_yield = buffer[:cut_idx].strip()
                            buffer = buffer[cut_idx:].lstrip()
                            yield chunk_to_yield
                        else:
                            break

            final_text = buffer.strip()
            if final_text:
                yield final_text

        except (asyncio.CancelledError, GeneratorExit):
            raise
        finally:
            if hasattr(stream, "response") and hasattr(stream.response, "aclose"):
                await stream.response.aclose()
            elif hasattr(stream, "close"):
                res = stream.close()
                if asyncio.iscoroutine(res):
                    await res
