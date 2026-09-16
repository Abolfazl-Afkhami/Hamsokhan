import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ProsodySegment:
    text: str
    tone: Optional[str] = None
    pause_ms: int = 0
    non_verbal: Optional[str] = None


class ProsodyTagParser:
    """
    Parses and extracts inline prosody, emotion, and timing control tags.
    Strips non-verbal tags (e.g. [sigh], [laugh]) from the text payload to prevent
    the neural TTS model from literally reading bracket/tag characters aloud.
    """

    TAG_PATTERN = re.compile(
        r"\[(tone:\s*[\w\-]+|pause:\s*\d+ms|sigh|laugh|gasp|breath)\]",
        re.IGNORECASE,
    )

    def parse_chunk(self, chunk: str, current_tone: Optional[str] = None) -> List[ProsodySegment]:
        segments: List[ProsodySegment] = []
        last_idx = 0
        active_tone = current_tone

        for match in self.TAG_PATTERN.finditer(chunk):
            start, end = match.span()
            tag_content = match.group(1).lower().strip()

            if start > last_idx:
                preceding_text = chunk[last_idx:start].strip()
                if preceding_text:
                    segments.append(ProsodySegment(text=preceding_text, tone=active_tone))

            if tag_content.startswith("tone:"):
                active_tone = tag_content.split(":", 1)[1].strip()
            elif tag_content.startswith("pause:"):
                ms_val = int(re.search(r"\d+", tag_content).group(0))
                segments.append(ProsodySegment(text="", tone=active_tone, pause_ms=ms_val))
            else:
                # Acoustic non-verbal action: text payload is intentionally empty
                segments.append(
                    ProsodySegment(text="", tone=active_tone, non_verbal=tag_content)
                )

            last_idx = end

        if last_idx < len(chunk):
            trailing_text = chunk[last_idx:].strip()
            if trailing_text:
                segments.append(ProsodySegment(text=trailing_text, tone=active_tone))

        return segments

    def strip_tags(self, text: str) -> str:
        """Strip all prosody tags completely."""
        return self.TAG_PATTERN.sub("", text).strip()
