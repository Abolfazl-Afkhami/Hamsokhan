import re
from typing import Optional

try:
    from num2fawords import words as num2words_fa
except ImportError:
    num2words_fa = None


class PersianTextNormalizer:
    """
    Deterministic Text Normalizer (TN) for speech synthesis:
    1. Standardizes Arabic glyphs to standard Persian equivalents.
    2. Strips bidirectional and invisible control characters (LRM, RLM, ZWSP).
    3. Strips markdown styling formatting (*, _, #, `).
    4. Expands numerals to spoken Persian words.
    5. Cleans and regularizes zero-width non-joiners (ZWNJ).
    No heuristic verbal morphology or rule-based inflection guessing.
    """

    ARABIC_TO_PERSIAN = {
        ord("ي"): "ی",
        ord("ى"): "ی",
        ord("ك"): "ک",
        ord("ة"): "ه",
        ord("ۀ"): "ه",
        ord("ؤ"): "و",
        ord("إ"): "ا",
        ord("أ"): "ا",
        ord("ء"): "",
    }

    PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
    ENGLISH_DIGITS = "0123456789"
    ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"

    def __init__(self, convert_numbers: bool = True):
        self.convert_numbers = convert_numbers

    def clean_characters(self, text: str) -> str:
        """Map Arabic glyphs, strip directional marks and markdown formatting."""
        text = text.translate(self.ARABIC_TO_PERSIAN)
        # Strip LRM, RLM, zero-width spaces
        text = re.sub(r"[\u200e\u200f\u200b]", "", text)
        # Strip markdown syntax symbols that should not be vocalized
        text = re.sub(r"[*_#`~>]+", "", text)
        return text

    def normalize_zwnj(self, text: str) -> str:
        """Collapse multiple ZWNJs and strip dangling ZWNJs adjacent to spaces."""
        text = re.sub(r"\u200c+", "\u200c", text)
        text = re.sub(r"\s+\u200c|\u200c\s+", " ", text)
        return text

    def normalize_numbers_to_words(self, text: str) -> str:
        """Convert consecutive digit sequences to spoken Persian words."""
        if not num2words_fa:
            return text

        def replace_num(match):
            raw_digits = match.group(0)
            normalized_digits = "".join(
                str(self.PERSIAN_DIGITS.index(ch)) if ch in self.PERSIAN_DIGITS else (
                    str(self.ARABIC_DIGITS.index(ch)) if ch in self.ARABIC_DIGITS else ch
                )
                for ch in raw_digits
            )
            try:
                val = int(normalized_digits)
                return " " + num2words_fa(val) + " "
            except ValueError:
                return match.group(0)

        pattern = r"[\d" + self.PERSIAN_DIGITS + self.ARABIC_DIGITS + r"]+"
        return re.sub(pattern, replace_num, text)

    def normalize(self, text: str) -> str:
        """Execute deterministic normalization."""
        if not text:
            return ""

        text = self.clean_characters(text)
        text = self.normalize_zwnj(text)

        if self.convert_numbers:
            text = self.normalize_numbers_to_words(text)

        text = re.sub(r"\s+", " ", text)
        return text.strip()
