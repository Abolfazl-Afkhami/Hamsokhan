import re
from typing import Dict, Optional


class PersianG2PDiacritizer:
    """
    Micro-G2P and Homograph Diacritizer for Persian Neural TTS:
    1. Static homograph disambiguation lexicon targeting common phonetic collisions.
    2. Vowel-ending ezafe resolution for nouns ending in silent heh.
    3. Prosody and non-verbal tag bypass (preserves [tone: ...], [pause: ...], [sigh], etc.).
    """

    HOMOGRAPH_LEXICON: Dict[str, str] = {
        "کشت": "کُشت",
        "سرد": "سَرد",
        "پست": "پُست",
        "گفت": "گُفت",
        "برد": "بُرد",
        "شد": "شُد",
        "کرد": "کَرد",
        "دست": "دَست",
        "درد": "دَرد",
        "بزرگ": "بُزُرگ",
        "تند": "تُند",
        "پر": "پُر",
        "پل": "پُل",
        "سر": "سَر",
    }

    TAG_SPLIT_REGEX = re.compile(r"(\[[^\]]*\])")
    WORD_TOKEN_REGEX = re.compile(r"([^\W\d_]+(?:[\u200c][^\W\d_]+)*)")

    # Ezafe patterns:
    # 1. Silent heh with hamza above (U+0654 / U+0674)
    EZAFE_HE_HAMZA = re.compile(r"([^\W\d_]+ه)[\u0654\u0674]")
    # 2. Silent heh with detached or fused heh-ye (U+06C0)
    EZAFE_HEH_YEH = re.compile(r"([^\W\d_]+)ۀ")
    # 3. Silent heh followed by space and yeh
    EZAFE_HE_SPACE_YE = re.compile(r"([^\W\d_]+ه)\s+ی(?![\u064b-\u0652])\b")
    # 4. Silent heh followed by ZWNJ and yeh
    EZAFE_HE_ZWNJ_YE = re.compile(r"([^\W\d_]+ه)\u200cی(?![\u064b-\u0652])\b")
    # 5. Long vowels (alef or vav) followed by detached ezafe yeh
    EZAFE_VOWEL_SPACE_YE = re.compile(r"([^\W\d_]+[او])\s+ی(?![\u064b-\u0652])\b")

    def __init__(self, custom_lexicon: Optional[Dict[str, str]] = None):
        self.lexicon = dict(self.HOMOGRAPH_LEXICON)
        if custom_lexicon:
            self.lexicon.update(custom_lexicon)

    def add_homograph(self, word: str, diacritized: str):
        self.lexicon[word] = diacritized

    def _apply_ezafe_rules(self, text: str) -> str:
        # Resolve hamza on silent heh to standard phonetized ezafe
        text = self.EZAFE_HE_HAMZA.sub(lambda m: m.group(1) + "\u200cی\u0650", text)
        text = self.EZAFE_HEH_YEH.sub(lambda m: m.group(1) + "ه\u200cی\u0650", text)
        # Resolve detached yeh after silent heh
        text = self.EZAFE_HE_SPACE_YE.sub(lambda m: m.group(1) + "\u200cی\u0650", text)
        # Attach kasreh to existing ZWNJ ezafe on silent heh
        text = self.EZAFE_HE_ZWNJ_YE.sub(lambda m: m.group(1) + "\u200cی\u0650", text)
        # Resolve detached yeh after long vowels
        text = self.EZAFE_VOWEL_SPACE_YE.sub(lambda m: m.group(1) + "\u200cی\u0650", text)
        return text

    def _replace_word(self, match: re.Match) -> str:
        w = match.group(0)
        return self.lexicon.get(w, w)

    def diacritize(self, text: str) -> str:
        if not text:
            return ""

        # Split around bracketed tags to preserve prosody / non-verbal cues
        segments = self.TAG_SPLIT_REGEX.split(text)
        processed = []

        for seg in segments:
            if not seg:
                continue
            if seg.startswith("[") and seg.endswith("]"):
                processed.append(seg)
                continue

            seg_ezafe = self._apply_ezafe_rules(seg)
            seg_diacritized = self.WORD_TOKEN_REGEX.sub(self._replace_word, seg_ezafe)
            processed.append(seg_diacritized)

        return "".join(processed)
