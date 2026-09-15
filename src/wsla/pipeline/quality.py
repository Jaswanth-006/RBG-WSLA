"""Stage 4 — quality: spot OCR output that is present but wrong.

A page can be full of characters and still be unusable. `WLA_Mensajeria
Corporativa_Final.pdf` is a 2009 scan carrying a garbled OCR layer — "Marfa" for
*María*, "105 alcances" for *los alcances* — with 232-2068 characters per page.
A character count alone can never flag it.

For Spanish the sharpest cheap signal is the accented-character rate. Measured
per page across the WSLA sample documents:

    PaddleOCR output           2.02% - 2.40%   (document level, good)
    Docling + EasyOCR          0.45% - 3.49%   (per page, acceptable)
    Embedded 2009 text layer   0.00% - 0.70%   (per page, garbled)

Hence the 0.8% default in ``settings.fallback_min_accent_rate``: it catches every
garbled page without flagging healthy ones.
"""

import re

_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_ACCENTED = re.compile(r"[áéíóúüñÁÉÍÓÚÜÑ¿¡]")


def count_alnum(text: str) -> int:
    """Count alphanumeric characters — the 'is there any text here' metric.

    Whitespace, punctuation and table pipes are ignored so a markdown table
    skeleton or a row of dashes cannot pass as readable text.
    """
    return sum(ch.isalnum() for ch in text)


def letter_count(text: str) -> int:
    """Count letters only, digits excluded."""
    return len(_LETTER.findall(text))


def accent_rate(text: str) -> float | None:
    """Accented characters as a fraction of letters, or None if there are none."""
    letters = letter_count(text)
    if letters == 0:
        return None
    return len(_ACCENTED.findall(text)) / letters


def is_garbled(text: str, *, min_accent_rate: float, min_letters: int) -> bool:
    """True when text looks like broken Spanish rather than missing Spanish.

    Short fragments are never judged: a caption or a table of part numbers can
    legitimately carry no accents, so pages with fewer than ``min_letters``
    letters are left alone.
    """
    letters = letter_count(text)
    if letters < min_letters:
        return False
    return len(_ACCENTED.findall(text)) / letters < min_accent_rate


def quality_score(text: str) -> float:
    """Rank two candidate texts for the same page; higher wins.

    Length dominates — more text read is usually better — but accented Spanish
    breaks the tie, so a slightly shorter read with proper accents beats a
    longer one that flattened them.
    """
    letters = letter_count(text)
    if letters == 0:
        return 0.0
    rate = len(_ACCENTED.findall(text)) / letters
    return letters * (1.0 + 4.0 * min(rate, 0.05))
