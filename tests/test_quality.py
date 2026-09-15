"""Stage 4 quality signals: telling garbled Spanish from good Spanish."""

from wsla.config import settings
from wsla.pipeline import quality


def _is_garbled(text: str) -> bool:
    return quality.is_garbled(
        text,
        min_accent_rate=settings.fallback_min_accent_rate,
        min_letters=settings.quality_min_letters,
    )


def test_good_spanish_is_not_flagged(good_spanish):
    assert quality.accent_rate(good_spanish) > settings.fallback_min_accent_rate
    assert not _is_garbled(good_spanish)


def test_garbled_spanish_is_flagged(garbled_spanish):
    assert quality.accent_rate(garbled_spanish) < settings.fallback_min_accent_rate
    assert _is_garbled(garbled_spanish)


def test_short_fragments_are_never_judged():
    # A page footer carries no accents but is far too short to judge.
    assert not quality.is_garbled("Pagina 1 de 8", min_accent_rate=0.008, min_letters=60)


def test_accented_text_outscores_flattened_text(good_spanish, garbled_spanish):
    assert quality.quality_score(good_spanish) > quality.quality_score(garbled_spanish)


def test_table_pipes_do_not_count_as_text():
    assert quality.count_alnum("| a | b |\n|---|---|") == 2
