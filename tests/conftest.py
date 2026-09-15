"""Shared fixtures: the same Spanish sentence, read correctly and read badly.

The garbled version mirrors the real defects in the embedded 2009 text layer of
WLA_Mensajeria Corporativa_Final.pdf: "prop6sito", "105 alcances", "Marfa", and
accents flattened throughout.
"""

import pytest

GOOD_SPANISH = (
    "El propósito del presente documento es definir los alcances y actividades "
    "que tendrán cada una de las áreas involucradas en MENSAJERÍA CORPORATIVA. "
    "Elaboró María Rosa Elva Mendoza, revisión de la Gerencia de Administración."
)

GARBLED_SPANISH = (
    "EI prop6sito del presente documento es definir 105 alcances y actividades "
    "que tend ran cada una de las areas involucradas en MENSAJERIA CORPORATIVA. "
    "Elaboro Marfa Rosa Elva Mendoza, revision de la Gerencia de Administracion."
)


@pytest.fixture
def good_spanish() -> str:
    return GOOD_SPANISH


@pytest.fixture
def garbled_spanish() -> str:
    return GARBLED_SPANISH
