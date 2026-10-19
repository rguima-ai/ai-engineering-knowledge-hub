"""Normalización de texto compartida por BM25 y por los embeddings de prueba."""

from __future__ import annotations

import re
import unicodedata

# Palabras vacías del español, sin tildes porque se comparan después de normalizar.
STOPWORDS = frozenset(
    """
a al algo algun alguna ante como con contra cual cuales cuando de del desde donde e el ella ellos
en entre era es esa ese eso esta este esto estos hay la las le les lo los mas me mi mis muy ni no
nos o otra otro para pero por porque puede que se sea ser si sin sobre son su sus tambien tan te
tiene tienen todo todos tu un una uno unos y ya yo hace hacer sirve uso the of and to in is
""".split()
)

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Minúsculas, sin tildes, sin signos y sin palabras vacías.

    'asyncio.gather' se parte en 'asyncio' y 'gather': matchea con y sin el módulo.
    """

    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in _TOKEN.findall(text) if t not in STOPWORDS and len(t) > 1]
