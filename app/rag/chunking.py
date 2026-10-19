"""Fragmentación medida en tokens (tiktoken), no en caracteres. Viene de la Pre-entrega 4."""

from __future__ import annotations

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.models import Chunk, DocType, ParsedDocument

# Para código los cortes naturales son clases y funciones antes que párrafos.
_CODE_SEPARATORS = ["\nclass ", "\nasync def ", "\ndef ", "\n\n", "\n", " ", ""]


class TokenChunker:
    """Parte el texto en chunks de como máximo `chunk_size` tokens con solapamiento."""

    def __init__(self, chunk_size: int = 450, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def _splitter(self, limit: int, code: bool) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=limit,
            chunk_overlap=self.chunk_overlap,
            length_function=self.count_tokens,
            separators=_CODE_SEPARATORS if code else None,
        )

    def split_text(self, text: str, *, code: bool = False) -> list[str]:
        """Garantiza que ningún chunk supere chunk_size tokens.

        El splitter suma los tokens de cada pedazo por separado y los tokens no se suman así
        ("palabra" suelta y " palabra" en una frase tokenizan distinto): un chunk puede pasarse
        por 1 o 2 tokens. Si pasa, se vuelve a partir con un límite interno más chico.
        """

        limit = self.chunk_size
        while True:
            parts = [p for p in self._splitter(limit, code).split_text(text) if p.strip()]
            if all(self.count_tokens(p) <= self.chunk_size for p in parts):
                return parts
            limit -= 5

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        code = document.metadata.doc_type in (DocType.CODE, DocType.NOTEBOOK)
        return [
            Chunk.build(document.metadata, index, text)
            for index, text in enumerate(self.split_text(document.text, code=code))
        ]
