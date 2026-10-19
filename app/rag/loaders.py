"""Extracción de texto por tipo de archivo: PDF, HTML, Markdown, Python y notebooks.

Son funciones puras sobre bytes (no tocan disco ni red), así sirven igual para la ingesta
inicial desde carpetas y para los archivos que llegan por la API. Parsear un PDF es trabajo
de CPU: quien llama desde código async lo corre con asyncio.to_thread.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import PurePath

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.rag.models import DocType, DocumentMetadata, ParsedDocument

EXTENSIONS: dict[str, DocType] = {
    ".pdf": DocType.PDF,
    ".html": DocType.HTML,
    ".htm": DocType.HTML,
    ".md": DocType.MARKDOWN,
    ".markdown": DocType.MARKDOWN,
    ".py": DocType.CODE,
    ".ipynb": DocType.NOTEBOOK,
    ".txt": DocType.TEXT,
}

_MODULE = re.compile(r"(?:modulo|módulo|module|clase|entrega)[\s_-]*0*(\d{1,2})", re.IGNORECASE)
_BLANKS = re.compile(r"[ \t\f\v]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_MD_TITLE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


class UnsupportedFileError(ValueError):
    """La extensión no está entre las soportadas."""


class EmptyDocumentError(ValueError):
    """El archivo no tiene texto extraíble (ej. un PDF escaneado sin OCR)."""


def doc_type_for(filename: str) -> DocType:
    suffix = PurePath(filename).suffix.lower()
    if suffix not in EXTENSIONS:
        allowed = ", ".join(sorted(EXTENSIONS))
        raise UnsupportedFileError(
            f"Extensión no soportada: {suffix or '(ninguna)'}. Usá {allowed}"
        )
    return EXTENSIONS[suffix]


def infer_module(path: str) -> int:
    """'modulo_05_langgraph/x.md' -> 5. La carpeta más cercana al archivo gana."""

    for part in reversed(PurePath(path).parts):
        match = _MODULE.search(part)
        if match:
            return int(match.group(1))
    return 0


def clean_text(text: str) -> str:
    lines = [_BLANKS.sub(" ", line).rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def _pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    # La marca de página queda en el texto: permite citar "p. 3" sin otro campo de metadata.
    return "\n\n".join(f"[p. {n}]\n{text}" for n, text in enumerate(pages, start=1) if text)


def _html_text(data: bytes) -> tuple[str, str | None]:
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else None
    # Los bloques de código conservan su indentación; el resto se separa por líneas.
    return soup.get_text("\n"), title or None


def _notebook_text(data: bytes) -> str:
    notebook = json.loads(data.decode("utf-8"))
    parts: list[str] = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        text = "".join(source) if isinstance(source, list) else str(source)
        if not text.strip():
            continue
        if cell.get("cell_type") == "code":
            parts.append(f"```python\n{text.strip()}\n```")
        else:
            parts.append(text.strip())
    # Las salidas de las celdas se descartan: suelen ser ruido (tablas, trazas, imágenes).
    return "\n\n".join(parts)


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def parse_document(
    data: bytes,
    source: str,
    *,
    module: int | None = None,
    topic: str = "",
    uploaded_by: str = "seed",
    title: str | None = None,
) -> ParsedDocument:
    """Convierte los bytes de un archivo en texto limpio con sus metadatos."""

    doc_type = doc_type_for(source)
    detected_title: str | None = None
    if doc_type is DocType.PDF:
        raw = _pdf_text(data)
    elif doc_type is DocType.HTML:
        raw, detected_title = _html_text(data)
    elif doc_type is DocType.NOTEBOOK:
        raw = _notebook_text(data)
    elif doc_type is DocType.CODE:
        # El nombre del archivo va adentro del texto para que BM25 lo encuentre por nombre.
        raw = f"# Archivo: {PurePath(source).name}\n{_decode(data)}"
    else:
        raw = _decode(data)
        if doc_type is DocType.MARKDOWN and (match := _MD_TITLE.search(raw)):
            detected_title = match.group(1).strip()

    text = clean_text(raw)
    if not text:
        raise EmptyDocumentError(f"{source} no tiene texto extraíble")

    metadata = DocumentMetadata(
        doc_id=DocumentMetadata.make_id(source),
        source=source,
        title=(title or detected_title or PurePath(source).stem)[:200],
        doc_type=doc_type,
        module=infer_module(source) if module is None else module,
        topic=topic,
        uploaded_by=uploaded_by,
    )
    return ParsedDocument(metadata=metadata, text=text)
