import json

import pytest

from app.rag.chunking import TokenChunker
from app.rag.loaders import (
    EmptyDocumentError,
    UnsupportedFileError,
    infer_module,
    parse_document,
)
from app.rag.models import DocType
from tests.conftest import make_pdf


def test_markdown_toma_el_titulo_del_primer_h1() -> None:
    doc = parse_document(b"# Cliente async\n\nTexto   con  espacios.", "modulo_01_x/a.md")
    assert doc.metadata.title == "Cliente async"
    assert doc.metadata.doc_type is DocType.MARKDOWN
    assert doc.metadata.module == 1
    assert "Texto con espacios." in doc.text


def test_html_descarta_scripts_y_usa_el_title() -> None:
    html = (
        b"<html><head><title>Glosario</title><script>alert(1)</script></head>"
        b"<body><h1>RRF</h1><p>Fusiona rankings.</p></body></html>"
    )
    doc = parse_document(html, "general/glosario.html")
    assert doc.metadata.title == "Glosario"
    assert "alert" not in doc.text
    assert "Fusiona rankings." in doc.text


def test_notebook_conserva_markdown_y_codigo_pero_no_salidas() -> None:
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Demo\n", "Explicación"]},
            {"cell_type": "code", "source": "print('hola')", "outputs": [{"text": "SALIDA"}]},
        ]
    }
    doc = parse_document(json.dumps(nb).encode(), "modulo_06/demo.ipynb")
    assert "Explicación" in doc.text
    assert "```python\nprint('hola')\n```" in doc.text
    assert "SALIDA" not in doc.text


def test_codigo_incluye_el_nombre_del_archivo() -> None:
    doc = parse_document(b"def f():\n    return 1\n", "modulo_05_langgraph/graph.py")
    assert doc.text.startswith("# Archivo: graph.py")
    assert doc.metadata.doc_type is DocType.CODE


def test_pdf_extrae_texto_con_marca_de_pagina() -> None:
    doc = parse_document(make_pdf(["Clase 4 RAG avanzado", "Pinecone serverless"]), "x.pdf")
    assert doc.text.startswith("[p. 1]")
    assert "Pinecone serverless" in doc.text


def test_extension_no_soportada() -> None:
    with pytest.raises(UnsupportedFileError):
        parse_document(b"x", "foto.png")


def test_archivo_vacio() -> None:
    with pytest.raises(EmptyDocumentError):
        parse_document(b"   \n\n ", "vacio.md")


@pytest.mark.parametrize(
    ("path", "module"),
    [
        ("sample_docs/modulo_05_langgraph/a.md", 5),
        ("knowledge_base/presentaciones/AI_Engineering_Clase_4_RAG.pdf", 4),
        ("Entrega 7/README.md", 7),
        ("general/glosario.html", 0),
    ],
)
def test_infer_module(path: str, module: int) -> None:
    assert infer_module(path) == module


def test_modulo_explicito_gana_sobre_la_ruta() -> None:
    doc = parse_document(b"texto", "modulo_02/a.md", module=6)
    assert doc.metadata.module == 6


def test_chunks_respetan_el_limite_de_tokens_y_tienen_ids_estables() -> None:
    chunker = TokenChunker(chunk_size=60, chunk_overlap=10)
    text = "\n\n".join(f"Párrafo {i}: " + "palabra " * 40 for i in range(6))
    doc = parse_document(text.encode(), "modulo_03/largo.md")
    chunks = chunker.chunk(doc)
    assert len(chunks) > 3
    assert all(chunker.count_tokens(c.text) <= 60 for c in chunks)
    assert [c.id for c in chunks] == [f"{doc.metadata.doc_id}::{i}" for i in range(len(chunks))]
    assert chunks == chunker.chunk(doc)
