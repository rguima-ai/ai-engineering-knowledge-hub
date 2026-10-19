# RAG Local End-to-End

Pre-Entrega 3 de AI Engineering (Coderhouse). Un sistema **RAG local** que responde preguntas sobre los apuntes del curso usando **solo** lo que está en los documentos, y dice exactamente `"No lo sé."` cuando la respuesta no está.

- **Ingesta:** los `.md` de `./data` se limpian, se parten en chunks de 500 tokens con 50 de overlap (medidos con tiktoken) y se persisten en **ChromaDB** (`./vectorstore`), con embeddings locales de `intfloat/multilingual-e5-small`.
- **Recuperación:** top-4 por similitud coseno, con un umbral mínimo de relevancia.
- **Generación grounded:** una cadena LCEL `prompt | llm | PydanticOutputParser`, con OpenAI o Anthropic intercambiables.

```python
resultado = await get_rag_response("¿Qué tamaño de chunk y cuánto overlap se recomiendan?")
resultado.respuesta    # texto basado solo en los fragmentos
resultado.referencias  # [Referencia(fragmento=1, fuente='clase3_embeddings_chromadb.md',
                       #             chunk_id='clase3_embeddings_chromadb.md::2', score=0.871)]
```

## Cómo correrlo

Requiere **Python 3.12**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env              # completá LLM_PROVIDER y la key de ese proveedor
python ingest.py                  # indexa ./data en ./vectorstore (solo si algo cambió)
python rag.py --solo-recuperacion # fragmentos recuperados y scores, SIN llamar al LLM (no necesita keys)
python rag.py                     # las dos pruebas: pregunta con respuesta + pregunta trampa
```

> La primera ingesta **descarga el modelo de embeddings desde HuggingFace (unos 470 MB)** y tarda un par de minutos. Después queda en `~/.cache/huggingface` y la ingesta funciona sin internet. Si los documentos no cambiaron, `ingest.py` no reindexa.

`rag.py` también corre la ingesta al arrancar, así que alcanza con `python rag.py`.

## Variables de entorno

| Variable | Obligatoria | Default | Para qué |
|---|---|---|---|
| `LLM_PROVIDER` | no | `openai` | Proveedor que elige el Factory: `openai` o `anthropic` |
| `OPENAI_API_KEY` | si usás OpenAI | | Key de OpenAI (también para los embeddings de OpenAI) |
| `ANTHROPIC_API_KEY` | si usás Anthropic | | Key de Anthropic |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | Modelo de OpenAI |
| `ANTHROPIC_MODEL` | no | `claude-haiku-4-5-20251001` | Modelo de Anthropic |
| `LLM_MAX_TOKENS` | no | `1024` | Tope de tokens de la respuesta |
| `LLM_TIMEOUT_S` | no | `30` | Timeout por llamada, en segundos |
| `EMBEDDINGS_PROVIDER` | no | `huggingface` | `huggingface` (local, sin key) u `openai` |
| `HF_EMBEDDINGS_MODEL` | no | `intfloat/multilingual-e5-small` | Modelo de embeddings local |
| `OPENAI_EMBEDDINGS_MODEL` | no | `text-embedding-3-small` | Modelo de embeddings de OpenAI |
| `TOP_K` | no | `4` | Fragmentos a recuperar (validado entre 3 y 5) |
| `MIN_RELEVANCE` | no | `0.80` | Relevancia mínima (1 − distancia coseno). Calibrada para e5 |
| `LOG_LEVEL` | no | `INFO` | Nivel de logging |

## Ejemplo de salida

### Recuperación (verificada, sin keys)

`python rag.py --solo-recuperacion` sobre el índice real (27 chunks, e5). ✅ supera `MIN_RELEVANCE=0.80` y ❌ no lo supera:

```
=== ¿Qué tamaño de chunk y cuánto overlap se recomiendan como práctica común?
  ✅ 0.871  clase3_embeddings_chromadb.md::2         ### El proceso de limpieza Los documentos reales (PDFs, HTMLs) traen ruido...
  ✅ 0.866  clase4_rag_avanzado_pinecone.md::6       ### Optimización: diagnóstico antes que parche - ¿Falta contexto? Aumentar el overlap...
  ✅ 0.845  clase3_embeddings_chromadb.md::5         ### Errores comunes a evitar en la Pre-Entrega 3 - **El "contexto infinito"**...
  ✅ 0.842  clase4_rag_avanzado_pinecone.md::5       El **re-ranking** es una segunda etapa en la que un Cross-Encoder evalúa...
  -> 4 de 4 superan MIN_RELEVANCE=0.8

=== ¿En qué año se creó ChromaDB y qué empresa lo desarrolla?
  ✅ 0.842  clase3_embeddings_chromadb.md::3         Para recordar: los chunks muy pequeños pierden el contexto global...
  ✅ 0.835  clase3_embeddings_chromadb.md::4         Regla clave: nunca consultar una colección con un modelo de embeddings distinto...
  ✅ 0.828  clase4_rag_avanzado_pinecone.md::1       Pinecone separa el cómputo del almacenamiento...
  ✅ 0.818  clase4_rag_avanzado_pinecone.md::0       # Clase 4 — Escalabilidad documental: RAG avanzado y Pinecone...
  -> 4 de 4 superan MIN_RELEVANCE=0.8

=== ¿Cuál es la receta tradicional de las empanadas salteñas?
  ❌ 0.788  clase4_rag_avanzado_pinecone.md::6       ### Optimización: diagnóstico antes que parche...
  ❌ 0.781  clase2_langchain_lcel.md::4              En los parsers, el flujo es: Pydantic detecta si la estructura...
  ❌ 0.780  clase3_embeddings_chromadb.md::2         ### El proceso de limpieza...
  ❌ 0.780  clase3_embeddings_chromadb.md::1         ### Embeddings estáticos vs. contextuales...
  -> 0 de 4 superan MIN_RELEVANCE=0.8
```

| Pregunta | Mejor score | Pasan el umbral | Qué hace el sistema |
|---|---|---|---|
| Chunk y overlap recomendados | 0.871 | 4 de 4 | El top-1 (`clase3::2`) contiene "500 tokens con 50 tokens de overlap": va al LLM |
| **Trampa:** año de creación de ChromaDB | 0.842 | 4 de 4 | Recupera fragmentos de ChromaDB, pero ninguno tiene la respuesta: el LLM tiene que decir `"No lo sé."` |
| Fuera de tema: empanadas | 0.788 | 0 de 4 | `"No lo sé."` **sin llamar al LLM** |

La pregunta trampa es a propósito cercana al tema: pasa el filtro de relevancia y llega al LLM, que conoce la respuesta por su entrenamiento. Es la prueba de que el prompt lo obliga a no usar conocimiento general.

### Respuestas del LLM (las dos pruebas)

Salida real de `python rag.py` con `LLM_PROVIDER=openai` (`gpt-4o-mini`):

```
=== Prueba 1: pregunta con respuesta en los documentos
Pregunta: ¿Qué tamaño de chunk y cuánto overlap se recomiendan como práctica común?
{
  "respuesta": "Se recomienda usar chunks de 500 tokens con 50 tokens de overlap como práctica común.",
  "referencias": [
    {
      "fragmento": 1,
      "fuente": "clase3_embeddings_chromadb.md",
      "chunk_id": "clase3_embeddings_chromadb.md::2",
      "score": 0.8711
    }
  ]
}

=== Prueba 2: pregunta trampa (sin respuesta en los documentos)
Pregunta: ¿En qué año se creó ChromaDB y qué empresa lo desarrolla?
{
  "respuesta": "No lo sé.",
  "referencias": []
}
```

- **Prueba 1:** de los 4 fragmentos que recibió, el LLM citó solo el que tiene la respuesta (`clase3::2`, el top-1). Por eso las referencias no listan todo lo recuperado, como pasaba en la pista.
- **Prueba 2:** los 4 fragmentos superaron el umbral (de 0.818 a 0.842) y llegaron al LLM. Aunque `gpt-4o-mini` conoce la respuesta, contestó exactamente `"No lo sé."`, sin referencias: **no alucinó**.

Los logs de esa corrida (una sola llamada al LLM por pregunta, sin reintentos):

```
INFO ingest ingesta=omitida motivo=sin_cambios chunks=27 fingerprint=3e53529fecb2
INFO rag recuperacion top_k=4 relevantes=4 umbral=0.80 scores=[clase3_embeddings_chromadb.md::2=0.871, clase4_rag_avanzado_pinecone.md::6=0.866, ...]
INFO rag ok latencia_ms=2684 referencias=['clase3_embeddings_chromadb.md::2']
INFO rag recuperacion top_k=4 relevantes=4 umbral=0.80 scores=[clase3_embeddings_chromadb.md::3=0.842, clase3_embeddings_chromadb.md::4=0.835, ...]
INFO rag ok latencia_ms=942 referencias=[]
```

Sin key, `python rag.py` hace la ingesta y termina con un aviso claro, sin traceback:

```
Índice: 27 chunks (sin cambios)
⚠️ No se pudo configurar el LLM: OpenAIError: Missing credentials. Please pass an `api_key`, ...
```

## Tests

```bash
pytest
```

Son 28 tests y **no usan internet ni API keys**. Los embeddings son `DeterministicFakeEmbedding`, que arma el vector a partir del hash del texto: si la pregunta es el texto exacto de un chunk, su relevancia es 1.0, y contra otro texto da cerca de 0. El LLM es un `FakeListChatModel` que cuenta sus llamadas y guarda el prompt que recibió. ChromaDB corre en una carpeta temporal por test. La única descarga es la del encoding `cl100k_base` de tiktoken, la primera vez; después queda en caché.

| Grupo | Qué prueba |
|---|---|
| DocumentProcessor | Ningún chunk de los apuntes supera 500 tokens y cada archivo se parte en 3 o más. Los IDs son `archivo::n`. Un texto largo se parte con overlap. `clean_text` colapsa espacios y 3+ saltos de línea, conserva los párrafos y quita patrones de ruido. |
| Persistencia | La primera ingesta indexa todo con `cosine` y guarda modelo y huella. La segunda **no** reindexa (no se vectoriza nada). Si cambia un documento, reindexa y conserva `cosine`. Un documento borrado no deja chunks viejos. Un modelo distinto lanza `EmbeddingMismatchError`. Se avisa cuando un chunk supera el límite del modelo. |
| RAG | Respuesta con referencias válidas (fuente, `chunk_id` y score reales). El prompt numera los fragmentos y **no incluye nombres de archivo**. Un número de fragmento inexistente se descarta con WARNING y los duplicados se ignoran. `"No lo sé."` sale sin referencias. Sin fragmentos relevantes **no se llama al LLM**. Solo llegan al prompt los fragmentos que superan el umbral. Un JSON inválido se reintenta. |
| Schemas y prompt | Variantes como `"no lo sé"` se normalizan a `"No lo sé."`. La respuesta vacía se rechaza. El prompt solo espera `contexto` y `pregunta`. |
| Factory | OpenAI con `temperature=0` y `max_retries=0`. Anthropic sin `temperature`. Proveedor desconocido rechazado. |

## Estructura

```
config.py              .env, rutas, TOP_K/MIN_RELEVANCE, get_embeddings(), Factory get_llm()
document_processor.py  DocumentProcessor: clean_text, fragmentación por tokens, IDs archivo::n
ingest.py              ingest(): huella, reindexado condicional, EmbeddingMismatchError
schemas.py             RespuestaLLM (salida del LLM), Referencia y RAGResponse (salida final)
rag.py                 Prompt, cadena LCEL, RAGPipeline, get_rag_response() y main con las pruebas
data/*.md              Apuntes de las clases 1 a 4 (el dataset)
tests/test_rag.py
```

## Cómo funciona

```
get_rag_response(query)
 ├─ a) asimilarity_search_with_relevance_scores(query, k=4)   ← embedding con "query: "
 │     └─ descarta score < MIN_RELEVANCE ── si no queda nada → "No lo sé." (sin LLM)
 ├─ b) prompt con los fragmentos numerados [1]..[n], sin nombres de archivo
 ├─ c) await (PROMPT | llm | PydanticOutputParser).with_retry(...).ainvoke(...)
 └─ d) RespuestaLLM{respuesta, fragmentos:[1,3]} → RAGResponse{respuesta, referencias}
        (números → fuente y chunk_id desde los metadatos; inexistentes → WARNING y se descartan)
```

## Decisiones de diseño

**Embeddings: `multilingual-e5-small` con prefijos.** Es multilingüe (los apuntes están en español), corre local sin key y acepta hasta 512 tokens. e5 fue entrenado con prefijos: `"passage: "` al indexar (`encode_kwargs`) y `"query: "` al consultar (`query_encode_kwargs`), con `normalize_embeddings=True` para que la similitud coseno sea correcta. Hay un solo `get_embeddings()` en `config.py` para indexar y consultar, así que no se puede mezclar modelos por accidente. Como alternativa, `EMBEDDINGS_PROVIDER=openai` usa `text-embedding-3-small`.

**Chunks de 500 tokens, con techo garantizado.** El splitter recibe `length_function` en tokens de `cl100k_base`. Pero arma cada chunk sumando los tokens de sus pedazos por separado, y los tokens no se suman así: se encontró un caso de 501 tokens. `DocumentProcessor.split_text` mide cada chunk terminado y, si alguno se pasa, vuelve a partir con un límite interno menor. Hay un test que lo fija.

**Truncamiento de e5, visible.** Los 500 tokens se miden con tiktoken, pero e5 usa el tokenizer de XLM-R, que parte el español en más pedazos. Un chunk de estos apuntes (`clase2_langchain_lcel.md::1`, 451 tokens de tiktoken) tiene 524 tokens para e5, que trunca a 512. No se pierde información, porque esos tokens finales se repiten al inicio del chunk siguiente por el overlap. Igual, la ingesta lo loguea con un WARNING en vez de truncar en silencio.

**Persistencia con huella.** La colección guarda en sus metadatos el modelo de embeddings y un sha256 del texto e ID de cada chunk, más los parámetros de chunking. Al arrancar, `ingest()` decide así:

| Situación | Qué hace |
|---|---|
| Modelo guardado ≠ modelo configurado | `EmbeddingMismatchError` con instrucciones (consultar con otro modelo da resultados aleatorios) |
| Misma huella y misma cantidad de chunks | No reindexa |
| Huella distinta (cambió, se agregó o se borró un documento, o el chunking) | Borra la colección y la recrea |

Para reindexar se usa `delete_collection` y **no** `reset_collection`, porque este último recrea la colección sin metadatos y la métrica vuelve de `cosine` a `l2` (se verificó en chromadb 1.5.9 y hay un test que lo fija). Los IDs son determinísticos (`archivo::n`) y la métrica es `"hnsw:space": "cosine"`.

**Umbral de relevancia.** e5 da scores altos incluso a textos no relacionados: la pregunta de empanadas llega a 0.788 contra estos apuntes, y las del tema dan entre 0.82 y 0.87. Sin umbral, cualquier pregunta traería 4 fragmentos. Con `MIN_RELEVANCE=0.80`, si nada lo supera, se responde `"No lo sé."` **sin llamar al LLM**: no cuesta nada y no hay posibilidad de alucinar.

**Referencias verificables.** El LLM nunca ve nombres de archivo: recibe los fragmentos numerados `[1]..[n]` y devuelve los números que usó. El código los traduce a `fuente`, `chunk_id` y `score` reales desde los metadatos de ChromaDB. Un número que no existe (el LLM dice `[7]` y hay 4) se descarta con WARNING. Si la respuesta es `"No lo sé."`, las referencias se vacían aunque el LLM haya listado alguna.

**Dos esquemas separados.** `RespuestaLLM` es lo que genera el modelo, y sus `description` viajan en las instrucciones del `PydanticOutputParser`. `RAGResponse` es lo que recibe el usuario, con referencias armadas por el código. Un validador normaliza `"no lo sé"`, `"No lo sé"` y variantes al texto exacto `"No lo sé."`.

**Reintentos filtrados.** La cadena entera va en `.with_retry()`, pero solo para `OutputParserException` (JSON mal formado o que no valida) y para los errores transitorios de `langchain_core.exceptions` (`ModelRateLimitError`, `ModelConnectionError`, `ModelTimeoutError`, `ModelAPIError`). Una key inválida o un request mal formado fallan enseguida. Los modelos van con `max_retries=0`, para que los reintentos del SDK no se multipliquen con los nuestros.

**Factory de LLM** (mismo patrón que la Pre-Entrega 2). `get_llm()` elige el constructor según `LLM_PROVIDER`. OpenAI va con `temperature=0`. Anthropic va sin `temperature`, porque los Claude recientes la rechazan con un 400. El LLM se crea recién cuando hace falta, así `--solo-recuperacion` y la rama "sin fragmentos relevantes" funcionan sin key.

**Asincronía de punta a punta.** La búsqueda (`asimilarity_search_with_relevance_scores`) y el LLM (`ainvoke`) son asíncronos. La ingesta es síncrona y pesada (calcula embeddings), así que se corre con `asyncio.to_thread` para no bloquear el event loop. El pipeline por defecto se crea una sola vez, con un `asyncio.Lock` por si llegan dos preguntas juntas al arrancar.

## Errores de la pista oficial que este proyecto evita

- **k=4 con documentos de un solo chunk.** Los documentos de la pista son cortos, así que cada uno es un solo chunk y con k=4 siempre se recupera todo. Acá cada apunte tiene más de 2400 tokens y se parte en 6 o 7 chunks (27 en total), y además hay un umbral de relevancia.
- **`all-MiniLM-L6-v2`.** Está pensado casi solo para inglés y tiene un límite de 256 tokens: con chunks de 500 trunca en silencio. Acá se usa e5 multilingüe (512 tokens) y se avisa si algún chunk igual se pasa.
- **Persistencia que solo mira si la carpeta existe.** Si cambia un documento, la pista no reindexa. Acá se compara una huella del contenido.
- **Sin verificación del modelo de embeddings.** Acá el modelo se guarda en la colección y un mismatch lanza `EmbeddingMismatchError`.
- **Sin filtro de relevancia.** Acá se aplica `MIN_RELEVANCE` y, si nada lo supera, no se llama al LLM.
- **Fuentes = todos los archivos recuperados.** Acá las referencias son solo los fragmentos que el LLM dijo usar, validados contra lo recuperado.

## Limitaciones

- **El margen del umbral es chico.** La pregunta fuera de tema quedó en 0.788, contra el umbral de 0.80. Con otros documentos o preguntas puede hacer falta recalibrarlo mirando `python rag.py --solo-recuperacion`. Con `EMBEDDINGS_PROVIDER=openai` la escala de scores es otra y 0.80 no sirve: hay que recalibrar.
- **Cambiar de modelo de embeddings obliga a reindexar.** Hay que borrar `./vectorstore`; el error lo indica.

## Sobre los documentos de `data/`

Los archivos de `data/` son apuntes propios del curso, redactados a partir del material de clase. Se incluyen solo como dataset de ejemplo para esta entrega.
