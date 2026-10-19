# RAG escalable con Pinecone: recuperación híbrida BM25 + vectorial

Pre-Entrega 4 del curso AI Engineering (Coderhouse). Un módulo de recuperación que:

1. **Crea o verifica** un índice Serverless en Pinecone (aws/us-east-1, coseno), con la dimensión calculada desde el modelo de embeddings.
2. **Ingesta** 8 documentos: los fragmenta con `RecursiveCharacterTextSplitter` (450 tokens, 50 de overlap), genera embeddings locales con `intfloat/multilingual-e5-small` y los sube por lotes a un namespace, con metadatos validados por Pydantic e IDs determinísticos.
3. **Recupera** con `RAGSystem`: un `EnsembleRetriever` que fusiona BM25 (palabras exactas) y Pinecone (significado) con Reciprocal Rank Fusion y devuelve el top-5 sin duplicados.
4. **Evalúa** con `evaluate.py`: Precision@5 y Recall@5 sobre un golden set de 10 preguntas, comparando BM25, vectorial y el híbrido con tres pesos.

## Cómo replicar el índice desde cero

**1. Cuenta de Pinecone.** Creá una cuenta gratuita en [app.pinecone.io](https://app.pinecone.io) y copiá la API key desde *API Keys*. El plan Starter alcanza: el índice ocupa 25 vectores de 384 dimensiones.

**2. Entorno** (Python 3.12):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. `.env`.** Copiá la plantilla y completá la key:

```bash
cp .env.example .env
# editá .env: PINECONE_API_KEY=... (INDEX_NAME por defecto: rag-stdlib-python)
```

**4. Setup del índice.** Lo crea si no existe y, si existe, verifica dimensión y métrica:

```bash
python setup_index.py
# INFO setup_index creando_indice name=rag-stdlib-python dimension=384 metric=cosine cloud=aws region=us-east-1
# Índice 'rag-stdlib-python' listo: dimension=384 metric=cosine vectores=0
```

La primera vez descarga el modelo e5 (unos 470 MB) desde Hugging Face.

**5. Ingesta:**

```bash
python ingest.py
```

**6. Evaluación:**

```bash
python evaluate.py
```

**7. Tests** (offline, sin key ni internet):

```bash
pytest
```

## Variables de entorno

| Variable | Default | Para qué |
|---|---|---|
| `PINECONE_API_KEY` | (obligatoria) | Autenticación con Pinecone |
| `INDEX_NAME` | `rag-stdlib-python` | Nombre del índice |
| `PINECONE_NAMESPACE` | `stdlib-dev` | Partición lógica dentro del índice |
| `PINECONE_CLOUD` / `PINECONE_REGION` | `aws` / `us-east-1` | Ubicación del índice Serverless |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | o `text-embedding-3-small` (necesita `OPENAI_API_KEY` y otro `INDEX_NAME`, porque son 1536 dimensiones) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `450` / `50` | Tamaño de los chunks en tokens de tiktoken |
| `BATCH_SIZE` | `100` | Vectores por upsert |
| `TOP_K` | `5` | Resultados por consulta |
| `LOG_LEVEL` | `INFO` | Nivel de logging |

## Ingesta: re-ingestar no duplica

Salida real de las dos ingestas seguidas contra Pinecone (sin las líneas de carga del modelo):

```text
===== INGESTA 1
INFO setup_index indice_existente name=rag-stdlib-python dimension=384 metric=cosine ok
INFO document_processor chunking source=archivos_json.md tokens=1160 chunks=3
INFO document_processor chunking source=archivos_pathlib.md tokens=1168 chunks=3
INFO document_processor chunking source=concurrencia_asyncio.md tokens=1179 chunks=3
INFO document_processor chunking source=diagnostico_logging.md tokens=1051 chunks=3
INFO document_processor chunking source=estructuras_collections.md tokens=1144 chunks=3
INFO document_processor chunking source=estructuras_dataclasses.md tokens=1096 chunks=3
INFO document_processor chunking source=funcional_functools.md tokens=1096 chunks=3
INFO document_processor chunking source=funcional_itertools.md tokens=1126 chunks=4
INFO ingest lote=1 vectores=25 namespace=stdlib-dev
INFO ingest vector_count=25 namespace=stdlib-dev
INFO ingest ingesta_completa chunks=25 archivos=8 namespace=stdlib-dev

===== INGESTA 2
INFO setup_index indice_existente name=rag-stdlib-python dimension=384 metric=cosine ok
INFO ingest lote=1 vectores=25 namespace=stdlib-dev
INFO ingest vector_count=25 namespace=stdlib-dev
INFO ingest ingesta_completa chunks=25 archivos=8 namespace=stdlib-dev

# consulta aparte, después de la segunda ingesta:
total_vector_count = 25 | namespaces = {'stdlib-dev': 25}
```

La segunda ingesta vuelve a subir 25 vectores y el índice sigue teniendo 25. Cada chunk tiene un ID fijo, `archivo.md::n`, y un upsert con un ID existente lo sobrescribe en lugar de agregar uno nuevo. La pista no pasaba IDs, así que Pinecone generaba IDs al azar y cada corrida duplicaba los vectores.

Con 25 chunks y lotes de 100 alcanza un solo lote. Los tests prueban el corte con `batch_size=10`, que da lotes de 10, 10 y 5.

## Métricas

Resultado real de `python evaluate.py` contra Pinecone (10 preguntas, k=5):

| Configuración | Precision@5 | P@5 / techo | Recall@5 | Hit@1 | P@5 exactas | P@5 parafraseadas |
|---|---|---|---|---|---|---|
| BM25 | 0.46 | 0.74 | 1.00 | 0.90 | 0.44 | 0.48 |
| Vectorial | 0.52 | 0.84 | 1.00 | 0.90 | 0.52 | 0.52 |
| Híbrido 0.3/0.7 | 0.52 | 0.84 | 1.00 | 0.90 | 0.52 | 0.52 |
| **Híbrido 0.5/0.5** | **0.54** | **0.87** | 1.00 | **1.00** | 0.52 | **0.56** |
| Híbrido 0.7/0.3 | 0.46 | 0.74 | 1.00 | 1.00 | 0.44 | 0.48 |

Cómo leer cada columna:
- Los pesos del híbrido se escriben BM25/vectorial.
- **Precision@5** = chunks del documento esperado entre los 5 / 5.
- **Recall@5** = 1 si el documento esperado aparece entre los 5.
- **Techo**: la Precision@5 máxima posible para cada pregunta. Cada documento tiene 3 chunks (4 en itertools), así que ni un sistema perfecto supera 3/5 = 0,60. El techo medio del golden set es 0,62, y "P@5 / techo" muestra qué parte de ese máximo se alcanzó.
- **Hit@1** = 1 si el primer resultado es del documento esperado. No lo pide la consigna; lo agregué porque el Recall@5 no discrimina (ver más abajo).

Detalle del híbrido 0.5/0.5, la configuración ganadora. Cada lista es el top-5 en orden, como `módulo#chunk`:

```text
P=0.40/0.60 exacta  ¿Para qué sirve lru_cache?                      [functools#0, itertools#3, functools#1, logging#2, pathlib#2]
P=0.60/0.60 exacta  ¿Qué hace asyncio.gather?                       [asyncio#1, asyncio#2, asyncio#0, itertools#3, json#1]
P=0.60/0.60 exacta  ¿Cómo uso Path.glob para buscar archivos?       [pathlib#1, pathlib#2, pathlib#0, json#0, functools#1]
P=0.60/0.60 exacta  ¿Qué diferencia hay entre json.dumps y json.dump? [json#0, json#1, json#2, collections#2, dataclasses#2]
P=0.40/0.60 exacta  ¿Para qué sirve Counter?                        [collections#0, logging#2, asyncio#2, json#1, collections#1]
P=0.60/0.60 paraf.  ¿Cómo evito recalcular una función cara...?     [functools#0, functools#1, dataclasses#2, functools#2, dataclasses#1]
P=0.60/0.60 paraf.  ¿Cómo lanzo varias descargas de red...?         [asyncio#0, asyncio#1, asyncio#2, logging#2, collections#1]
P=0.40/0.60 paraf.  ¿Cómo dejo constancia de lo que hace...?        [logging#0, itertools#1, logging#2, itertools#2, functools#1]
P=0.60/0.60 paraf.  ¿Cómo defino una clase que solo agrupa...?      [dataclasses#0, collections#2, dataclasses#1, dataclasses#2, itertools#1]
P=0.60/0.80 paraf.  ¿Cómo recorro todas las parejas posibles...?    [itertools#2, collections#0, itertools#1, itertools#0, functools#2]
```

`python evaluate.py` imprime el mismo detalle para las cinco configuraciones.

## Análisis

**Ganó el híbrido 0.5/0.5, pero por poco.** Le saca 0,02 de Precision@5 al vectorial, que equivale a **un solo chunk útil más** en 50 posiciones (pregunta de itertools). Con 10 preguntas esa diferencia no es concluyente. Lo que sí es claro es el **Hit@1**: el 0.5/0.5 es la única configuración que pone primero un chunk del documento correcto en las 10 preguntas. BM25 y el vectorial fallan una cada uno, y no es la misma pregunta. Ese es el argumento real a favor del híbrido: **los dos métodos se equivocan en preguntas distintas**, y la fusión corrige a cada uno con el otro.

**Dónde falló cada configuración y por qué:**

| Pregunta | Qué pasó | Por qué |
|---|---|---|
| `¿Para qué sirve lru_cache?` | BM25: P=0,20 | `lru_cache` aparece literal en **un solo chunk** (functools#0). Después del preprocesamiento la pregunta queda en un solo token, así que los otros 4 lugares los ocupan chunks con score 0, en orden arbitrario (itertools#3, logging#2, json…). El vectorial sí encuentra functools#1, que habla de `@cache` sin nombrar `lru_cache`. |
| `¿Para qué sirve Counter?` | BM25: P=0,20 | El mismo efecto: `Counter` solo está en collections#0. **En preguntas de un solo término, la precisión de BM25 tiene como techo la cantidad de chunks que contienen ese término**, no la cantidad de chunks del documento. |
| `¿Cómo defino una clase que solo agrupa valores…?` | BM25: Hit@1 falla (collections#2 primero) | Trampa léxica. collections#2 (namedtuple y ChainMap) contiene "clase", "agrupa", "valores", "solo" y "mano", que son 5 de los 8 tokens de la pregunta, uno más que dataclasses#0. BM25 no sabe que "crea una clase de tuplas" no responde la pregunta. El vectorial pone dataclasses#0 primero. |
| `¿Cómo dejo constancia… archivo que se corte cuando crece mucho?` | Vectorial: Hit@1 falla (itertools#1 primero) | La paráfrasis evita "log" y "handler", y e5 se queda con "archivo" y "corte". itertools#1 empieza con "algo que los **corte**" y tiene `islice(archivo, 10)`. Acá BM25 sí acierta: logging#0 tiene "gravedad" y "programa", términos raros en el resto del corpus. El 0.5/0.5 pone logging#0 primero. |
| `¿Cómo recorro todas las parejas posibles… dos listas…?` | Todas: P=0,40, salvo 0.5/0.5 con 0,60 (techo 0,80) | Los tokens útiles son comunes. "dos", "elementos" y "listas" aparecen mucho en collections#0 ("dos extremos", "las listas…"), así que BM25 la pone segunda. `product` está en itertools#2, que todas las configuraciones ponen primero, pero el resto del documento habla de otras cosas (cortar, agrupar). Solo la fusión 0.5/0.5 junta 3 de los 4 chunks de itertools. |

**Por qué el híbrido no le gana más claramente a cada rama sola, con este dataset:**
- **Los documentos son muy distintos entre sí y están en español.** e5 multilingüe resuelve casi todas las paráfrasis, y BM25 resuelve casi todos los nombres de funciones. Quedan pocos casos donde uno falla y el otro no, que es justo donde el híbrido gana algo.
- **Con pesos desbalanceados, el híbrido copia a la rama dominante.** RRF suma `peso / (60 + posición)`. Con c=60 las posiciones 1 a 5 valen casi lo mismo (1/61 contra 1/65), así que pesa mucho más el peso que la posición, y con 0.7/0.3 la rama de 0,7 decide casi todo el orden. Por eso el 0.3/0.7 tiene exactamente las métricas del vectorial y el 0.7/0.3 la Precision@5 de BM25. El 0.7/0.3 conserva igual el Hit@1 = 1,00. En la pregunta de logging manda BM25, que acierta. En la de dataclasses, BM25 pone collections#2 primero y dataclasses#0 segundo, y el vectorial pone dataclasses#0 primero y collections#2 quinto. Los puntajes quedan en 0,7/62 + 0,3/61 = 0,01621 contra 0,7/61 + 0,3/65 = 0,01609: el 0,3 del vectorial alcanza justo para dar vuelta un empate tan ajustado.
- **El techo de la métrica comprime las diferencias.** Con 3 chunks por documento, la Precision@5 de todas las configuraciones queda entre 0,46 y 0,54 sobre un máximo de 0,62. No hay mucho espacio para separar sistemas.

## Límites de las métricas con 8 documentos y top-5

**El Recall@5 no discrimina.** Da 1,00 en las cinco configuraciones. La razón es combinatoria: hay 25 chunks y la respuesta trae 5, o sea el 20% del corpus. Un recuperador **completamente aleatorio** ya tendría un Recall@5 de:

- 1 − C(22,5)/C(25,5) ≈ **0,50** para un documento de 3 chunks;
- ≈ **0,62** para itertools, que tiene 4.

Así que el Recall@5 solo separa "inútil" de "razonable". Entre sistemas razonables satura enseguida, porque alcanza con que **uno** de los 5 resultados sea del documento esperado. Un 1,00 acá no prueba que el sistema sea excelente: solo que no está roto.

**Con la Precision@5 pasa algo parecido, pero desde arriba.** Con un solo documento esperado por pregunta y 3 chunks por documento, el máximo es 0,60. Por eso el reporte muestra el techo y la precisión relativa. Un 0,54 suena bajo, pero es el 87% de lo posible. El azar daría 3/25 ≈ 0,12.

**El Hit@1 es la métrica que más discrimina con este dataset.** El azar daría 0,12, y acá es donde las configuraciones se diferencian (0,90 contra 1,00).

**Para que las métricas digan más** habría que hacer alguna de estas cosas:
- usar un corpus más grande, para que 5 lugares no cubran el 20% del total;
- armar un golden set **a nivel chunk** ("la respuesta está en functools#0"), así la precisión no tiene techo artificial;
- agregar métricas de ranking como MRR o nDCG, que premian poner lo correcto arriba;
- sumar más preguntas, porque con 10 cada pregunta mueve el promedio en 0,02 a 0,06.

## Estructura

```text
config.py              único módulo que lee el .env; embeddings y dimensión
schemas.py             ChunkMetadata (metadata validada) y RetrievedChunk
document_processor.py  limpieza y chunking por tokens (el de la Entrega 3 + process_file)
setup_index.py         get_client, ensure_index, IndexMismatchError
ingest.py              carga, lotes, embeddings, upsert asíncrono, borrado de huérfanos
retriever.py           preprocess, PineconeRetriever, RAGSystem
evaluate.py            Precision@5, Recall@5, Hit@1, techo y reporte en consola
golden_set.json        10 preguntas: 5 con términos exactos, 5 parafraseadas
data/                  8 documentos sobre la biblioteca estándar de Python
tests/                 FakePineconeIndex y 40 tests offline
```

## Decisiones de diseño

- **Dimensión desde el modelo.** `embedding_dimension()` hace `len(embeddings.embed_query("dimension"))`. Para e5-small da 384, pero nunca está escrita en el código: si cambiás a OpenAI (1536), el índice se crea con la dimensión correcta.
- **Verificación del índice existente.** Si dimensión o métrica no coinciden, `ensure_index` lanza `IndexMismatchError` antes de subir nada. Con otra dimensión el upsert fallaría a mitad de camino. Con otra métrica ni siquiera fallaría: daría rankings distintos sin avisar.
- **Chunks de 450 tokens de tiktoken.** e5 trunca en 512 tokens de **su** tokenizer, que parte el español en más pedazos que tiktoken. El chunk más largo mide 435 tokens de tiktoken y **497 de e5**. Con 500 de tiktoken hubiera superado el límite y perdido el final del chunk.
- **e5 con prefijos.** `"passage: "` al indexar, `"query: "` al consultar, y `normalize_embeddings=True` para que el coseno funcione sobre vectores de norma 1.
- **Metadatos con `ChunkMetadata`.** Los campos son `text`, `source`, `chunk_id`, `category`, `tokens` e `ingested_at`, con `extra="forbid"`: un campo mal escrito, como `categoria` en lugar de `category`, es un error y no un campo nuevo silencioso. El texto viaja en la metadata, así una consulta trae vector, texto y fuente juntos.
- **Categoría desde el nombre del archivo** (`funcional_itertools.md` → `funcional`). Es una sola regla y 3 categorías agrupan 2 documentos, así el filtro se puede probar con casos reales.
- **IDs `archivo.md::n` y borrado de huérfanos.** El upsert solo sobrescribe los IDs que se vuelven a subir. Si un documento se achica de 4 chunks a 3, o se borra, la ingesta lista los IDs del namespace y elimina los que ya no existen. Sin eso, los chunks viejos seguirían apareciendo en las búsquedas.
- **Upsert con `asyncio.to_thread`.** El SDK de Pinecone es síncrono. En un hilo aparte no bloquea el event loop mientras viaja el request, y un test verifica que el upsert corra fuera del hilo del loop.
- **Espera a que el conteo se vea.** Pinecone Serverless es eventualmente consistente. La ingesta espera hasta que `describe_index_stats` muestre el conteo esperado, para que un `evaluate.py` corrido enseguida no mida un índice a medias.
- **BM25 con `preprocess` propio.** Pasa a minúsculas, saca tildes, signos y palabras vacías del español. El default (`text.split()`) deja "¿Qué" distinto de "que" y "lru_cache?" distinto de "lru_cache". Además, `asyncio.gather` se parte en `asyncio` y `gather`.
- **Retriever vectorial propio** (`PineconeRetriever`, un `BaseRetriever` sobre el SDK nativo), con namespace y filtro opcional. **El mismo filtro se aplica al corpus de BM25** con `matches_filter`: si solo filtrara Pinecone, la rama léxica traería chunks de otras categorías.
- **`EnsembleRetriever(id_key="id")`.** Fusiona por ID de chunk; por defecto compara el texto completo. El ensemble puede devolver hasta 10 resultados (5 por rama) y `RAGSystem` deduplica y recorta a 5.
- **Pesos por llamada.** `rag.aretrieve(pregunta, mode="hybrid", weights=(0.7, 0.3))` arma el ensemble con esos pesos sin reconstruir BM25. Así `evaluate.py` compara configuraciones sobre el mismo sistema.

## Tests

```bash
pytest     # 40 passed, sin internet ni cuenta de Pinecone
```

Usan `FakePineconeIndex`, un índice en memoria con la interfaz del SDK 10: `upsert` con namespace, `query` con filtros `$eq`/`$in`/`$gt`, `describe_index_stats`, `delete` y `list`. Igual que Pinecone, devuelve los números de la metadata como `float`. Los embeddings son `DeterministicFakeEmbedding`. Cubren:

- `preprocess` normaliza signos, tildes y palabras vacías, y BM25 con `preprocess` encuentra lo que el default pierde;
- la ingesta sube lotes del tamaño correcto, en el namespace correcto y fuera del hilo del event loop;
- re-ingestar no duplica, y los chunks huérfanos se borran;
- `IndexMismatchError` por dimensión y por métrica; creación Serverless en aws/us-east-1 e idempotencia;
- el filtro por categoría (`$eq`, valor directo y `$in`) en BM25, vectorial e híbrido;
- `RAGSystem` devuelve como máximo 5 resultados sin duplicados, aunque el ensemble crudo traiga más;
- Precision@5 y Recall@5 con casos armados a mano (el esperado en el puesto 5, en el 6, lista corta, lista vacía) y el techo de precisión;
- `ChunkMetadata` rechaza campos con otro nombre y valores inválidos.

Un detalle que salió en los tests: con un corpus de 2 documentos, BM25 da score 0 a todo. `rank_bm25` calcula el IDF como `log((N − n + 0.5) / (n + 0.5))`, y con N=2, n=1 eso es `log(1) = 0`. En corpus diminutos BM25 no puede discriminar, por eso el test usa 4 documentos.

## Errores de la pista oficial que este proyecto evita

1. **La ingesta duplica.** `PineconeVectorStore.from_documents` sin IDs genera IDs al azar, y cada corrida suma los mismos chunks otra vez. Acá los IDs son `archivo.md::n`.
2. **La Precision@5 queda fija en 0,25.** Con 4 documentos de un solo chunk y un documento esperado por pregunta, aunque el sistema sea perfecto solo 1 de los 5 resultados puede ser útil. Acá hay 8 documentos y un techo explícito en el reporte.
3. **Divide la precisión por `len(top_k)` y no por k.** Si llegan menos de 5 resultados, la precisión sube artificialmente. Acá siempre se divide por 5.
4. **BM25 sin preprocesamiento.** Con `text.split()`, los signos de pregunta y las tildes rompen las coincidencias.
5. **`all-MiniLM-L6-v2` con chunks de 600 tokens.** El modelo trunca en 256 tokens, así que más de la mitad de cada chunk no se usa. Además es casi solo inglés y los documentos están en español.
6. **No verifica el índice existente.** Solo mira si el nombre existe: con otra dimensión falla recién en el upsert, y con otra métrica no falla nunca.
7. **Dimensión hardcodeada** (`EMBEDDING_DIM = 384`). Si se cambia el modelo y no esa constante, se crea un índice incompatible.

## Notas de versiones

- **pinecone 10.0.** Las respuestas usan atributos (`res.matches`, `m.metadata`, `stats.namespaces[ns].vector_count`) y `index.list()` devuelve páginas con `.vectors[i].id`. La guía de la clase está probada con 7.3, que aceptaba `res["matches"]`.
- **langchain-community 0.4.2** avisa al importarse que está en *sunset*. Se usa igual porque la consigna pide su `BM25Retriever`; el aviso se silencia en `pytest.ini`.

## Limitaciones

- La evaluación mide solo recuperación, no respuestas generadas: el proyecto no incluye LLM.
- 10 preguntas son pocas para afirmar que una configuración es mejor: la diferencia entre el híbrido 0.5/0.5 y el vectorial es un chunk.
- BM25 vive en memoria y se reconstruye desde `data/` en cada arranque. Con millones de documentos habría que usar vectores sparse en Pinecone (búsqueda híbrida nativa) o un motor léxico aparte.
- El borrado de huérfanos lista todos los IDs del namespace, lo que está bien para cientos de vectores pero no para millones.

## Sobre los documentos de `data/`

Los 8 documentos (asyncio, pathlib, json, logging, dataclasses, collections, itertools y functools) se escribieron de cero para este proyecto, en español. No copian la documentación oficial de Python ni usan material del curso.
