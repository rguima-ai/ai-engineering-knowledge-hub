# Pipeline de Extracción de Entidades Técnicas

Pre-Entrega 2 de AI Engineering (Coderhouse). Un pipeline con **LangChain (LCEL)** que recibe un texto técnico sin procesar (un log de error, un incidente o una descripción de arquitectura) y devuelve un objeto **validado con Pydantic**, con OpenAI o Anthropic de forma intercambiable.

```python
resultado = await process_text("Nuestra API en FastAPI tira timeouts porque Redis se satura...")
resultado.tecnologias          # ['FastAPI', 'Redis']
resultado.nivel_de_criticidad  # NivelCriticidad.ALTA
resultado.resumen_tecnico      # 'La API ...'
```

## Cómo correrlo

Requiere **Python 3.12**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # y completá la key del proveedor que vayas a usar
python prueba.py
```

`prueba.py` procesa dos textos con el proveedor de `LLM_PROVIDER`: un log técnico claro (FastAPI, Redis, PostgreSQL, SQLAlchemy) y un texto ambiguo sin ningún detalle técnico, como prueba de estrés. Si un texto falla, lo informa y sigue con el siguiente.

## Variables de entorno

| Variable | Obligatoria | Default | Para qué |
|---|---|---|---|
| `LLM_PROVIDER` | no | `openai` | Proveedor que elige el Factory: `openai` o `anthropic` |
| `OPENAI_API_KEY` | si usás OpenAI | | Key de OpenAI |
| `ANTHROPIC_API_KEY` | si usás Anthropic | | Key de Anthropic |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | Modelo de OpenAI |
| `ANTHROPIC_MODEL` | no | `claude-haiku-4-5-20251001` | Modelo de Anthropic |
| `LLM_MAX_TOKENS` | no | `1024` | Tope de tokens de la respuesta |
| `LLM_TIMEOUT_S` | no | `30` | Timeout por llamada, en segundos |
| `LOG_LEVEL` | no | `INFO` | Nivel de logging de `prueba.py` |

## Ejemplo de salida (ejecución real)

Salida de `python prueba.py` para el texto claro, corrida el 2026-10-01 con los dos proveedores. El código es el mismo; lo único que cambia es `LLM_PROVIDER`.

**OpenAI** (`gpt-4o-mini`, `LLM_PROVIDER=openai`):

```json
{
  "tecnologias": [
    "FastAPI",
    "Redis",
    "PostgreSQL",
    "SQLAlchemy"
  ],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "La API en FastAPI está experimentando timeouts intermitentes debido a la saturación del caché en Redis y a la agotamiento de conexiones a PostgreSQL por un mal dimensionamiento del pool de SQLAlchemy. Esto está impactando a los usuarios en producción."
}
```

**Anthropic** (`claude-haiku-4-5-20251001`, `LLM_PROVIDER=anthropic`):

```json
{
  "tecnologias": [
    "FastAPI",
    "Redis",
    "PostgreSQL",
    "SQLAlchemy"
  ],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "La API en FastAPI presenta timeouts intermitentes en producción debido a saturación del caché Redis en picos de tráfico y agotamiento del pool de conexiones PostgreSQL por subdimensionamiento en SQLAlchemy."
}
```

Los dos validaron al primer intento, con las mismas 4 tecnologías y la misma criticidad. Solo cambia la redacción del resumen (los errores de redacción son del modelo y se dejaron tal cual). Latencias de esa corrida: 1,7 s (OpenAI) y 1,6 s (Anthropic).

Los logs de una ejecución con un rate limit y un JSON inválido antes del éxito (tomados de los tests):

```
INFO     pipeline inicio caracteres=55
WARNING  pipeline llamada_al_modelo=fallida error=langchain_openai.chat_models.base.OpenAIRateLimitError: Rate limit reached
WARNING  pipeline validacion=rechazada motivo=no pasó el parseo/validación: Invalid json output
INFO     pipeline validacion=ok tecnologias=2 criticidad=alta
INFO     pipeline ok latencia_ms=7 resultado={"tecnologias":["FastAPI","Redis"],...}
```

## Qué pasó con el texto ambiguo

Texto: *"El sistema anda medio raro últimamente, no sé bien qué está pasando."*

**Falló de forma explícita tras 3 intentos, con los dos proveedores. Ningún modelo inventó tecnologías.** En los 3 intentos, tanto `gpt-4o-mini` como `claude-haiku-4-5` siguieron la instrucción del prompt y devolvieron `tecnologias: []`. El validador rechazó la lista vacía, `with_retry` volvió a llamar al modelo con backoff exponencial (en los timestamps se ve que el hueco entre intentos crece) y, agotados los intentos, `process_text` logueó el fallo definitivo y relanzó `SalidaInvalidaError`. `prueba.py` lo atrapó y siguió.

Logs reales (OpenAI):

```
14:56:35,776 INFO    pipeline inicio caracteres=68
14:56:36,855 WARNING pipeline validacion=rechazada motivo=no pasó el parseo/validación: tecnologias: Value error, La lista de tecnologías no puede estar vacía
14:56:39,396 WARNING pipeline validacion=rechazada motivo=no pasó el parseo/validación: tecnologias: Value error, La lista de tecnologías no puede estar vacía
14:56:43,281 WARNING pipeline validacion=rechazada motivo=no pasó el parseo/validación: tecnologias: Value error, La lista de tecnologías no puede estar vacía
14:56:43,282 ERROR   pipeline fallo_definitivo tipo=reintentos_agotados error=SalidaInvalidaError: no pasó el parseo/validación: tecnologias: Value error, La lista de tecnologías no puede estar vacía
```

Logs reales (Anthropic):

```
14:56:46,043 INFO    pipeline inicio caracteres=68
14:56:47,420 WARNING pipeline validacion=rechazada motivo=no pasó el parseo/validación: tecnologias: Value error, La lista de tecnologías no puede estar vacía
14:56:50,526 WARNING pipeline validacion=rechazada motivo=no pasó el parseo/validación: tecnologias: Value error, La lista de tecnologías no puede estar vacía
14:56:54,676 WARNING pipeline validacion=rechazada motivo=no pasó el parseo/validación: tecnologias: Value error, La lista de tecnologías no puede estar vacía
14:56:54,676 ERROR   pipeline fallo_definitivo tipo=reintentos_agotados error=SalidaInvalidaError: no pasó el parseo/validación: tecnologias: Value error, La lista de tecnologías no puede estar vacía
```

### Por qué fallar es la respuesta correcta

La slide de la consigna pide un pipeline que devuelva *"siempre un objeto validado, no importa cuán ambiguo sea el input"*. Leído al pie de la letra, eso choca con el contrato: si el texto no menciona ninguna tecnología, la única forma de producir un `EntidadesTecnicas` válido es que el modelo **invente** una para cumplir el requisito de lista no vacía. Sería una alucinación inducida por el propio contrato.

El objetivo de fondo de la slide es otro: que **nunca llegue al resto del sistema un objeto inválido o inventado**. Un fallo explícito y controlado también lo cumple:

- El código que llama a `process_text` recibe o un `EntidadesTecnicas` que cumple el contrato, o una excepción tipada (`SalidaInvalidaError`). Nunca un objeto a medio llenar, `None` ni datos inventados.
- El fallo no rompe el programa: queda logueado con el motivo y el que llama decide qué hacer (pedir más contexto, mandar el texto a revisión humana, etc.).
- Antes de rendirse se reintenta 3 veces, así que un fallo ocasional del modelo se recupera. El que no se recupera es el que no tiene solución: el texto no trae la información.

Las alternativas se descartaron a propósito. Un valor centinela como `["no especificada"]` pasaría la validación con un dato que no es una tecnología. Un fallback con un objeto por defecto (`with_fallbacks`) devolvería algo "válido" que el resto del sistema no podría distinguir de una extracción real. La guía de la clase llega a la misma conclusión: ante este texto, que el pipeline lo informe es el comportamiento correcto; que el modelo invente es el incorrecto.

## Tests

```bash
pytest
```

Los tests (32) **no usan internet ni API keys**. La mayoría reemplaza el LLM por un `FakeModel` cuyo `with_structured_output()` devuelve un `RunnableLambda` que sigue un guion de respuestas. Los de integración usan `ChatOpenAI` y `ChatAnthropic` reales y solo reemplazan `_agenerate`, el punto donde salen a la API: así se prueba el parseo de verdad de LangChain, no una imitación. Los errores de API son los mismos que lanza `ChatOpenAI` en producción (`OpenAIRateLimitError`, `OpenAIAuthenticationError` de `langchain_openai`), y el backoff se pone en 0 para que los tests no tarden.

| Grupo | Qué prueba |
|---|---|
| Schema | Enum inválido (`"high"`) rechazado, `" Alta "` normalizado a `alta`, lista vacía (o solo espacios) rechazada, duplicados eliminados sin distinguir mayúsculas y conservando la primera aparición, resumen vacío rechazado, restricciones fuera del JSON schema. |
| Prompt | Tiene roles `system` y `human`, su única variable de entrada es `texto` y las instrucciones de formato quedaron fijadas. |
| Factory | OpenAI con `temperature=0` y `max_retries=0`; Anthropic sin `temperature` y con `max_retries=0`; proveedor y modelo leídos del entorno; proveedor desconocido rechazado. |
| Resiliencia | Camino feliz (el `{texto}` llega al modelo). Respuesta cortada (`finish_reason="length"` de OpenAI y `stop_reason="max_tokens"` de Anthropic) se reintenta y se recupera. JSON inválido y salida que no valida se reintentan. Se rinde tras 3 intentos. Rate limit se reintenta. Key inválida **no** se reintenta: una sola llamada. |
| Integración (OpenAI y Anthropic) | El pedido viaja como herramienta (`function_calling`), no como `response_format`. Con el parseo real de LangChain, una lista vacía se reintenta y se recupera, una respuesta cortada se reintenta, y un texto sin tecnologías falla tras 3 intentos con `SalidaInvalidaError`. |
| Logging | Se loguean el inicio, cada validación, los reintentos, el resultado y el fallo definitivo. |

## Estructura

```
schemas.py            NivelCriticidad (Enum) y EntidadesTecnicas (Pydantic + field_validator)
chain.py              Prompt, get_model (Factory), verificar_salida, build_chain y process_text
prueba.py             Demo asíncrona: texto claro + texto ambiguo
tests/test_pipeline.py
```

## Cómo funciona la cadena

```
PROMPT | model.with_structured_output(EntidadesTecnicas, method="function_calling", include_raw=True) | verificar_salida
└──────────────────────────────────────── .with_retry(3 intentos) ────────────────────────────────────────┘
```

1. **Prompt.** `ChatPromptTemplate` con rol `system` (el analista y las instrucciones de formato, fijadas con `.partial()`) y rol `human` (el `{texto}`). Sin f-strings: LangChain completa las variables al invocar.
2. **Salida estructurada.** `with_structured_output` le pasa el schema al modelo como herramienta. Con `include_raw=True` no lanza si el parseo falla: devuelve `{raw, parsed, parsing_error}`, y eso nos permite inspeccionar la respuesta cruda.
3. **Verificación.** `verificar_salida` lanza `SalidaInvalidaError` si la respuesta vino cortada por límite de tokens, si hubo `parsing_error` (JSON mal formado o un `field_validator` que no se cumple) o si `parsed` es `None`.
4. **Reintentos.** La cadena **entera** va envuelta en `.with_retry()`: si la verificación rechaza la salida, se vuelve a llamar al modelo.

## Decisiones de diseño

**Qué se reintenta y qué no.** `with_retry()` reintenta *cualquier* excepción por defecto, incluido un 401. Acá se le pasa `retry_if_exception_type` explícito:

| Tipo | Clases | Qué hace |
|---|---|---|
| Salida inválida | `SalidaInvalidaError` (propia) | Reintenta |
| Transitorio | `ModelRateLimitError`, `ModelConnectionError`, `ModelTimeoutError`, `ModelAPIError` (5xx, 529) | Reintenta con backoff exponencial y jitter (~1 s, 2 s, 4 s…) |
| Permanente | `ModelAuthenticationError` (401), `ModelInvalidRequestError` (400), `ModelNotFoundError` (404)… | Falla enseguida |

Las clases son de `langchain_core.exceptions` y sirven para los dos proveedores: `langchain_openai` y `langchain_anthropic` envuelven los errores de sus SDKs en subclases que heredan a la vez del error del SDK y del de `langchain_core` (se verificó la jerarquía en langchain-openai 1.6.6 y langchain-anthropic 1.7.4, y hay un test que la fija).

**`method="function_calling"` explícito.** `ChatOpenAI` usa `json_schema` por defecto. En ese modo, el SDK de OpenAI valida con Pydantic **dentro de la llamada al modelo**: un `ValidationError`, o un `LengthFinishReasonError` si la respuesta se corta, sale crudo, sin pasar por `include_raw` ni por `verificar_salida`, y como no está en la lista de reintentables, falla al primer intento. Lo descubrimos en la primera ejecución real: el texto ambiguo cortó con un `ValidationError` sin un solo reintento. Con `function_calling`, el parseo lo hace LangChain después de la llamada, el error llega como `parsing_error`, `finish_reason` queda visible para la verificación, y es el mismo método que usa Anthropic por defecto. Los tests de integración fijan este comportamiento.

**`max_retries=0` en los modelos.** `ChatOpenAI` y `ChatAnthropic` reintentan por su cuenta a través de sus SDKs. Si además reintenta `with_retry`, los intentos se multiplican (3 × 3 = 9) sin que se note. Así, la única política de reintentos es la nuestra.

**Restricciones en `field_validator`, no en el JSON schema.** Los `Field` solo tienen `description`, que es lo que ve el LLM. Las reglas (lista no vacía, sin duplicados, resumen con contenido) se aplican al parsear, y si no se cumplen la salida se reintenta.

**Factory con registro.** `get_model()` elige el constructor según `LLM_PROVIDER` en un diccionario `proveedor → constructor`. Sumar un proveedor es agregar una línea. Los modelos salen del `.env`, así un modelo retirado se cambia sin tocar código.

**Logging sin estado compartido.** Cada verificación loguea `WARNING` si rechaza e `INFO` si pasa. Los errores de la API (429, timeouts) no pasan por la verificación, así que se loguean con un listener `on_error` sobre la llamada al modelo. No hay contadores globales: si se procesan varios textos a la vez, los logs no se mezclan.

## Errores de la pista oficial que este proyecto evita

- **`with_retry()` sin filtro** reintenta también una key inválida. Acá solo se reintentan salidas inválidas y errores transitorios.
- **Respuesta cortada no detectada.** La pista confía en que el retry la atrape, pero un JSON truncado puede parsear igual. Acá se revisa `finish_reason` / `stop_reason`.
- **`temperature` en Anthropic.** LangChain la manda a la API si se la pasás, y los Claude recientes la rechazan con un 400. OpenAI sí usa `temperature=0`.
- **Modelos hardcodeados** (`claude-sonnet-4-6` fijo en el código). Acá salen del `.env`, con `gpt-4o-mini` y `claude-haiku-4-5-20251001` por defecto.
- **`min_length` en `Field`**, que termina en el JSON schema. Acá las restricciones van en `field_validator`.
- **Sin instrucciones de formato en el prompt.** Acá se fijan con `.partial()`.
- **`with_structured_output` con el método por defecto.** Con `ChatOpenAI` es `json_schema`, y una salida que no valida o viene cortada se escapa de `include_raw` y no se reintenta (ver *Decisiones de diseño*). El pipeline de referencia de la guía de clase tiene el mismo problema.
- **Gemini** se sacó: la consigna pide OpenAI y Anthropic.
