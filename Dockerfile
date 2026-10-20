FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

# tiktoken descarga su vocabulario la primera vez que se usa, con `requests` sincrónico: eso
# metía una llamada de red bloqueante en el arranque async y hacía depender el seed de internet
# (falló una vez por DNS). Se descarga al construir la imagen y se lee del caché.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY app ./app
COPY sample_docs ./sample_docs

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
