# Lecciones aprendidas en producción (Pre-entrega 7)

Apunte propio con problemas reales que aparecieron al llevar el agente a Docker.

## redis-py 8 y BLPOP

redis-py 8 trae por defecto `socket_timeout=5`. Si el worker hace `BLPOP` con un timeout de
5 segundos, el socket corta antes que Redis y aparece un `TimeoutError` intermitente.
La solución es usar un `socket_timeout` mayor que el timeout del comando bloqueante, o
`None` para la conexión del worker. `fakeredis` no reproduce el problema: solo aparece con
un Redis real.

## Phoenix: costo duplicado

Con `auto_instrument=True` y los instrumentadores de LangChain y de OpenAI instalados a la
vez, cada llamada al LLM se registra dos veces y el costo que muestra Phoenix se duplica.
Conviene instrumentar solo LangChain. Además, la interfaz muestra los costos menores a un
centavo como "<$0.01"; el costo exacto se obtiene con la query GraphQL `costSummary`.

## Señales y trazas perdidas

`docker compose restart` manda SIGTERM. Si el proceso muere sin vaciar el exportador de
OpenTelemetry, el último lote de spans nunca llega a Phoenix. El worker tiene que atrapar la
señal, terminar el job en curso y llamar a `force_flush()` antes de salir.
