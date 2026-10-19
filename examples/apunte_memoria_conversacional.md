# Memoria conversacional: el historial que le llega al LLM

Apunte propio sobre un bug real del Knowledge Hub (Entrega Final).

## El síntoma

El checkpointer guardaba la conversación completa, pero ante la repregunta "recordame los
errores que encontraste" el tutor listaba 4 de 5. El estado persistido estaba bien: el
problema era cuánto de ese estado entraba al prompt.

## La causa

La función que arma el historial para el prompt cortaba cada mensaje anterior en 1 500
caracteres. Una respuesta con revisión de código y cuestionario ronda los 4 000, así que el
último error quedaba fuera del prompt aunque estuviera en Redis.

## La lección

Persistir el estado no alcanza: hay que revisar qué parte del estado se le pasa al modelo.
Un recorte por mensaje tiene que ser mayor que la respuesta típica del sistema, y el límite
total se controla con la cantidad de mensajes (ventana de los últimos 6), no truncando cada uno.
Además, una respuesta equivocada que queda en el historial contamina los turnos siguientes:
el modelo tiende a repetir la respuesta más reciente.
