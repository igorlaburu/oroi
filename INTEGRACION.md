# Integrar Oroi en tu agente — manual básico

Guía para dar memoria asociativa a un agente conversacional que ya existe. Supone Oroi
instalado (`pip install oroi`, Python ≥ 3.12 con extensiones SQLite) y credenciales
configuradas (`.env` local o `~/.oroi/.env`; Azure OpenAI o un endpoint OpenAI-compatible
como Ollama — ver README).

## La idea en una frase

Tu agente conserva su LLM conversacional y su lógica; Oroi le añade una memoria que **percibe**
cada turno, **recuerda** lo pertinente ante cada mensaje y **consolida** en los descansos. Toda
la integración pasa por una sola clase (`Mind`) y, si quieres el ciclo completo ya orquestado,
por `ChatSession`.

## Regla nº 1: una mente por relación

Cada base de datos es **una** relación usuario↔agente (un fichero SQLite). En multiusuario:
un fichero por usuario (`memoria/<usuario>.db`), jamás usuarios mezclados en una base. La base
fija además su modelo de embeddings el primer día y se niega a mezclarlo — no cambies de
proveedor de embeddings a mitad de vida de una memoria.

```python
from oroi import Mind
from oroi.extraction.extractor import TurnExtractor
from oroi.providers.azure import AzureEmbedder, AzureLLM
from oroi.providers.settings import ProviderSettings

settings = ProviderSettings()          # lee .env / ~/.oroi/.env / variables de entorno
llm = AzureLLM(settings)               # el LLM rápido: extractor y juez de consolidación
mind = Mind(f"memoria/{user_id}.db", AzureEmbedder(settings), TurnExtractor(llm), judge=llm)
```

(Con Ollama u OpenAI directo: `OpenAICompatLLM`/`OpenAICompatEmbedder` de
`oroi.providers.openai_compat`, o simplemente `MEMORY_PROVIDER=openai` en el `.env` si usas
los constructores de `oroi.chat.loop.build_mind`.)

## El ciclo del turno (si orquestas tú)

El patrón que usa el propio Oroi (es exactamente lo que hace `ChatSession.turn`):

```python
# 1) PERCIBIR el turno completo: la respuesta previa del agente + el mensaje nuevo,
#    con marcas de hablante. Una sola extracción por turno; la asimetría de hablante
#    (el protagonista es el usuario) la aplica el extractor gracias a las marcas.
perceived = f"[asistente] {ultima_respuesta}\n[usuario] {mensaje}" if ultima_respuesta else mensaje
mind.perceive(perceived)

# 2) RECORDAR con el mensaje como pregunta. Devuelve texto plano listo para inyectar
#    ("" si nada viene a la mente: NO inyectes nada — el silencio es correcto).
memoria = mind.recall(mensaje, window_turns=turnos_en_ventana)

# 3) (opcional) La señal de sorpresa: aviso graduado de que la conversación cambió de rumbo.
alerta = mind.surprise_alert()
if alerta:
    memoria = f"{alerta}\n{memoria}" if memoria else alerta

# 4) RESPONDER con TU LLM de siempre. La memoria va pegada al MENSAJE DE USUARIO,
#    nunca al system prompt (preserva la caché de prompts del proveedor):
#    contenido_usuario = f"{memoria}\n\n{mensaje}" if memoria else mensaje
respuesta = tu_llm(system, historial, contenido_usuario)
```

Notas del ciclo:

- **`window_turns`**: los números de turno que ya están en tu historial visible, para que el
  recall no repita lo que el modelo ya ve (dedupe). Si mantienes ventana propia, pásalos;
  `Mind.turn` te da el contador tras cada perceive.
- **El system prompt de tu agente** debe decirle qué es el bloque `[memoria asociativa]`: son
  recuerdos suyos de la relación, que use con naturalidad y sin mencionar el mecanismo. Copia
  el `SYSTEM` de `oroi/chat/session.py` como punto de partida.
- **Latencia**: perceive ≈ 1 llamada al LLM rápido (~1-2 s), recall ≈ 1 llamada + 1 embedding
  (~1 s). Si te aprieta, el perceive del par (respuesta previa + mensaje) puede hacerse tras
  enviar la respuesta.

## El ciclo del turno (si prefieres no orquestar)

`ChatSession` hace todo lo anterior — ventana conversacional con truncado en bloque incluida —
y solo le das tu `Chat` (cualquier objeto con
`reply(system, window, memory, user_text) -> str`):

```python
from oroi.chat.session import ChatSession
session = ChatSession(mind, tu_chat)     # tu_chat implementa el Protocol Chat
respuesta = session.turn(mensaje)        # percibe + recuerda + responde
```

## Dormir y despertar

- `mind.wake()` al abrir sesión. No reinicia nada (reabrir es continuar); prepara la mente.
- `mind.sleep()` cuando la conversación queda inactiva y al cerrar: consolida (funde duplicados,
  promueve asociaciones repetidas, transfiere a memoria permanente, poda). Usa LLM: no lo
  llames en el camino caliente de un turno. En un servidor: un temporizador de inactividad por
  conversación (el REPL usa `IdleSleeper`, `oroi/chat/loop.py`, como referencia).

## Operación

- **Copia de seguridad** = copiar el fichero `.db` (con la conversación parada, o vía
  `sqlite3 .backup`). La memoria entera vive ahí: portable por diseño.
- **Depuración**: `oroi info --db memoria/u1.db` (resumen sin credenciales),
  `oroi viz --db ...` (la red en HTML interactivo), `oroi serve --db ...` (visor en vivo).
- **Concurrencia**: una `Mind` por proceso y por base; `ChatSession` serializa turnos con su
  lock. No compartas el mismo fichero entre procesos escritores.
- **Config**: todos los parámetros de la dinámica viven en `DynamicsConfig` (inyectable en
  `Mind(..., config=...)`). Los defectos son los del preprint; los que más tocarás:
  `attention_budget` (tokens máximos de memoria inyectada), `window_max_turns`/`window_keep_turns`
  (ventana), `episode_context_turns` (±N turnos vecinos alrededor de cada recuerdo — útil en
  régimen de diálogo, contraproducente con hechos confusables consecutivos).

## La voz (experimental, desde 0.1.2): qué tiene la mente en mente

Observabilidad de solo lectura: un pensamiento en primera persona que verbaliza los conceptos
más activos de la red, con **valencia** (−2 muy negativo … +2 muy positivo) y **sorpresa**
(el último turno rompió el hilo). No escribe en el grafo, no toca la dinámica, y su texto
jamás se re-percibe. Cuesta **una llamada extra al LLM rápido por pensamiento** — por eso
está **apagada por defecto** y se decide al instanciar.

```python
from oroi import DynamicsConfig, Mind, Thought

# Opción A — bajo demanda (sin flag, sin coste por turno): pides el pensamiento cuando quieras.
mind = Mind(db, embedder, extractor, judge=llm)      # el judge (LLM rápido) es también la voz
thought = mind.consciousness()                        # Thought | None (None si la red está fría)
if thought:
    print(thought.text)      # "sigo dándole vueltas a la mudanza, y eso me lleva a…"
    print(thought.valence)   # -2..+2 — graficable turno a turno
    print(thought.surprise)  # True = giro de conversación (revisa el hilo a fondo)
    print(thought.chain)     # los conceptos verbalizados: los recibos del pensamiento

# Opción B — automática por turno (ChatSession): se activa en la config al instanciar.
config = DynamicsConfig(consciousness_enabled=True,   # apagada por defecto
                        consciousness_every=1)        # cadencia en turnos
mind = Mind(db, embedder, extractor, judge=llm, config=config)
session = ChatSession(mind, tu_chat, on_thought=lambda t: guardar_o_pintar(t))
# on_thought se dispara en un hilo aparte DESPUÉS de entregar la respuesta:
# el turno nunca espera a la voz; si el pensamiento anterior sigue en curso, se salta el ciclo.

# El diario (lectura pura, sin LLM): los últimos pensamientos, cronológicos.
for t in mind.thoughts(limit=50):
    print(t.turn, t.valence, t.text)                  # la serie de valencia, lista para graficar
```

Notas: `Thought` es serializable (`model_dump()`); el diario persiste en la tabla `thoughts`
de la propia base (`oroi thoughts --db …` lo lee sin credenciales, y `oroi serve` lo expone
en `GET /consciousness`); si la coalición activa no tiene asociaciones ni recuerdos que hilar,
el pensamiento es mecánico («tengo en la cabeza: …») y no gasta LLM.

## Lo que Oroi no hace (a propósito)

- No responde por ti: no toca tu LLM conversacional ni tu prompt de sistema.
- No inyecta nada si nada resuena: `recall()` vacío significa «calla», no «falla».
- No mezcla usuarios, no llama a casa, no necesita red más allá de tus proveedores: la memoria
  es un fichero local del que tú eres dueño.

## Checklist de integración

1. ☐ Un `.db` por usuario/relación, en disco tuyo.
2. ☐ `wake()` al abrir; `sleep()` en inactividad y al cerrar.
3. ☐ Perceive del turno completo con marcas `[usuario]`/`[asistente]`.
4. ☐ Recall con el mensaje como query; si devuelve `""`, no inyectar.
5. ☐ La memoria pegada al mensaje de usuario, nunca al system prompt.
6. ☐ System prompt de tu agente instruido sobre el bloque `[memoria asociativa]`.
7. ☐ Backup del fichero; visor para depurar lo que la memoria cree saber.
