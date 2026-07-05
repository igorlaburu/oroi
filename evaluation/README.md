# evaluation — banco de pruebas de oroi (Fase 5)

Pone a prueba la hipótesis del proyecto:

> *La recuperación sensible a activación supera a la similitud vectorial pura en
> conversaciones largas con referencias recurrentes y cambios de tema.*

**Aislado de la librería a propósito.** Este directorio vive **fuera** de
`src/oroi/` y la importa como cualquier consumidor. Dependencia
unidireccional: `evaluation` → `oroi`, nunca al revés. Las tecnologías de
contraste (RAG, re-ranker) son de aquí, no de la librería.

## Ejecutar

```bash
uv run python -m evaluation.run
```

Usa providers reales (Azure) porque reproduce conversaciones de verdad. Imprime
una tabla Recall@contexto (condición × escenario) y el veredicto.

## Las cuatro condiciones (`baselines.py`)

| | Condición | Recupera |
|---|---|---|
| a | sin memoria | solo la ventana (últimos turnos) |
| b | RAG híbrido | denso (coseno) + léxico (BM25) fusionados por RRF — baseline **fuerte**, no de juguete |
| c | oroi | subgrafo activado por resonancia + propagación |
| d | re-ranker | k-NN reordenado por activación de la red (plan B) |

El RAG es híbrido a propósito: si oroi solo le ganara a un k-NN básico, la
conclusión sería floja. BM25 está implementado a mano (sin dependencias nuevas).

## Métrica (`metrics.py`)

**Recall@contexto**: el contexto recuperado debe contener todos los fragmentos
`expect_context` y ninguno de `avoid_context` (el dato viejo, el distractor).
Objetiva y determinista, sin LLM-juez. Mide la recuperación, que es la hipótesis.

*Pendiente (extensión):* acierto de respuesta end-to-end con LLM-juez.

## Corpus a escala (`generate.py`)

Para tamaño de muestra de verdad, el generador paramétrico instancia plantillas
con variación aleatoria (determinista por semilla):

```bash
uv run python -m evaluation.generate --per-type 50 --seed 7   # 200 escenarios
uv run python -m evaluation.run                                # usa corpus/gen si existe
```

Cuatro fenómenos: recurrencia, actualización, distractores y **multi-sesión**
(con `session_break` → ejercita el sueño/consolidación y la persistencia a largo
plazo). El ground truth sale de la propia generación. La posición del dato, la
longitud y el nº de distractores se varían al azar, incluido el caso difícil.

**Significancia:** `run.py` reporta el test de **McNemar** (apareado, exacto) entre
oroi y RAG híbrido, con su p-valor. Para `gpt-4o-mini` conviene `temperature=0`
y/o repeticiones (el extractor no es determinista).

## Corpus curado a mano (`corpus/*.jsonl`)

Datos separados del código. Un fichero por escenario; cada línea un evento:

```
{"type":"meta","name":"coche-rojo","goal":"..."}
{"type":"turn","user":"texto del usuario"}
{"type":"session_break"}                 # dormir + despertar (multi-sesión)
{"type":"probe","question":"...","expect_context":["rojo"],"avoid_context":["azul"],"expect_answer":"rojo"}
```

Añadir un escenario = añadir un `.jsonl`. Sin tocar código.

## Cómo leer el resultado

`recall@contexto` por condición y escenario, 0–1 (media de las probes). El
escenario `actualizacion` es el más diagnóstico: solo una técnica sensible a
recencia/activación distingue el dato vigente del obsoleto.
