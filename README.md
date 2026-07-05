# Oroi

> **Oroi** (euskera: raíz de *oroitu*, «recordar») — una red semántica para agentes conversacionales.
> *Memoria para una conversación infinita.*

Oroi es una memoria asociativa para chatbots y agentes LLM que almacena la conversación como una
**red semántica con dinámica de activación** inspirada en la memoria humana: lo mencionado se
activa, la activación se propaga a lo asociado, todo decae con los turnos, las asociaciones que se
repiten se refuerzan y un proceso en segundo plano consolida lo importante — como la memoria humana
consolida durante el sueño.

**La hipótesis**: la recuperación sensible a la activación supera a la recuperación por similitud
vectorial pura (RAG) en conversaciones largas con referencias recurrentes y deriva temática. En una
evaluación de 224 escenarios, Oroi supera al RAG híbrido 0,75 frente a 0,55 en recall de
recuperación (gana en 43 escenarios y no pierde en ninguno; p<10⁻⁴), con la ventaja concentrada en
los casos que confunden a la búsqueda por similitud: distractores, datos que cambian y cadenas de
tres asociaciones.

*Oroi is an associative memory for LLM agents: a semantic network with activation dynamics
inspired by human memory. Spanish-first project; English preprint available below.*

## El preprint

- 📄 [Oroi: una red semántica para agentes conversacionales (castellano)](https://oroi.gako.ai/oroi-preprint-v1-es.pdf)
- 📄 [Oroi: a semantic network for conversational agents (English)](https://oroi.gako.ai/oroi-preprint-v1-en.pdf)
- 🔗 DOI: [10.5281/zenodo.21208930](https://doi.org/10.5281/zenodo.21208930)

## Instalación

Requiere Python ≥ 3.12. Desde el repositorio (la release en PyPI llegará como 0.1.0):

```bash
git clone https://github.com/igorlaburu/oroi && cd oroi
uv sync
cp .env.example .env   # y rellena tus credenciales (Azure OpenAI para extracción y embeddings)
```

## Uso en diez líneas

```python
from oroi import Mind
from oroi.extraction.extractor import TurnExtractor
from oroi.providers.azure import AzureEmbedder, AzureLLM
from oroi.providers.settings import ProviderSettings

settings = ProviderSettings()                      # lee .env
llm = AzureLLM(settings)
mind = Mind("memoria.db", AzureEmbedder(settings), TurnExtractor(llm), judge=llm)

mind.perceive("mi oficina está en Madrid")         # codificar un turno
print(mind.recall("¿dónde trabajo?"))              # recuperar: reconocimiento + evocación
```

La memoria persiste en `memoria.db`: reabrir es continuar, no empezar de cero. `Mind.sleep()`
consolida (fusión de duplicados, promoción de asociaciones repetidas, poda); en el REPL y el
servidor ocurre solo al quedar inactiva la conversación.

## La línea de comandos

```bash
uv run oroi-chat                        # REPL conversacional con memoria (proveedor por .env)
uv run oroi viz --db memoria.db         # la película de la memoria, en HTML autocontenido
uv run oroi serve --db memoria.db       # visor en vivo + chat contra la mente real
uv run oroi replay --db memoria.db      # reconstruye la película desde los episodios
uv run oroi consolidate --db memoria.db # consolidación bajo demanda
```

¿Sin conversación propia todavía? `uv run python examples/demo_conversation.py` graba una
conversación cotidiana de demostración en `examples/demo.db`.

## Proveedores

| Pieza | Soporte hoy |
|---|---|
| Extracción (LLM rápido) | Azure OpenAI, u OpenAI-compatible (OpenAI directo, **Ollama**, vLLM...) |
| Embeddings | Azure OpenAI, u OpenAI-compatible con un modelo de embeddings en el endpoint |
| Conversador | Claude (API o sesión local de Claude Code), Azure, u OpenAI-compatible |

Para correr **100 % en local** con Ollama: `MEMORY_PROVIDER=openai`,
`OPENAI_BASE_URL=http://localhost:11434/v1`, `OPENAI_FAST_MODEL=qwen3.6:latest` (o tu modelo), y
un modelo de embeddings servido (p. ej. `nomic-embed-text`, con `OPENAI_EMBEDDING_DIM=768`).

El núcleo no depende de ningún proveedor: `oroi/providers/base.py` define los `Protocol`s
(`Embedder`, `Extractor`, `Chat`) y todo se inyecta por constructor — cualquier otro proveedor es
implementar dos o tres métodos.

## Evaluación

El protocolo del preprint es reproducible: `evaluation/` contiene el corpus sintético
(224 escenarios, 7 fenómenos), las condiciones de contraste (RAG híbrido, RAG sobre hechos,
re-ranker) y las dos métricas (Recall@contexto y Answer@judge).

```bash
uv run python -m evaluation.run        # tabla por fenómeno + test de McNemar + costes
```

## Estado y licencia

Código bajo [Apache-2.0](LICENSE); el nombre «Oroi» queda fuera de la concesión de licencia.
Proyecto de investigación en desarrollo activo: la API puede cambiar. Próximas piezas declaradas
en el preprint: reelaboración de recuerdos y validación sobre benchmark externo.

## Contacto

Igor Laburu · [Gako AI](https://gako.ai) · oroi@gako.ai
