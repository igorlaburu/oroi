"""La voz (consciencia de solo lectura): verbaliza lo que la red tiene en mente.

Un observador, no un actor: lee la coalición activa y la convierte en un pensamiento
legible con valencia y sorpresa. NO escribe en el grafo (solo archiva en `thoughts`),
no toca la física, y su prosa JAMÁS se re-percibe — los efectos de pensar quedan
explícitamente fuera de esta pieza (decisión jul 2026: primero observabilidad).
Vive fuera de core/ porque usa LLM, como extraction/ y consolidation/.
"""

import json
import threading

from pydantic import BaseModel

from .core.config import DynamicsConfig
from .core.graph import Graph
from .core.store import Store

VOICE_PROMPT = """\
Eres la voz interior de una memoria asociativa: enuncias lo que la red tiene en mente \
AHORA MISMO. Recibes el MATERIAL del pensamiento: los conceptos activos (de más a menos), \
las asociaciones entre ellos y fragmentos literales de los recuerdos que los respaldan.
Devuelve SOLO este JSON: {{"text": "...", "valence": 0}}
- "text": un RESUMEN objetivo del material en COMO MÁXIMO {max_words} PALABRAS: qué está en \
el foco y cómo se conecta. Prosa factual, sin listas y SIN NINGÚN ASPECTO POÉTICO: nada de \
valoraciones ni coletillas («es un paso importante», «es curioso»), ni muletillas \
introspectivas («sigo dándole vueltas»), ni emociones, ni moralejas.
- ANCLAJE ESTRICTO: solo puedes mencionar lo que aparece en el material. PROHIBIDO añadir \
conocimiento de mundo, datos externos, escenas, sentimientos o conclusiones que el material \
no contenga; una implicación solo si el material la afirma.
- Conserva nombres e identificadores EXACTOS del material (códigos, referencias, matrículas: \
tal cual aparecen, nunca parafraseados). EXCEPCIÓN: [usuario]/[asistente] son marcas internas \
de hablante — jamás las escribas; di «el usuario» u omite el sujeto.
- Material escaso = resumen más corto. Nunca se rellena para llegar al máximo.
- Escribe en el idioma del material.
- SOLO si el material incluye la marca [GIRO] (el último turno rompió el hilo): la primera \
frase lo constata literalmente («La conversación ha girado hacia …»). Sin esa marca, JAMÁS \
menciones giros ni sorpresas.
- "valence": el tono emocional del CONTENIDO activo, entero de -2 (muy negativo) a +2 \
(muy positivo); 0 = neutro. La emoción se declara en este número, NUNCA en el texto.
Devuelve SOLO el JSON, sin comentarios."""


class Thought(BaseModel):
    """Un pensamiento de la voz — serializable (API web mañana, como NetworkSnapshot)."""

    turn: int
    text: str
    valence: int  # -2 (muy negativo) .. +2 (muy positivo)
    surprise: bool
    chain: list[str] = []  # los labels de la coalición verbalizada: los recibos


def coalition(graph: Graph, config: DynamicsConfig) -> tuple[str, list[str]]:
    """El material del pensamiento (pura, sin LLM): la coalición dominante serializada.

    Top-K nodos por activación, las aristas entre ellos y fragmentos de episodios
    testigo del más activo. Todo procede del grafo: nada más existe para la voz.
    Devuelve (material, labels) — material "" si la red está fría (sin material no
    hay pensamiento: el silencio es correcto, como en el recall).
    """
    active = graph.activations(floor=config.activation_floor)
    top = sorted(active, key=active.get, reverse=True)[: config.consciousness_top_k]
    if not top:
        return "", []
    labels = graph.labels(top)
    ordered = [labels[nid] for nid in top if nid in labels]
    edges = graph.edges_among(top)
    fragments = graph.node_episodes(top[0], config.consciousness_max_episodes)
    if not edges and not fragments:
        return "", ordered  # conceptos sueltos sin contexto: no hay nada que hilar (ver reflect)
    lines = ["[CONCEPTOS ACTIVOS] (de más a menos)"]
    lines += [f"- {label}" for label in ordered]
    if edges:
        lines.append("[ASOCIACIONES]")
        lines += [f"- {labels[e.src]} --{e.rel or 'con'}--> {labels[e.dst]}" for e in edges]
    if fragments:
        lines.append("[RECUERDOS]")
        lines += [f"- «{f}»" for f in fragments]
    return "\n".join(lines), ordered


class Consciousness:
    """Genera el pensamiento del estado actual. Patrón de locking del proyecto:
    leer la coalición con el lock (rápido), llamar al LLM sin él, escribir con lock."""

    def __init__(self, graph: Graph, store: Store, config: DynamicsConfig,
                 lock: threading.Lock, llm=None):
        self.graph, self.store, self.config = graph, store, config
        self.lock, self.llm = lock, llm

    def reflect(self, turn: int, surprise: bool) -> Thought | None:
        """Un pensamiento, o None. NUNCA lanza: la voz no puede romper la conversación."""
        if self.llm is None:
            return None  # sin LLM rápido no hay palabra
        try:
            with self.lock:
                material, chain = coalition(self.graph, self.config)
            if not material:
                # Sin nada que hilar no se llama al LLM: o silencio (red fría), o un
                # pensamiento mecánico de conceptos sueltos — anclado por construcción.
                if not chain:
                    return None
                thought = Thought(turn=turn, text=f"tengo en la cabeza: {', '.join(chain)}",
                                  valence=0, surprise=surprise, chain=chain)
                with self.lock:
                    self.store.add_thought(thought.turn, thought.text, thought.valence,
                                           thought.surprise, json.dumps(thought.chain))
                return thought
            if surprise:
                material += "\n[GIRO] El último turno ha roto el hilo que se venía siguiendo."
            thought = self._verbalize(material, chain, turn, surprise)
            if thought:
                with self.lock:
                    self.store.add_thought(thought.turn, thought.text, thought.valence,
                                           thought.surprise, json.dumps(thought.chain))
            return thought
        except Exception:
            return None

    def _verbalize(self, material: str, chain: list[str], turn: int,
                   surprise: bool) -> Thought | None:
        prompt = VOICE_PROMPT.format(max_words=self.config.consciousness_max_words)
        data = json.loads(self.llm.complete_json(prompt, material))
        text = str(data["text"]).strip()
        if not text:
            return None
        valence = max(-2, min(2, int(data.get("valence", 0))))
        return Thought(turn=turn, text=text, valence=valence, surprise=surprise, chain=chain)


def thoughts_from_store(store: Store, limit: int) -> list[Thought]:
    """El diario: lectura pura de la tabla, sin LLM (cronológico, el más reciente al final)."""
    rows = store.recent_thoughts(limit)
    return [Thought(turn=r["turn"], text=r["text"], valence=r["valence"],
                    surprise=bool(r["surprise"]), chain=json.loads(r["chain"]))
            for r in reversed(rows)]
