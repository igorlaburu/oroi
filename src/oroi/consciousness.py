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
Eres la voz interior de una memoria. Recibes lo que está EN LA MENTE ahora (los conceptos \
más presentes, de más a menos) y LO QUE SE RECUERDA de ellos: fragmentos LITERALES de \
conversaciones pasadas. Tu tarea es enunciar, en un pensamiento breve, qué tiene la mente en \
mente — un resumen fiel de esos recuerdos, no una interpretación.
Devuelve SOLO este JSON: {{"text": "...", "valence": 0}}
- "text": COMO MÁXIMO {max_words} PALABRAS, con verbos ACTIVOS y sujeto concreto, como una \
mente pensándolo en el momento («La junta nueva no ha aguantado — la envasadora de línea 1 \
vuelve a gotear.»). PROHIBIDAS de raíz las pasivas y el tono de acta («ha sido generado», «se \
remite», «se considera», «contiene información sobre»): di «el informe de hoy recoge la fuga…», \
no «el informe ha sido generado y contiene…».
- QUIÉN PIENSA. La voz es la MENTE que recuerda, no la persona recordada. El USUARIO y \
cualquier otra persona van SIEMPRE en TERCERA persona, por su nombre: «Igor trabaja en el \
proyecto de calidad y esquía en Cauterets». JAMÁS hables como si fueras el usuario («estoy \
trabajando», «me gusta esquiar»): eso es suplantarlo. La primera persona («yo») solo para un \
acto de la propia mente/asistente si el recuerdo lo dice («generé el informe»); ante la duda, \
tercera persona o sujeto impersonal del hecho.
- FIDELIDAD, NO INTERPRETACIÓN. Di SOLO lo que los fragmentos afirman, con sus mismas \
palabras cuando puedas. PROHIBIDO inventar relaciones, causas, pertenencias o conclusiones que \
los recuerdos no digan; que dos conceptos aparezcan juntos NO los relaciona. Sin conocimiento \
de mundo ni datos externos. Nombres e identificadores EXACTOS y COMPLETOS (códigos, \
referencias, matrículas, nombres: tal cual, nunca truncados ni parafraseados).
- Sobrio: sin poesía, sin valoraciones ni coletillas, sin muletillas rituales («sigo dándole \
vueltas»), sin moralejas.
- NADA DE META: piensas el MUNDO del que se ha hablado, nunca la conversación como acto. \
Ignora las frases sobre la sesión misma («esto es una prueba», «prueba finalizada», «seguimos \
luego», saludos, «vale», «de acuerdo»). Si tras descartarlas no queda sustancia, "text": "".
- [usuario]/[asistente] son marcas internas de hablante: jamás las escribas; di «el usuario» \
u omite el sujeto.
- Escaso = más corto; nunca se rellena para llegar al máximo. Escribe en el idioma del material.
- SOLO si el material incluye [GIRO] (el último turno rompió el hilo): la primera frase lo \
constata literalmente («La conversación ha girado hacia …»). Sin esa marca, JAMÁS menciones \
giros ni sorpresas.
- "valence": el tono emocional del contenido, entero de -2 a +2. Anclas: -2 desgracia o \
emoción fuerte negativa · -1 contratiempo, avería, retraso, preocupación · 0 neutro/informativo \
· +1 buena noticia, avance, resolución · +2 alegría o logro celebrado. No te pegues al 0: un \
contratiempo ES -1 aunque se cuente con calma. La emoción se declara en este número, NUNCA \
en el texto.
EJEMPLOS (recuerdo → "text"):
- «[usuario] me acabo de mudar a madrid a trabajar» → "El usuario se acaba de mudar a Madrid \
a trabajar." (tercera persona; la marca [usuario] NUNCA aparece escrita)
- «[usuario] mañana instalan la correa de maq-etq-001» → "Mañana instalan la correa de \
maq-etq-001." (el sujeto del hecho se conserva: quien instala NO es el usuario, aunque él lo cuente)
- «[usuario] esto es una prueba del sistema, luego seguimos» → "" (meta-conversación: no hay \
mundo del que pensar)
Devuelve SOLO el JSON, sin comentarios."""


class Thought(BaseModel):
    """Un pensamiento de la voz — serializable (API web mañana, como NetworkSnapshot)."""

    turn: int
    text: str
    valence: int  # -2 (muy negativo) .. +2 (muy positivo)
    surprise: bool
    chain: list[str] = []  # los labels de la coalición verbalizada: los recibos


def coalition(graph: Graph, config: DynamicsConfig) -> tuple[str, list[str]]:
    """El material del pensamiento (pura, sin LLM): los nodos activos y su CONTENIDO.

    Top-K nodos por activación sobre un suelo relativo al líder, y los fragmentos
    LITERALES de recuerdo de cada uno. NO se pasan aristas: las relaciones que la
    consolidación extrae son ruidosas (una co-activación cementada como «parte_de»
    contamina la voz), así que la voz piensa desde el TEXTO real, no desde el grafo
    interpretado. Todo procede de la memoria: nada más existe para la voz.
    Devuelve (material, labels) — material "" si la red está fría o sin recuerdos.
    """
    active = graph.activations(floor=config.activation_floor)
    ranked = sorted(active, key=active.get, reverse=True)[: config.consciousness_top_k]
    if not ranked:
        return "", []
    # Suelo RELATIVO al líder: focaliza en la parte caliente y descarta la cola,
    # pero deja intervenir a un nodo de menor grado si es comparable al más activo.
    lead = active[ranked[0]]
    floor = lead * config.consciousness_focus_ratio
    top = [n for n in ranked if active[n] >= floor]
    if len(top) < 2:
        top = ranked[:2]  # al menos un par (si lo hay): hace falta par para hilar
    labels = graph.labels(top)
    ordered = [labels[nid] for nid in top if nid in labels]
    # Fragmentos de recuerdo de los nodos activos (contenido real, sin duplicar).
    fragments: list[str] = []
    seen: set[str] = set()
    for nid in top:
        for f in graph.node_episodes(nid, config.consciousness_max_episodes):
            if f not in seen:
                seen.add(f)
                fragments.append(f)
    if not fragments:
        return "", ordered  # conceptos sueltos sin recuerdo: nada que resumir (ver reflect)
    lines = ["[EN LA MENTE AHORA] (de más a menos presente)"]
    lines += [f"- {label}" for label in ordered]
    lines.append("[LO QUE SE RECUERDA DE ELLO]")
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
