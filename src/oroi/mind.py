"""Fachada neurocognitiva: la única puerta de entrada al sistema (SPEC §5 y §6).

Mind es también la frontera de la futura API (FastAPI + Next.js): entradas y
salidas serializables, sin estado global.
"""

import threading
from collections.abc import Iterable

from .consciousness import Consciousness, Thought, thoughts_from_store
from .consolidation.consolidator import ConsolidationReport, Consolidator
from .core import activation as dynamics
from .core import retrieval
from .core.config import DynamicsConfig
from .core.graph import Graph, NetworkSnapshot
from .core.store import Store
from .extraction.schema import MiniNetwork
from .providers.base import Embedder, Extractor


class Mind:
    def __init__(self, db_path: str, embedder: Embedder, extractor: Extractor,
                 judge=None, config: DynamicsConfig | None = None):
        self.config = config or DynamicsConfig()
        self.embedder = embedder
        self.extractor = extractor
        self.store = Store(db_path, embedder.model, embedder.dim)
        self.graph = Graph(self.store)
        # Mutex del patrón "proponer sin lock / aplicar con lock" (SPEC §5, Concurrencia).
        self._lock = threading.Lock()
        # judge: el LLM del consolidador (fusiones dudosas y episodios); None = solo mecánica.
        self.consolidator = Consolidator(self.graph, self.store, self.config,
                                         self._lock, judge, embedder)
        # La voz reutiliza el mismo LLM rápido; sin él, consciousness() devuelve None.
        self._consciousness = Consciousness(self.graph, self.store, self.config,
                                            self._lock, judge)
        self.last_surprise = 0.0  # error de predicción del último turno percibido (0 = encajó)

    @property
    def turn(self) -> int:
        """El turno conversacional actual — por la fachada, sin alcanzar el store."""
        return self.store.turn

    def perceive(self, text: str, role: str = "user") -> None:
        """Percibe un turno: cierra el anterior (decay) y ejecuta el pipeline del SPEC §5."""
        # Ventana reciente como contexto para desambiguar correferencias del turno (lectura, fuera del
        # lock como la llamada al LLM). (NO se canonicaliza contra "conceptos conocidos": probado y
        # RETIRADO —colapsaba valores/entidades distintos de superficie parecida: todos los importes a
        # un mismo nodo, Correos Express→correos; el coste superó al beneficio anti-fragmentación.)
        mini = self.extractor.extract(text, self._recent_context())
        with self._lock:
            dynamics.decay(self.graph, self.config)
            # Campo de pre-activación: los focos calientes que la red "esperaba" antes del impulso.
            hot = self.graph.activations(floor=self.config.activation_floor)
            hot_vectors = self.graph.embeddings(hot.keys())
            turn = self.store.advance_turn()
            episode = self.graph.add_episode(turn, role, text)
            if not mini.nodes:
                self.last_surprise = 0.0
                return
            vectors = self._embed_labels(mini)
            # Error de predicción: ¿lo que llega encaja con algún hilo vivo, o es ajeno a todos? (S1→S2)
            self.last_surprise = dynamics.surprise(mini, vectors, hot_vectors, sum(hot.values()), self.config)
            matched = dynamics.resonate(self.graph, mini, vectors, self.config, turn)
            self.graph.link_episode(matched.values(), episode)
            self._imprint_flashbulbs(mini, matched)
            dynamics.spread_activation(self.graph, self.config)
            dynamics.homeostasis(self.graph, self.config)
            dynamics.wire_together(self.graph, self.config, turn)

    def _recent_context(self) -> str:
        """La ventana deslizante de turnos previos como texto, para que el extractor resuelva
        referencias del turno actual (el turno aún no está en episodes: extraer va antes de añadirlo)."""
        episodes = self.graph.recent_episodes(self.config.window_keep_turns)
        return "\n".join(f"(turno {t}) {text}" for t, text in episodes)

    def recall(self, query_text: str | None = None, window_turns: Iterable[int] = ()) -> str:
        """Recuerda lo relevante como texto plano; "" si nada viene a la mente (gating).

        Con `query_text` (la pregunta del usuario), extrae sus pistas y las resuena para
        sembrar la evocación — además del contexto reciente del hilo.
        """
        with self._lock:
            cue_nodes = self._resonate_cues(query_text) if query_text else ()
            return retrieval.recall(self.graph, self.config, frozenset(window_turns),
                                    self.store.turn, cue_nodes)

    def _resonate_cues(self, text: str) -> tuple[int, ...]:
        cues = self.extractor.extract_cues(text)
        if not cues:
            return ()
        matched = []
        for cue, vec in zip(cues, self.embedder.embed(cues)):
            node_id = self.graph.recognize(cue, vec, self.config.sim_threshold,
                                           self.config.lexical_floor, self.config.lexical_k)
            if node_id is not None:
                matched.append(node_id)
        return tuple(matched)

    def surprise_alert(self) -> str:
        """Gancho S1→S2: el error de predicción del último turno como aviso subjetivo graduado,
        listo para anteponer al contexto del recall. "" si el turno encajó en el hilo activo.

        Es la señal con que la red asociativa (automática) pide al LLM (racional) que se detenga
        a razonar un giro, en vez de seguir en piloto automático. Un integrador externo puede
        leer el escalar crudo (`last_surprise`) o usar este aviso ya redactado.
        """
        if self.last_surprise >= self.config.surprise_threshold_high:
            return "[esto rompe el hilo que veníamos siguiendo; un giro que no esperaba]"
        if self.last_surprise >= self.config.surprise_threshold:
            return "[esto se aparta de lo que la conversación venía sugiriendo]"
        return ""

    def wake(self) -> None:
        """Abre sesión. Por diseño NO toca la activación: el tiempo es conversacional, no de reloj
        (SPEC §5), y la activación se persiste en SQLite — reabrir es CONTINUAR donde se dejó, no un
        reinicio. Lo no usado se enfría solo, turno a turno; lo consolidado conserva su base_strength
        y, si se enfrió, lo reenciende la evocación. Se mantiene como hook de la fachada."""

    def introspect(self) -> NetworkSnapshot:
        """Foto del estado completo de la red — visualización hoy, endpoint web mañana."""
        with self._lock:
            return self.graph.snapshot(self.store.turn)

    def consciousness(self) -> Thought | None:
        """La voz (consciencia de SOLO LECTURA): verbaliza la coalición activa en un
        pensamiento con valencia (−2..+2) y sorpresa. Una llamada al LLM rápido; no
        escribe en el grafo (solo archiva en `thoughts`) y su prosa jamás se re-percibe.
        None si la red está fría o no hay LLM. Disponible siempre bajo demanda; el
        disparo automático por turno lo gobierna `consciousness_enabled` (ChatSession)."""
        return self._consciousness.reflect(self.store.turn,
                                           self.last_surprise >= self.config.surprise_threshold)

    def thoughts(self, limit: int = 20) -> list[Thought]:
        """El diario de la voz: los últimos pensamientos, cronológicos. Lectura pura."""
        with self._lock:
            return thoughts_from_store(self.store, limit)

    def sleep(self) -> ConsolidationReport:
        """Consolida ("sueño"): fusión, promoción, strength, poda, abstracción.

        No retiene el lock global: el consolidador propone sin lock y aplica en
        transacciones cortas — el pipeline de turnos siempre tiene prioridad.
        """
        return self.consolidator.consolidate()

    def _embed_labels(self, mini: MiniNetwork) -> dict[str, list[float]]:
        # Embeddings en lote: una sola llamada por turno (SPEC §5, paso 2).
        labels = [n.label for n in mini.nodes]
        return dict(zip(labels, self.embedder.embed(labels)))

    def _imprint_flashbulbs(self, mini: MiniNetwork, matched: dict[str, int]) -> None:
        """Emoción → memoria, como rampa (no solo el acantilado flash): la salience extrema
        imprime de golpe; una mención saliente normal deposita base_strength proporcional, que si
        cruza `consolidation_floor` queda permanente. Lo trivial no llega al suelo (se enfría)."""
        for n in mini.nodes:
            if n.salience >= self.config.flashbulb_threshold:
                dynamics.imprint(self.graph, matched[n.label], n.salience, self.config)
            elif (deposit := n.salience * self.config.salience_deposit) >= self.config.consolidation_floor:
                self.graph.raise_strength(matched[n.label], deposit)  # saliente: cruza el suelo de golpe
            # lo trivial no deposita: solo consolida si se REVISITA (transferencia del sueño)
