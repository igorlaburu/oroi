"""El job de "sueño": consolidación en segundo plano (SPEC §5, Consolidación).

Patrón "proponer sin lock / aplicar con lock": el mutex nunca se retiene durante
una llamada al LLM; cada aplicación re-valida sus precondiciones en una
transacción corta (concurrencia optimista). Si la re-validación falla, la
propuesta se descarta — el siguiente ciclo la recalculará si sigue pertinente.
El pipeline de turnos tiene prioridad.
"""

import threading
from collections import defaultdict

from pydantic import BaseModel

from ..core.config import DynamicsConfig
from ..core.graph import Graph
from ..core.store import Store

MERGE_PROMPT = """\
Eres el consolidador ("sueño") de una memoria asociativa. Recibes UN par de
etiquetas de conceptos con embeddings parecidos. Decide si nombran el MISMO
concepto —sinónimos, variantes de superficie, singular/plural, el mismo
referente— o conceptos distintos que solo se parecen (p.ej. dos importes, dos
clases, dos sectores). Responde SOLO este JSON: {"same": true} o {"same": false}"""

EPISODE_PROMPT = """\
Eres el consolidador ("sueño") de una memoria asociativa. Recibes los conceptos
de un clúster que se ha co-activado en la conversación. Si forman una unidad
temática coherente, devuelve una etiqueta corta (3-6 palabras, minúsculas) que
los resuma como episodio; si son un cajón de sastre, null.
Responde SOLO este JSON: {"label": "..." | null}"""


class MergeVerdict(BaseModel):
    same: bool = False


class EpisodeVerdict(BaseModel):
    label: str | None = None


class ConsolidationReport(BaseModel):
    """Resultado serializable de un ciclo de sueño (Mind es la frontera de la API)."""

    merged: list[str] = []
    promoted: int = 0
    faded: int = 0
    episode: str | None = None


class Consolidator:
    def __init__(self, graph: Graph, store: Store, config: DynamicsConfig,
                 lock: threading.Lock, judge=None, embedder=None):
        self.graph, self.store, self.config = graph, store, config
        self.lock = lock
        self.judge = judge        # LLM con complete_json(system, user) -> str; None = solo mecánica
        self.embedder = embedder  # para el embedding de los episodios abstraídos

    def consolidate(self) -> ConsolidationReport:
        report = ConsolidationReport()
        self._merge_concepts(report)
        with self.lock:
            report.promoted = self._promote_coactivations()
            self.graph.decay_strength(self.config.strength_decay, self.config.consolidation_floor)
            self.graph.transfer_strength(self.config.strength_transfer)
            report.faded = self.fade()
        self._abstract_episode(report)
        return report

    # ── fusión de conceptos duplicados ──────────────────────────────────

    def _merge_concepts(self, report: ConsolidationReport) -> None:
        """Similitud muy alta fusiona sola; la zona dudosa la decide el LLM."""
        with self.lock:
            pairs = self.graph.similar_pairs(self.config.merge_review_sim)
        accepted = [p for p in pairs if p[2] >= self.config.merge_auto_sim]
        accepted += self._judge_merges([p for p in pairs if p[2] < self.config.merge_auto_sim])
        for a, b, _, label_a, label_b in accepted:
            self._apply_merge(a, b, {a: label_a, b: label_b}, report)

    def _judge_merges(self, doubtful: list) -> list:
        """Juzga UN par por llamada. Probado: en lote, los pares genuinamente distintos (importes,
        clases, sectores) contaminan el juicio y arrastran a «False» al par que sí debía fundirse
        (p.ej. «cliente»~«clientes»); a solas, el juez acierta. La decisión atómica es un par; así no
        hay ni desalineación de lista ni contaminación de contexto. Un par que no valida no arrastra."""
        if not self.judge:
            return []
        accepted = []
        for pair in doubtful:
            _, _, _, la, lb = pair
            try:
                reply = self.judge.complete_json(MERGE_PROMPT, f"«{la}» ≡ «{lb}»")
                if MergeVerdict.model_validate_json(reply).same:
                    accepted.append(pair)
            except Exception:
                continue
        return accepted

    def _apply_merge(self, a: int, b: int, labels: dict[int, str], report: ConsolidationReport) -> None:
        with self.lock:
            if not (self.graph.alive(a) and self.graph.alive(b)):
                return  # alguien lo fusionó/podó desde el snapshot: descartar
            keep, drop = self._elect(a, b)
            self.graph.merge_nodes(keep, drop)
        report.merged.append(f"{labels[drop]} → {labels[keep]}")

    def _elect(self, a: int, b: int) -> tuple[int, int]:
        """Sobrevive el más consolidado; a igualdad, el más antiguo."""
        sa, sb = self.graph.node(a).base_strength, self.graph.node(b).base_strength
        return (a, b) if (sa, -a) >= (sb, -b) else (b, a)

    # ── promoción de co-activaciones repetidas a arista ─────────────────

    def _promote_coactivations(self) -> int:
        promoted = 0
        for a, b in self.graph.associative_coactivations(self.config.coact_promote_count, self.config.coact_assoc_min):
            if self.graph.alive(a) and self.graph.alive(b) and not self.graph.edge_between(a, b):
                self.graph.add_edge(a, b, None, symmetric=True, turn=self.store.turn)
                promoted += 1
            self.graph.forget_coactivation(a, b)
        return promoted

    # ── olvido a largo plazo: poda lógica ───────────────────────────────

    def fade(self) -> int:
        """Marca, no borra: la memoria nunca olvida del todo (SPEC §5)."""
        horizon = self.store.turn - self.config.fade_grace_turns
        return self.graph.fade_nodes(self.config.fade_threshold, horizon)

    # ── abstracción: clúster de hechos → nodo episodio ──────────────────

    def _abstract_episode(self, report: ConsolidationReport) -> None:
        if not (self.judge and self.embedder):
            return
        with self.lock:
            cluster = self._hot_cluster()
            if len(cluster) < self.config.episode_min_cluster or self.graph.has_episode_parent(cluster):
                return
        label = self._name_episode(cluster)  # LLM, sin lock
        if not label:
            return
        vector = self.embedder.embed([label])[0]
        with self.lock:
            if all(self.graph.alive(n) for n in cluster) and not self.graph.has_episode_parent(cluster):
                self._birth_episode(label, vector, cluster)
                report.episode = label

    def _hot_cluster(self) -> list[int]:
        """El mayor componente conexo del subgrafo caliente."""
        hot = set(self.graph.activations(floor=self.config.coact_threshold))
        neighbors: dict[int, set[int]] = defaultdict(set)
        for e in self.graph.edges_among(hot):
            neighbors[e.src].add(e.dst)
            neighbors[e.dst].add(e.src)
        best: set[int] = set()
        pending = set(neighbors)
        while pending:
            node = pending.pop()
            component, frontier = {node}, [node]
            while frontier:
                fresh = neighbors[frontier.pop()] - component
                component |= fresh
                frontier += fresh
                pending -= fresh
            best = max(best, component, key=len)
        return sorted(best)

    def _name_episode(self, cluster: list[int]) -> str | None:
        labels = self.graph.labels(cluster)
        try:
            reply = self.judge.complete_json(EPISODE_PROMPT, " · ".join(labels[n] for n in cluster))
            return EpisodeVerdict.model_validate_json(reply).label
        except Exception:
            return None

    def _birth_episode(self, label: str, vector: list[float], cluster: list[int]) -> None:
        turn = self.store.turn
        episode = self.graph.add_node(label, "episode", vector, 0.5, turn)
        self.graph.raise_strength(episode, self.graph.mean_strength(cluster))
        for member in cluster:
            self.graph.add_edge(member, episode, "parte_de", symmetric=False, turn=turn)
