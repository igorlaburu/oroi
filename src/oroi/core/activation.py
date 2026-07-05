"""Dinámica de activación: los procesos cognitivos de la red (nomenclatura normativa, SPEC §5)."""

from collections import defaultdict
from itertools import combinations

import numpy as np

from ..extraction.schema import MiniNetwork
from .config import DynamicsConfig
from .graph import Graph


def resonate(graph: Graph, mini: MiniNetwork, vectors: dict[str, list[float]],
             config: DynamicsConfig, turn: int) -> dict[str, int]:
    """Confronta la mini-red del turno con la memoria: lo que resuena se activa, lo nuevo nace."""
    matched: dict[str, int] = {}
    gains: dict[int, float] = {}
    known: set[int] = set()
    for n in mini.nodes:
        hit = _best_match(graph, n.label, vectors[n.label], config)
        if hit:
            node_id, sim = hit
            gains[node_id] = config.boost * sim
            known.add(node_id)
        else:
            node_id = graph.add_node(n.label, n.kind, vectors[n.label], n.salience, turn)
            gains[node_id] = config.boost
        matched[n.label] = node_id
    _match_structure(graph, mini, matched, gains, known, config, turn)
    graph.add_activation(gains, turn)
    return matched


def _best_match(graph: Graph, text: str, vector: list[float], config: DynamicsConfig) -> tuple[int, float] | None:
    """Reconocimiento mixto vector+léxico: el match no exige coseno alto si el nombre coincide en
    superficie ("guggen"→"museo guggenheim"). Devuelve (nodo, coseno) — el coseno modula el impulso."""
    node_id = graph.recognize(text, vector, config.sim_threshold, config.lexical_floor, config.lexical_k)
    if node_id is None:
        return None
    return node_id, graph.similarities(vector, [node_id]).get(node_id, config.sim_threshold)


def _match_structure(graph: Graph, mini: MiniNetwork, matched: dict[str, int],
                     gains: dict[int, float], known: set[int],
                     config: DynamicsConfig, turn: int) -> None:
    """Bonus si la arista de la mini-red ya existía en memoria; si no, el extractor la crea."""
    bonused: set[int] = set()
    for e in mini.edges:
        a, b = matched.get(e.src), matched.get(e.dst)
        if a is None or b is None or a == b:
            continue
        if a in known and b in known and graph.edge_between(a, b):
            for node_id in (a, b):
                if node_id not in bonused:
                    gains[node_id] *= config.struct_bonus
                    bonused.add(node_id)
        else:
            graph.add_edge(a, b, e.rel, e.symmetric, turn)


def spread_activation(graph: Graph, config: DynamicsConfig) -> None:
    """Propagación amortiguada por rondas síncronas: los ciclos convergen, nunca se recorren."""
    spread = _equalize_from if config.spread_equalizes else _spread_from
    for _ in range(config.rounds):
        snapshot = graph.activations(floor=config.activation_floor)
        deltas: dict[int, float] = defaultdict(float)
        for node_id in snapshot:
            spread(graph, node_id, snapshot, config, deltas)
        if deltas:
            graph.add_activation(dict(deltas))


def _spread_from(graph: Graph, node_id: int, snapshot: dict[int, float],
                 config: DynamicsConfig, deltas: dict[int, float]) -> None:
    """Histórico: reparte una fracción de la activación SIN cederla (suma sin restar) —
    un subgrafo conectado se realimenta y puede saturar sin input (ver ablación)."""
    act = snapshot[node_id]
    edges = graph.top_edges(node_id, config.fanout_k)
    total_weight = sum(e.weight for e in edges)
    for e in edges:
        share = config.damping * act * e.weight / total_weight
        if e.src == node_id:
            deltas[e.dst] += share
        else:
            deltas[e.src] += share if e.symmetric else share * config.back_factor


def _equalize_from(graph: Graph, node_id: int, snapshot: dict[int, float],
                   config: DynamicsConfig, deltas: dict[int, float]) -> None:
    """Equiparación por gradiente (difusión): el calor fluye solo del más caliente al más frío,
    proporcional a la diferencia, y el origen CEDE lo que fluye. Al igualarse, el flujo es cero —
    nunca se cruzan: lo percibido queda siempre por encima de lo que prima. El spreading solo
    redistribuye (nunca crea energía), así que en silencio el decay siempre gana y todo se enfría.
    Contra la flecha, atenuado por back_factor."""
    act = snapshot[node_id]
    edges = graph.top_edges(node_id, config.fanout_k)
    total_weight = sum(e.weight for e in edges)
    back = config.back_factor if config.spread_back_factor is None else config.spread_back_factor
    for e in edges:
        neighbor = e.dst if e.src == node_id else e.src
        gap = act - snapshot.get(neighbor, 0.0)
        if gap <= 0:
            continue  # el calor nunca sube
        flow = config.damping * (e.weight / total_weight) * gap
        if e.src != node_id and not e.symmetric:
            flow *= back
        deltas[neighbor] += flow
        deltas[node_id] -= flow


def homeostasis(graph: Graph, config: DynamicsConfig) -> None:
    """Saturación, inhibición lateral y suelo: si todo está caliente, nada es saliente."""
    graph.cap_activations(config.activation_cap)
    total = graph.total_activation()
    if total > config.activation_budget:
        graph.scale_activations(config.activation_budget / total)
    graph.zero_below(config.activation_floor)


def wire_together(graph: Graph, config: DynamicsConfig, turn: int) -> None:
    """Regla de Hebb: lo que se activa junto se cablea junto — solo sobre aristas existentes.

    La co-activación sin arista no crea nada (puntual = ruido), pero deja traza:
    si se repite, el sueño la promueve a arista (repetida = estructura, SPEC §5).
    La traza solo cuenta pares re-percibidos EN este turno (resonancia directa):
    seguir caliente de turnos anteriores no es repetirse — sin esto, una sola
    mención conjunta inflaría el contador durante todo su enfriamiento. Y solo
    entre el FOCO del turno (los `coact_focus_k` más activos): un turno con muchos
    conceptos no debe cablear todos contra todos (anti-hairball).
    """
    hot = graph.activations(floor=config.coact_threshold)
    wired: set[frozenset[int]] = set()
    for e in graph.edges_among(hot):
        graph.strengthen_edge(e.id, config.eta * hot[e.src] * hot[e.dst], turn)
        wired.add(frozenset((e.src, e.dst)))
    focus = sorted(graph.freshly_activated(turn) & hot.keys(), key=hot.get, reverse=True)[:config.coact_focus_k]
    loose = [p for p in combinations(sorted(focus), 2) if frozenset(p) not in wired]
    graph.note_coactivations(loose, focus, turn)


def decay(graph: Graph, config: DynamicsConfig) -> None:
    """Olvido de la memoria de trabajo al cerrar cada turno — por turnos, nunca por reloj."""
    graph.scale_activations(config.decay)
    graph.zero_below(config.activation_floor)


def surprise(mini: MiniNetwork, vectors: dict[str, list[float]], hot_vectors: list,
             expectation: float, config: DynamicsConfig) -> float:
    """Error de predicción del turno [0-1]: lo entrante relevante que está LEJOS, en significado,
    de TODOS los focos calientes (los hilos vivos). Encaja si se parece a CUALQUIER foco; solo
    sorprende si está lejos de todos. Solo cuenta si había expectativa fuerte (red caliente).
    La señal con que el Sistema 1 despierta al Sistema 2."""
    if expectation < config.surprise_min_expectation or not mini.nodes or len(hot_vectors) == 0:
        return 0.0
    hot = np.array(hot_vectors, dtype=float)
    hot /= np.linalg.norm(hot, axis=1, keepdims=True) + 1e-9
    # Afinidad de referencia: cuánto encaja un miembro TÍPICO del hilo con sus vecinos. Medir en
    # RELATIVO a esto descuenta el "suelo" de los embeddings (dos temas distintos ya dan ~0.3).
    if len(hot) >= 2:
        sims = hot @ hot.T
        np.fill_diagonal(sims, -1.0)
        ref = float(sims.max(axis=1).mean()) + 1e-9
    else:
        ref = 1.0
    scored = []
    for n in mini.nodes:
        v = np.array(vectors[n.label], dtype=float)
        v /= np.linalg.norm(v) + 1e-9
        affinity = float((hot @ v).max())                       # encaje de lo nuevo con su mejor foco
        coldness = min(max((ref - affinity) / ref, 0.0), 1.0)   # encaja menos que un miembro típico = frío
        scored.append(n.salience * coldness)                    # relevante Y ajeno al hilo = sorprendente
    return min(sum(scored) / len(scored), 1.0)


def imprint(graph: Graph, node_id: int, salience: float, config: DynamicsConfig) -> None:
    """Memoria flash: la emoción extrema consolida en el acto, sin esperar al sueño."""
    graph.raise_strength(node_id, salience)
    graph.set_activation(node_id, config.activation_cap)
