"""Recordar: del subgrafo activado al texto plano que se inyecta al Conversador (SPEC §5, paso 8)."""

from collections import defaultdict

from .config import DynamicsConfig
from .graph import Graph

CHARS_PER_TOKEN = 4  # aproximación suficiente para acotar el presupuesto


def comes_to_mind(graph: Graph, config: DynamicsConfig) -> bool:
    """Gating: ¿hay algo por encima del umbral de recuperación?"""
    return bool(graph.effective_activations(floor=config.retrieval_threshold))


def recall(graph: Graph, config: DynamicsConfig, window_turns: frozenset[int] = frozenset(),
           turn: int | None = None, cue_nodes: tuple[int, ...] = ()) -> str:
    """El recuerdo como texto. Silencio ("") por defecto cuando nada viene a la mente.

    La activación ILUMINA, no ordena: un nodo es candidato si está encendido —por recencia
    (caliente) o porque la pista lo evoca (frío o caliente)—, pero el ORDEN lo deciden la
    RELEVANCIA a la pregunta (evocación) y la IMPORTANCIA consolidada (`base_strength`), no
    cuánto de caliente esté. Así un hecho muy activo pero irrelevante no gana por estar activo,
    y uno frío pero relevante aflora en cuanto la pista lo enciende ("se calienta al preguntar").
    Lo reciente, además, ya viaja literal en la ventana: el recall aporta el sujeto, no la recencia.
    """
    # Compuerta (¿qué está DISPONIBLE?): encendido por recencia ∪ evocado por la pista ∪ las pistas.
    # La evocación se siembra de los conceptos de la pregunta; si es elíptica, del hilo reciente.
    seeds = set(cue_nodes)
    if not seeds and turn is not None:
        seeds = graph.recently_activated(turn - config.context_window)
    relevance = _evoke_scores(graph, config, seeds)
    candidates = set(graph.effective_activations(floor=config.retrieval_threshold))
    candidates |= relevance.keys() | set(cue_nodes)
    if not candidates:
        return ""
    # Orden (¿qué es lo que buscas?): relevancia + importancia. NO entra la magnitud de activación.
    # Las pistas (el sujeto preguntado) anclan al frente; el resto, por puntuación.
    strength = graph.strengths(candidates)
    gain = config.recall_relevance_gain  # gradación relevancia↔importancia (1.0 = histórico)
    score = {n: gain * relevance.get(n, 0.0) + strength.get(n, 0.0) for n in candidates}
    anchors = list(cue_nodes)
    rest = [n for n in sorted(candidates, key=score.get, reverse=True) if n not in set(anchors)]
    ordered = anchors + rest[:config.recall_max_nodes]
    facts = _facts(graph, ordered)
    memories = _recollect(graph, ordered, window_turns, config)
    text = _compose(facts, memories, budget_chars=config.attention_budget * CHARS_PER_TOKEN)
    if text:
        # Efecto testeo: lo que se recuerda, se consolida un poco (escala lenta, no activación).
        graph.strengthen_nodes(ordered, config.testing_effect)
    return text


def _evoke_scores(graph: Graph, config: DynamicsConfig, seeds: set[int]) -> dict[int, float]:
    """Spreading activation MULTI-FUENTE desde las pistas (`seeds`): cuánta RELEVANCIA recibe cada
    nodo. No muta la activación (consulta de lectura). Camina aristas y pesos —no calor—, así que
    reenciende lo frío; donde varias pistas confluyen, la acumulación lo eleva solo (convergencia,
    sin contarla); el alcance (rondas) llega a varios saltos. Excluye las propias pistas (la
    respuesta es lo evocado, no lo preguntado)."""
    cues = set(seeds)
    if not cues:
        return {}
    received: dict[int, float] = defaultdict(float)
    for cue in cues:
        frontier = {cue: 1.0}
        for _ in range(config.evoke_rounds):     # alcance: la ola llega a varios saltos
            nxt: dict[int, float] = defaultdict(float)
            for node, act in frontier.items():
                edges = graph.top_edges(node, config.fanout_k)
                total = sum(e.weight for e in edges) or 1.0
                for e in edges:
                    if e.src == node:
                        neighbor, factor = e.dst, 1.0
                    else:
                        neighbor, factor = e.src, (1.0 if e.symmetric else config.back_factor)
                    nxt[neighbor] += act * config.evoke_damping * factor * e.weight / total
            for node, a in nxt.items():
                received[node] += a
            frontier = nxt
    return {n: r for n, r in received.items() if n not in cues}


def evoke(graph: Graph, config: DynamicsConfig, seeds: set[int]) -> list[int]:
    """Los nodos evocados, ordenados por relevancia recibida (envoltorio de `_evoke_scores`)."""
    scores = _evoke_scores(graph, config, seeds)
    return sorted(scores, key=scores.get, reverse=True)


def _facts(graph: Graph, node_ids: list[int]) -> list[str]:
    """Hechos SOLO entre los nodos calientes (edges_among, no _touching): un hub caliente
    no arrastra a sus vecinos fríos — anti-distractor.

    Cada hecho lleva su turno de última afirmación (la co-mención más reciente de sus dos
    extremos, no el eco hebbiano) y se listan del más reciente al más antiguo: ante dos
    valores en conflicto (residencia es X · residencia es Y), el que rige se distingue solo
    — el dato viejo no se oculta (a veces es información), se fecha."""
    edges = graph.edges_among(node_ids)
    asserted = graph.last_comention_turns(edges)
    recency = {e.id: asserted.get(e.id) or e.last_turn or 0 for e in edges}
    edges.sort(key=lambda e: recency[e.id], reverse=True)
    labels = graph.labels({e.src for e in edges} | {e.dst for e in edges} | set(node_ids))
    seen: set[str] = set()
    lines = []
    for e in edges:
        fact = f"{labels[e.src]} {e.rel or 'se asocia con'} {labels[e.dst]}"
        if fact in seen:
            continue                      # duplicado: se queda la versión más reciente
        seen.add(fact)
        lines.append(f"{fact} (turno {recency[e.id]})" if recency[e.id] else fact)
    return lines


def _recollect(graph: Graph, node_ids: list[int], window_turns: frozenset[int],
               config: DynamicsConfig) -> list[str]:
    """Evoca los episodios LITERALES (el acta entera del turno, ambos hablantes), ordenados por
    CO-MENCIÓN PONDERADA: cada episodio suma el peso de los nodos recordados que menciona, y el
    peso decae con el puesto del nodo en el recall (mencionar al ancla o a lo más relevante vale
    más que mencionar a dos nodos rasos — el relleno reciente no desplaza al eslabón antiguo de
    una cadena). Recencia solo de desempate. Con `episode_context_turns` > 0 cada episodio viene
    arropado por sus turnos vecinos, en orden cronológico: la implicatura del diálogo vive en el
    hilo adyacente. El texto no se mutila: el grafo encuentra, el texto habla."""
    rank = {nid: i for i, nid in enumerate(node_ids)}
    rows = graph.episodes_ranked(node_ids, exclude_turns=window_turns)

    def weight(row) -> float:
        mentioned = {int(n) for n in row["nids"].split(",")}
        return sum(1.0 / (rank[n] + 1.0) for n in mentioned if n in rank)

    chosen, seen = [], set()
    for r in sorted(rows, key=lambda r: (weight(r), r["turn"]), reverse=True):
        if r["turn"] in seen:
            continue
        seen.add(r["turn"])
        chosen.append(r)
        if len(chosen) >= config.recall_max_episodes:
            break
    radius = config.episode_context_turns
    if not radius:
        return [f"- (turno {r['turn']}, {r['role']}) «{r['text']}»" for r in chosen]
    # arropar cada episodio con su hilo vecino, sin duplicar y sin re-traer la ventana
    turns = {t for r in chosen for t in range(r["turn"] - radius, r["turn"] + radius + 1)}
    turns -= set(window_turns)
    return [f"- (turno {r['turn']}, {r['role']}) «{r['text']}»"
            for r in graph.episodes_by_turns(turns)]


def _compose(facts: list[str], memories: list[str], budget_chars: int) -> str:
    parts = ["[memoria asociativa]"]
    if facts:
        parts.append("hechos (más reciente primero): " + " · ".join(facts))
    if memories:
        parts.append("recuerdos:")
        parts.extend(memories)
    kept, used = [], 0
    for part in parts:
        if used + len(part) + 1 > budget_chars:
            break
        kept.append(part)
        used += len(part) + 1
    return "\n".join(kept)
