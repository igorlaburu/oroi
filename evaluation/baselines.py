"""Las cuatro condiciones que se comparan. Tecnologías de CONTRASTE (RAG, re-ranker)
viven aquí, fuera de la librería: solo consumen `Mind` y su base, no la modifican.

Cada condición es una función `retrieve(mind, question) -> str` que devuelve el
contexto que esa técnica inyectaría al Conversador. La métrica mira ese texto.
"""

import math
import re

import numpy as np

# ── helpers de acceso a la base que expone la Mind (solo lectura) ──────────────

def _episodes(mind):
    rows = mind.store.conn.execute("SELECT id, text FROM episodes ORDER BY turn").fetchall()
    return [(r["id"], r["text"]) for r in rows]


def _unit(vectors):
    arr = np.array(vectors, dtype=float)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)


def _similar_episodes(mind, question, pool):
    """k-NN puro de texto: embebe la pregunta y los episodios, ordena por coseno."""
    eps = _episodes(mind)
    if not eps:
        return []
    texts = [t for _, t in eps]
    emb = _unit(mind.embedder.embed(texts))
    qv = _unit(mind.embedder.embed([question]))[0]
    order = (emb @ qv).argsort()[::-1][:pool]
    return [(eps[i][0], texts[i]) for i in order]


def _episode_activation(mind, episode_id):
    row = mind.store.conn.execute(
        "SELECT COALESCE(MAX(n.activation), 0) AS a FROM node_sources ns "
        "JOIN nodes n ON n.id = ns.node_id WHERE ns.episode_id = ?", (episode_id,)).fetchone()
    return row["a"]


# ── las cuatro condiciones ─────────────────────────────────────────────────────

def no_memory(mind, question, keep=2):
    """(a) Sin memoria: solo la ventana conversacional (últimos turnos)."""
    rows = mind.store.conn.execute(
        "SELECT text FROM episodes ORDER BY turn DESC LIMIT ?", (keep,)).fetchall()
    return "\n".join(r["text"] for r in reversed(rows))


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def _bm25_ranking(question, docs, k1=1.5, b=0.75):
    """Ranking léxico Okapi BM25 (a mano, sin dependencias): índices ordenados desc."""
    toks = [_tokenize(d) for d in docs]
    n, avgdl = len(toks), sum(len(t) for t in toks) / max(len(toks), 1)
    df = {}
    for t in toks:
        for term in set(t):
            df[term] = df.get(term, 0) + 1
    q = _tokenize(question)
    scores = []
    for doc in toks:
        tf = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1
        s = 0.0
        for term in q:
            if term not in df:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            f = tf.get(term, 0)
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(doc) / (avgdl or 1)))
        scores.append(s)
    return sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)


def _rrf(rankings, k=60):
    """Reciprocal Rank Fusion: combina varios rankings sin normalizar scores."""
    fused = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0) + 1 / (k + pos + 1)
    return sorted(fused, key=fused.get, reverse=True)


def rag(mind, question, k=3):
    """(b) RAG híbrido: denso (coseno) + léxico (BM25), fusionados por RRF.

    Baseline fuerte y justo —lo que se usa en producción—, no un k-NN de juguete:
    el denso capta sinónimos/paráfrasis, BM25 capta coincidencia literal de términos.
    """
    eps = _episodes(mind)
    if not eps:
        return ""
    texts = [t for _, t in eps]
    emb = _unit(mind.embedder.embed(texts))
    qv = _unit(mind.embedder.embed([question]))[0]
    dense = list((emb @ qv).argsort()[::-1])
    lexical = _bm25_ranking(question, texts)
    fused = _rrf([dense, lexical])[:k]
    return "\n".join(f"- «{texts[i]}»" for i in fused)


def oroi(mind, question, k=3):
    """(c) oroi: la pregunta se procesa como CONSULTA (extrae pistas, resuena y evoca),
    no como turno a percibir; recupera reconocimiento + evocación desde pistas y contexto."""
    return mind.recall(question)


def rag_facts(mind, question, k=8, max_ep=4):
    """(e) RAG sobre HECHOS: misma red extraída que oroi, pero selección por SIMILITUD
    (k-NN sobre los embeddings de los nodos), no por activación. Aísla la contribución de
    la dinámica: misma materia prima y misma serialización; lo único distinto es el criterio.
    Hereda el coste de extracción (la red la construye el mismo extractor que oroi)."""
    rows = mind.store.conn.execute("SELECT id, embedding FROM nodes WHERE faded = 0").fetchall()
    if not rows:
        return ""
    ids = [r["id"] for r in rows]
    embs = _unit([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    qv = _unit(mind.embedder.embed([question]))[0]
    top = [ids[i] for i in (embs @ qv).argsort()[::-1][:k]]   # nodos más SIMILARES a la pregunta
    return _serialize(mind.graph, top, max_ep)


def _serialize(graph, node_ids, max_ep):
    """Misma prosa que recall() (hechos entre nodos + episodios literales), para comparar limpio."""
    edges = graph.edges_among(node_ids)
    labels = graph.labels({e.src for e in edges} | {e.dst for e in edges} | set(node_ids))
    facts, seen = [], set()
    for e in edges:
        line = f"{labels[e.src]} {e.rel or 'se asocia con'} {labels[e.dst]}"
        if line not in seen:
            seen.add(line)
            facts.append(line)
    eps, turns = [], set()
    for nid in node_ids:
        for r in graph.episodes_for([nid]):
            if r["turn"] not in turns:
                turns.add(r["turn"])
                eps.append(f"- «{r['text']}»")
            if len(eps) >= max_ep:
                break
        if len(eps) >= max_ep:
            break
    parts = ["[memoria asociativa]"]
    if facts:
        parts.append("hechos: " + " · ".join(facts))
    if eps:
        parts += ["recuerdos:", *eps]
    return "\n".join(parts)


def reranker(mind, question, k=3, pool=20):
    """(d) Plan B: recupera por similitud y REORDENA por activación de la red."""
    hits = _similar_episodes(mind, question, pool)
    ranked = sorted(hits, key=lambda h: _episode_activation(mind, h[0]), reverse=True)
    return "\n".join(f"- «{t}»" for _, t in ranked[:k])


CONDITIONS = {
    "a · sin memoria": no_memory,
    "b · RAG crudo": rag,            # texto crudo de los turnos
    "e · RAG hechos": rag_facts,     # misma red extraída, selección por similitud
    "c · oroi": oroi,        # misma red, selección por activación
    "d · re-ranker": reranker,
}
