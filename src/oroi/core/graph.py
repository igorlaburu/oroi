"""Operaciones de datos sobre nodos y aristas. Los procesos cognitivos viven en activation.py."""

import json
import re
import sqlite3
from collections.abc import Iterable

import numpy as np
from pydantic import BaseModel

from .store import Store, to_blob


class Node(BaseModel):
    id: int
    label: str
    kind: str
    activation: float
    base_strength: float
    salience: float


class Edge(BaseModel):
    id: int
    src: int
    dst: int
    rel: str | None
    symmetric: bool
    weight: float
    last_turn: int | None = None  # última percepción/refuerzo: la recencia del hecho


class SnapshotNode(BaseModel):
    id: int
    label: str
    kind: str
    activation: float
    base_strength: float
    salience: float
    faded: bool
    created_turn: int
    # Coordenadas semánticas: proyección 2D del embedding con ejes fijados en meta.
    # Deterministas y estables para siempre: el mapa es el significado.
    x: float | None = None
    y: float | None = None


class EpisodeSource(BaseModel):
    turn: int
    role: str
    text: str


class NetworkSnapshot(BaseModel):
    """Foto serializable de la red en un turno — visualización y futura API web."""

    turn: int
    nodes: list[SnapshotNode] = []
    edges: list[Edge] = []
    # El texto literal que respalda cada nodo: el grafo encuentra, el texto habla (SPEC §4).
    sources: dict[int, list[EpisodeSource]] = {}
    # Lo que se dijo EN este turno: deja ver el replay como conversación + red sincronizadas.
    episode: list[EpisodeSource] = []


def _placeholders(ids: Iterable[int]) -> tuple[str, list[int]]:
    ids = list(ids)
    return ", ".join("?" * len(ids)), ids


class Graph:
    def __init__(self, store: Store):
        self.db = store.conn

    # ── nodos ──────────────────────────────────────────────────────────

    def add_node(self, label: str, kind: str, embedding: list[float], salience: float, turn: int) -> int:
        cur = self.db.execute(
            "INSERT INTO nodes (label, kind, embedding, salience, created_turn) VALUES (?, ?, ?, ?, ?)",
            (label, kind, to_blob(embedding), salience, turn),
        )
        self.db.execute("INSERT INTO nodes_fts(rowid, label) VALUES (?, ?)", (cur.lastrowid, label))
        return cur.lastrowid

    def node(self, node_id: int) -> Node:
        row = self.db.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return Node(**{k: row[k] for k in Node.model_fields})

    def by_label(self, label: str) -> Node | None:
        row = self.db.execute("SELECT id FROM nodes WHERE label = ? AND faded = 0", (label,)).fetchone()
        return self.node(row["id"]) if row else None

    def labels(self, node_ids: Iterable[int]) -> dict[int, str]:
        marks, ids = _placeholders(node_ids)
        rows = self.db.execute(f"SELECT id, label FROM nodes WHERE id IN ({marks})", ids)
        return {r["id"]: r["label"] for r in rows}

    def node_episodes(self, node_id: int, limit: int, max_chars: int = 300) -> list[str]:
        """Fragmentos literales recientes que atestiguan un nodo (el texto habla)."""
        rows = self.db.execute(
            "SELECT substr(e.text, 1, ?) AS t FROM episodes e "
            "JOIN node_sources ns ON ns.episode_id = e.id "
            "WHERE ns.node_id = ? ORDER BY e.turn DESC LIMIT ?",
            (max_chars, node_id, limit),
        )
        return [r["t"] for r in rows]

    def snapshot(self, turn: int) -> NetworkSnapshot:
        """Toda la red de un vistazo, desvanecidos incluidos (en la viz son fantasmas)."""
        projection = self._semantic_projection()
        nodes = []
        for r in self.db.execute("SELECT * FROM nodes"):
            fields = {k: r[k] for k in SnapshotNode.model_fields if k not in ("x", "y")}
            if projection is not None:
                center, axes = projection
                point = (np.frombuffer(r["embedding"], dtype=np.float32) - center) @ axes.T
                fields["x"], fields["y"] = round(float(point[0]), 2), round(float(point[1]), 2)
            nodes.append(SnapshotNode(**fields))
        edges = [self._edge(r) for r in self.db.execute("SELECT * FROM edges")]
        return NetworkSnapshot(turn=turn, nodes=nodes, edges=edges,
                               sources=self._sources(), episode=self._episode(turn))

    def _episode(self, turn: int, max_chars: int = 800) -> list[EpisodeSource]:
        rows = self.db.execute(
            "SELECT turn, role, substr(text, 1, ?) AS text FROM episodes WHERE turn = ? ORDER BY id",
            (max_chars, turn),
        )
        return [EpisodeSource(turn=r["turn"], role=r["role"], text=r["text"]) for r in rows]

    def _semantic_projection(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Ejes 2D del mapa semántico, fijados en meta la primera vez: estabilidad > optimalidad."""
        row = self.db.execute("SELECT value FROM meta WHERE key = 'viz_projection'").fetchone()
        if row:
            data = json.loads(row["value"])
            return np.array(data["center"]), np.array(data["axes"])
        vectors = [np.frombuffer(r["embedding"], dtype=np.float32)
                   for r in self.db.execute("SELECT embedding FROM nodes")]
        if not vectors:
            return None  # sin nodos no hay con qué fijar ejes; se fijarán al nacer el primero
        center, axes = self._principal_axes(np.array(vectors))
        payload = {"center": [round(float(v), 5) for v in center],
                   "axes": [[round(float(v), 5) for v in axis] for axis in axes]}
        self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('viz_projection', ?)",
                        (json.dumps(payload),))
        self.db.commit()
        return center, axes

    @staticmethod
    def _principal_axes(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(vectors) < 3:  # PCA degenerada: ejes aleatorios reproducibles
            axes = np.random.default_rng(7).standard_normal((2, vectors.shape[1]))
            axes /= np.linalg.norm(axes, axis=1, keepdims=True)
            return np.zeros(vectors.shape[1]), axes
        center = vectors.mean(axis=0)
        _, _, vt = np.linalg.svd(vectors - center, full_matrices=False)
        return center, vt[:2]

    def _sources(self, per_node: int = 6, max_chars: int = 400) -> dict[int, list[EpisodeSource]]:
        rows = self.db.execute(
            "SELECT node_id, turn, role, substr(text, 1, ?) AS text FROM ("
            "  SELECT ns.node_id, e.turn, e.role, e.text,"
            "    ROW_NUMBER() OVER (PARTITION BY ns.node_id ORDER BY e.turn DESC) AS rn"
            "  FROM node_sources ns JOIN episodes e ON e.id = ns.episode_id"
            ") WHERE rn <= ?",
            (max_chars, per_node),
        )
        sources: dict[int, list[EpisodeSource]] = {}
        for r in rows:
            sources.setdefault(r["node_id"], []).append(
                EpisodeSource(turn=r["turn"], role=r["role"], text=r["text"]))
        return sources

    def find_similar(self, embedding: list[float], k: int = 1) -> list[tuple[int, float]]:
        rows = self.db.execute(
            "SELECT id, 1 - vec_distance_cosine(embedding, ?) AS sim "
            "FROM nodes WHERE faded = 0 ORDER BY sim DESC LIMIT ?",
            (to_blob(embedding), k),
        ).fetchall()
        return [(r["id"], r["sim"]) for r in rows]

    def find_lexical(self, text: str, k: int = 5) -> list[int]:
        """Candidatos por coincidencia LÉXICA (BM25 sobre los labels). Cada palabra se busca como
        prefijo ("guggen" → "guggen*"), así una mención parcial alcanza el nombre completo. Devuelve ids
        vivos por relevancia BM25; vacío si no hay términos usables."""
        terms = [w for w in re.findall(r"\w+", text.lower()) if len(w) >= 2]
        if not terms:
            return []
        query = " OR ".join(f"{w}*" for w in terms)
        try:
            rows = self.db.execute(
                "SELECT f.rowid AS id FROM nodes_fts f JOIN nodes n ON n.id = f.rowid "
                "WHERE nodes_fts MATCH ? AND n.faded = 0 ORDER BY bm25(nodes_fts) LIMIT ?",
                (query, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r["id"] for r in rows]

    def similarities(self, embedding: list[float], node_ids: Iterable[int]) -> dict[int, float]:
        """Coseno del vector contra nodos CONCRETOS (para desambiguar candidatos léxicos)."""
        ids = list(node_ids)
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.db.execute(
            f"SELECT id, 1 - vec_distance_cosine(embedding, ?) AS sim FROM nodes WHERE id IN ({marks})",
            (to_blob(embedding), *ids),
        ).fetchall()
        return {r["id"]: r["sim"] for r in rows}

    def recognize(self, text: str, embedding: list[float], sim_threshold: float,
                  lexical_floor: float, lexical_k: int = 5) -> int | None:
        """Reconoce a qué nodo se refiere una mención/pista con señal MIXTA vector+léxica — como un
        cerebro reconoce una palabra parcial o mal escrita. El vector manda: si hay match semántico
        sobre el umbral, ese. Si no, el léxico (BM25) RESCATA nombres propios/parciales que el coseno
        no alcanza ("guggen"→"museo guggenheim"); el coseno DESAMBIGUA entre los candidatos léxicos y pone
        un suelo (un token compartido no basta). None si nada convence."""
        if any(c.isdigit() for c in text):
            # Los VALORES (importes, fechas, cantidades) se reconocen por IDENTIDAD, no por parecido
            # semántico: "15.000 euros" y "60.000 euros" embeben casi igual (cos>0.85) y se fundirían
            # en un solo nodo, muddleando todo lo numérico. Coincidencia exacta de etiqueta o nada.
            row = self.db.execute(
                "SELECT id FROM nodes WHERE label = ? COLLATE NOCASE AND faded = 0 LIMIT 1",
                (text.strip(),),
            ).fetchone()
            return row["id"] if row else None
        best = self.find_similar(embedding, k=1)
        if best and best[0][1] >= sim_threshold:
            return best[0][0]
        lexical = self.find_lexical(text, lexical_k)
        if not lexical:
            return None
        sims = self.similarities(embedding, lexical)
        winner = max(lexical, key=lambda i: sims.get(i, 0.0))
        return winner if sims.get(winner, 0.0) >= lexical_floor else None

    # ── activación (operaciones de dato; la semántica está en activation.py) ──

    def activations(self, floor: float = 0.0) -> dict[int, float]:
        rows = self.db.execute("SELECT id, activation FROM nodes WHERE activation >= ?", (floor,))
        return {r["id"]: r["activation"] for r in rows}

    def embeddings(self, node_ids: Iterable[int]) -> list:
        marks, ids = _placeholders(node_ids)
        if not ids:
            return []
        rows = self.db.execute(f"SELECT embedding FROM nodes WHERE id IN ({marks})", ids)
        return [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]

    def strengths(self, node_ids: Iterable[int]) -> dict[int, float]:
        marks, ids = _placeholders(node_ids)
        if not ids:
            return {}
        rows = self.db.execute(f"SELECT id, base_strength FROM nodes WHERE id IN ({marks})", ids)
        return {r["id"]: r["base_strength"] for r in rows}

    def add_activation(self, deltas: dict[int, float], turn: int | None = None) -> None:
        self.db.executemany(
            "UPDATE nodes SET activation = activation + ?, "
            "last_activated_turn = COALESCE(?, last_activated_turn) WHERE id = ?",
            [(delta, turn, nid) for nid, delta in deltas.items()],
        )
        self.db.commit()

    def set_activation(self, node_id: int, value: float) -> None:
        self.db.execute("UPDATE nodes SET activation = ? WHERE id = ?", (value, node_id))
        self.db.commit()

    def scale_activations(self, factor: float) -> None:
        self.db.execute("UPDATE nodes SET activation = activation * ?", (factor,))
        self.db.commit()

    def effective_activations(self, floor: float) -> dict[int, float]:
        """Recuperabilidad ACT-R: activación actual + fuerza consolidada, con resonancia presente."""
        rows = self.db.execute(
            "SELECT id, activation + base_strength AS effective FROM nodes "
            "WHERE faded = 0 AND activation > 0 AND activation + base_strength >= ?",
            (floor,),
        )
        return {r["id"]: r["effective"] for r in rows}

    def cap_activations(self, cap: float) -> None:
        self.db.execute("UPDATE nodes SET activation = ? WHERE activation > ?", (cap, cap))
        self.db.commit()

    def zero_below(self, floor: float) -> None:
        self.db.execute("UPDATE nodes SET activation = 0 WHERE activation < ?", (floor,))
        self.db.commit()

    def total_activation(self) -> float:
        return self.db.execute("SELECT COALESCE(SUM(activation), 0) AS t FROM nodes").fetchone()["t"]

    def raise_strength(self, node_id: int, value: float) -> None:
        self.db.execute(
            "UPDATE nodes SET base_strength = MAX(base_strength, ?) WHERE id = ?", (value, node_id)
        )
        self.db.commit()

    def strengthen_nodes(self, node_ids: Iterable[int], delta: float) -> None:
        """Efecto testeo: recordar consolida. Refuerza la escala LENTA de los nodos recordados —
        nunca la activación: la fuerza no se propaga, así que el refuerzo no puede realimentarse."""
        marks, ids = _placeholders(node_ids)
        if not ids or delta <= 0:
            return
        self.db.execute(
            f"UPDATE nodes SET base_strength = base_strength + ? WHERE id IN ({marks})",
            [delta] + ids,
        )
        self.db.commit()

    # ── aristas ────────────────────────────────────────────────────────

    def add_edge(self, src: int, dst: int, rel: str | None, symmetric: bool, turn: int) -> None:
        if src == dst or self.edge_between(src, dst):
            return
        self.db.execute(
            "INSERT INTO edges (src, dst, rel, symmetric, created_turn) VALUES (?, ?, ?, ?, ?)",
            (src, dst, rel, int(symmetric), turn),
        )
        self.db.commit()

    def edge_between(self, a: int, b: int) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM edges WHERE (src = ? AND dst = ?) OR (src = ? AND dst = ?) LIMIT 1",
            (a, b, b, a),
        ).fetchone()
        return row is not None

    def top_edges(self, node_id: int, k: int) -> list[Edge]:
        rows = self.db.execute(
            "SELECT * FROM edges WHERE src = ? OR dst = ? ORDER BY weight DESC LIMIT ?",
            (node_id, node_id, k),
        )
        return [self._edge(r) for r in rows]

    def last_comention_turns(self, edges: Iterable[Edge]) -> dict[int, int]:
        """Recencia de un HECHO = turno del episodio más reciente que menciona a AMBOS extremos
        (su última afirmación conjunta) — no el eco hebbiano de haberse mantenido caliente."""
        turns: dict[int, int] = {}
        for e in edges:
            row = self.db.execute(
                "SELECT MAX(ep.turn) AS t FROM node_sources a "
                "JOIN node_sources b ON a.episode_id = b.episode_id "
                "JOIN episodes ep ON ep.id = a.episode_id "
                "WHERE a.node_id = ? AND b.node_id = ?",
                (e.src, e.dst),
            ).fetchone()
            if row and row["t"] is not None:
                turns[e.id] = row["t"]
        return turns

    def edges_among(self, node_ids: Iterable[int]) -> list[Edge]:
        marks, ids = _placeholders(node_ids)
        if not ids:
            return []
        rows = self.db.execute(
            f"SELECT * FROM edges WHERE src IN ({marks}) AND dst IN ({marks})", ids + ids
        )
        return [self._edge(r) for r in rows]

    def edges_touching(self, node_ids: Iterable[int], limit: int = 50) -> list[Edge]:
        marks, ids = _placeholders(node_ids)
        if not ids:
            return []
        rows = self.db.execute(
            f"SELECT * FROM edges WHERE src IN ({marks}) OR dst IN ({marks}) "
            "ORDER BY weight DESC LIMIT ?",
            ids + ids + [limit],
        )
        return [self._edge(r) for r in rows]

    def edge_weight(self, src: int, dst: int) -> float | None:
        row = self.db.execute(
            "SELECT weight FROM edges WHERE src = ? AND dst = ?", (src, dst)
        ).fetchone()
        return row["weight"] if row else None

    def strengthen_edge(self, edge_id: int, delta: float, turn: int) -> None:
        self.db.execute(
            "UPDATE edges SET weight = weight + ?, last_reinforced_turn = ? WHERE id = ?",
            (delta, turn, edge_id),
        )
        self.db.commit()

    # ── co-activaciones: la traza que el sueño convierte en estructura ──

    def note_coactivations(self, pairs: Iterable[tuple[int, int]], focus: Iterable[int], turn: int) -> None:
        if focus:  # eventos de co-activación por nodo: denominador de la asociación (frecuencia)
            self.db.executemany("UPDATE nodes SET coact_events = coact_events + 1 WHERE id = ?",
                                [(n,) for n in focus])
        self.db.executemany(
            "INSERT INTO coactivations (a, b, last_turn) VALUES (?, ?, ?) "
            "ON CONFLICT(a, b) DO UPDATE SET count = count + 1, last_turn = excluded.last_turn",
            [(min(a, b), max(a, b), turn) for a, b in pairs],
        )
        self.db.commit()

    def freshly_activated(self, turn: int) -> set[int]:
        rows = self.db.execute(
            "SELECT id FROM nodes WHERE last_activated_turn = ? AND faded = 0", (turn,)
        )
        return {r["id"] for r in rows}

    def recently_activated(self, since_turn: int) -> set[int]:
        """Conceptos aún vivos del hilo reciente: pistas de contexto para la evocación."""
        rows = self.db.execute(
            "SELECT id FROM nodes WHERE last_activated_turn >= ? AND faded = 0", (since_turn,)
        )
        return {r["id"] for r in rows}

    def associative_coactivations(self, min_count: int, min_assoc: float) -> list[tuple[int, int]]:
        """Pares co-activados ASOCIATIVOS: `count ≥ min_count` y `count/√(eventos_a·eventos_b) ≥
        min_assoc`. La normalización por frecuencia descarta los hubs (un nodo caliente-en-todo
        tiene muchos eventos → asociación baja), promoviendo solo conexiones representativas."""
        rows = self.db.execute(
            "SELECT c.a, c.b, c.count, na.coact_events ea, nb.coact_events eb FROM coactivations c "
            "JOIN nodes na ON na.id = c.a JOIN nodes nb ON nb.id = c.b WHERE c.count >= ?", (min_count,)
        ).fetchall()
        return [(r["a"], r["b"]) for r in rows
                if r["count"] / ((r["ea"] * r["eb"]) ** 0.5 or 1) >= min_assoc]

    def prune_weak_associations(self, min_weight: float) -> int:
        """Utilidad de migración: poda aristas SIN rel (co-activación) de peso bajo —el ruido de
        un hairball que nunca se reforzó—, conservando las reforzadas. No la usa el ciclo normal."""
        cur = self.db.execute("DELETE FROM edges WHERE rel IS NULL AND weight < ?", (min_weight,))
        self.db.commit()
        return cur.rowcount

    def forget_coactivation(self, a: int, b: int) -> None:
        self.db.execute("DELETE FROM coactivations WHERE a = ? AND b = ?", (min(a, b), max(a, b)))
        self.db.commit()

    # ── consolidación: operaciones de dato que ejecuta el "sueño" ───────

    def alive(self, node_id: int) -> bool:
        row = self.db.execute("SELECT 1 FROM nodes WHERE id = ? AND faded = 0", (node_id,)).fetchone()
        return row is not None

    def similar_pairs(self, min_sim: float) -> list[tuple[int, int, float, str, str]]:
        """Pares vivos con similitud ≥ min_sim, como (a, b, sim, label_a, label_b). O(n²): escala experimental."""
        rows = self.db.execute(
            "SELECT a.id AS a, b.id AS b, a.label AS la, b.label AS lb, "
            "1 - vec_distance_cosine(a.embedding, b.embedding) AS sim "
            "FROM nodes a JOIN nodes b ON a.id < b.id "
            "WHERE a.faded = 0 AND b.faded = 0 "
            "AND 1 - vec_distance_cosine(a.embedding, b.embedding) >= ?",
            (min_sim,),
        )
        return [(r["a"], r["b"], r["sim"], r["la"], r["lb"]) for r in rows]

    def merge_nodes(self, keep: int, drop: int) -> None:
        """Fusión: drop vierte aristas, fuentes, fuerza y su FORMA DE SUPERFICIE en keep, y se
        desvanece. El label del drop queda como alias léxico en nodes_fts: así una pista que solo
        casaba al fusionado (p.ej. la plural «clientes» cuyo prefijo no alcanza al singular «cliente»
        superviviente) sigue reconociendo al concepto. La fusión es donde la red APRENDE que dos
        formas nombran lo mismo; el índice léxico lo recoge."""
        self._merge_lexical(keep, drop)
        self._rewire_edges(keep, drop)
        self.db.execute("UPDATE OR IGNORE node_sources SET node_id = ? WHERE node_id = ?", (keep, drop))
        self.db.execute("DELETE FROM node_sources WHERE node_id = ?", (drop,))
        self.db.execute(
            "UPDATE nodes SET "
            "activation = MAX(activation, (SELECT activation FROM nodes WHERE id = :d)), "
            "base_strength = MAX(base_strength, (SELECT base_strength FROM nodes WHERE id = :d)), "
            "salience = MAX(salience, (SELECT salience FROM nodes WHERE id = :d)) WHERE id = :k",
            {"d": drop, "k": keep},
        )
        self.db.execute("UPDATE nodes SET faded = 1, activation = 0 WHERE id = ?", (drop,))
        self.db.execute("DELETE FROM coactivations WHERE a = :d OR b = :d", {"d": drop})
        self.db.commit()

    def _merge_lexical(self, keep: int, drop: int) -> None:
        """Funde la fila FTS del drop en la del keep: el label sobrante pasa a ser alias buscable."""
        drop_label = self.db.execute("SELECT label FROM nodes WHERE id = ?", (drop,)).fetchone()
        if drop_label:
            self.db.execute("UPDATE nodes_fts SET label = label || ' ' || ? WHERE rowid = ?",
                            (drop_label["label"], keep))
        self.db.execute("DELETE FROM nodes_fts WHERE rowid = ?", (drop,))

    def _rewire_edges(self, keep: int, drop: int) -> None:
        for e in self.edges_touching([drop], limit=100_000):
            src = keep if e.src == drop else e.src
            dst = keep if e.dst == drop else e.dst
            if src == dst or self.edge_between(src, dst):
                self.db.execute("DELETE FROM edges WHERE id = ?", (e.id,))
            else:
                self.db.execute("UPDATE edges SET src = ?, dst = ? WHERE id = ?", (src, dst, e.id))

    def transfer_strength(self, rate: float) -> None:
        self.db.execute(
            "UPDATE nodes SET base_strength = base_strength + ? * activation * salience "
            "WHERE faded = 0 AND activation > 0",
            (rate,),
        )
        self.db.commit()

    def decay_strength(self, factor: float, floor: float) -> None:
        """Decae base_strength, pero lo CONSOLIDADO (≥ floor) no baja del suelo: si cruzó, se queda.
        Lo no consolidado decae libre hacia 0 (ruido transitorio)."""
        self.db.execute(
            "UPDATE nodes SET base_strength = CASE WHEN base_strength >= ? "
            "THEN MAX(base_strength * ?, ?) ELSE base_strength * ? END WHERE faded = 0",
            (floor, factor, floor, factor),
        )
        self.db.commit()

    def fade_nodes(self, strength_floor: float, before_turn: int) -> int:
        """Poda lógica del ruido transitorio. NUNCA poda un nodo que participa en una relación
        AFIRMADA (arista con rel): lo que el usuario expresó es permanente (lo consolidado, con
        base_strength ≥ consolidation_floor > strength_floor, ya queda fuera por el umbral)."""
        cur = self.db.execute(
            "UPDATE nodes SET faded = 1 WHERE faded = 0 AND activation = 0 "
            "AND base_strength < ? AND COALESCE(last_activated_turn, created_turn) < ? "
            "AND id NOT IN (SELECT src FROM edges WHERE rel IS NOT NULL "
            "               UNION SELECT dst FROM edges WHERE rel IS NOT NULL)",
            (strength_floor, before_turn),
        )
        self.db.commit()
        return cur.rowcount

    def has_episode_parent(self, node_ids: Iterable[int]) -> bool:
        marks, ids = _placeholders(node_ids)
        row = self.db.execute(
            f"SELECT 1 FROM edges e JOIN nodes n ON n.id = e.dst "
            f"WHERE e.src IN ({marks}) AND n.kind = 'episode' AND n.faded = 0 LIMIT 1",
            ids,
        ).fetchone()
        return row is not None

    def mean_strength(self, node_ids: Iterable[int]) -> float:
        marks, ids = _placeholders(node_ids)
        row = self.db.execute(
            f"SELECT COALESCE(AVG(base_strength), 0) AS s FROM nodes WHERE id IN ({marks})", ids
        ).fetchone()
        return row["s"]

    @staticmethod
    def _edge(row) -> Edge:
        return Edge(
            id=row["id"], src=row["src"], dst=row["dst"], rel=row["rel"],
            symmetric=bool(row["symmetric"]), weight=row["weight"],
            last_turn=row["last_reinforced_turn"] or row["created_turn"],
        )

    # ── episodios ──────────────────────────────────────────────────────

    def add_episode(self, turn: int, role: str, text: str) -> int:
        cur = self.db.execute(
            "INSERT INTO episodes (turn, role, text) VALUES (?, ?, ?)", (turn, role, text)
        )
        self.db.commit()
        return cur.lastrowid

    def recent_episodes(self, limit: int) -> list[tuple[int, str]]:
        """Los últimos `limit` turnos ya percibidos, en orden cronológico: la ventana deslizante que
        el extractor usa como CONTEXTO (resolver correferencias), nunca como fuente de extracción."""
        rows = self.db.execute(
            "SELECT turn, text FROM (SELECT turn, text FROM episodes ORDER BY id DESC LIMIT ?) ORDER BY turn",
            (limit,),
        ).fetchall()
        return [(r["turn"], r["text"]) for r in rows]

    def link_episode(self, node_ids: Iterable[int], episode_id: int) -> None:
        self.db.executemany(
            "INSERT OR IGNORE INTO node_sources (node_id, episode_id) VALUES (?, ?)",
            [(nid, episode_id) for nid in node_ids],
        )
        self.db.commit()

    def episodes_ranked(self, node_ids: Iterable[int], exclude_turns: Iterable[int] = (), limit: int = 50):
        """Episodios que mencionan a los nodos dados, con QUÉ nodos menciona cada uno
        (`nids`): el llamador pondera la co-mención con el ranking del recall — mencionar
        al ancla vale más que mencionar a dos nodos rasos. La pura recencia dejaría que un
        hub (el hablante de un diálogo, presente en cientos de turnos) o el relleno
        reciente acapararan los huecos."""
        marks, ids = _placeholders(node_ids)
        if not ids:
            return []
        ex_marks, ex_turns = _placeholders(exclude_turns)
        exclusion = f"AND e.turn NOT IN ({ex_marks})" if ex_turns else ""
        return self.db.execute(
            f"SELECT e.id, e.turn, e.role, e.text, GROUP_CONCAT(DISTINCT ns.node_id) AS nids "
            f"FROM episodes e JOIN node_sources ns ON ns.episode_id = e.id "
            f"WHERE ns.node_id IN ({marks}) {exclusion} "
            f"GROUP BY e.id ORDER BY COUNT(DISTINCT ns.node_id) DESC, e.turn DESC LIMIT ?",
            ids + ex_turns + [limit],
        ).fetchall()

    def episodes_by_turns(self, turns: Iterable[int]):
        """Los episodios de los turnos dados, en orden cronológico (para inyectar el hilo
        vecino de un episodio recordado: la implicatura vive en los turnos adyacentes)."""
        marks, ids = _placeholders(turns)
        if not ids:
            return []
        return self.db.execute(
            f"SELECT id, turn, role, text FROM episodes WHERE turn IN ({marks}) ORDER BY turn, id",
            ids,
        ).fetchall()

    def episodes_for(self, node_ids: Iterable[int], exclude_turns: Iterable[int] = (), limit: int = 20):
        marks, ids = _placeholders(node_ids)
        ex_marks, ex_turns = _placeholders(exclude_turns)
        exclusion = f"AND e.turn NOT IN ({ex_marks})" if ex_turns else ""
        return self.db.execute(
            f"SELECT DISTINCT e.id, e.turn, e.role, e.text FROM episodes e "
            f"JOIN node_sources ns ON ns.episode_id = e.id "
            f"WHERE ns.node_id IN ({marks}) {exclusion} ORDER BY e.turn DESC LIMIT ?",
            ids + ex_turns + [limit],
        ).fetchall()
