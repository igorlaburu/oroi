"""Persistencia SQLite + sqlite-vec. Capa técnica: aquí termina la metáfora (SPEC §5)."""

import sqlite3
import struct

import sqlite_vec

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS nodes (
    id                  INTEGER PRIMARY KEY,
    label               TEXT NOT NULL,
    kind                TEXT NOT NULL,
    embedding           BLOB NOT NULL,
    activation          REAL DEFAULT 0,
    base_strength       REAL DEFAULT 0,
    salience            REAL DEFAULT 0.5,
    last_activated_turn INTEGER,
    created_turn        INTEGER NOT NULL,
    faded               INTEGER DEFAULT 0,
    coact_events        INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS edges (
    id                   INTEGER PRIMARY KEY,
    src                  INTEGER NOT NULL REFERENCES nodes(id),
    dst                  INTEGER NOT NULL REFERENCES nodes(id),
    rel                  TEXT,
    symmetric            INTEGER DEFAULT 0,
    weight               REAL DEFAULT 1.0,
    last_reinforced_turn INTEGER,
    created_turn         INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE TABLE IF NOT EXISTS episodes (
    id         INTEGER PRIMARY KEY,
    turn       INTEGER NOT NULL,
    role       TEXT NOT NULL,
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS node_sources (
    node_id    INTEGER REFERENCES nodes(id),
    episode_id INTEGER REFERENCES episodes(id),
    PRIMARY KEY (node_id, episode_id)
);
CREATE TABLE IF NOT EXISTS coactivations (
    a         INTEGER NOT NULL REFERENCES nodes(id),  -- siempre a < b
    b         INTEGER NOT NULL REFERENCES nodes(id),
    count     INTEGER NOT NULL DEFAULT 1,
    last_turn INTEGER,
    PRIMARY KEY (a, b)
);
-- Índice léxico (BM25) sobre los labels: el reconocimiento es mixto vector+léxico,
-- para que "guggen" reconozca "museo guggenheim" (los embeddings flaquean con nombres propios).
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(label, tokenize='unicode61');
"""


def to_blob(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


class Store:
    def __init__(self, path: str, embedding_model: str, embedding_dim: int):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._load_vec_extension()
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.conn.executescript(SCHEMA)
        self._pin_embedding(embedding_model, embedding_dim)
        self._index_labels()

    def _index_labels(self) -> None:
        """Backfill del índice léxico para bases creadas antes de añadirlo (idempotente)."""
        if self.conn.execute("SELECT COUNT(*) c FROM nodes_fts").fetchone()["c"] == 0:
            self.conn.execute("INSERT INTO nodes_fts(rowid, label) SELECT id, label FROM nodes")
            self.conn.commit()

    def _load_vec_extension(self) -> None:
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

    def _pin_embedding(self, model: str, dim: int) -> None:
        # La base queda ligada a un único modelo de embedding: vectores de
        # modelos distintos no son comparables (SPEC §2).
        pinned = self.get_meta("embedding_model")
        if pinned is None:
            self.set_meta("embedding_model", model)
            self.set_meta("embedding_dim", str(dim))
        elif pinned != model or int(self.get_meta("embedding_dim") or 0) != dim:
            raise ValueError(
                f"esta base está ligada al embedding '{pinned}'; "
                f"no se mezclan modelos (recibido '{model}')"
            )

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    @property
    def turn(self) -> int:
        return int(self.get_meta("current_turn") or 0)

    def advance_turn(self) -> int:
        new_turn = self.turn + 1
        self.set_meta("current_turn", str(new_turn))
        return new_turn
