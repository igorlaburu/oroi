"""Dobles de prueba: el núcleo se testea sin LLM ni red (SPEC §7, Fase 1)."""

import itertools
import json
import random

import pytest

from oroi import DynamicsConfig, Mind
from oroi.extraction.schema import MiniEdge, MiniNetwork, MiniNode

DIM = 64
_counter = itertools.count()


class FakeEmbedder:
    """Vectores unitarios deterministas por etiqueta; los sinónimos comparten vector."""

    model = "fake-64d"
    dim = DIM

    def __init__(self, synonyms: dict[str, str] | None = None):
        self.synonyms = synonyms or {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vector(t) for t in texts]

    def vector(self, label: str) -> list[float]:
        rng = random.Random(self.synonyms.get(label, label))
        raw = [rng.gauss(0, 1) for _ in range(DIM)]
        norm = sum(x * x for x in raw) ** 0.5
        return [x / norm for x in raw]


class FakeJudge:
    """Veredictos enlatados del consolidador: una fusión dudosa por llamada y etiqueta de episodio."""

    def __init__(self, same: list[bool] | None = None, episode_label: str | None = None):
        self.same = list(same or [])  # se consumen en orden, uno por par juzgado
        self.episode_label = episode_label

    def complete_json(self, system: str, user: str) -> str:
        if "episodio" in system:
            return json.dumps({"label": self.episode_label})
        return json.dumps({"same": self.same.pop(0) if self.same else False})


class FakeExtractor:
    """Devuelve las mini-redes preparadas por el test, en orden; luego, mini-redes vacías."""

    def __init__(self, minis: list[MiniNetwork] | None = None):
        self.queue = list(minis or [])

    def extract(self, text: str, context: str = "") -> MiniNetwork:
        return self.queue.pop(0) if self.queue else MiniNetwork()

    def extract_cues(self, text: str) -> list[str]:
        return []  # los tests siembran la evocación vía el contexto reciente, no por cues


def mini(nodes=(), edges=()) -> MiniNetwork:
    return MiniNetwork(
        nodes=[MiniNode(label=label, kind=kind, salience=salience) for label, kind, salience in nodes],
        edges=[MiniEdge(src=src, dst=dst, rel=rel) for src, dst, rel in edges],
    )


def seed_chain(mind: Mind, labels: list[str], rel: str = "leads_to") -> dict[str, int]:
    """Inyecta a mano una cadena A→B→C... en la memoria (sin pasar por el extractor)."""
    ids = {}
    for label in labels:
        ids[label] = mind.graph.add_node(label, "entity", mind.embedder.vector(label), 0.5, 0)
    for a, b in zip(labels, labels[1:]):
        mind.graph.add_edge(ids[a], ids[b], rel, False, 0)
    return ids


@pytest.fixture
def make_mind(tmp_path):
    def _make(minis=(), config: DynamicsConfig | None = None,
              synonyms: dict[str, str] | None = None, embedder=None, judge=None) -> Mind:
        return Mind(
            db_path=str(tmp_path / f"mind-{next(_counter)}.db"),
            embedder=embedder or FakeEmbedder(synonyms),
            extractor=FakeExtractor(list(minis)),
            judge=judge,
            config=config or DynamicsConfig(),
        )

    return _make
