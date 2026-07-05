import pytest

from oroi import Mind
from tests.conftest import FakeEmbedder, FakeExtractor


def test_embedding_model_is_pinned(tmp_path):
    """La base se liga al modelo de embedding; mezclar modelos debe fallar (SPEC §2)."""
    db = str(tmp_path / "pinned.db")
    Mind(db, FakeEmbedder(), FakeExtractor())

    class OtherEmbedder(FakeEmbedder):
        model = "otro-modelo"

    with pytest.raises(ValueError, match="no se mezclan modelos"):
        Mind(db, OtherEmbedder(), FakeExtractor())


def test_turns_advance_and_persist(tmp_path):
    db = str(tmp_path / "turns.db")
    mind = Mind(db, FakeEmbedder(), FakeExtractor())
    mind.perceive("hola")
    mind.perceive("seguimos")
    assert mind.store.turn == 2

    reopened = Mind(db, FakeEmbedder(), FakeExtractor())
    assert reopened.store.turn == 2
