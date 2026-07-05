"""Instrumentación: snapshots por turno, diario, export HTML y replay (SPEC §7, Fase 4)."""

from oroi.viz import graph_view
from oroi.viz.replay import replay
from tests.conftest import mini


def test_introspect_returns_serializable_snapshot(make_mind):
    mind = make_mind(minis=[mini(nodes=[("coche", "entity", 0.6), ("rojo", "attribute", 0.5)],
                                edges=[("coche", "rojo", "tiene_color")])])
    mind.perceive("me he comprado un coche rojo")

    snap = mind.introspect()

    assert snap.turn == 1
    assert {"coche", "rojo"} <= {n.label for n in snap.nodes}
    assert any(e.rel == "tiene_color" for e in snap.edges)
    assert snap.model_dump_json()  # serializable: listo para la API web


def test_snapshot_carries_literal_sources(make_mind):
    """El grafo encuentra, el texto habla: cada nodo viaja con sus episodios literales."""
    mind = make_mind(minis=[mini(nodes=[("coche", "entity", 0.6)])])
    mind.perceive("mi coche es rojo")

    snap = mind.introspect()

    coche = mind.graph.by_label("coche").id
    assert any("mi coche es rojo" in s.text for s in snap.sources[coche])


def test_semantic_coordinates_pinned_and_stable(make_mind):
    """Los ejes se fijan una vez: la coordenada de un nodo no se mueve cuando nacen otros."""
    mind = make_mind(minis=[mini(nodes=[("A", "entity", 0.5)]), mini(nodes=[("B", "entity", 0.5)])])
    mind.perceive("hablemos de A")
    before = next(n for n in mind.introspect().nodes if n.label == "A")

    mind.perceive("llega B")
    snap = mind.introspect()

    after = next(n for n in snap.nodes if n.label == "A")
    assert (before.x, before.y) == (after.x, after.y)
    assert all(n.x is not None for n in snap.nodes)


def test_timeline_records_and_reloads(make_mind, tmp_path):
    mind = make_mind(minis=[mini(nodes=[("A", "entity", 0.5)]), mini(nodes=[("B", "entity", 0.5)])])
    journal = tmp_path / "timeline.jsonl"

    for text in ("hablemos de A", "y ahora de B"):
        mind.perceive(text)
        graph_view.record(mind.introspect(), journal)

    timeline = graph_view.load_timeline(journal)
    assert [s.turn for s in timeline] == [1, 2]
    assert len(timeline[0].nodes) == 1  # en el turno 1, B aún no existía
    assert len(timeline[1].nodes) == 2


def test_timeline_survives_truncated_line(make_mind, tmp_path):
    mind = make_mind(minis=[mini(nodes=[("A", "entity", 0.5)])])
    mind.perceive("A")
    journal = tmp_path / "timeline.jsonl"
    graph_view.record(mind.introspect(), journal)
    with open(journal, "a") as f:
        f.write('{"turn": 99, "nodes": [{"id":')  # un kill a media escritura

    timeline = graph_view.load_timeline(journal)

    assert [s.turn for s in timeline] == [1]  # la línea rota se ignora, el diario vive


def test_replay_rebuilds_timeline_from_texts(make_mind):
    mind = make_mind(minis=[mini(nodes=[("A", "entity", 0.5)]),
                            mini(nodes=[("B", "entity", 0.5)])])

    timeline = replay(["hablemos de A", "y de B"], mind)

    assert [s.turn for s in timeline] == [1, 2]
    assert {n.label for n in timeline[-1].nodes} == {"A", "B"}


def test_export_html_is_self_contained(make_mind, tmp_path):
    mind = make_mind(minis=[mini(nodes=[("coche", "entity", 0.6)])])
    mind.perceive("mi coche")

    out = graph_view.export_html([mind.introspect()], tmp_path / "red.html")

    html = out.read_text(encoding="utf-8")
    assert "d3" in html
    assert "coche" in html  # los datos viajan embebidos
    assert "__TIMELINE__" not in html and "__JOURNAL__" not in html  # plantilla resuelta
