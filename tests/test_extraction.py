"""La mini-red del extractor: red de seguridad de aristas (sin LLM, determinista)."""

from oroi.extraction.schema import MiniNetwork


def _net(nodes, edges):
    return MiniNetwork.model_validate({
        "nodes": [{"label": label} for label in nodes],
        "edges": [{"src": s, "dst": d} for s, d in edges],
    })


def test_dangling_edge_to_phantom_subject_is_dropped():
    """La arista con un extremo que no es nodo declarado (el "usuario" fantasma) se descarta."""
    net = _net(["trabajo", "madrid"], [("usuario", "madrid"), ("trabajo", "madrid")])
    assert [(e.src, e.dst) for e in net.edges] == [("trabajo", "madrid")]


def test_self_loop_is_dropped():
    net = _net(["coche"], [("coche", "coche")])
    assert net.edges == []


def test_edge_endpoints_are_canonicalized_to_node_label():
    """Si el extractor desvía el case ('Pablo' en la arista, 'pablo' en el nodo), se canoniza."""
    net = _net(["pablo", "enfermero"], [("Pablo", "ENFERMERO")])
    assert [(e.src, e.dst) for e in net.edges] == [("pablo", "enfermero")]


def test_valid_edge_survives():
    net = _net(["coche", "rojo"], [("coche", "rojo")])
    assert [(e.src, e.dst) for e in net.edges] == [("coche", "rojo")]
