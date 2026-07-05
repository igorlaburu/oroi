"""El sueño: fusión, promoción de co-activaciones, largo plazo y poda (SPEC §7, Fase 3)."""

import math
import time

from oroi import DynamicsConfig
from oroi.chat.loop import IdleSleeper
from tests.conftest import DIM, FakeJudge, mini, seed_chain


def unit(first: float, axis: int = 0) -> list[float]:
    """Vector unitario en el plano (axis, axis+1) con coseno `first` respecto al eje."""
    v = [0.0] * DIM
    v[axis], v[axis + 1] = first, math.sqrt(1 - first * first)
    return v


def test_synthetic_duplicates_merge_when_judge_agrees(make_mind):
    mind = make_mind(judge=FakeJudge(same=[True]))
    coche = mind.graph.add_node("coche", "entity", unit(1.0), 0.5, 0)
    auto = mind.graph.add_node("mi auto", "entity", unit(0.75), 0.5, 0)  # zona dudosa
    rojo = mind.graph.add_node("rojo", "attribute", unit(1.0, axis=2), 0.5, 0)
    mind.graph.add_edge(auto, rojo, "tiene_color", False, 0)

    report = mind.sleep()

    assert report.merged == ["mi auto → coche"]
    assert mind.graph.by_label("mi auto") is None  # desvanecido, no borrado
    assert mind.graph.edge_between(coche, rojo)  # la arista se re-cableó al superviviente


def test_obvious_duplicates_merge_without_judge(make_mind):
    mind = make_mind()  # sin LLM: solo fusión automática por similitud extrema
    mind.graph.add_node("coche", "entity", unit(1.0), 0.5, 0)
    mind.graph.add_node("coches", "entity", unit(0.95), 0.5, 0)

    report = mind.sleep()

    assert len(report.merged) == 1
    assert mind.graph.by_label("coche") is not None  # sobrevive el más antiguo
    assert mind.graph.by_label("coches") is None


def test_merged_label_survives_as_lexical_alias(make_mind):
    """Tras fusionar «clientes»→«cliente», la pista plural —cuyo coseno (0.75) no llega al umbral
    (0.80) y cuyo prefijo «clientes*» no alcanza al singular— sigue reconociendo al superviviente:
    el label fusionado quedó como alias en el índice léxico. Sin esto, fusionar rompería el recall."""
    mind = make_mind(judge=FakeJudge(same=[True]))
    cliente = mind.graph.add_node("cliente", "entity", unit(1.0), 0.5, 0)
    mind.graph.add_node("clientes", "entity", unit(0.75), 0.5, 0)  # zona dudosa

    report = mind.sleep()
    assert report.merged == ["clientes → cliente"]

    cfg = mind.config
    # La pista plural, por vector, no cruza el umbral contra el singular superviviente...
    assert mind.graph.find_similar(unit(0.75), k=1)[0][1] < cfg.sim_threshold
    # ...pero el reconocimiento mixto la rescata por el alias léxico y la apunta al superviviente.
    assert mind.graph.recognize("clientes", unit(0.75), cfg.sim_threshold, cfg.lexical_floor) == cliente
    assert mind.graph.by_label("clientes") is None  # el nodo plural se desvaneció en la fusión


def test_doubtful_pair_without_judge_stays_apart(make_mind):
    mind = make_mind()
    mind.graph.add_node("coche", "entity", unit(1.0), 0.5, 0)
    mind.graph.add_node("camión", "entity", unit(0.75), 0.5, 0)

    report = mind.sleep()

    assert report.merged == []
    assert mind.graph.by_label("camión") is not None


def test_repeated_coactivation_becomes_edge(make_mind):
    config = DynamicsConfig()
    minis = [mini(nodes=[("A", "entity", 0.5), ("B", "entity", 0.5)])] * config.coact_promote_count
    mind = make_mind(minis=minis, config=config)
    ids = {**seed_chain(mind, ["A"]), **seed_chain(mind, ["B"])}

    for _ in range(config.coact_promote_count - 1):
        mind.perceive("A y B aparecen juntos")
    mind.sleep()
    assert not mind.graph.edge_between(ids["A"], ids["B"])  # aún no madura: puntual = ruido

    mind.perceive("A y B aparecen juntos otra vez")
    report = mind.sleep()
    assert report.promoted == 1
    assert mind.graph.edge_between(ids["A"], ids["B"])  # repetida = estructura


def test_frequency_hub_is_not_promoted(make_mind):
    """Asociación, no frecuencia: un par exclusivo (pocos eventos) se promueve; un par que solo
    coincide porque un extremo está caliente-en-todo (muchos eventos) no — anti-hairball."""
    mind = make_mind()
    g = mind.graph
    ids = {label: g.add_node(label, "entity", mind.embedder.vector(label), 0.5, 0)
           for label in ("a", "b", "hub", "x")}
    for t in range(3):                                   # a-b: exclusivos, count 3, eventos 3/3
        g.note_coactivations([(ids["a"], ids["b"])], [ids["a"], ids["b"]], t)
        g.note_coactivations([(ids["hub"], ids["x"])], [ids["hub"], ids["x"]], t)  # hub-x: count 3
    for t in range(50):                                  # el hub estuvo caliente en muchos turnos más
        g.note_coactivations([], [ids["hub"]], t)

    promoted = set(g.associative_coactivations(3, mind.config.coact_assoc_min))
    assert (ids["a"], ids["b"]) in promoted              # exclusivo → asociativo
    assert (ids["hub"], ids["x"]) not in promoted        # hub caliente-en-todo → no se cablea


def test_focus_limits_coactivation_pairs(make_mind):
    """Un turno con muchos conceptos no cablea todos contra todos: solo el foco (top-k)."""
    config = DynamicsConfig(coact_focus_k=3)
    mind = make_mind(minis=[mini(nodes=[(c, "entity", 0.5) for c in "abcdef"])], config=config)
    mind.perceive("a b c d e f")
    pairs = mind.store.conn.execute("SELECT COUNT(*) c FROM coactivations").fetchone()["c"]
    assert pairs <= 3                                    # C(3,2), no C(6,2)=15


def test_prune_removes_weak_relless_edges_only(make_mind):
    """La poda quita el ruido del hairball (aristas sin rel y de peso bajo), no lo reforzado ni lo etiquetado."""
    mind = make_mind()
    g = mind.graph
    a, b, c, d, e = (g.add_node(label, "entity", mind.embedder.vector(label), 0.5, 0) for label in "abcde")
    conn = mind.store.conn
    conn.execute("INSERT INTO edges(src,dst,rel,symmetric,weight,created_turn) VALUES(?,?,NULL,1,1.0,0)", (a, b))
    conn.execute("INSERT INTO edges(src,dst,rel,symmetric,weight,created_turn) VALUES(?,?,NULL,1,9.0,0)", (c, d))
    conn.execute("INSERT INTO edges(src,dst,rel,symmetric,weight,created_turn) VALUES(?,?,'vive_en',0,1.0,0)", (a, e))
    conn.commit()

    assert g.prune_weak_associations(1.5) == 1           # solo la débil sin rel
    assert not g.edge_between(a, b)                       # podada
    assert g.edge_between(c, d) and g.edge_between(a, e)  # fuerte-sin-rel y con-rel sobreviven


def test_sustained_activation_alone_does_not_promote(make_mind):
    """Una sola mención conjunta deja el par caliente varios turnos: eso NO es repetirse."""
    mind = make_mind(minis=[mini(nodes=[("A", "entity", 0.5), ("B", "entity", 0.5)])])
    ids = {**seed_chain(mind, ["A"]), **seed_chain(mind, ["B"])}

    mind.perceive("A y B juntos, una única vez")
    for _ in range(4):  # siguen calientes mientras se enfrían
        mind.perceive("relleno")
    mind.sleep()

    assert not mind.graph.edge_between(ids["A"], ids["B"])


def test_revisited_concepts_outlive_unrevisited_twins(make_mind):
    """La hipótesis del olvido: B se revisita y vive; A, su gemelo de nacimiento, se desvanece.
    Salience trivial (no cruza el suelo de golpe): solo la REVISITA consolida (vía transferencia)."""
    # Secuencia de mini-redes alineada con cada perceive: relleno = vacío, revisita = solo B.
    minis = [mini(nodes=[("A", "entity", 0.3), ("B", "entity", 0.3)])]
    for _ in range(3):
        minis += [mini()] * 6
        minis += [mini(nodes=[("B", "entity", 0.3)])]
    mind = make_mind(minis=minis)
    mind.perceive("A y B nacen juntos")

    for _ in range(3):  # tres "días": relleno, una revisita a B, y a dormir
        for _ in range(6):
            mind.perceive("relleno")
        mind.perceive("B otra vez")
        mind.sleep()

    assert mind.graph.by_label("A") is None  # no revisitado: podado
    assert mind.graph.by_label("B") is not None  # revisitado: consolidado
    assert mind.graph.by_label("B").base_strength > 0


def test_consolidated_facts_survive_long_decay_and_trivia_fades(make_mind):
    config = DynamicsConfig()
    mind = make_mind(minis=[mini(nodes=[("dato importante", "concept", 0.8)]),
                            mini(nodes=[("dato trivial", "concept", 0.2)])])
    mind.perceive("algo importante")
    mind.perceive("algo trivial")
    mind.sleep()  # transfiere activación × salience a base_strength

    for _ in range(40):
        mind.perceive("otro tema")
    mind.sleep()  # decaimiento lento del strength + poda lógica

    importante = mind.graph.by_label("dato importante")
    assert importante.activation == 0  # la memoria de trabajo lo olvidó
    assert importante.base_strength >= config.fade_threshold  # el largo plazo no
    assert mind.graph.by_label("dato trivial") is None  # lo trivial se podó


def test_idle_sleeper_naps_once_when_conversation_goes_quiet():
    class CountingMind:
        naps = 0

        def sleep(self):
            self.naps += 1

    mind = CountingMind()
    watcher = IdleSleeper(mind, idle_seconds=0.05, poll=0.01)
    watcher.start()
    watcher.touch()

    time.sleep(0.3)  # la conversación se queda quieta

    assert mind.naps == 1  # durmió una vez, y no en bucle: sin turnos nuevos no se re-duerme


def test_hot_cluster_abstracts_to_episode(make_mind):
    mind = make_mind(
        minis=[mini(nodes=[("coche", "entity", 0.6), ("rojo", "attribute", 0.5),
                           ("concesionario", "entity", 0.5)],
                    edges=[("coche", "rojo", "tiene_color"),
                           ("coche", "concesionario", "comprado_en")])],
        judge=FakeJudge(episode_label="la compra del coche"),
    )
    mind.perceive("compré un coche rojo en el concesionario")

    report = mind.sleep()

    assert report.episode == "la compra del coche"
    episode = mind.graph.by_label("la compra del coche")
    assert episode.kind == "episode"
    for label in ("coche", "rojo", "concesionario"):
        assert mind.graph.edge_between(mind.graph.by_label(label).id, episode.id)

    assert mind.sleep().episode is None  # un segundo sueño no duplica la abstracción
