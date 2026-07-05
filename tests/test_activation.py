"""La física de la red, con mini-redes y embeddings deterministas (SPEC §7, Fase 1)."""

import math

from oroi import DynamicsConfig
from tests.conftest import mini, seed_chain


def test_resonance_activates_matching_node_only(make_mind):
    mind = make_mind(minis=[mini(nodes=[("coche", "entity", 0.6), ("rojo", "attribute", 0.5)],
                                edges=[("coche", "rojo", "tiene_color")])])
    seed_chain(mind, ["coche"])
    seed_chain(mind, ["bicicleta"])

    mind.perceive("me he comprado un coche rojo")

    assert mind.graph.by_label("coche").activation > 0
    assert mind.graph.by_label("bicicleta").activation == 0
    assert mind.graph.by_label("rojo") is not None  # lo nuevo nace


def test_synonyms_resonate_to_same_node(make_mind):
    mind = make_mind(minis=[mini(nodes=[("coche", "entity", 0.5)]),
                            mini(nodes=[("auto", "entity", 0.5)])],
                     synonyms={"auto": "coche"})
    mind.perceive("tengo un coche")
    mind.perceive("mi auto es viejo")

    assert mind.graph.by_label("auto") is None  # no se creó un duplicado
    assert mind.graph.by_label("coche").activation > 1.0  # re-resonó y se reforzó


def test_propagation_decreases_along_chain(make_mind):
    mind = make_mind(minis=[mini(nodes=[("A", "entity", 0.5)])])
    seed_chain(mind, ["A", "B", "C"])

    mind.perceive("hablemos de A")

    a, b, c = (mind.graph.by_label(x).activation for x in "ABC")
    assert a > b > c > 0


def test_backward_propagation_is_attenuated(make_mind):
    mind = make_mind(minis=[mini(nodes=[("A", "entity", 0.5), ("D", "entity", 0.5)])])
    seed_chain(mind, ["A", "B"])  # forward: A activa a B
    seed_chain(mind, ["C", "D"])  # backward: D activa a C

    mind.perceive("A y D")

    forward = mind.graph.by_label("B").activation
    backward = mind.graph.by_label("C").activation
    assert forward > backward > 0


def test_cycles_converge(make_mind):
    config = DynamicsConfig(rounds=6)
    mind = make_mind(minis=[mini(nodes=[("A", "entity", 0.5)])], config=config)
    ids = seed_chain(mind, ["A", "B", "C"])
    mind.graph.add_edge(ids["C"], ids["A"], "cierra_ciclo", False, 0)

    mind.perceive("A")  # no debe divergir ni colgarse

    total = mind.graph.total_activation()
    assert math.isfinite(total)
    assert total <= config.activation_budget + 1e-6


def test_decay_forgets_working_memory(make_mind):
    mind = make_mind(minis=[mini(nodes=[("A", "entity", 0.5)])])
    mind.perceive("A")
    assert mind.graph.by_label("A").activation > 0

    for _ in range(40):  # turnos sin mencionar A: el extractor devuelve mini-redes vacías
        mind.perceive("otro tema")

    assert mind.graph.by_label("A").activation == 0  # cayó bajo el suelo y se olvidó


def test_hebbian_reinforces_existing_edges_only(make_mind):
    mind = make_mind(minis=[mini(nodes=[("A", "entity", 0.5), ("B", "entity", 0.5)]),
                            mini(nodes=[("C", "entity", 0.5), ("D", "entity", 0.5)])])
    chain = seed_chain(mind, ["A", "B"])
    c_d = {**seed_chain(mind, ["C"]), **seed_chain(mind, ["D"])}

    mind.perceive("A y B juntos")  # co-activación con arista → refuerza
    assert mind.graph.edge_weight(chain["A"], chain["B"]) > 1.0

    mind.perceive("C y D juntos")  # co-activación sin arista → no crea nada
    assert not mind.graph.edge_between(c_d["C"], c_d["D"])


def test_sequence_priming(make_mind):
    """A→B→C→D: activar A,B,C deja D pre-activado por encima de un control aislado."""
    config = DynamicsConfig(rounds=3)
    mind = make_mind(minis=[mini(nodes=[("A", "entity", 0.5), ("B", "entity", 0.5),
                                        ("C", "entity", 0.5)])], config=config)
    seed_chain(mind, ["A", "B", "C", "D"])
    seed_chain(mind, ["E"])  # control no conectado

    mind.perceive("A, B y C")

    assert mind.graph.by_label("D").activation > mind.graph.by_label("E").activation
    assert mind.graph.by_label("E").activation == 0


def test_homeostasis_caps_and_inhibits(make_mind):
    config = DynamicsConfig(boost=10.0, activation_budget=2.0)
    mind = make_mind(minis=[mini(nodes=[("A", "entity", 0.5), ("B", "entity", 0.5),
                                        ("C", "entity", 0.5)])], config=config)
    mind.perceive("A, B y C")

    acts = [mind.graph.by_label(x).activation for x in "ABC"]
    assert all(a <= config.activation_cap for a in acts)  # saturación
    assert mind.graph.total_activation() <= config.activation_budget + 1e-6  # inhibición


def test_imprint_survives_long_decay(make_mind):
    """Memoria flash: salience extrema consolida en el acto, sin pasar por el sueño."""
    mind = make_mind(minis=[mini(nodes=[("soy ingeniero informático", "concept", 0.95)])])
    mind.perceive("es muy importante que recuerdes que soy ingeniero informático")

    node = mind.graph.by_label("soy ingeniero informático")
    assert node.base_strength >= 0.95

    for _ in range(50):
        mind.perceive("otro tema")

    node = mind.graph.by_label("soy ingeniero informático")
    assert node.activation == 0  # la memoria de trabajo lo olvidó
    assert node.base_strength >= 0.95  # el largo plazo no


# ── Equiparación por gradiente (spread_equalizes) — las sondas de jun 2026, como tests ──────────


def test_equalize_kills_the_perpetual_motion(make_mind):
    """Dos nodos conectados, mencionados UNA vez, y la conversación sigue por temas ajenos
    (el spreading corre cada turno, como en uso real). El histórico se realimenta (suma sin
    restar) y la pareja queda caliente para siempre; la equiparación solo redistribuye,
    así que el decay siempre gana y se enfría hasta el olvido."""
    def run(config):
        others = [mini(nodes=[(f"tema{i}", "concept", 0.5)]) for i in range(25)]
        mind = make_mind(minis=[mini(nodes=[("hub", "entity", 0.5), ("a", "entity", 0.5)],
                                     edges=[("hub", "a", "tiene")])] + others, config=config)
        mind.perceive("el hub tiene a")
        for i in range(25):
            mind.perceive(f"hablemos de tema{i}")
        return mind.graph.by_label("hub").activation, mind.graph.by_label("a").activation

    hot_hub, hot_a = run(DynamicsConfig(spread_equalizes=False, boost=1.0))  # histórico: el bombeo vence al decay
    cold_hub, cold_a = run(DynamicsConfig(spread_equalizes=True))
    assert hot_hub > 1.0 and hot_a > 1.0
    assert cold_hub == 0 and cold_a == 0


def test_equalize_redistributes_without_creating_energy(make_mind):
    """El spreading solo redistribuye: la energía total no crece, ni siquiera con ciclos."""
    from oroi.core.activation import spread_activation
    config = DynamicsConfig(spread_equalizes=True, rounds=6)
    mind = make_mind(config=config)
    ids = seed_chain(mind, ["A", "B", "C"])
    mind.graph.add_edge(ids["C"], ids["A"], "cierra_ciclo", False, 0)
    mind.graph.add_activation({ids["A"]: 3.0}, 1)

    before = mind.graph.total_activation()
    spread_activation(mind.graph, config)

    assert mind.graph.total_activation() <= before + 1e-9
    assert mind.graph.by_label("B").activation > 0  # pero sí circula


def test_equalize_source_and_receiver_meet_never_cross(make_mind):
    """El que cede no cae por debajo del que recibe: se igualan, no se cruzan —
    lo percibido directamente queda siempre por encima de lo primado."""
    from oroi.core.activation import spread_activation
    config = DynamicsConfig(spread_equalizes=True)
    mind = make_mind(config=config)
    ids = seed_chain(mind, ["A", "B"])
    mind.graph.add_activation({ids["A"]: 4.0}, 1)

    spread_activation(mind.graph, config)

    a, b = mind.graph.by_label("A").activation, mind.graph.by_label("B").activation
    assert b > 0
    assert a >= b - 1e-9


def test_equalize_heat_never_flows_uphill(make_mind):
    """Un nodo tibio no inyecta energía al que ya está más caliente (el histórico sí lo hacía):
    solo el más caliente cede, también contra la flecha (atenuado)."""
    from oroi.core.activation import spread_activation
    config = DynamicsConfig(spread_equalizes=True, rounds=1)
    mind = make_mind(config=config)
    ids = seed_chain(mind, ["A", "B"])  # arista A→B, con B más caliente
    mind.graph.add_activation({ids["A"]: 1.0, ids["B"]: 3.0}, 1)

    spread_activation(mind.graph, config)

    a, b = mind.graph.by_label("A").activation, mind.graph.by_label("B").activation
    assert a > 1.0  # A solo recibe (backward atenuado desde B)
    assert b < 3.0  # B solo cede
    assert b >= a


def test_equalize_forward_only_via_spread_back_factor(make_mind):
    """spread_back_factor=0 → el calor solo viaja a favor de la flecha (variante de la
    ablación), sin tocar back_factor: la evocación conserva su backward."""
    from oroi.core.activation import spread_activation
    config = DynamicsConfig(spread_equalizes=True, spread_back_factor=0.0)
    mind = make_mind(config=config)
    ids = seed_chain(mind, ["A", "B"])  # solo la arista A→B
    mind.graph.add_activation({ids["B"]: 3.0}, 1)

    spread_activation(mind.graph, config)

    assert mind.graph.by_label("A").activation == 0  # contra la flecha, nada
    assert abs(mind.graph.by_label("B").activation - 3.0) < 1e-9  # y B no pierde lo que no fluye


def test_equalize_priming_still_decreases_along_chain(make_mind):
    """La equiparación conserva el priming en cadena: degradado con el foco al frente."""
    config = DynamicsConfig(spread_equalizes=True, rounds=3)
    mind = make_mind(minis=[mini(nodes=[("A", "entity", 0.5)])], config=config)
    seed_chain(mind, ["A", "B", "C"])

    mind.perceive("hablemos de A")

    a, b, c = (mind.graph.by_label(x).activation for x in "ABC")
    assert a > b > c > 0


def test_equalize_fact_still_recalled_next_turn(make_mind):
    """El coste de no fabricar energía, compensado: con boost recalibrado (2.0), un hecho
    mencionado UNA vez sigue aflorando en el recall al turno siguiente (sonda de junio)."""
    config = DynamicsConfig(spread_equalizes=True, boost=2.0)
    mind = make_mind(minis=[mini(nodes=[("coche", "entity", 0.5), ("rojo", "attribute", 0.5)],
                                 edges=[("coche", "rojo", "tiene_color")])], config=config)
    mind.perceive("me he comprado un coche rojo")
    mind.perceive("qué tiempo hace hoy")  # un turno de trámite por medio

    assert "coche" in mind.recall("¿de qué color es mi coche?")


def test_surprise_relative_to_the_hot_thread():
    """Sorpresa relativa: lo afín al hilo caliente apenas sorprende; lo ajeno, sí — descontando
    el 'suelo' de los embeddings midiendo contra la cohesión del propio hilo."""
    from oroi.core.activation import surprise
    from oroi.extraction.schema import MiniNetwork, MiniNode

    config = DynamicsConfig()
    hot = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]          # dos focos del MISMO hilo (cohesivo)
    afin = MiniNetwork(nodes=[MiniNode(label="x", salience=0.8)])
    giro = MiniNetwork(nodes=[MiniNode(label="y", salience=0.8)])
    s_afin = surprise(afin, {"x": [0.95, 0.05, 0.0]}, hot, 5.0, config)   # encaja en el hilo
    s_giro = surprise(giro, {"y": [0.0, 0.0, 1.0]}, hot, 5.0, config)     # ajeno al hilo
    assert s_giro > s_afin
    assert surprise(afin, {"x": [0.95, 0.05, 0.0]}, hot, 0.0, config) == 0.0  # sin expectativa, nada sorprende


def test_surprise_alert_grades_the_hook(make_mind):
    """El gancho público S1→S2 traduce el escalar de sorpresa en aviso graduado (o silencio)."""
    mind = make_mind()
    cfg = mind.config

    mind.last_surprise = 0.0
    assert mind.surprise_alert() == ""                                   # encaja: sin aviso
    mind.last_surprise = (cfg.surprise_threshold + cfg.surprise_threshold_high) / 2
    assert "se aparta" in mind.surprise_alert()                          # desviación leve
    mind.last_surprise = cfg.surprise_threshold_high + 0.1
    assert "rompe el hilo" in mind.surprise_alert()                      # ruptura marcada
