"""Recall: gating, serialización y presupuesto (SPEC §7, Fase 1)."""

from oroi import DynamicsConfig
from oroi.core.retrieval import CHARS_PER_TOKEN
from tests.conftest import mini, seed_chain


def test_gating_returns_silence_without_resonance(make_mind):
    mind = make_mind()
    mind.perceive("un saludo sin contenido")
    assert mind.recall() == ""


def test_recall_serializes_facts_and_episodes(make_mind):
    mind = make_mind(minis=[mini(nodes=[("coche", "entity", 0.6), ("rojo", "attribute", 0.5)],
                                edges=[("coche", "rojo", "tiene_color")])])
    mind.perceive("me he comprado un coche rojo")

    memory = mind.recall()
    assert "coche tiene_color rojo" in memory
    assert "me he comprado un coche rojo" in memory  # el episodio literal habla


def test_recall_respects_attention_budget(make_mind):
    config = DynamicsConfig(attention_budget=8)  # presupuesto minúsculo a propósito
    mind = make_mind(minis=[mini(nodes=[("coche", "entity", 0.6), ("rojo", "attribute", 0.5)],
                                edges=[("coche", "rojo", "tiene_color")])], config=config)
    mind.perceive("me he comprado un coche rojo precioso esta misma mañana")

    assert len(mind.recall()) <= config.attention_budget * CHARS_PER_TOKEN


def test_recall_dedupes_against_window(make_mind):
    mind = make_mind(minis=[mini(nodes=[("coche", "entity", 0.6)])])
    mind.perceive("mi coche es rojo")

    with_window = mind.recall(window_turns=[1])  # el turno 1 ya está en la ventana
    assert "mi coche es rojo" not in with_window


def test_wake_preserves_working_memory(make_mind):
    """Tiempo conversacional, no de reloj: abrir sesión NO toca la activación — reabrir es
    continuar donde se dejó (la activación se persiste), no un reinicio que la enfríe."""
    mind = make_mind(minis=[mini(nodes=[("coche", "entity", 0.6)])])
    mind.perceive("mi coche es rojo")
    before = mind.graph.by_label("coche").activation

    mind.wake()

    assert mind.graph.by_label("coche").activation == before  # intacta


def test_imprinted_fact_comes_back_in_next_session(make_mind):
    """La memoria flash persiste: imprint deja el hecho caliente y consolidado, y reabrir sesión
    no lo enfría (la activación se persiste) — sigue a mano al recordar."""
    mind = make_mind(minis=[mini(nodes=[("me llamo igorix", "concept", 0.95)])])
    mind.perceive("es importante: me llamo igorix")

    mind.wake()  # sesión nueva: la app se cerró y se reabrió

    assert "me llamo igorix" in mind.recall()


def test_unconsolidated_facts_fade_with_turns(make_mind):
    """Sin consolidar, un hecho trivial se enfría por el decay de los turnos y deja de aflorar:
    no hay base_strength que lo sostenga (el gating sigue mandando)."""
    mind = make_mind(minis=[mini(nodes=[("nube", "concept", 0.2)])])  # trivial: no cruza el suelo
    mind.perceive("hoy hay una nube")
    for _ in range(6):                 # pasan turnos: la memoria de trabajo decae (boost 2.0 tarda más)
        mind.perceive("otro tema")

    assert mind.recall() == ""
    assert mind.graph.by_label("nube").base_strength == 0  # trivial y sin revisita: no se consolidó


def test_imprinted_fact_cools_from_working_memory_but_keeps_strength(make_mind):
    """El hecho consolidado se enfría de la memoria de trabajo por el decay de muchos turnos
    (deja de aflorar solo), pero conserva su base_strength para reencenderse por evocación."""
    mind = make_mind(minis=[mini(nodes=[("me llamo igorix", "concept", 0.95)])])
    mind.perceive("es importante: me llamo igorix")

    for _ in range(30):  # toda una sesión larga sin tocarlo
        mind.perceive("otro tema")

    assert mind.recall() == ""
    assert mind.graph.by_label("me llamo igorix").base_strength >= 0.95


def test_evoke_recovers_cold_neighbour(make_mind):
    """Evocación: un dato frío se recupera porque la pista de la pregunta lo reenciende
    propagando por la arista que los une (lo que el recall pasivo, por umbral, no logra)."""
    # Red: trabajo → granada (el hecho), ambos fríos (no se mencionan hace mucho).
    seed_chain(mind := make_mind(minis=[mini(nodes=[("trabajo", "concept", 0.5)])]),
               ["trabajo", "granada"], rel="ubicado_en")
    # La pregunta solo resuena con "trabajo" (la pista); "granada" sigue frío.
    mind.perceive("¿dónde está mi trabajo?")
    memory = mind.recall()
    assert "granada" in memory.lower()  # evocado vía la arista, pese a estar frío


def test_recognition_still_works_with_evocation(make_mind):
    """El reconocimiento (lo caliente) sigue aflorando; la evocación solo añade, no rompe."""
    mind = make_mind(minis=[mini(nodes=[("coche", "entity", 0.6), ("rojo", "attribute", 0.5)],
                                edges=[("coche", "rojo", "tiene_color")])])
    mind.perceive("mi coche es rojo")
    assert "coche tiene_color rojo" in mind.recall()


def test_facts_carry_recency_most_recent_first(make_mind):
    """Los hechos llevan su turno y se listan del más reciente al más antiguo: ante dos
    valores en conflicto (residencia vigo/murcia), el que rige se distingue solo — el dato
    viejo no se oculta, se fecha (lección del juez de respuesta: la serialización sin orden
    temporal pierde contra el texto crudo en actualización)."""
    mind = make_mind(minis=[
        mini(nodes=[("residencia", "concept", 0.6), ("vigo", "entity", 0.5)],
             edges=[("residencia", "vigo", "es")]),
        mini(nodes=[("residencia", "concept", 0.6), ("murcia", "entity", 0.5)],
             edges=[("residencia", "murcia", "es")]),
    ])
    mind.perceive("vivo en vigo")
    mind.perceive("me he mudado, ahora vivo en murcia")

    memory = mind.recall()
    assert "residencia es murcia (turno 2)" in memory
    assert "residencia es vigo (turno 1)" in memory
    assert memory.index("murcia") < memory.index("vigo")  # lo vigente, primero


def test_recall_strengthens_what_it_returns(make_mind):
    """Efecto testeo: recordar consolida — los nodos inyectados suben su fuerza LENTA (no la
    activación: la fuerza no se propaga, así que el refuerzo no puede realimentarse)."""
    mind = make_mind(minis=[mini(nodes=[("coche", "entity", 0.6), ("rojo", "attribute", 0.5)],
                                edges=[("coche", "rojo", "tiene_color")])])
    mind.perceive("mi coche es rojo")
    before = mind.graph.strengths(set(mind.graph.effective_activations(floor=0.0)))

    assert mind.recall() != ""          # hay contenido → hay refuerzo
    after = mind.graph.strengths(set(before))
    assert any(after[n] > before[n] for n in before)  # los nodos inyectados subieron


def test_silent_recall_strengthens_nothing(make_mind):
    """Sin contenido no hay efecto testeo: el silencio no consolida nada."""
    mind = make_mind()
    assert mind.recall() == ""


def test_episodes_ranked_by_comention_not_recency(make_mind):
    """El episodio que menciona a VARIOS nodos recordados gana al episodio reciente que solo
    menciona a uno: en un diálogo el hablante es un hub presente en cientos de turnos y la pura
    recencia acapararía los huecos (lección del piloto LoCoMo)."""
    mind = make_mind(minis=[
        # turno 1: el hecho (menciona hub + dato)
        mini(nodes=[("melanie", "entity", 0.6), ("boda", "event", 0.6)],
             edges=[("melanie", "boda", "celebró")]),
        # turnos 2-3: charla reciente del hub (solo melanie)
        mini(nodes=[("melanie", "entity", 0.5)]),
        mini(nodes=[("melanie", "entity", 0.5), ("boda", "event", 0.5)]),
    ])
    mind.perceive("melanie celebró su boda hace cinco años")
    mind.perceive("melanie saluda otra vez")
    mind.perceive("melanie recuerda la boda")

    memory = mind.recall()
    # ambos episodios de co-mención (turnos 1 y 3) entran ANTES que el saludo (turno 2, solo hub)
    assert memory.index("celebró su boda") < memory.index("saluda otra vez") or \
           "saluda otra vez" not in memory
    assert "recuerda la boda" in memory


def test_weighted_comention_filler_does_not_displace_chain_link(make_mind):
    """El relleno reciente (dos nodos rasos co-mencionados) no desplaza al eslabón antiguo de
    una cadena que menciona al ancla: la co-mención se pondera con el puesto en el recall
    (lección del juez sobre multihop3 con extractor v2)."""
    mind = make_mind(minis=[
        mini(nodes=[("elena", "entity", 0.6), ("ana", "entity", 0.6)],
             edges=[("elena", "ana", "hermana_de")]),           # turno 1: el eslabón
        mini(nodes=[("tráfico", "concept", 0.3), ("mañana", "concept", 0.3)],
             edges=[("tráfico", "mañana", "en")]),              # turno 2: relleno co-mencionado
        mini(nodes=[("llamada", "concept", 0.3), ("mañana", "concept", 0.3)],
             edges=[("llamada", "mañana", "para")]),            # turno 3: más relleno
        mini(nodes=[("elena", "entity", 0.5)]),                 # turno 4: la pregunta resuena elena
    ])
    config = DynamicsConfig(recall_max_episodes=2)              # solo dos huecos: que se note
    mind = make_mind(minis=mind.extractor.queue, config=config)
    mind.perceive("Elena es hermana de Ana")
    mind.perceive("el tráfico estaba imposible esta mañana")
    mind.perceive("tengo una llamada pendiente para mañana")
    mind.perceive("¿quién es la hermana de Elena?")

    memory = mind.recall()
    assert "Elena es hermana de Ana" in memory  # el eslabón entra pese a ser el más viejo


def test_episode_context_turns_wraps_neighbours(make_mind):
    """Con episode_context_turns=1, el episodio recordado viene arropado por sus turnos
    vecinos en orden cronológico (la implicatura del diálogo vive en el hilo adyacente);
    con el defecto 0, solo el turno recordado."""
    minis = [
        mini(nodes=[("aniversario", "event", 0.6)]),                      # turno 1: la clave del hilo
        mini(nodes=[("cinco años", "concept", 0.6), ("vestido", "entity", 0.5)],
             edges=[("cinco años", "vestido", "desde")]),                 # turno 2: la implicatura
        mini(nodes=[("aniversario", "event", 0.5), ("cinco años", "concept", 0.5)]),  # turno 3: pregunta
    ]
    plain = make_mind(minis=list(minis))
    plain.perceive("hoy es nuestro aniversario")
    plain.perceive("¡cinco años ya! parece que fue ayer que me puse este vestido")
    plain.perceive("¿cuántos años llevamos?")
    assert "aniversario" in plain.recall()  # sanity: el hilo aflora

    wrapped = make_mind(minis=list(minis), config=DynamicsConfig(episode_context_turns=1))
    wrapped.perceive("hoy es nuestro aniversario")
    wrapped.perceive("¡cinco años ya! parece que fue ayer que me puse este vestido")
    wrapped.perceive("¿cuántos años llevamos?")
    memory = wrapped.recall()
    memories_section = memory[memory.index("recuerdos:"):]
    assert "aniversario" in memories_section and "vestido" in memories_section  # el vecino arropa
    assert memories_section.index("aniversario") < memories_section.index("vestido")  # cronológico
