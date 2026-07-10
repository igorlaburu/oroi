"""La voz (consciencia de solo lectura): observador, jamás actor (plan jul 2026, §1).

Los tests garantizan el contrato: solo-lectura sobre el grafo, invarianza con el
flag apagado, cadencia por turnos, sorpresa desde la física, robustez (la voz nunca
rompe un turno) y anti-eco (su prosa jamás llega a los episodios).
"""

import json

from oroi import DynamicsConfig
from oroi.chat.session import ChatSession
from oroi.consciousness import coalition

from .conftest import mini, seed_chain


class FakeVoice:
    """LLM rápido enlatado para la voz: respuesta fija y contador de llamadas."""

    def __init__(self, response: dict | str | None = None):
        self.calls = 0
        self.last_material = ""
        self.response = response if response is not None else {"text": "pienso en ello", "valence": 1}

    def complete_json(self, system: str, user: str) -> str:
        self.calls += 1
        self.last_material = user
        return self.response if isinstance(self.response, str) else json.dumps(self.response)


class FakeChat:
    def reply(self, system, window, memory, user_text) -> str:
        return "vale"


def _activate(mind, ids: dict[str, int], levels: dict[str, float]) -> None:
    for label, value in levels.items():
        mind.graph.set_activation(ids[label], value)


def test_coalition_is_text_only_with_relative_floor(make_mind):
    config = DynamicsConfig(consciousness_top_k=5)
    mind = make_mind(config=config)
    ids = seed_chain(mind, ["bilbao", "mudanza", "batman"])
    # batman queda bajo el suelo RELATIVO al líder (3.0 · 0.35 > 0.5), aun cabiendo en top_k.
    _activate(mind, ids, {"bilbao": 3.0, "mudanza": 2.0, "batman": 0.5})
    episode = mind.graph.add_episode(1, "user", "trabajé años en bilbao")
    mind.graph.link_episode([ids["bilbao"]], episode)

    material, chain = coalition(mind.graph, config)

    assert chain == ["bilbao", "mudanza"]            # el foco caliente, de más a menos
    assert "batman" not in material                  # cortado por el suelo relativo
    assert "-->" not in material                     # SIN aristas: la voz piensa desde el texto
    assert "trabajé años en bilbao" in material      # el recuerdo literal es el contenido


def test_coalition_keeps_a_pair_to_weave(make_mind):
    """El suelo relativo nunca deja al líder solo si hay un segundo nodo (par para hilar)."""
    config = DynamicsConfig()
    mind = make_mind(config=config)
    ids = seed_chain(mind, ["lider", "debil"])
    _activate(mind, ids, {"lider": 5.0, "debil": 0.2})   # débil MUY por debajo del suelo
    episode = mind.graph.add_episode(1, "user", "hablamos del líder")
    mind.graph.link_episode([ids["lider"]], episode)

    _, chain = coalition(mind.graph, config)
    assert chain == ["lider", "debil"]


def test_consciousness_is_read_only(make_mind):
    voice = FakeVoice()
    mind = make_mind(judge=voice)
    ids = seed_chain(mind, ["a", "b"])
    _activate(mind, ids, {"a": 2.0, "b": 1.0})

    def state():
        nodes = mind.graph.db.execute(
            "SELECT id, activation, base_strength, salience FROM nodes ORDER BY id").fetchall()
        edges = mind.graph.db.execute("SELECT id, weight FROM edges ORDER BY id").fetchall()
        return [tuple(r) for r in nodes], [tuple(r) for r in edges]

    before = state()
    thought = mind.consciousness()
    assert thought is not None and thought.chain == ["a", "b"]
    assert state() == before                         # ni un bit del grafo cambia
    assert len(mind.thoughts()) == 1                 # lo único que crece es el diario


def test_disabled_flag_means_no_voice_calls(make_mind):
    voice = FakeVoice()
    mind = make_mind(minis=[mini(nodes=[("casa", "entity", 0.5)])], judge=voice)
    session = ChatSession(mind, FakeChat())          # consciousness_enabled=False (defecto)
    session.turn("hola")
    session.turn("sigo aquí")
    assert voice.calls == 0                          # invarianza: sin flag no hay llamada extra


def test_cadence_fires_every_n_turns(make_mind):
    voice = FakeVoice()
    minis = [mini(nodes=[(f"n{i}", "entity", 0.5)]) for i in range(4)]
    config = DynamicsConfig(consciousness_enabled=True, consciousness_every=2)
    mind = make_mind(minis=minis, config=config, judge=voice)
    heard = []
    session = ChatSession(mind, FakeChat(), on_thought=heard.append)
    for i in range(4):
        session.turn(f"mensaje {i}")
        if session._voice_thread:
            session._voice_thread.join()             # el test espera; la conversación jamás
    assert voice.calls == 2                          # turnos 2 y 4
    assert [t.turn for t in heard] == [2, 4]


def test_surprise_flag_mirrors_the_physics(make_mind):
    mind = make_mind(judge=FakeVoice())
    ids = seed_chain(mind, ["a"])
    _activate(mind, ids, {"a": 1.0})
    mind.last_surprise = mind.config.surprise_threshold + 0.01
    assert mind.consciousness().surprise is True
    mind.last_surprise = 0.0
    assert mind.consciousness().surprise is False


def _with_episode(mind, ids: dict[str, int], text: str) -> None:
    episode = mind.graph.add_episode(1, "user", text)
    mind.graph.link_episode(ids.values(), episode)


def test_voice_never_breaks_valence_clamped_and_bad_json_is_none(make_mind):
    mind = make_mind(judge=FakeVoice({"text": "x", "valence": 7}))
    ids = seed_chain(mind, ["a", "b"])
    _activate(mind, ids, {"a": 1.0, "b": 0.5})
    _with_episode(mind, ids, "a y b")                # con recuerdo: material real → pasa por el LLM
    assert mind.consciousness().valence == 2         # clamp a [-2, 2]

    mind_bad = make_mind(judge=FakeVoice("esto no es json"))
    ids = seed_chain(mind_bad, ["a", "b"])
    _activate(mind_bad, ids, {"a": 1.0, "b": 0.5})
    _with_episode(mind_bad, ids, "a y b")
    assert mind_bad.consciousness() is None          # sin excepción: la voz no rompe nada
    assert mind_bad.thoughts() == []


def test_bare_concepts_get_a_mechanical_thought_without_llm(make_mind):
    """Conceptos sueltos (sin asociaciones ni recuerdos): pensamiento mecánico, cero LLM —
    anclado por construcción (la sonda real mostró que el LLM rellena con vivencias)."""
    voice = FakeVoice()
    mind = make_mind(judge=voice)
    ids = seed_chain(mind, ["gato"])                 # un solo nodo: ni arista ni episodio
    _activate(mind, ids, {"gato": 1.0})
    thought = mind.consciousness()
    assert thought.text == "tengo en la cabeza: gato"
    assert voice.calls == 0                          # ni una llamada: gratis y sin inventos


def test_voice_prose_never_reaches_episodes(make_mind):
    marker = "ZWXQ_PENSAMIENTO_INTERNO"
    voice = FakeVoice({"text": f"{marker} sobre la charla", "valence": 0})
    minis = [mini(nodes=[(f"n{i}", "entity", 0.5)]) for i in range(3)]
    config = DynamicsConfig(consciousness_enabled=True)
    mind = make_mind(minis=minis, config=config, judge=voice)
    session = ChatSession(mind, FakeChat())
    for i in range(3):
        session.turn(f"mensaje {i}")
        if session._voice_thread:
            session._voice_thread.join()
    assert voice.calls >= 1                          # la voz pensó de verdad
    echoes = mind.graph.db.execute(
        "SELECT COUNT(*) c FROM episodes WHERE text LIKE ?", (f"%{marker}%",)).fetchone()["c"]
    assert echoes == 0                               # y su prosa jamás se percibió (anti-eco)
