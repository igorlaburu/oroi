"""La sesión de conversación: orquesta Mind + Conversador sin tocar el núcleo (Fase 4)."""

from oroi.chat.session import ChatSession, Window
from oroi.core.config import DynamicsConfig
from tests.conftest import mini


class EchoChat:
    """Conversador de prueba: devuelve lo que ve, para inspeccionar qué se le inyecta."""

    def __init__(self):
        self.calls = []

    def reply(self, system, window, memory, user_text):
        self.calls.append({"window": window, "memory": memory, "user_text": user_text})
        return f"recibido: {user_text}"


def test_turn_perceives_recalls_and_replies(make_mind):
    mind = make_mind(minis=[mini(nodes=[("coche", "entity", 0.6), ("rojo", "attribute", 0.5)],
                                edges=[("coche", "rojo", "tiene_color")])])
    chat = EchoChat()
    snapshots = []
    session = ChatSession(mind, chat, system="S", on_turn=snapshots.append)

    reply = session.turn("mi coche es rojo")

    assert reply == "recibido: mi coche es rojo"
    assert mind.turn == 1                       # percibió un turno (acceso normalizado)
    assert snapshots and snapshots[0].turn == 1  # instrumentación: snapshot tras percibir
    assert "coche tiene_color rojo" in chat.calls[0]["memory"]  # recall inyectado al Conversador


def test_previous_reply_folds_into_next_perception(make_mind):
    mind = make_mind(minis=[mini(nodes=[("a", "entity", 0.5)]), mini(nodes=[("b", "entity", 0.5)])])
    session = ChatSession(mind, EchoChat(), system="S")

    session.turn("hola")
    session.turn("¿y bien?")

    # El segundo turno percibe la respuesta previa + el mensaje nuevo (una sola extracción, SPEC §5).
    assert session.window.messages()[-2:] == [("user", "¿y bien?"), ("assistant", "recibido: ¿y bien?")]


def test_window_truncates_in_block_not_sliding():
    config = DynamicsConfig(window_max_turns=4, window_keep_turns=2)
    window = Window(config)
    for turn in range(1, 6):
        window.add(turn, "user", f"t{turn}")
    kept = sorted(window.turn_numbers())
    assert kept == [4, 5]  # al superar el máximo, conserva solo los últimos keep_turns
