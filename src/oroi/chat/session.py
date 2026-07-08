"""Una conversación contra una Mind: orquesta percepción, recall y respuesta.

Es la ÚNICA pieza que conduce un turno de conversación, y la comparten el REPL
y el servidor web. Así ninguna interfaz mete lógica de conversación en el núcleo:
todo acceso a la mente pasa por su fachada pública (`perceive`/`recall`/`introspect`),
nunca por `core`. La instrumentación (grabar el snapshot para la visualización)
entra por un callback inyectado — la sesión no sabe nada de viz ni de HTTP.
"""

import threading

from ..core.config import DynamicsConfig
from ..mind import Mind
from ..providers.base import Chat

SYSTEM = """\
Eres un asistente conversacional con memoria asociativa de largo plazo. Cuando el \
mensaje del usuario incluye un bloque [memoria asociativa], son recuerdos tuyos de \
esta relación: úsalos con naturalidad, sin mencionar el mecanismo.
Las líneas entre corchetes [...] son señales internas tuyas (recuerdos, avisos sobre el rumbo \
de la charla): déjate guiar por ellas, pero NUNCA las menciones ni las repitas literalmente.
Tu mundo es ESTA relación: ante preguntas como «¿qué ha pasado hoy?» o «¿qué te he \
contado?», responde desde la conversación y tus recuerdos del usuario — nunca desde \
conocimiento general, actualidad inventada o noticias que no te haya dado él.
Responde en el idioma del usuario, con cercanía y MUY CORTO: 1-3 frases, salvo que \
te pidan detalle explícitamente."""


class Window:
    """Ventana conversacional: crece y se TRUNCA en bloque, nunca desliza (SPEC §5)."""

    def __init__(self, config: DynamicsConfig):
        self.config = config
        self.turns: list[tuple[int, str, str]] = []  # (turno, rol, texto)

    def add(self, turn: int, role: str, text: str) -> None:
        self.turns.append((turn, role, text))
        spoken_turns = {t for t, _, _ in self.turns}
        if len(spoken_turns) > self.config.window_max_turns:
            keep_from = sorted(spoken_turns)[-self.config.window_keep_turns]
            self.turns = [t for t in self.turns if t[0] >= keep_from]

    def messages(self) -> list[tuple[str, str]]:
        return [(role, text) for _, role, text in self.turns]

    def turn_numbers(self) -> set[int]:
        return {t for t, _, _ in self.turns}


class ChatSession:
    """Un turno: percibe (respuesta previa + mensaje), recall, y responde el Conversador.

    `on_turn(snapshot)` es instrumentación opcional tras percibir (la viz graba ahí).
    `on_thought(thought)` recibe la voz (consciencia de solo lectura) cuando
    `consciousness_enabled` está activo — asíncrona: el turno nunca la espera.
    Los turnos se serializan con un lock: el REPL es secuencial, la web concurrente.
    """

    def __init__(self, mind: Mind, chat: Chat, system: str = SYSTEM, on_turn=None,
                 on_thought=None):
        self.mind = mind
        self.chat = chat
        self.system = system
        self.on_turn = on_turn
        self.on_thought = on_thought
        self.window = Window(mind.config)
        self.last_reply = ""
        self._lock = threading.Lock()
        self._voice_thread: threading.Thread | None = None

    def turn(self, user_text: str) -> str:
        with self._lock:
            # Una sola extracción por turno completo: respuesta previa + mensaje actual (SPEC §5).
            perceived = (f"[asistente] {self.last_reply}\n[usuario] {user_text}"
                         if self.last_reply else user_text)
            self.mind.perceive(perceived)
            if self.on_turn:
                self.on_turn(self.mind.introspect())
            memory = self.mind.recall(user_text, window_turns=self.window.turn_numbers())
            alert = self.mind.surprise_alert()                      # señal interna S1→S2, graduada
            if alert:
                memory = f"{alert}\n{memory}" if memory else alert
            reply = self.chat.reply(self.system, self.window.messages(), memory, user_text)
            turn = self.mind.turn
            self.window.add(turn, "user", user_text)
            self.window.add(turn, "assistant", reply)
            self.last_reply = reply
            self._voice(turn)
            return reply

    def _voice(self, turn: int) -> None:
        """La voz, fuera del camino caliente: hilo daemon tras entregar la respuesta.
        Si el pensamiento anterior sigue en curso, este ciclo se salta (nunca se encolan)."""
        config = self.mind.config
        if not config.consciousness_enabled or turn % config.consciousness_every != 0:
            return
        if self._voice_thread and self._voice_thread.is_alive():
            return

        def _reflect():
            thought = self.mind.consciousness()  # nunca lanza; None si la red está fría
            if thought and self.on_thought:
                self.on_thought(thought)

        self._voice_thread = threading.Thread(target=_reflect, daemon=True)
        self._voice_thread.start()
