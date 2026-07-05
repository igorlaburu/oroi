"""REPL de consola mínimo: el experimento conversando (SPEC §7, Fase 2).

input()/print() a propósito — una interfaz; la lógica de conversación vive en
session.py (compartida con el servidor web) y todo acceso a la mente pasa por
la fachada Mind. Aquí solo está el bucle de consola y el cableado de providers.
"""

import threading
import time

from ..extraction.extractor import TurnExtractor
from ..mind import Mind
from ..providers.anthropic_chat import ClaudeChat
from ..providers.azure import AzureChat, AzureEmbedder, AzureLLM
from ..providers.base import Chat
from ..providers.settings import ProviderSettings
from ..viz import graph_view
from .session import ChatSession

DB_PATH = "mind.db"


def build_mind(db_path: str = DB_PATH) -> Mind:
    settings = ProviderSettings()
    if settings.memory_provider == "openai":
        from ..providers.openai_compat import OpenAICompatEmbedder, OpenAICompatLLM
        llm = OpenAICompatLLM(settings)
        return Mind(db_path, OpenAICompatEmbedder(settings), TurnExtractor(llm), judge=llm)
    azure = AzureLLM(settings)
    return Mind(db_path, AzureEmbedder(settings), TurnExtractor(azure), judge=azure)


def build_chat(settings: ProviderSettings) -> Chat:
    if settings.chat_provider == "azure":
        return AzureChat(settings)
    if settings.chat_provider == "openai":
        from ..providers.openai_compat import OpenAICompatChat
        return OpenAICompatChat(settings)
    if settings.claude_auth == "session":
        from ..providers.claude_session import ClaudeSessionChat
        return ClaudeSessionChat(settings)
    return ClaudeChat(settings)


class IdleSleeper(threading.Thread):
    """Como los humanos: la mente se duerme cuando la conversación se queda quieta.

    `touch()` en cada turno marca actividad; pasado `idle_seconds` sin tocar,
    un único ciclo de sueño en segundo plano — y no vuelve a dormir hasta que
    haya turnos nuevos que consolidar.
    """

    def __init__(self, mind: Mind, idle_seconds: float, poll: float = 5.0):
        super().__init__(daemon=True)
        self.mind, self.idle, self.poll = mind, idle_seconds, poll
        self.last_activity = time.monotonic()
        self.pending = False

    def touch(self) -> None:
        self.last_activity = time.monotonic()
        self.pending = True

    def nap_due(self) -> bool:
        return self.pending and time.monotonic() - self.last_activity >= self.idle

    def run(self) -> None:
        while True:
            time.sleep(self.poll)
            if self.nap_due():
                self.pending = False
                self.mind.sleep()


def main() -> None:
    mind = build_mind()
    mind.wake()
    sleeper = IdleSleeper(mind, mind.config.idle_sleep_seconds)
    sleeper.start()
    journal = graph_view.timeline_path(DB_PATH)
    session = ChatSession(mind, build_chat(ProviderSettings()),
                          on_turn=lambda snap: graph_view.record(snap, journal))
    print("oroi · chat experimental (escribe /salir para terminar)\n")
    while True:
        try:
            user_text = input("tú> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text or user_text in {"/salir", "/exit"}:
            break
        try:
            reply = session.turn(user_text)
        except Exception as error:  # un turno fallido (timeout, red) no tira la sesión
            print(f"\nmente> (me he quedado en blanco: {error})\n")
            continue
        print(f"\nmente> {reply}\n")
        sleeper.touch()
    if sleeper.pending:  # al despedirse, la mente duerme lo que quedara fresco
        print("(consolidando recuerdos antes de salir…)")
        try:
            mind.sleep()
        except Exception as error:
            print(f"(el sueño falló, queda para la próxima: {error})")


if __name__ == "__main__":
    main()
