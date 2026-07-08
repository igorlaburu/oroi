"""REPL de consola mínimo: el experimento conversando (SPEC §7, Fase 2).

input()/print() a propósito — una interfaz; la lógica de conversación vive en
session.py (compartida con el servidor web) y todo acceso a la mente pasa por
la fachada Mind. Aquí solo está el bucle de consola y el cableado de providers.
"""

import sys
import threading
import time
from pathlib import Path

from ..extraction.extractor import TurnExtractor
from ..mind import Mind
from ..providers.anthropic_chat import ClaudeChat
from ..providers.azure import AzureChat, AzureEmbedder, AzureLLM
from ..providers.base import Chat
from ..providers.settings import ProviderSettings
from ..viz import graph_view
from .session import ChatSession

def default_db() -> str:
    """La memoria vive en ~/.oroi/memoria.db salvo que se pida otra: abrir la base que
    haya en la carpeta de turno sería una emboscada (charlarías con una memoria ajena
    sin saberlo). Explícito > implícito."""
    home = Path.home() / ".oroi"
    home.mkdir(exist_ok=True)
    return str(home / "mind.db")


def build_mind(db_path: str | None = None) -> Mind:
    db_path = db_path or default_db()
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


def main(db_path: str | None = None, voice: bool = False) -> None:
    settings = ProviderSettings()
    if not settings.has_credentials():
        sys.exit("oroi: no encuentro credenciales de ningún proveedor.\n"
                 "  Copia .env.example a .env (en esta carpeta o en ~/.oroi/) y rellena tus claves\n"
                 "  — Azure OpenAI, o un endpoint OpenAI-compatible como Ollama para correr en local.\n"
                 "  Guía: https://github.com/igorlaburu/oroi#instalación")
    db_path = db_path or default_db()
    mind = build_mind(db_path)
    mind.wake()
    sleeper = IdleSleeper(mind, mind.config.idle_sleep_seconds)
    sleeper.start()
    journal = graph_view.timeline_path(db_path)
    session = ChatSession(mind, build_chat(settings),
                          on_turn=lambda snap: graph_view.record(snap, journal))
    print(f"oroi · memoria: {db_path} · {mind.turn} turnos vividos")
    print("chat experimental (escribe /salir para terminar)\n")
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
            print(f"\nasistente> (me he quedado en blanco: {error})\n")
            continue
        print(f"\nasistente> {reply}\n")
        if voice:  # en el REPL la voz se pide en el sitio (síncrona): salida limpia, sin carreras
            _print_voice(mind)
        sleeper.touch()
    if sleeper.pending:  # al despedirse, la mente duerme lo que quedara fresco
        print("(consolidando recuerdos antes de salir…)")
        try:
            mind.sleep()
        except Exception as error:
            print(f"(el sueño falló, queda para la próxima: {error})")


def _print_voice(mind: Mind) -> None:
    """La voz atenuada bajo la respuesta: qué tiene la mente en mente, con su valencia."""
    thought = mind.consciousness()
    if thought:
        mark = " · ⚡giro" if thought.surprise else ""
        print(f"\033[2m〔 voz · valencia {thought.valence:+d}{mark} · {thought.text} 〕\033[0m\n")


if __name__ == "__main__":
    main()
