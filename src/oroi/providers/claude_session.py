"""El Conversador usando la sesión local de Claude Code (cuenta Max), sin API key.

Cada turno es una consulta de un solo paso, sin herramientas: el contexto que ve
el modelo es exactamente el que construye el experimento (ventana + memoria),
nunca un historial interno del SDK — si no, la red nunca trabajaría.

La sesión se lanza HERMÉTICA: cwd neutro y sin fuentes de configuración, para
que no cargue CLAUDE.md, memoria de proyecto ni settings del directorio desde
el que corre el REPL — esa fuga contaminaría el experimento.
"""

import asyncio
import tempfile

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

from .settings import ProviderSettings


class ClaudeSessionChat:
    def __init__(self, settings: ProviderSettings):
        self.model = settings.claude_model

    def reply(self, system: str, window: list[tuple[str, str]], memory: str, user_text: str) -> str:
        prompt = self._compose(window, memory, user_text)
        options = ClaudeAgentOptions(
            system_prompt=system, model=self.model, allowed_tools=[], max_turns=1,
            cwd=tempfile.gettempdir(), setting_sources=[],
        )
        return asyncio.run(self._ask(prompt, options))

    @staticmethod
    async def _ask(prompt: str, options: ClaudeAgentOptions) -> str:
        parts = []
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                parts.extend(b.text for b in message.content if isinstance(b, TextBlock))
        return "".join(parts)

    @staticmethod
    def _compose(window: list[tuple[str, str]], memory: str, user_text: str) -> str:
        """Serializa ventana + memoria + turno actual en un único prompt de un paso."""
        lines = [f"[{role}] {text}" for role, text in window]
        if memory:
            lines.append(memory)
        lines.append(f"[user] {user_text}")
        lines.append("Responde directamente al último mensaje del usuario.")
        return "\n\n".join(lines)
