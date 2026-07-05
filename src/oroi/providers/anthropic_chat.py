"""El Conversador: Claude Opus por llamada directa al API de Anthropic."""

import anthropic

from .settings import ProviderSettings


class ClaudeChat:
    def __init__(self, settings: ProviderSettings):
        # Sin api_key explícita, el SDK resuelve credenciales del entorno.
        # Timeout corto: una petición colgada no debe congelar el REPL (ver azure.py).
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None,
                                          timeout=120.0, max_retries=1)
        self.model = settings.claude_model

    def reply(self, system: str, window: list[tuple[str, str]], memory: str, user_text: str) -> str:
        """Responde con la ventana de turnos literales + la memoria inyectada en el turno de usuario.

        La memoria va en el último mensaje de usuario, nunca en el system prompt:
        el prefijo (system + ventana) queda estable para el caché (SPEC §5).
        """
        content = f"{memory}\n\n{user_text}" if memory else user_text
        messages = [{"role": role, "content": text} for role, text in window]
        messages.append({"role": "user", "content": content})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=messages,
        )
        return "".join(block.text for block in response.content if block.type == "text")
