"""Proveedor OpenAI-compatible: OpenAI directo, Ollama, vLLM o cualquier endpoint /v1.

Con `openai_base_url` vacío habla con api.openai.com (requiere OPENAI_API_KEY); apuntándolo
a un servidor propio (p. ej. Ollama: http://localhost:11434/v1) funciona sin clave real.
Probado contra Ollama (chat + JSON mode); los embeddings requieren un modelo de embeddings
servido en el mismo endpoint (p. ej. `nomic-embed-text`), con su dimensión en
`openai_embedding_dim`.
"""

from openai import OpenAI

from .settings import ProviderSettings

FAST_TIMEOUT = 60.0
CHAT_TIMEOUT = 120.0


def _client(settings: ProviderSettings, timeout: float) -> OpenAI:
    return OpenAI(
        api_key=settings.openai_api_key or "sin-clave",  # Ollama y compañía la ignoran
        base_url=settings.openai_base_url or None,
        timeout=timeout, max_retries=1,
    )


class OpenAICompatLLM:
    """Una sola operación: completar en JSON con el modelo rápido (extractor y juez)."""

    def __init__(self, settings: ProviderSettings):
        self.client = _client(settings, FAST_TIMEOUT)
        self.deployment = settings.openai_fast_model

    def complete_json(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or "{}"


class OpenAICompatEmbedder:
    """Embeddings en lote. La dimensión se declara en config (los modelos varían)."""

    def __init__(self, settings: ProviderSettings):
        self.client = _client(settings, FAST_TIMEOUT)
        self.model = settings.openai_embedding_model
        self.dim = settings.openai_embedding_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class OpenAICompatChat:
    """El Conversador (CHAT_PROVIDER=openai): misma composición que AzureChat/ClaudeChat —
    la memoria va en el último turno de usuario y el prefijo queda estable para el caché."""

    def __init__(self, settings: ProviderSettings):
        self.client = _client(settings, CHAT_TIMEOUT)
        self.deployment = settings.openai_chat_model

    def reply(self, system: str, window: list[tuple[str, str]], memory: str, user_text: str) -> str:
        content = f"{memory}\n\n{user_text}" if memory else user_text
        messages = [{"role": "system", "content": system}]
        messages += [{"role": role, "content": text} for role, text in window]
        messages.append({"role": "user", "content": content})
        response = self.client.chat.completions.create(model=self.deployment, messages=messages)
        return response.choices[0].message.content or ""
