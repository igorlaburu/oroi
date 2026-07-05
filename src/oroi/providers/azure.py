"""Clientes finos sobre Azure OpenAI: chat JSON (extractor) y embeddings."""

from openai import AzureOpenAI

from .settings import ProviderSettings

EMBEDDING_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

# El SDK trae 600 s + reintentos por defecto: una petición colgada congela el REPL
# durante muchos minutos. Mejor fallar pronto y que el turno siga su vida.
FAST_TIMEOUT = 60.0   # extractor y embeddings: tareas cortas
CHAT_TIMEOUT = 120.0  # el Conversador puede pensar más


class AzureLLM:
    """Una sola operación: completar en JSON con el deployment rápido (gpt-4o-mini)."""

    def __init__(self, settings: ProviderSettings):
        self.client = AzureOpenAI(
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
            azure_endpoint=settings.azure_api_base,
            timeout=FAST_TIMEOUT, max_retries=1,
        )
        self.deployment = settings.azure_fast_deployment

    def complete_json(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or "{}"


class AzureChat:
    """El Conversador por Azure OpenAI (CHAT_PROVIDER=azure, deployment "smart": gpt-5.4).

    Misma composición que ClaudeChat: la memoria va en el último turno de usuario,
    el prefijo (system + ventana) queda estable para el caché de prompts (SPEC §5).
    """

    def __init__(self, settings: ProviderSettings):
        self.client = AzureOpenAI(
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
            azure_endpoint=settings.azure_api_base,
            timeout=CHAT_TIMEOUT, max_retries=1,
        )
        self.deployment = settings.azure_smart_deployment

    def reply(self, system: str, window: list[tuple[str, str]], memory: str, user_text: str) -> str:
        content = f"{memory}\n\n{user_text}" if memory else user_text
        messages = [{"role": "system", "content": system}]
        messages += [{"role": role, "content": text} for role, text in window]
        messages.append({"role": "user", "content": content})
        response = self.client.chat.completions.create(model=self.deployment, messages=messages)
        return response.choices[0].message.content or ""


class AzureEmbedder:
    """Embeddings en lote contra el deployment de embeddings (SPEC §5, paso 2)."""

    def __init__(self, settings: ProviderSettings):
        self.client = AzureOpenAI(
            api_key=settings.azure_embedding_key or settings.azure_api_key,
            api_version=settings.azure_api_version,
            azure_endpoint=settings.azure_embedding_endpoint or settings.azure_api_base,
            timeout=FAST_TIMEOUT, max_retries=1,
        )
        self.model = settings.azure_embedding_deployment
        self.dim = EMBEDDING_DIMS[self.model]

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
