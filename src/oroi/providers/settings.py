"""Configuración de proveedores: .env de la carpeta actual, con ~/.oroi/.env de respaldo."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderSettings(BaseSettings):
    # el .env local tiene prioridad (último gana); el global evita copiar credenciales por carpeta
    model_config = SettingsConfigDict(
        env_file=(str(Path.home() / ".oroi" / ".env"), ".env"), extra="ignore")

    def has_credentials(self) -> bool:
        """¿Hay algún proveedor de memoria configurado? (para fallar amable, no con traceback)"""
        return bool(self.azure_api_key or (self.memory_provider == "openai"
                                           and (self.openai_api_key or self.openai_base_url)))

    # Azure OpenAI: extractor y embeddings (mismos identificadores que web.agent.simple)
    azure_api_base: str = ""
    azure_api_key: str = ""
    azure_api_version: str = "2024-10-21"
    azure_fast_deployment: str = "gpt-4o-mini"
    azure_embedding_endpoint: str = ""    # si está vacío se usa azure_api_base
    azure_embedding_key: str = ""         # si está vacío se usa azure_api_key
    azure_embedding_deployment: str = "text-embedding-3-small"

    # Memoria (extractor + embeddings): "azure" (por defecto) u "openai" (OpenAI directo,
    # Ollama o cualquier endpoint /v1 compatible vía openai_base_url).
    memory_provider: str = "azure"
    openai_base_url: str = ""             # vacío = api.openai.com; Ollama: http://host:11434/v1
    openai_api_key: str = ""
    openai_fast_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536
    openai_chat_model: str = "gpt-4o"

    # El Conversador: "claude" (por defecto), "azure" (deployment "smart") u "openai".
    chat_provider: str = "claude"
    azure_smart_deployment: str = "gpt-5.4"

    # Solo con chat_provider=claude: "session" usa la sesión local de Claude Code
    # (cuenta Max); "api" llama directamente al API (requiere ANTHROPIC_API_KEY).
    claude_auth: str = "session"
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-4-8"
