"""Contratos de los proveedores externos. El núcleo solo conoce estos protocolos (SPEC §2)."""

from typing import Protocol

from ..extraction.schema import MiniNetwork


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class Extractor(Protocol):
    """Transforma texto en mini-red (hechos del turno) y extrae las pistas de una pregunta."""

    def extract(self, text: str, context: str = "") -> MiniNetwork: ...

    def extract_cues(self, text: str) -> list[str]: ...


class Chat(Protocol):
    """El Conversador. La memoria se inyecta en el turno de usuario, nunca en el system (SPEC §5)."""

    def reply(self, system: str, window: list[tuple[str, str]], memory: str, user_text: str) -> str: ...
