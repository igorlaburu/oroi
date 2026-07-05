"""oroi: memoria asociativa experimental para chatbots.

Red semántica con dinámica de activación inspirada en la memoria humana.
La única interfaz pública es la fachada `Mind` (SPEC §6, Principios); los
proveedores de LLM/embeddings se inyectan tras `Protocol`s (ver `providers.base`),
de modo que la librería es agnóstica del chatbot y del proveedor que la use.

Autor: Igor Laburu — Gako AI <oroi@gako.ai>. Licencia Apache-2.0.
"""

from .core.config import DynamicsConfig
from .mind import Mind
from .providers.base import Chat, Embedder, Extractor

__all__ = ["Mind", "DynamicsConfig", "Embedder", "Extractor", "Chat"]

__version__ = "0.1.0"
__author__ = "Igor Laburu — Gako AI <oroi@gako.ai>"
