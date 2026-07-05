"""Métrica principal: Recall@contexto — objetiva, determinista, sin LLM.

Mide directamente la hipótesis (¿la recuperación trae el hecho correcto?): el
contexto recuperado debe contener TODOS los fragmentos esperados y NINGUNO de
los que debe evitar (el dato viejo en escenarios de actualización, el distractor).
"""

from .scenario import Probe


def recall_at_context(context: str, probe: Probe) -> bool:
    ctx = context.lower()
    found = all(frag.lower() in ctx for frag in probe.expect_context)
    leaked = any(frag.lower() in ctx for frag in probe.avoid_context)
    return found and not leaked
