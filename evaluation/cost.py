"""Medición de coste — envuelve los providers inyectados, sin tocar la librería.

Cuenta llamadas al LLM (extractor/juez), llamadas y textos de embedding, tokens
ESTIMADOS por longitud (los SDKs no exponen el usage a través de la fachada, así
que es una aproximación honesta ≈ chars/4) y tiempo de pared. Permite separar el
coste de ESCRITURA (construir/mantener la red) del de LECTURA (cada recuperación).
"""

import time


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.llm_calls = self.embed_calls = self.embed_texts = 0
        self.tokens = 0
        self.wall = 0.0

    def snapshot(self) -> dict:
        return {"llm": self.llm_calls, "emb": self.embed_calls,
                "emb_txt": self.embed_texts, "tok": self.tokens, "wall": self.wall}


class CountingLLM:
    """Envuelve un LLM con complete_json(system, user) -> str (extractor y juez)."""

    def __init__(self, inner, meter: Meter):
        self.inner, self.meter = inner, meter

    def complete_json(self, system: str, user: str) -> str:
        t = time.perf_counter()
        out = self.inner.complete_json(system, user)
        self.meter.wall += time.perf_counter() - t
        self.meter.llm_calls += 1
        self.meter.tokens += _est_tokens(system) + _est_tokens(user) + _est_tokens(out)
        return out


class CountingEmbedder:
    """Envuelve un Embedder con embed(texts) -> list[list[float]], .model y .dim."""

    def __init__(self, inner, meter: Meter):
        self.inner, self.meter = inner, meter
        self.model, self.dim = inner.model, inner.dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        t = time.perf_counter()
        out = self.inner.embed(texts)
        self.meter.wall += time.perf_counter() - t
        self.meter.embed_calls += 1
        self.meter.embed_texts += len(texts)
        self.meter.tokens += sum(_est_tokens(x) for x in texts)
        return out
