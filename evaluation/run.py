"""Punto de entrada del banco de evaluación (Fase 5).

    uv run python -m evaluation.run [--equalizes]

Reproduce el corpus contra las condiciones e imprime: (1) tabla de Recall@contexto
por fenómeno + global, (2) test de McNemar (apareado) de oroi frente a los dos
RAG, y (3) tabla de COSTE (llamadas LLM, embeddings, tokens estimados, tiempo),
separando escritura de la red (la pagan oroi/re-ranker/RAG-hechos) de lectura.

`--equalizes` corre oroi con la dinámica corregida (equiparación por gradiente,
boost 2.0 — la ganadora de la ablación de spreading) sin tocar el defecto de la
librería; las condiciones RAG no usan la dinámica y no cambian.
"""

import math
import sys

from oroi import DynamicsConfig, Mind
from oroi.extraction.extractor import TurnExtractor
from oroi.providers.azure import AzureEmbedder, AzureLLM
from oroi.providers.settings import ProviderSettings

from .cost import CountingEmbedder, CountingLLM, Meter
from .harness import evaluate
from .scenario import load_corpus


def make_open_at(meter: Meter):
    settings = ProviderSettings()
    azure = CountingLLM(AzureLLM(settings), meter)
    embedder = CountingEmbedder(AzureEmbedder(settings), meter)
    if "--equalizes" in sys.argv:
        config = DynamicsConfig(spread_equalizes=True, boost=2.0)
    else:
        config = DynamicsConfig()
    def open_at(path: str) -> Mind:
        return Mind(path, embedder, TurnExtractor(azure), judge=azure, config=config)
    return open_at


def _mean(bools):
    return sum(bools) / len(bools) if bools else 0.0


def print_recall(raw):
    types = sorted(next(iter(raw.values())).keys())
    w = max((len(t) for t in types), default=8)
    head = "condición".ljust(16) + "".join(f"  {t:>{max(w, 6)}}" for t in types) + "   global  (n)"
    print("\nRECALL@CONTEXTO\n" + head)
    print("─" * len(head))
    for cond, per in raw.items():
        cells = "".join(f"  {_mean(per.get(t, [])):>{max(w, 6)}.2f}" for t in types)
        allb = [b for t in types for b in per.get(t, [])]
        print(cond.ljust(16) + cells + f"   {_mean(allb):.2f}   {len(allb)}")


def mcnemar_exact(a_hits, b_hits):
    n01 = sum(1 for a, b in zip(a_hits, b_hits) if not a and b)
    n10 = sum(1 for a, b in zip(a_hits, b_hits) if a and not b)
    n, lo = n01 + n10, min(n01, n10)
    if n == 0:
        return n01, n10, 1.0
    return n01, n10, min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n))


def verdict(raw):
    types = sorted(next(iter(raw.values())).keys())
    flat = {c: [b for t in types for b in per[t]] for c, per in raw.items()}
    mind = next(c for c in raw if "oroi" in c)
    print("\noroi vs cada RAG (Recall@contexto, McNemar apareado):")
    for rag in [c for c in raw if "RAG" in c]:
        n01, n10, p = mcnemar_exact(flat[rag], flat[mind])   # a=rag, b=mind
        # n01 = RAG falla y oroi acierta = victoria de oroi; n10 = al revés = derrota.
        print(f"  vs {rag:<14} oroi {_mean(flat[mind]):.2f} · RAG {_mean(flat[rag]):.2f} · "
              f"gana {n01}/pierde {n10} · p={p:.4f}")


def print_cost(cost):
    w, turns, q = cost["write"], cost["turns"] or 1, cost["queries"] or 1
    print("\nCOSTE")
    print("  Escritura de la red (la pagan oroi, re-ranker y RAG-hechos; NO a/b):")
    print(f"    por turno: {w['llm']/turns:.2f} llamadas LLM · {w['emb']/turns:.2f} embeds · "
          f"{w['tok']/turns:.0f} tokens~ · {w['wall']/turns*1000:.0f} ms")
    print("  Lectura por consulta, por condición:")
    print(f"    {'condición':<16}{'LLM':>6}{'embeds':>8}{'txt-emb':>9}{'tokens~':>9}{'ms':>8}")
    for cond, r in cost["read"].items():
        print(f"    {cond:<16}{r['llm']/q:>6.2f}{r['emb']/q:>8.2f}{r['emb_txt']/q:>9.1f}"
              f"{r['tok']/q:>9.0f}{r['wall']/q*1000:>8.0f}")
    print("  (tokens estimados ≈ chars/4; el RAG crudo re-embebe episodios en lectura — en producción se precomputaría)")


def main():
    scenarios, kind = load_corpus()
    print(f"corpus {kind}: {len(scenarios)} escenarios")
    meter = Meter()

    def bar(done, total):
        print(f"  evaluando… {done}/{total}", flush=True)

    raw, cost = evaluate(scenarios, make_open_at(meter), meter, on_progress=bar)
    print_recall(raw)
    verdict(raw)
    print_cost(cost)


if __name__ == "__main__":
    main()
