"""Ablación de la EQUIPARACIÓN en el spreading (el bombeo que impedía enfriar, ver CLAUDE.md
y RUNBOOK-spreading.md — la corrida larga la pilota un modelo económico):

    uv run python -m evaluation.spreading

El spreading histórico SUMA a los vecinos sin restar al origen → un subgrafo conectado gana
energía sin input y satura (nada se enfría; test_equalize_kills_the_perpetual_motion).
`spread_equalizes=True` lo cambia por difusión por gradiente: el calor solo baja, el origen
cede lo que fluye y al igualarse el flujo para. Baja el nivel absoluto tras una mención, así
que se recalibra `boost` (2.0, ver test_equalize_fact_still_recalled_next_turn). La tercera
variante prueba 'solo a favor de la flecha' (spread_back_factor=0, la evocación intacta).
Aquí medimos el efecto REAL sobre Recall@contexto.

A diferencia de grading.py, la equiparación afecta a la ESCRITURA (cómo se acumula la
activación al percibir), así que cada variante REHACE su red — más caro: la corrida larga
la lanza el usuario. Respeta la fachada (perceive/recall).
"""

import tempfile
import time
from pathlib import Path

from oroi import DynamicsConfig, Mind
from oroi.extraction.extractor import TurnExtractor
from oroi.providers.azure import AzureEmbedder, AzureLLM
from oroi.providers.settings import ProviderSettings

from .metrics import recall_at_context
from .scenario import load_corpus

# (nombre, config): histórico vs equiparación (backward atenuado, el diseño propuesto) vs
# equiparación solo-a-favor-de-la-flecha (¿pierde la asociación lateral categoría→hermanos?).
VARIANTS = {
    "histórico":      DynamicsConfig(),
    "equipara+b2.0":  DynamicsConfig(spread_equalizes=True, boost=2.0),
    "eq-flecha+b2.0": DynamicsConfig(spread_equalizes=True, boost=2.0, spread_back_factor=0.0),
}


def _window_turns(scenario) -> frozenset[int]:
    win: set[int] = set()
    for i in range(len(scenario.turns)):
        win.add(i + 1)
        if i in scenario.breaks_after:
            win = set()
    return frozenset(win)


def _build(config):
    settings = ProviderSettings()
    azure = AzureLLM(settings)
    embedder = AzureEmbedder(settings)
    def open_at(path: str) -> Mind:
        return Mind(path, embedder, TurnExtractor(azure), judge=azure, config=config)
    return open_at


def _replay(open_at, scenario, path: str, retries: int = 3):
    for attempt in range(retries):
        try:
            mind = open_at(path)
            mind.wake()
            for i, turn in enumerate(scenario.turns):
                mind.perceive(turn)
                if i in scenario.breaks_after:
                    mind.sleep()
                    mind.wake()
            return mind
        except Exception:
            for suffix in ("", "-wal", "-shm"):
                Path(path + suffix).unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
            time.sleep(3)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def measure(scenarios, tmp: Path):
    """results[variante][familia] = [aciertos...]; cada variante rehace la red (la dinámica difiere)."""
    results = {v: {} for v in VARIANTS}
    for vname, config in VARIANTS.items():
        open_at = _build(config)
        for si, sc in enumerate(scenarios):
            path = str(tmp / f"{vname}-s{si}.db")
            try:
                mind = _replay(open_at, sc, path)
            except Exception as error:
                print(f"  ⚠️  [{vname}] «{sc.name}» omitido: {error}")
                continue
            window = _window_turns(sc)
            for probe in sc.probes:
                ctx = mind.recall(probe.question, window_turns=window)
                results[vname].setdefault(sc.name, []).append(recall_at_context(ctx, probe))
            mind.store.conn.close()
            print(f"  [{vname}] {si + 1}/{len(scenarios)} {sc.name}", flush=True)
    return results


def report(results):
    families = sorted({f for v in VARIANTS for f in results[v]})
    head = "familia".ljust(14) + "".join(f"{v:>16}" for v in VARIANTS)
    print("\nRECALL@CONTEXTO por variante de spreading\n" + head)
    print("─" * len(head))
    for fam in families:
        row = fam.ljust(14) + "".join(f"{_mean(results[v].get(fam, [])):>16.2f}" for v in VARIANTS)
        print(row)
    print("─" * len(head))
    glob = "GLOBAL".ljust(14)
    for v in VARIANTS:
        allb = [b for f in results[v] for b in results[v][f]]
        glob += f"{_mean(allb):>16.2f}"
    print(glob)
    print("\nLectura: la equiparación debería ENFRIAR sin perder recall. Si el GLOBAL de 'equipara' se "
          "mantiene o sube, el fix estructural es sano; si baja, recalibrar boost (1.5/2.5/3.0). Si "
          "'eq-flecha' pierde (esp. en multihop), el backward atenuado se queda. Criterios completos "
          "y registro de resultados: evaluation/RUNBOOK-spreading.md")


def main():
    scenarios, kind = load_corpus()
    print(f"corpus {kind}: {len(scenarios)} escenarios · variantes {list(VARIANTS)}")
    tmp = Path(tempfile.mkdtemp(prefix="spreading-"))
    report(measure(scenarios, tmp))


if __name__ == "__main__":
    main()
