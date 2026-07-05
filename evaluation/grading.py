"""Ablación de la GRADACIÓN relevancia↔importancia del recall (el "dragón", SPEC §5):

    uv run python -m evaluation.grading

Barre `recall_relevance_gain` sobre el corpus ENTERO y mide Recall@contexto por familia
y global. `score = gain·relevancia_evocada + base_strength`: a gain=1.0 (histórico) la
importancia consolidada domina y entierra miembros relevantes pero poco consolidados;
subirlo deja que la relevancia a la pregunta pese de verdad.

Mide el efecto GLOBAL, no un caso suelto: un gain que solo ayuda a una familia y daña
otras es SOBREAJUSTE, no mejora (memoria 'generalize-not-overfit'). El gain solo afecta
la LECTURA, así que cada escenario se construye UNA vez (lo caro) y se lee a cada gain
(barato), igual que ablation.py. Respeta la fachada: flip de `mind.config` + `mind.recall`.
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

GAINS = [1.0, 6.0, 12.0]  # 1.0 = histórico; >1 deja competir a la relevancia


def _window_turns(scenario) -> frozenset[int]:
    """Los turnos que el ChatSession real tendría en la ventana al preguntar (se vacía al dormir):
    lo que el LLM YA ve y `recall` deduplica. Igual que en ablation.py."""
    win: set[int] = set()
    for i in range(len(scenario.turns)):
        win.add(i + 1)
        if i in scenario.breaks_after:
            win = set()
    return frozenset(win)


def _replay(open_at, scenario, path: str, retries: int = 3):
    """Reproduce el escenario en una base fresca, reintentando ante blips de la API."""
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


def _build():
    settings = ProviderSettings()
    azure = AzureLLM(settings)
    embedder = AzureEmbedder(settings)
    base = DynamicsConfig()
    def open_at(path: str) -> Mind:
        return Mind(path, embedder, TurnExtractor(azure), judge=azure, config=base)
    return open_at, base


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def grade(scenarios, open_at, base, tmp: Path):
    """results[gain][familia] = [aciertos...]; cada red se construye una vez y se lee a cada gain."""
    results = {g: {} for g in GAINS}
    for si, sc in enumerate(scenarios):
        path = str(tmp / f"s{si}.db")
        try:
            mind = _replay(open_at, sc, path)
        except Exception as error:                       # blip persistente: omite, no abortes
            print(f"  ⚠️  «{sc.name}» omitido: {error}")
            continue
        window = _window_turns(sc)
        fam = sc.name  # el nombre de escenario ya es la familia (del meta): 8 por familia
        for g in GAINS:
            mind.config = base.model_copy(update={"recall_relevance_gain": g})
            for probe in sc.probes:
                ctx = mind.recall(probe.question, window_turns=window)
                results[g].setdefault(fam, []).append(recall_at_context(ctx, probe))
        mind.store.conn.close()
        print(f"  {si + 1}/{len(scenarios)} {sc.name}", flush=True)
    return results


def report(results):
    families = sorted({f for g in GAINS for f in results[g]})
    head = "familia".ljust(14) + "".join(f"{'g=' + str(g):>9}" for g in GAINS)
    print("\nRECALL@CONTEXTO por gain de relevancia\n" + head)
    print("─" * len(head))
    for fam in families:
        row = fam.ljust(14) + "".join(f"{_mean(results[g].get(fam, [])):>9.2f}" for g in GAINS)
        print(row)
    print("─" * len(head))
    glob = "GLOBAL".ljust(14)
    for g in GAINS:
        allb = [b for f in results[g] for b in results[g][f]]
        glob += f"{_mean(allb):>9.2f}"
    n = sum(len(v) for v in results[GAINS[0]].values())
    print(glob + f"   (n={n})")
    print("\nLectura: un gain alto que sube el GLOBAL es señal real; si solo sube una familia "
          "y baja otras, es sobreajuste — no tocar el defecto del core.")


def main():
    scenarios, kind = load_corpus()
    print(f"corpus {kind}: {len(scenarios)} escenarios · gains {GAINS}")
    open_at, base = _build()
    tmp = Path(tempfile.mkdtemp(prefix="grading-"))
    report(grade(scenarios, open_at, base, tmp))


if __name__ == "__main__":
    main()
