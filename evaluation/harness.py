"""El runner: reproduce cada escenario UNA vez y clona la base para cada condición.

Reproducir la conversación es lo caro (una llamada al extractor por turno). Antes
se repetía por cada condición; ahora se reproduce una sola vez en una base, se le
hace checkpoint, y para cada condición se CLONA el fichero SQLite (copia barata) y
se abre encima. ~4× menos llamadas al LLM. Cada condición opera sobre su clon, así
que puede mutar (oroi percibe la pregunta) sin afectar a las demás.
"""

import shutil
import tempfile
import time
from pathlib import Path

from .baselines import CONDITIONS
from .metrics import recall_at_context
from .scenario import Scenario


def _discard(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(path + suffix).unlink(missing_ok=True)


def _replay(scenario: Scenario, open_at, path: str, retries: int = 3):
    """Reproduce el escenario en una base fresca. Reintenta ante errores transitorios (un blip
    de la API no debe tirar una corrida de 10 min); cada reintento parte de cero. Si se agotan,
    propaga para que el llamador omita el escenario en vez de abortar todo."""
    for attempt in range(retries):
        try:
            mind = open_at(path)
            mind.wake()
            for i, turn in enumerate(scenario.turns):
                mind.perceive(turn)
                if i in scenario.breaks_after:   # frontera de sesión: dormir y despertar
                    mind.sleep()
                    mind.wake()
            return mind
        except Exception:
            _discard(path)                       # base parcial fuera: el reintento empieza limpio
            if attempt == retries - 1:
                raise
            time.sleep(3)


def evaluate(scenarios, open_at, meter=None, on_progress=None):
    """Devuelve (raw, cost). raw = {condición: {tipo: [aciertos...]}}.
    cost = coste de ESCRITURA (construir la red, por turno) y de LECTURA (por consulta y condición)."""
    raw = {name: {} for name in CONDITIONS}
    write = {"llm": 0, "emb": 0, "emb_txt": 0, "tok": 0, "wall": 0.0}
    read = {name: dict(write) for name in CONDITIONS}
    turns_total, queries = 0, 0
    tmp = Path(tempfile.mkdtemp(prefix="eval-"))
    done, total = 0, len(scenarios) * len(CONDITIONS)

    def accumulate(acc, snap):
        for key in acc:
            acc[key] += snap[key]

    for si, scenario in enumerate(scenarios):
        if meter:
            meter.reset()
        base = tmp / f"s{si}.db"
        try:
            mind = _replay(scenario, open_at, str(base))      # 1 sola reproducción (lo caro = escritura)
        except Exception as error:                            # blip de red persistente: omite, no abortes
            print(f"  ⚠️  escenario «{scenario.name}» omitido: {error}")
            continue
        mind.store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        mind.store.conn.commit()
        mind.store.conn.close()
        if meter:
            accumulate(write, meter.snapshot())
            turns_total += len(scenario.turns)
        for ci, (cond_name, retrieve) in enumerate(CONDITIONS.items()):
            for pi, probe in enumerate(scenario.probes):
                clone = tmp / f"s{si}-c{ci}-p{pi}.db"
                shutil.copy(base, clone)                       # clon barato por condición
                m = open_at(str(clone))
                if meter:
                    meter.reset()
                for attempt in range(3):                       # sobrevive a blips transitorios de la API
                    try:
                        context = retrieve(m, probe.question)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(3)
                if meter:
                    accumulate(read[cond_name], meter.snapshot())
                    queries += 1
                m.store.conn.close()
                raw[cond_name].setdefault(scenario.name, []).append(recall_at_context(context, probe))
            done += 1
            if on_progress:
                on_progress(done, total)
    cost = {"write": write, "read": read, "turns": turns_total, "queries": queries // len(CONDITIONS)}
    return raw, cost
