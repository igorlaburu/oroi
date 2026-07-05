"""Segundo nivel de medición: Answer@judge — puntúa la RESPUESTA, no la chuleta.

    uv run python -m evaluation.judge <dir-eval-temporal> [--equalizes]

Recall@contexto audita el paso intermedio (¿el contexto contiene lo esperado y
nada de lo prohibido?) y castiga contextos que traen la evidencia vecina aunque
basten para responder bien (convergencia, actualización). Este módulo mide el
final de la cadena: con el contexto recuperado, un respondedor (gpt-4o-mini)
genera la respuesta y un juez LLM la compara con `expect_answer`.

No repite la fase cara (reproducir los turnos): reutiliza las bases `s<i>.db`
que `evaluation.run` deja en su directorio temporal `eval-*`, en el orden
determinista de `load_corpus()`. Pasar el MISMO flag de dinámica que la corrida
original para que la lectura de oroi opere igual. Imprime Answer@judge por
fenómeno y condición junto al Recall@contexto de control (si el control no cuadra
con la corrida original, el mapeo escenario↔base está mal y no hay que fiarse).
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from oroi import DynamicsConfig, Mind
from oroi.extraction.extractor import TurnExtractor
from oroi.providers.azure import AzureEmbedder, AzureLLM
from oroi.providers.settings import ProviderSettings

from .baselines import CONDITIONS
from .metrics import recall_at_context
from .scenario import load_corpus

ANSWER_SYSTEM = (
    "Eres un asistente con memoria. Responde a la pregunta del usuario usando SOLO el contexto "
    "de memoria proporcionado (si está vacío, di que no lo sabes). Sé directo: la respuesta en "
    "pocas palabras, sin explicación, y EN EL IDIOMA DE LA PREGUNTA. "
    'Devuelve JSON: {"answer": "..."}'
)

JUDGE_SYSTEM = (
    "Eres un juez de evaluación. Se te da una respuesta esperada y una respuesta dada. Decide si "
    "la dada expresa el mismo hecho que la esperada (mismas entidades/valores; da igual la "
    'redacción). Devuelve JSON: {"correct": true|false}'
)


def _ask_json(llm: AzureLLM, system: str, user: str, key: str, retries: int = 3):
    for attempt in range(retries):
        try:
            return json.loads(llm.complete_json(system, user)).get(key)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3)


def judge_probe(llm: AzureLLM, context: str, question: str, expected: str,
                answer_llm: AzureLLM | None = None) -> bool:
    """`answer_llm` permite responder con un modelo distinto del juez (p.ej. el conversador
    de producción para benchmarks que exigen razonamiento; el juez sigue siendo el rápido)."""
    user = f"[contexto de memoria]\n{context or '(vacío)'}\n\n[pregunta]\n{question}"
    answer = str(_ask_json(answer_llm or llm, ANSWER_SYSTEM, user, "answer") or "")
    verdict = _ask_json(llm, JUDGE_SYSTEM, f"esperada: {expected}\ndada: {answer}", "correct")
    return bool(verdict)


def _mean(bools):
    return sum(bools) / len(bools) if bools else 0.0


def print_table(title, raw):
    types = sorted({t for per in raw.values() for t in per})
    w = max((len(t) for t in types), default=8)
    head = "condición".ljust(16) + "".join(f"  {t:>{max(w, 6)}}" for t in types) + "   global  (n)"
    print(f"\n{title}\n" + head)
    print("─" * len(head))
    for cond, per in raw.items():
        cells = "".join(f"  {_mean(per.get(t, [])):>{max(w, 6)}.2f}" for t in types)
        allb = [b for t in types for b in per.get(t, [])]
        print(cond.ljust(16) + cells + f"   {_mean(allb):.2f}   {len(allb)}")


def main():
    eval_dir = Path(sys.argv[1])
    scenarios, kind = load_corpus()
    print(f"corpus {kind}: {len(scenarios)} escenarios · bases: {eval_dir}")

    settings = ProviderSettings()
    llm = AzureLLM(settings)
    embedder = AzureEmbedder(settings)
    if "--equalizes" in sys.argv:
        config = DynamicsConfig(spread_equalizes=True, boost=2.0)
    else:
        config = DynamicsConfig()

    def open_at(path: str) -> Mind:
        return Mind(path, embedder, TurnExtractor(llm), judge=llm, config=config)

    answer = {name: {} for name in CONDITIONS}
    control = {name: {} for name in CONDITIONS}
    tmp = Path(tempfile.mkdtemp(prefix="judge-"))
    total = len(scenarios) * len(CONDITIONS)
    done = 0
    # Punto de control: cada sonda juzgada se apunta; al relanzar, lo hecho se salta
    # (una interrupción —tapa del portátil, reposo— no cuesta trabajo ya pagado).
    ckpt_path = Path(__file__).parent / f"judge-checkpoint-{eval_dir.name}.jsonl"
    ckpt = {}
    if ckpt_path.exists():
        for line in ckpt_path.read_text().splitlines():
            r = json.loads(line)
            ckpt[(r["si"], r["cond"], r["pi"])] = r
        print(f"  retomando: {len(ckpt)} sondas ya juzgadas ({ckpt_path.name})")
    with ckpt_path.open("a") as ckpt_file:
        for si, scenario in enumerate(scenarios):
            base = eval_dir / f"s{si}.db"
            if not base.exists():
                print(f"  ⚠️  falta {base.name}: escenario «{scenario.name}» omitido")
                continue
            for ci, (cond_name, retrieve) in enumerate(CONDITIONS.items()):
                for pi, probe in enumerate(scenario.probes):
                    if not probe.expect_answer:
                        continue
                    prev = ckpt.get((si, cond_name, pi))
                    if prev is None:
                        clone = tmp / f"s{si}-c{ci}-p{pi}.db"
                        shutil.copy(base, clone)
                        m = open_at(str(clone))
                        context = retrieve(m, probe.question)
                        m.store.conn.close()
                        prev = {"si": si, "cond": cond_name, "pi": pi,
                                "control": recall_at_context(context, probe),
                                "answer": judge_probe(llm, context, probe.question, probe.expect_answer)}
                        ckpt_file.write(json.dumps(prev) + "\n")
                        ckpt_file.flush()
                    control[cond_name].setdefault(scenario.name, []).append(prev["control"])
                    answer[cond_name].setdefault(scenario.name, []).append(prev["answer"])
                done += 1
                print(f"  juzgando… {done}/{total}", flush=True)

    print_table("ANSWER@JUDGE (la respuesta, juzgada por LLM)", answer)
    print_table("RECALL@CONTEXTO (control: debe cuadrar con la corrida original)", control)


if __name__ == "__main__":
    main()
