"""Validación externa: LoCoMo (Maharana et al., 2024) sobre la tubería propia.

    uv run python -m evaluation.locomo --data locomo10.json [--conversations 2]
        [--per-category 5] [--equalizes] [--dry]

Adapta el benchmark a nuestros escenarios: cada conversación LoCoMo (dos hablantes,
~19 sesiones fechadas, ~600 turnos) se percibe turno a turno — hablante y fecha van
dentro del texto del turno; cada frontera de sesión duerme y despierta la mente — y
sus preguntas se convierten en sondas. Se comparan las condiciones RAG crudo y
oroi con dos métricas: Answer@judge (la respuesta, juzgada contra la de
referencia) y, como control, contexto-contiene-evidencia (algún turno de evidencia
anotado por el dataset aparece en el contexto recuperado).

`--dry` valida el adaptador sin una sola llamada (carga, mapea y resume).
Checkpoint por sonda (jsonl): una interrupción no repite trabajo pagado.
La escritura de cada conversación es cara (~600 extracciones): el piloto por
defecto son 2 conversaciones y 5 preguntas por categoría.
"""

import argparse
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

from oroi import DynamicsConfig, Mind
from oroi.extraction.extractor import TurnExtractor
from oroi.providers.azure import AzureEmbedder, AzureLLM
from oroi.providers.settings import ProviderSettings

from .baselines import CONDITIONS
from .judge import judge_probe, print_table
from .scenario import Probe, Scenario

PILOT_CONDITIONS = ("b · RAG crudo", "c · oroi")
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "inferencia", 4: "single-hop", 5: "adversarial"}
NO_ANSWER = "no se menciona en la conversación"


def load_locomo(path: str, max_conversations: int | None = None) -> list[dict]:
    """Cada conversación → turnos con hablante y fecha, fronteras de sesión y su QA."""
    out = []
    for c in json.loads(Path(path).read_text())[:max_conversations]:
        conv = c["conversation"]
        turns, breaks, dia_text = [], [], {}
        i = 1
        while f"session_{i}" in conv:
            date = conv.get(f"session_{i}_date_time", "")
            for t in conv[f"session_{i}"]:
                dia_text[t["dia_id"]] = t["text"]
                turns.append(f"[{date}] {t['speaker']}: {t['text']}")
            breaks.append(len(turns) - 1)          # tras el último turno de la sesión
            i += 1
        out.append({"id": c["sample_id"], "turns": turns, "breaks": set(breaks[:-1]),
                    "dia_text": dia_text, "qa": c["qa"]})
    return out


def to_probes(conv: dict, per_category: int, rng: random.Random) -> list[tuple[str, Probe]]:
    """Muestra estratificada por categoría. La evidencia del dataset (dia_ids) se vuelve
    expect_context; en las adversariales (cat. 5) la respuesta correcta es la abstención."""
    by_cat: dict[int, list[dict]] = {}
    for q in conv["qa"]:
        by_cat.setdefault(int(q.get("category", 0)), []).append(q)
    picked = []
    for cat, questions in sorted(by_cat.items()):
        for q in rng.sample(questions, min(per_category, len(questions))):
            evidence = [conv["dia_text"][d] for d in _evidence_ids(q) if d in conv["dia_text"]]
            answer = str(q.get("answer") or q.get("adversarial_answer") or NO_ANSWER)
            probe = Probe(question=q["question"], expect_context=evidence[:1],
                          expect_answer=answer)
            picked.append((CATEGORIES.get(cat, str(cat)), probe))
    return picked


def _evidence_ids(q: dict) -> list[str]:
    ev = q.get("evidence") or []
    if isinstance(ev, str):                        # viene serializado como "['D1:3']"
        ev = [e.strip(" '\"") for e in ev.strip("[]").split(",") if e.strip(" '\"")]
    return ev


def _perceive_resumable(scenario: Scenario, open_at, path: str, retries: int = 3):
    """Percibe el escenario RETOMANDO donde se quedó: la mente es persistente y `store.turn`
    dice cuántos turnos lleva — una interrupción a mitad de conversación no repaga nada.
    Los blips transitorios de la API se reintentan sin perder el progreso."""
    mind = open_at(path)
    mind.wake()
    start = mind.store.turn
    if start:
        print(f"  retomando «{scenario.name}» desde el turno {start + 1}", flush=True)
    else:
        print(f"  percibiendo «{scenario.name}» ({len(scenario.turns)} turnos)…", flush=True)
    for i, turn in enumerate(scenario.turns):
        if i < start:
            continue
        for attempt in range(retries):
            try:
                mind.perceive(turn)
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(3)
        if i in scenario.breaks_after:
            mind.sleep()
            mind.wake()
    return mind


def main():
    ap = argparse.ArgumentParser()
    # El dataset LoCoMo (Maharana et al., 2024) no se redistribuye aquí; descárgalo de su repo:
    #   curl -L -o evaluation/corpus/locomo10.json \
    #        https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
    ap.add_argument("--data", default=str(Path(__file__).parent / "corpus" / "locomo10.json"))
    ap.add_argument("--conversations", type=int, default=2)
    ap.add_argument("--per-category", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--equalizes", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--smart-answerer", action="store_true",
                    help="responder con el deployment smart (producción); el juez sigue en el rápido")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    convs = load_locomo(args.data, args.conversations)
    plan = [(conv, to_probes(conv, args.per_category, rng)) for conv in convs]
    n_turns = sum(len(c["turns"]) for c, _ in plan)
    n_probes = sum(len(p) for _, p in plan)
    print(f"LoCoMo: {len(plan)} conversaciones · {n_turns} turnos a percibir · "
          f"{n_probes} sondas × {len(PILOT_CONDITIONS)} condiciones")
    if args.dry:
        for conv, probes in plan:
            cats = {}
            for cat, _ in probes:
                cats[cat] = cats.get(cat, 0) + 1
            print(f"  {conv['id']}: {len(conv['turns'])} turnos, {len(conv['breaks'])} "
                  f"fronteras de sesión, sondas {cats}")
            print(f"    turno 1: {conv['turns'][0][:100]}")
        return

    settings = ProviderSettings()
    llm = AzureLLM(settings)
    answer_llm = None
    if args.smart_answerer:                            # el respondedor de producción
        answer_llm = AzureLLM(settings)
        answer_llm.deployment = settings.azure_smart_deployment
    embedder = AzureEmbedder(settings)
    # ±2 turnos vecinos: en LoCoMo la respuesta vive en la implicatura del hilo adyacente
    context_turns = {"episode_context_turns": 2}
    config = (DynamicsConfig(spread_equalizes=True, boost=2.0, **context_turns)
              if args.equalizes else DynamicsConfig(**context_turns))

    def open_at(path: str) -> Mind:
        return Mind(path, embedder, TurnExtractor(llm), judge=llm, config=config)

    ckpt_path = Path(__file__).parent / "locomo-checkpoint.jsonl"
    ckpt = {}
    if ckpt_path.exists():
        for line in ckpt_path.read_text().splitlines():
            r = json.loads(line)
            ckpt[(r["conv"], r["cond"], r["q"])] = r
        print(f"  retomando: {len(ckpt)} sondas ya juzgadas")

    answer = {name: {} for name in PILOT_CONDITIONS}
    control = {name: {} for name in PILOT_CONDITIONS}
    bases = Path(__file__).parent / ".locomo-bases"   # persistente: la escritura se paga una vez
    bases.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="locomo-"))    # los clones por sonda sí son desechables
    with ckpt_path.open("a") as ckpt_file:
        for conv, probes in plan:
            scenario = Scenario(name=conv["id"], goal="locomo", turns=conv["turns"],
                                probes=[p for _, p in probes], breaks_after=conv["breaks"])
            base = bases / f"{conv['id']}.db"
            done_mark = base.with_suffix(".done")     # sin marca = percepción sin terminar
            if not done_mark.exists():
                mind = _perceive_resumable(scenario, open_at, str(base))
                mind.store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                mind.store.conn.commit()
                mind.store.conn.close()
                done_mark.touch()
            for cond_name in PILOT_CONDITIONS:
                retrieve = CONDITIONS[cond_name]
                for qi, (cat, probe) in enumerate(probes):
                    prev = ckpt.get((conv["id"], cond_name, qi))
                    if prev is None:
                        clone = tmp / f"{conv['id']}-{cond_name[:1]}-{qi}.db"
                        shutil.copy(base, clone)
                        m = open_at(str(clone))
                        context = retrieve(m, probe.question)
                        m.store.conn.close()
                        has_evidence = any(f.lower() in context.lower() for f in probe.expect_context)
                        prev = {"conv": conv["id"], "cond": cond_name, "q": qi, "cat": cat,
                                "control": has_evidence,
                                "answer": judge_probe(llm, context, probe.question, probe.expect_answer, answer_llm)}
                        ckpt_file.write(json.dumps(prev) + "\n")
                        ckpt_file.flush()
                    answer[cond_name].setdefault(prev["cat"], []).append(prev["answer"])
                    control[cond_name].setdefault(prev["cat"], []).append(prev["control"])
                    print(f"  {conv['id']} · {cond_name} · {qi + 1}/{len(probes)}", flush=True)

    print_table("LoCoMo · ANSWER@JUDGE", answer)
    print_table("LoCoMo · contexto-contiene-evidencia (control)", control)


if __name__ == "__main__":
    main()
