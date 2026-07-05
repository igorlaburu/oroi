"""Carga de escenarios desde el corpus JSONL (datos separados del código)."""

import json
from dataclasses import dataclass, field
from pathlib import Path

CORPUS = Path(__file__).parent / "corpus"


@dataclass
class Probe:
    question: str
    expect_context: list[str]          # fragmentos que el contexto recuperado DEBE contener
    expect_answer: str = ""            # respuesta esperada (para el juez LLM)
    avoid_context: list[str] = field(default_factory=list)  # fragmentos que NO debería traer


@dataclass
class Scenario:
    name: str
    goal: str
    turns: list[str]                   # turnos de usuario, en orden
    probes: list[Probe]
    breaks_after: set[int] = field(default_factory=set)  # índices de turno tras los que duerme/despierta


def load(path: Path) -> Scenario:
    name, goal, turns, probes, breaks = path.stem, "", [], [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        match ev["type"]:
            case "meta":
                name, goal = ev.get("name", name), ev.get("goal", "")
            case "turn":
                turns.append(ev["user"])
            case "probe":
                probes.append(Probe(ev["question"], ev.get("expect_context", []),
                                    ev.get("expect_answer", ""), ev.get("avoid_context", [])))
            case "session_break":
                breaks.add(len(turns) - 1)  # dormir/despertar tras el último turno percibido
    return Scenario(name, goal, turns, probes, breaks)


def load_all(directory: Path = CORPUS) -> list[Scenario]:
    return [load(p) for p in sorted(directory.glob("*.jsonl"))]


def load_corpus() -> tuple[list[Scenario], str]:
    """Usa el corpus generado (corpus/gen) si existe; si no, los escenarios curados."""
    gen = CORPUS / "gen"
    if gen.exists() and any(gen.glob("*.jsonl")):
        return load_all(gen), "generado"
    return load_all(), "curado"
