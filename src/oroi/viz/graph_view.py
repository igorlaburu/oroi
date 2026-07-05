"""Visualización: la red "iluminándose" turno a turno (SPEC §7, Fase 4).

Capa de instrumentación desacoplada: consume `NetworkSnapshot` (JSON plano) y
renderiza un HTML autocontenido con Cytoscape.js (CDN). La futura interfaz
Next.js reutilizará la misma librería (react-cytoscapejs) sobre el mismo JSON
servido por la API — aquí solo cambia el contenedor.
"""

import json
from importlib import resources
from pathlib import Path

from ..core.graph import NetworkSnapshot


def timeline_path(db_path: str) -> Path:
    """El diario de la conversación vive junto a su base: <db>.viz.jsonl."""
    return Path(f"{db_path}.viz.jsonl")


def record(snapshot: NetworkSnapshot, path: str | Path) -> None:
    """Apunta la foto del turno al diario (jsonl, append)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(snapshot.model_dump_json() + "\n")


def load_timeline(path: str | Path) -> list[NetworkSnapshot]:
    """Una línea truncada (un kill a media escritura) no invalida el resto del diario."""
    snapshots = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            snapshots.append(NetworkSnapshot.model_validate_json(line))
        except ValueError:
            continue
    return snapshots


def export_html(timeline: list[NetworkSnapshot], out: str | Path,
                live_journal: str = "", chat_endpoint: str = "") -> Path:
    """HTML del visor. Sin live_journal va autocontenido (datos embebidos); con él,
    el visor lee el diario por HTTP y se recarga al crecer (en vivo). Con chat_endpoint
    además aparece el panel de chat que hace POST a esa ruta."""
    template = resources.files("oroi.viz").joinpath("template.html").read_text(encoding="utf-8")
    payload = json.dumps([s.model_dump() for s in timeline], ensure_ascii=False)
    out = Path(out)
    out.write_text(template.replace("__TIMELINE__", payload)
                   .replace("__JOURNAL__", live_journal)
                   .replace("__CHAT_ENDPOINT__", chat_endpoint), encoding="utf-8")
    return out
