"""Replay: reconstruir la película de una conversación re-percibiendo sus episodios.

La dinámica es determinista por turnos (SPEC §5): la misma secuencia produce la
misma red. Los episodios guardan la conversación literal, así que cualquier base
puede convertirse en línea temporal visualizable — y la Fase 5 reusará esto para
reproducir escenarios.
"""

import sqlite3

from ..core.graph import NetworkSnapshot
from ..mind import Mind


def episode_texts(db_path: str) -> list[str]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT text FROM episodes ORDER BY turn, id").fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def replay(texts: list[str], mind: Mind, on_turn=None) -> list[NetworkSnapshot]:
    """Re-percibe los turnos en una mente fresca, durmiendo a la cadencia del REPL."""
    timeline = []
    for i, text in enumerate(texts, 1):
        mind.perceive(text)
        timeline.append(mind.introspect())
        if i % mind.config.consolidate_every == 0:
            mind.sleep()
        if on_turn:
            on_turn(i, len(texts))
    return timeline
