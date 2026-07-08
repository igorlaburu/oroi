"""CLI: `oroi` — la puerta de entrada. `info` y `viz` funcionan sin credenciales;
`chat`, `serve`, `replay` y `consolidate` usan los proveedores de .env."""

import argparse
import sqlite3
from importlib.metadata import version
from pathlib import Path

from .viz import graph_view

BIENVENIDA = """\
Oroi — memoria continua para una conversación infinita  ·  https://oroi.gako.ai

Para empezar:
  oroi chat        conversar con memoria (credenciales en .env o ~/.oroi/.env)
  oroi info        resumen de tu memoria: tamaño, lo activo, lo consolidado
  oroi viz         la película de la memoria, en HTML interactivo

Tu memoria vive en ~/.oroi/mind.db (cámbiala con --db en cualquier comando).
Ayuda completa: oroi --help
"""


def main() -> None:
    args = _parse()
    if args.command is None:
        print(BIENVENIDA)
    elif args.command == "info":
        from .chat.loop import default_db
        print(_info(args.db or default_db()))
    elif args.command == "chat":
        from .chat.loop import main as chat_main
        chat_main(args.db, voice=args.voice)
    elif args.command == "thoughts":
        from .chat.loop import default_db
        print(_thoughts(args.db or default_db(), args.limit))
    elif args.command == "consolidate":
        from .chat.loop import build_mind
        print(build_mind(args.db).sleep().model_dump_json(indent=2))
    elif args.command == "viz":
        print(_export(args))
    elif args.command == "replay":
        print(_replay(args))
    elif args.command == "serve":
        from .chat.web import serve
        serve(args.db, args.port, args.host)


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="oroi", description="Oroi: memoria asociativa para agentes conversacionales")
    parser.add_argument("-V", "--version", action="version", version=f"oroi {version('oroi')}")
    commands = parser.add_subparsers(dest="command", required=False)
    info = commands.add_parser("info", help="resumen de la memoria (sin credenciales): tamaño, activo, consolidado")
    info.add_argument("--db", default=None, help="ruta de la base de memoria (por defecto ~/.oroi/mind.db)")
    chat = commands.add_parser("chat", help="REPL conversacional con memoria (equivale a oroi-chat)")
    chat.add_argument("--db", default=None, help="ruta de la base de memoria (por defecto ~/.oroi/mind.db)")
    chat.add_argument("--voice", action="store_true",
                      help="muestra la voz interior tras cada respuesta (experimental: "
                           "una llamada extra al LLM rápido por turno)")
    thoughts = commands.add_parser(
        "thoughts", help="el diario de la voz (sin credenciales): qué ha ido pensando la mente")
    thoughts.add_argument("--db", default=None, help="ruta de la base de memoria (por defecto ~/.oroi/mind.db)")
    thoughts.add_argument("-n", type=int, default=20, dest="limit", help="pensamientos a mostrar")
    consolidate = commands.add_parser("consolidate", help="ejecuta un ciclo de consolidación ('sueño')")
    consolidate.add_argument("--db", default="mind.db", help="ruta de la base de memoria")
    viz = commands.add_parser("viz", help="exporta la red a un HTML estático interactivo (D3.js)")
    viz.add_argument("--db", default="mind.db", help="ruta de la base de memoria")
    viz.add_argument("--out", default=None, help="ruta del HTML (por defecto <db>.viz.html)")
    rep = commands.add_parser("replay", help="reconstruye la película re-percibiendo los episodios (usa LLM)")
    rep.add_argument("--db", default="mind.db", help="base cuyos episodios se re-perciben")
    rep.add_argument("--out", default=None, help="ruta del HTML (por defecto <db>.replay.html)")
    serve = commands.add_parser("serve", help="servidor en vivo: visor + chat sobre la mente real")
    serve.add_argument("--db", default="mind.db", help="ruta de la base de memoria")
    serve.add_argument("--port", type=int, default=8765, help="puerto del servidor")
    serve.add_argument("--host", default="127.0.0.1",
                       help="interfaz de escucha (usa 0.0.0.0 en contenedor/despliegue)")
    return parser.parse_args()


def _export(args: argparse.Namespace):
    """Con diario de turnos usa la línea temporal completa; sin él, una foto del estado actual."""
    from .chat.loop import build_mind
    journal = graph_view.timeline_path(args.db)
    timeline = (graph_view.load_timeline(journal) if journal.exists()
                else [build_mind(args.db).introspect()])
    _backfill_sources(timeline, args.db)
    return graph_view.export_html(timeline, args.out or f"{args.db}.viz.html")


def _backfill_sources(timeline, db_path: str) -> None:
    """Los diarios de antes no llevaban textos por nodo: se rellenan con los actuales de la base."""
    if all(snap.sources for snap in timeline):
        return
    from .chat.loop import build_mind
    sources = build_mind(db_path).introspect().sources
    for snap in timeline:
        if not snap.sources:
            snap.sources = sources


def _replay(args: argparse.Namespace):
    """Re-percibe los episodios en una mente desechable y exporta la película."""
    import tempfile

    from .viz import replay as rep

    from .chat.loop import build_mind
    texts = rep.episode_texts(args.db)
    fresh = build_mind(f"{tempfile.mkdtemp()}/replay.db")
    timeline = rep.replay(texts, fresh, on_turn=lambda i, n: print(f"\rturno {i}/{n}", end=""))
    print()
    journal = Path(f"{args.db}.replay.jsonl")  # re-renderizar luego sale gratis
    journal.unlink(missing_ok=True)
    for snap in timeline:
        graph_view.record(snap, journal)
    return graph_view.export_html(timeline, args.out or f"{args.db}.replay.html")


def _thoughts(db_path: str, limit: int) -> str:
    """El diario de la voz, de solo lectura y sin proveedores (como `oroi info`)."""
    if not Path(db_path).exists():
        return f"no existe {db_path}"
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute("SELECT turn, text, valence, surprise FROM thoughts "
                          "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    except sqlite3.OperationalError:
        return "esta memoria aún no tiene voz (se estrena al abrirla con la versión con consciencia)"
    finally:
        db.close()
    if not rows:
        return ("la voz no ha pensado nada todavía — actívala con `oroi chat --voice` o "
                "con consciousness_enabled=True en la config")
    lines = [f"t{r['turn']:>4}  {r['valence']:+d}{'  ⚡giro' if r['surprise'] else ''}\n"
             f"      {r['text']}" for r in reversed(rows)]
    return "\n".join(lines)


def _info(db_path: str) -> str:
    """Resumen de solo lectura, sin proveedores: cualquiera puede asomarse a una memoria."""
    if not Path(db_path).exists():
        return f"no existe {db_path} — graba una con `oroi chat` o examples/demo_conversation.py"
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    def q(sql: str):
        return db.execute(sql).fetchone()[0]

    turn = q("SELECT COALESCE(value, 0) FROM meta WHERE key = 'current_turn'")
    lines = [
        f"memoria: {db_path}",
        f"  turnos vividos: {turn} · nodos: {q('SELECT COUNT(*) FROM nodes WHERE faded = 0')} · "
        f"asociaciones: {q('SELECT COUNT(*) FROM edges')} · episodios: {q('SELECT COUNT(*) FROM episodes')}",
        "  ahora mismo activo:",
    ]
    for r in db.execute("SELECT label, activation FROM nodes WHERE faded = 0 AND activation > 0 "
                        "ORDER BY activation DESC LIMIT 5"):
        lines.append(f"    {r['label']}  ({r['activation']:.2f})")
    lines.append("  lo más consolidado:")
    for r in db.execute("SELECT label, base_strength FROM nodes WHERE base_strength > 0 "
                        "ORDER BY base_strength DESC LIMIT 5"):
        lines.append(f"    {r['label']}  ({r['base_strength']:.2f})")
    db.close()
    return "\n".join(lines)
