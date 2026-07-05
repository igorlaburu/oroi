"""Evaluación LIGERA de los dos mecanismos añadidos (fuera de la librería; solo consume Mind):

  uv run python -m evaluation.ablation

1. ANTI-HUB — ablación on/off sobre Recall@contexto en escenarios de distractores y
   convergencia. El config solo afecta la LECTURA, así que la red se construye UNA vez por
   escenario (lo caro) y se lee dos veces: sin anti-hub (suelo y supresión de hub apagados)
   vs con anti-hub. Aísla exactamente la contribución de los dos suelos.
2. SORPRESA — separación entre un turno de CONTINUACIÓN y uno de GIRO sobre el mismo hilo
   caliente: la señal sirve si los giros puntúan claramente por encima de las continuaciones.
"""

from pathlib import Path

from oroi import DynamicsConfig, Mind
from oroi.extraction.extractor import TurnExtractor
from oroi.providers.azure import AzureEmbedder, AzureLLM
from oroi.providers.settings import ProviderSettings

from .metrics import recall_at_context
from .scenario import CORPUS, load

ON = DynamicsConfig()
OFF = DynamicsConfig(evoke_keep_ratio=0.0, hub_degree=10**9)  # sin suelo relativo ni supresión de hub

# Hilos con su continuación (encaja) y su giro (rompe el tema), para medir la sorpresa.
THREADS = [
    {"setup": ["vamos a planear el viaje a Japón", "quiero ver los templos de Kioto",
               "y comer ramen en Tokio"],
     "cont": "también me gustaría subir al monte Fuji",
     "giro": "por cierto, me he hecho daño en la rodilla jugando al fútbol"},
    {"setup": ["estoy aprendiendo a tocar la guitarra", "practico acordes cada noche",
               "me encantaría tocar blues"],
     "cont": "he comprado un cancionero de blues",
     "giro": "mañana tengo que llevar el coche al taller"},
    {"setup": ["este año quiero cuidar más la alimentación", "como más verdura y fruta",
               "he reducido el azúcar"],
     "cont": "ahora desayuno avena con fruta",
     "giro": "el equipo ha fichado a un delantero nuevo"},
]


def _build():
    settings = ProviderSettings()
    azure = AzureLLM(settings)
    return AzureEmbedder(settings), TurnExtractor(azure), azure


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _replay(mind, turns, breaks):
    mind.wake()
    for i, t in enumerate(turns):
        mind.perceive(t)
        if i in breaks:
            mind.sleep()
            mind.wake()


def _window_turns(scenario):
    """Los turnos que el ChatSession real tendría en la ventana conversacional al preguntar:
    todo lo hablado desde el último corte de sesión (la ventana se vacía al dormir/despertar).
    Es lo que el LLM YA ve — `recall` los deduplica, no los re-inyecta."""
    win = set()
    for i in range(len(scenario.turns)):
        win.add(i + 1)                       # advance_turn empieza en 1
        if i in scenario.breaks_after:
            win = set()                      # nueva sesión: ventana en blanco
    return frozenset(win)


def anti_hub(scenarios, embedder, extractor, azure, tmp):
    print("\nANTI-HUB · Recall@contexto (acierta si trae el hecho correcto y NINGÚN distractor)")
    print(f"  {'familia':<16}{'sin anti-hub':>14}{'con anti-hub':>14}{'n':>5}")
    print("  " + "─" * 49)
    fam = {}
    for idx, sc in enumerate(scenarios):
        mind = Mind(str(tmp / f"ah{idx}.db"), embedder, extractor, judge=azure, config=ON)
        _replay(mind, sc.turns, sc.breaks_after)
        window = _window_turns(sc)           # lo que el LLM ya ve: recall lo deduplica (como el REPL)
        for probe in sc.probes:
            mind.config = OFF
            off = recall_at_context(mind.recall(probe.question, window_turns=window), probe)
            mind.config = ON
            on = recall_at_context(mind.recall(probe.question, window_turns=window), probe)
            fam.setdefault(sc.name, {"off": [], "on": []})
            fam[sc.name]["off"].append(off)
            fam[sc.name]["on"].append(on)
        mind.store.conn.close()
    alloff, allon = [], []
    for name, h in fam.items():
        alloff += h["off"]
        allon += h["on"]
        print(f"  {name:<16}{_mean(h['off']):>14.2f}{_mean(h['on']):>14.2f}{len(h['on']):>5}")
    print("  " + "─" * 49)
    print(f"  {'GLOBAL':<16}{_mean(alloff):>14.2f}{_mean(allon):>14.2f}{len(allon):>5}")


def surprise_signal(embedder, extractor, azure, tmp):
    print("\nSORPRESA · sobre el mismo hilo caliente (mayor = más ajeno al tema)")
    print(f"  {'hilo':<10}{'continuación':>14}{'giro':>10}{'separación':>13}")
    print("  " + "─" * 47)
    conts, giros = [], []
    for i, th in enumerate(THREADS):
        s = {}
        for kind in ("cont", "giro"):                 # red fresca por medición: mismo estado caliente
            mind = Mind(str(tmp / f"s{i}{kind}.db"), embedder, extractor, judge=azure, config=ON)
            mind.wake()
            for t in th["setup"]:
                mind.perceive(t)
            mind.perceive(th[kind])
            s[kind] = mind.last_surprise
            mind.store.conn.close()
        conts.append(s["cont"])
        giros.append(s["giro"])
        print(f"  hilo {i+1:<5}{s['cont']:>14.3f}{s['giro']:>10.3f}{s['giro']-s['cont']:>13.3f}")
    print("  " + "─" * 47)
    print(f"  {'MEDIA':<10}{_mean(conts):>14.3f}{_mean(giros):>10.3f}{_mean(giros)-_mean(conts):>13.3f}")
    avisa = sum(g >= ON.surprise_threshold for g in giros)
    calla = sum(c < ON.surprise_threshold for c in conts)
    print(f"  umbral de aviso = {ON.surprise_threshold}: avisa en {avisa}/{len(giros)} giros, "
          f"calla en {calla}/{len(conts)} continuaciones")


def main():
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="ablation-"))
    embedder, extractor, azure = _build()
    scenarios = [load(CORPUS / "distractores.jsonl")] + \
                [load(p) for p in sorted((CORPUS / "gen").glob("convergencia-*.jsonl"))]
    print(f"escenarios anti-hub: {len(scenarios)} (distractores + convergencia)")
    anti_hub(scenarios, embedder, extractor, azure, tmp)
    surprise_signal(embedder, extractor, azure, tmp)


if __name__ == "__main__":
    main()
