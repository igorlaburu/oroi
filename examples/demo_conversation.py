"""Demo: una conversación cotidiana (3 sesiones, ~44 turnos) percibida por Oroi.

Temas diferenciados (trabajo/editorial, gimnasio, familia, el perro, un viaje) con
recurrencias a propósito, para ver consolidación y decaimiento. Requiere credenciales
de Azure en .env (ver .env.example). Al terminar:

    uv run oroi viz --db examples/demo.db     # la película de la memoria
    uv run oroi serve --db examples/demo.db   # visor + chat en vivo
"""

import os

from oroi.extraction.extractor import TurnExtractor
from oroi.mind import Mind
from oroi.providers.azure import AzureEmbedder, AzureLLM
from oroi.providers.settings import ProviderSettings
from oroi.viz import graph_view

DB = os.path.join(os.path.dirname(__file__), "demo.db")

SESSIONS = [
    # ── sesión 1: presentación de la vida ──
    ["trabajo en una editorial, mi oficina está en Madrid",
     "mi jefa se llama Lucía, llevamos cuatro años trabajando juntos",
     "vivo en el barrio de Chamberí, cerca del mercado",
     "los martes y jueves voy al gimnasio al salir del trabajo",
     "mi entrenador del gimnasio se llama Andrés y es muy exigente",
     "tengo un perro que se llama Truco, un mestizo tranquilo",
     "los sábados por la mañana llevo a Truco al parque del Retiro",
     "mi hermano Pablo vive en Valencia, trabaja de enfermero en el hospital La Fe",
     "Pablo tiene dos hijas, Carla y Vega, mis sobrinas",
     "estoy leyendo una novela de misterio que me recomendó Lucía",
     "los domingos me gusta cocinar arroces, me sale bien la paella",
     "compro el pescado en el mercado de Chamberí, en el puesto de Manoli",
     "este año quiero apuntarme a clases de inglés",
     "en la editorial estamos preparando la feria del libro",
     "la feria del libro es en junio y me toca organizar la caseta"],
    # ── sesión 2: avanzan los temas ──
    ["los martes y jueves sigo yendo al gimnasio, no fallo ninguna semana",
     "Andrés me ha puesto una rutina nueva de fuerza",
     "me he lesionado un poco el hombro con el press banca",
     "en el trabajo hemos cerrado el catálogo de otoño",
     "Lucía me ha pedido que presente el catálogo en la feria del libro",
     "estoy nervioso con la presentación, es mi primera vez hablando en público",
     "el mes que viene viajo a Valencia a ver a Pablo y a las niñas",
     "quiero llevar a Truco a Valencia, a mis sobrinas les encanta",
     "Carla, la mayor, quiere ser veterinaria por Truco",
     "he empezado las clases de inglés los lunes por la tarde",
     "mi profesora de inglés se llama Emma y es de Manchester",
     "el domingo hice paella para seis, vinieron mis amigos del gimnasio",
     "Manoli me guardó sepia fresca para el arroz negro",
     "he terminado la novela de misterio, el final era buenísimo",
     "le he devuelto la novela a Lucía y me ha prestado otra del mismo autor"],
    # ── sesión 3: lo reciente (esto queda caliente en la figura) ──
    ["la semana que viene es la feria del libro, ya está todo listo",
     "he ensayado la presentación del catálogo tres veces con Lucía",
     "Andrés dice que el hombro ya está casi recuperado",
     "en el gimnasio he vuelto a entrenar con normalidad",
     "después de la feria me tomaré unos días en Valencia con Pablo",
     "ya tengo los billetes del tren a Valencia para julio",
     "Truco viene conmigo en el tren, ya le compré el transportín",
     "mis sobrinas me han pedido que les enseñe a hacer paella",
     "Emma dice que mi inglés mejora rápido, ya mantengo conversaciones",
     "estoy pensando en apuntarme al viaje de la editorial a Lisboa en otoño",
     "el viaje a Lisboa es para la feria del libro portuguesa",
     "Lucía también viene a Lisboa, iremos juntos en avión",
     "este sábado llevo a Truco al Retiro y luego ensayo la presentación",
     "tengo muchas ganas de que llegue junio con la feria y el viaje"],
]

settings = ProviderSettings()
azure = AzureLLM(settings)
mind = Mind(DB, AzureEmbedder(settings), TurnExtractor(azure), judge=azure)
diary = graph_view.timeline_path(DB)

n = sum(len(s) for s in SESSIONS)
i = 0
for si, session in enumerate(SESSIONS):
    for text in session:
        i += 1
        mind.perceive(text)
        graph_view.record(mind.introspect(), diary)
        print(f"[{i:2}/{n}] {text[:60]}")
    if si < len(SESSIONS) - 1:
        report = mind.sleep()
        mind.wake()
        print(f"  … sueño tras sesión {si+1}: {report}")

print("hecho →", DB)
