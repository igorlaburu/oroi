"""Conduce una conversación larga multisesión contra la mente real (yo, como Igor, hablando de
Gako AI Labs) para observar cómo CRECE el grafo a lo largo de sesiones — ¿representativo y
acotado, o hairball? Imprime cada turno y, al cerrar la sesión, las métricas del grafo.

    CHAT_PROVIDER=azure uv run python -m evaluation.conversation_drive <índice_sesión>

Cada invocación = una sesión nueva (ventana fresca), continuando sobre la MISMA base (la
activación se persiste, así que reabrir es continuar). Graba el diario para el replay web.
Fuera de la librería: solo usa la fachada vía ChatSession.
"""

import sys
from pathlib import Path

from oroi.chat.loop import build_chat, build_mind
from oroi.chat.session import ChatSession
from oroi.providers.settings import ProviderSettings
from oroi.viz import graph_view

DB = "empresa.db"

SESSIONS = [
    # 0 — Presentación y la empresa
    ["Hola, soy Igor Leroux, fundador de Gako AI Labs. Te voy a ir contando mi proyecto de empresa.",
     "Gako AI Labs es una startup tecnológica que integra inteligencia artificial en los procesos de las empresas.",
     "La constituí en septiembre de 2025, en Aralde.",
     "Soy el CEO y el CTO, socio único de momento.",
     "La sede está en el NabarLab, el hub tecnológico de Nabar, en la Plaza Nagusia.",
     "El capital social es de 15.000 euros y el CIF es B00000000.",
     "La forma jurídica es una sociedad limitada unipersonal.",
     "Mi marca principal es Gako AI, y tengo Ekimen en proceso de registro.",
     "Iniciamos actividad comercial en noviembre de 2025.",
     "En enero de 2026 cerramos los primeros clientes de pago.",
     "La facturación inicial fue diez veces superior a la prevista en el escenario base.",
     "Mi filosofía es AI-First: los agentes inteligentes son los protagonistas del flujo de trabajo.",
     "Trabajo con metodología Lean Startup: construir, medir, aprender.",
     "¿Te queda claro qué es Gako AI Labs y cuándo la fundé?"],
    # 1 — Ekimen y las verticales
    ["Te cuento la tecnología. La plataforma se llama Ekimen.",
     "Ekimen es un runtime común multi-tenant de agentes de IA verticalizados.",
     "Cada cliente recibe un agente adaptado a su sector, sobre el mismo runtime.",
     "La diferenciación entre verticales está en la configuración, no en el código.",
     "La primera vertical es Ekimen Press, para automatización editorial de medios digitales.",
     "Ekimen Press está en producción comercial desde enero de 2026.",
     "La segunda vertical es Ekimen Tech, para automatización industrial de pymes.",
     "Ekimen Tech está en fase de prospección, con visitas a empresas industriales.",
     "También está Ekimen base, disponible para servicios y comercio.",
     "Ekimen tiene 81.439 líneas de código propio.",
     "Tiene 161 endpoints de API y 78 componentes de interfaz.",
     "El margen bruto objetivo del SaaS es superior al 90%.",
     "¿Qué verticales de Ekimen te he contado?"],
    # 2 — Clientes
    ["Vamos con los clientes. El cliente principal es Araba Post.",
     "Araba Post usa Ekimen Press y paga una suscripción SaaS recurrente.",
     "Otro cliente es Nabar, que entró por consultoría de automatización.",
     "También tengo a Velmar como cliente.",
     "Y a Berlan SLU.",
     "En mayo de 2026 tengo 4 clientes en cartera.",
     "La facturación anualizada de los contratos es de 28.725 euros.",
     "El modelo combina consultoría de entrada con suscripción SaaS recurrente.",
     "Nabar además es donde tengo la oficina, en su hub NabarLab.",
     "Quiero llegar a entre 7 y 18 clientes SaaS activos en la segunda mitad de 2026.",
     "El objetivo de MRR es superar los 6.600 euros al mes.",
     "¿Qué cliente usa Ekimen Press?",
     "¿Te acuerdas de cuántos clientes tengo ahora?"],
    # 3 — Financiación y tutela
    ["Te cuento la financiación pública, que es importante.",
     "Me concedieron el programa Hasi, con 60.000 euros de ayuda.",
     "Presenté la PE26, Emprender en la Provincia, por unos 44.100 euros.",
     "También presenté la solicitud Impulsa, de la Organización de Comercio.",
     "Me seleccionaron en el programa de la Fundación Industrial.",
     "Y gané la categoría industrial del concurso Tu Proyecto Cuenta, de Araba Emprende.",
     "Tengo créditos cloud por 24.000 dólares en total.",
     "De AWS son 3.000, de Microsoft 15.000 y de Google 6.000.",
     "Me tutoriza el Fomento de Emprendimiento; mi tutor allí es Dorey Aminio.",
     "Y en la Organización de Comercio mi contacto es David Renaud.",
     "La marca Gako AI está registrada en las clases 9, 35 y 42.",
     "¿Quién es mi tutor en el Fomento de Emprendimiento?",
     "¿Cuánto me concedieron en el Hasi?"],
    # 4 — Mercado y competencia
    ["Hablemos del mercado. En Euskadi hay unas 12.000 empresas de 10 a 250 empleados.",
     "Solo el 17,4% de las empresas vascas usa inteligencia artificial.",
     "Eso deja más de 9.500 pymes como mercado potencial.",
     "Mi mercado objetivo son pymes consolidadas en proceso de digitalización.",
     "La competencia está polarizada.",
     "Por un lado, plataformas SaaS globales como Helpix, Mensae, Flowzy, Linko y Botaria.",
     "Por otro, startups y RPA para grandes como Pathbot, Solventia y Cumbre.ai.",
     "También herramientas genéricas como NovaAI, BigCloud AI y ModelHub.",
     "Y wrappers locales, como Wrap IA SL, en Olatza.",
     "Mi hueco es el que dejan las plataformas rígidas y las consultoras caras.",
     "El EU AI Act favorece a actores locales con garantías de cumplimiento.",
     "¿Quién es mi competencia en RPA para grandes empresas?",
     "¿Cuál dirías que es mi mayor oportunidad de mercado?"],
    # 5 — Trayectoria y equipo
    ["Te cuento mi trayectoria. Trabajé 17 años como responsable técnico en Public Company.",
     "Public Company es la sociedad pública de la administración foral, de infraestructuras de transporte.",
     "Antes estuve 6 años en Teknobide, en Marea, gestionando proyectos de transporte inteligente.",
     "Soy Ingeniero en Informática por la universidad pública.",
     "Hice un MBA Executive en una escuela de negocios, con programa en una universidad estadounidense.",
     "Y un Executive Master en Inteligencia Artificial.",
     "Hice la Machine Learning Specialization de una universidad de prestigio en 2023.",
     "Compagino Gako AI con mi puesto en Public Company, bajo compatibilidad autorizada.",
     "El plan es incorporar hasta 3 personas: desarrollo, negocio y operaciones.",
     "Para 2027 hay dos escenarios: base de 108.000 euros y acelerado de 435.000.",
     "El escenario acelerado contempla un equipo de 4 personas.",
     "¿Dónde trabajé antes de fundar Gako AI?",
     "¿Cuántos años estuve en Public Company?"],
    # 6 — Repaso + actualizaciones + multi-hop
    ["Vamos a repasar lo que recuerdas de todo esto.",
     "¿Quién soy y qué empresa fundé?",
     "¿Cómo se llama mi plataforma tecnológica?",
     "¿Qué cliente usa Ekimen Press y paga suscripción?",
     "Te actualizo un dato: ya no tengo 4 clientes, ahora tengo 6 clientes.",
     "¿Cuántos clientes tengo ahora?",
     "¿Quién me tutoriza en el Fomento de Emprendimiento?",
     "¿En qué ciudad tengo la sede?",
     "¿Cuánto dinero me concedió el Hasi?",
     "Otra actualización: la PE26 ya me la han concedido, ya no está solo presentada.",
     "¿En qué estado está ahora la PE26?",
     "¿Qué vertical de Ekimen está en producción y cuál en prospección?",
     "¿Dónde trabajé 17 años antes de Gako AI?",
     "¿Quién es mi competencia local, la de Olatza?",
     "¿Te acuerdas de mi filosofía de producto?"],
]


def metrics(mind) -> str:
    c = mind.store.conn
    n = c.execute("SELECT COUNT(*) c FROM nodes WHERE faded=0").fetchone()["c"]
    e = c.execute("SELECT COUNT(*) c FROM edges").fetchone()["c"]
    nr = c.execute("SELECT COUNT(*) c FROM edges WHERE rel IS NULL").fetchone()["c"]
    return f"nodos={n} · aristas={e} · ar/nodo={e/max(n,1):.1f} · sin_rel={nr} ({100*nr//max(e,1)}%)"


def main() -> None:
    idx = int(sys.argv[1])
    mind = build_mind(DB)
    mind.wake()
    journal = graph_view.timeline_path(DB)
    session = ChatSession(mind, build_chat(ProviderSettings()),
                          on_turn=lambda snap: graph_view.record(snap, journal))
    state = Path(f"{DB}.lastreply")            # hila la última respuesta entre sesiones (procesos)
    if idx > 0 and state.exists():             # así el 1er turno de la sesión percibe la respuesta
        session.last_reply = state.read_text(encoding="utf-8")  # previa (no salen 2 turnos de usuario seguidos)
    print(f"\n===== SESIÓN {idx + 1}/{len(SESSIONS)} =====")
    for text in SESSIONS[idx]:
        try:
            reply = session.turn(text)             # el perceive ya ocurrió aunque el chat falle
        except Exception as error:                 # filtro de contenido / límite del proveedor: no tumbar el run
            session.last_reply = ""                # no arrastrar una respuesta previa colgada al siguiente turno
            print(f"\ntú> {text}\n⚠️ turno omitido ({type(error).__name__})")
            continue
        print(f"\ntú> {text}\nmente> {reply}")
    state.write_text(session.last_reply, encoding="utf-8")
    report = mind.sleep()                      # corte de sesión: consolida
    print(f"\n--- fin sesión {idx + 1} · GRAFO: {metrics(mind)}"
          f" · sueño: +{report.promoted} aristas, {len(report.merged)} fusiones, {report.faded} podas ---")


if __name__ == "__main__":
    main()
