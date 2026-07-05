"""El extractor: convierte cada turno en una mini-red semántica (SPEC §5, paso 1)."""

import json

from .schema import MiniNetwork

CUES_PROMPT = """\
Eres el sistema de evocación de una memoria asociativa. Dada una CONSULTA del usuario, \
extrae los CONCEPTOS-PISTA por los que rebuscar en la memoria: entidades, atributos, \
temas y el ASUNTO de la pregunta (incluido el verbo o la categoría: "trabajo", "ciudad", \
"color"). NO filtres por trámite: incluso una pregunta vaga tiene pistas. \
Devuelve SOLO este JSON: {"cues": ["...", "..."]}. Labels cortos, minúsculas, y SIEMPRE \
en el idioma de la consulta (consulta en inglés → pistas en inglés; nunca traduzcas)."""

EXTRACTION_PROMPT = """\
Eres el sistema de percepción de una memoria asociativa. Convierte el turno de \
conversación en una mini-red semántica con este JSON exacto:
{"nodes": [{"label": "...", "kind": "entity|attribute|event|concept", "salience": 0.5}],
 "edges": [{"src": "...", "dst": "...", "rel": "...", "symmetric": false}]}

Recibes dos secciones: [CONTEXTO RECIENTE] y [TURNO ACTUAL].

Reglas:
- RIGOR ABSOLUTO: genera nodos y aristas EXCLUSIVAMENTE a partir del [TURNO ACTUAL]. \
[CONTEXTO RECIENTE] es solo para entender referencias (correferencia, elipsis: "lo", \
"ese", "también"): NUNCA extraigas un hecho que solo aparezca en el contexto. Si el \
turno actual no afirma nada propio (p.ej. solo una pregunta o un "vale"), devuelve vacío.
- El turno actual puede traer líneas [usuario] y [asistente]. El protagonista es el \
USUARIO: extrae sus hechos, preferencias, eventos y personas. De lo dicho por el \
[asistente] extrae SOLO hechos nuevos que el usuario haya pedido o acogido, con \
salience ≤ 0.4 — ignora sus listados, explicaciones enciclopédicas y cortesías.
- Nodos: solo entidades, atributos, eventos o conceptos con valor de recuerdo. \
Labels cortos, en minúsculas, y SIEMPRE en el idioma del turno (turno en inglés → \
labels en inglés; NUNCA traduzcas).
- CONCRETO, no temático: prefiere lo NOMBRADO en el turno (personas, lugares, objetos, \
cantidades, fechas) a los temas abstractos. "fui a un grupo de apoyo y fue muy potente" \
→ nodo "grupo de apoyo" (evento concreto), NO nodos "apoyo"/"emoción". Un tema abstracto \
solo si es el asunto mismo del hecho. OJO: la concreción NO anula el filtro de trámite — \
un comentario pasajero (tráfico, tiempo, ganas, "tengo una llamada luego") sigue siendo \
{"nodes": [], "edges": []}: no crees nodos de circunstancias sin valor de recuerdo \
("mañana", "esta semana", "un rato").
- HABLANTE NOMBRADO: si la línea tiene el formato "Nombre: texto" (con o sin fecha \
delante), ese Nombre es un nodo OBLIGATORIO y es el SUJETO de todo hecho en primera \
persona de esa línea. "Melanie: I painted a sunrise" → nodos "melanie" y "sunrise", \
arista melanie→sunrise "painted". "Caroline: I went to a support group" → \
caroline→support group "went_to". La reificación de primera persona (ver abajo) aplica \
SOLO cuando el hablante es el [usuario] anónimo, nunca cuando tiene nombre.
- Las "rel" también en el idioma del turno (turno en inglés → "painted", "works_at"; \
nunca mezcles idiomas en una rel).
- Aristas: relaciones realmente expresadas en el texto, con "rel" descriptivo corto \
("tiene_color", "vive_en", "ocurre_antes_de"). "symmetric": true solo si la relación \
no tiene dirección ("hermano_de").
- CRÍTICO — cada extremo ("src"/"dst") DEBE ser un nodo declarado; NUNCA inventes un \
sujeto implícito ("usuario", "yo", "mi"). Distingue dos casos:
  · Relación entre DOS entidades NOMBRADAS → conéctalas DIRECTAMENTE, la acción es el \
"rel", sin nodo intermedio: "Diego trabaja en Repsol" → diego→repsol "trabaja_en".
  · Hecho en PRIMERA persona (sujeto implícito = el usuario) → reifica el CONCEPTO del \
hecho como un SUSTANTIVO declarado como nodo (nunca el verbo suelto) y conéctalo, para \
que sea recuperable: "trabajo en Madrid" → trabajo→madrid "ubicado_en"; "vivo en Granada" \
→ residencia→granada "en"; "mi coche es rojo" → coche→rojo "tiene_color".
  · PERTENENCIA a una categoría afirmada por el USUARIO ("X es cliente", "otro cliente es Y", \
"tengo de cliente a Z") → arista entidad→categoría con rel "es", la categoría como nodo en \
SINGULAR ("cliente", "proveedor", "amigo"): "Araba Post es cliente" → araba post→cliente "es". Así la \
categoría reúne a sus miembros y se pueden LISTAR. NUNCA crees pertenencia a partir de una línea \
[asistente]: sus suposiciones o listados no son hechos del usuario (envenenarían la red).
- "salience" (0-1) mide importancia y carga emocional del hecho. Reserva ≥ 0.9 \
EXCLUSIVAMENTE para emoción extrema o cuando el usuario marca explícitamente la \
importancia ("es muy importante que recuerdes que..."). Lo trivial: ≤ 0.3.
- Turnos de puro trámite ("vale", "sigue", saludos): devuelve {"nodes": [], "edges": []}.
Devuelve SOLO el JSON, sin comentarios."""


class TurnExtractor:
    """Llama al LLM rápido y valida contra el schema; un reintento y, si no, mini-red vacía."""

    def __init__(self, llm):
        self.llm = llm  # cualquier objeto con complete_json(system, user) -> str

    def extract(self, text: str, context: str = "") -> MiniNetwork:
        """`context` = ventana deslizante de turnos previos, SOLO para desambiguar correferencias del
        turno (no para extraer). La extracción se ancla SIEMPRE al turno actual (`text`); el rigor lo
        impone el prompt."""
        message = self._compose(text, context)
        for _ in range(2):
            try:
                return MiniNetwork.model_validate_json(self.llm.complete_json(EXTRACTION_PROMPT, message))
            except Exception:
                continue
        return MiniNetwork()  # mejor percibir nada que envenenar la red

    @staticmethod
    def _compose(text: str, context: str) -> str:
        parts = []
        if context:
            parts.append("[CONTEXTO RECIENTE] solo para resolver referencias; NO extraigas de aquí:\n" + context)
        parts.append("[TURNO ACTUAL] extrae nodos y aristas SOLO de este turno:\n" + text)
        return "\n\n".join(parts)

    def extract_cues(self, text: str) -> list[str]:
        """Modo consulta: los conceptos-pista de una pregunta (sin filtro de trámite),
        para sembrar la resonancia/evocación del recall — distinto de extraer hechos."""
        for _ in range(2):
            try:
                data = json.loads(self.llm.complete_json(CUES_PROMPT, text))
                return [c.strip().lower() for c in data.get("cues", []) if c.strip()]
            except Exception:
                continue
        return []
