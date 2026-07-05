"""Generador paramétrico de corpus: instancia plantillas con variación aleatoria.

    uv run python -m evaluation.generate --per-type 50 --seed 7

Produce muchos escenarios (corpus/gen/*.jsonl) con ground truth automático, para
una evaluación con tamaño de muestra de verdad. Cuatro fenómenos: recurrencia,
actualización, distractores y multi-sesión (que ejercita el sueño). Determinista
por semilla. NO favorece a ninguna condición: el dato/posición/longitud se varían
al azar, incluido el caso difícil (datos lejanos, muchos distractores).
"""

import argparse
import json
import random
from pathlib import Path

GEN = Path(__file__).parent / "corpus" / "gen"

NAMES = ["Pablo", "Laura", "Carlos", "Marta", "Ana", "Luis", "Sofía", "Diego", "Elena", "Javier"]
FEMALE = {"Laura", "Marta", "Ana", "Sofía", "Elena"}  # para concordar hermano/hermana en multihop3
COLORS = ["rojo", "azul", "verde", "negro", "amarillo", "naranja", "morado", "gris", "blanco"]
JOBS = ["enfermero", "profesora", "cocinero", "abogado", "fontanero", "dentista", "arquitecto", "panadero"]
CITIES = ["Bilbao", "Madrid", "Sevilla", "Valencia", "Vigo", "Granada", "Burgos", "Murcia", "Gijón"]
COMPANIES = ["Eroski", "Mercadona", "Telefónica", "Iberdrola", "Inditex", "Repsol", "Naturgy", "Cabify"]

FILLER = [
    "hoy ha hecho un día estupendo", "estoy un poco cansado esta semana",
    "ayer vi una película muy entretenida", "me apetece un café ahora mismo",
    "el tráfico estaba imposible esta mañana", "he quedado con unos amigos el sábado",
    "estoy leyendo un libro muy interesante", "tengo que hacer la compra luego",
    "me he comprado unas zapatillas nuevas", "el otro día fui al cine",
    "hace bastante frío para la época", "tengo ganas de que llegue el finde",
    "he empezado a hacer deporte por las tardes", "me gusta pasear por la mañana",
    "ayer cené algo ligero en casa", "estoy pensando en cambiar de móvil",
    "el jardín necesita un buen riego", "he dormido fatal esta noche",
    "tengo una llamada pendiente para mañana", "me he aficionado a los crucigramas",
]


def _filler(rng, n):
    return [{"type": "turn", "user": rng.choice(FILLER)} for _ in range(n)]


def recurrencia(rng):
    color = rng.choice(COLORS)
    turns = [{"type": "turn", "user": f"me he comprado un coche de color {color}"}]
    turns += _filler(rng, rng.randint(6, 16))   # cambio de tema, longitud variable
    probe = {"type": "probe", "question": "¿de qué color es mi coche?",
             "expect_context": [color], "expect_answer": color}
    return [{"type": "meta", "name": "recurrencia", "goal": "dato temprano, pregunta tardía"}, *turns, probe]


def actualizacion(rng):
    old, new = rng.sample(CITIES, 2)
    turns = [{"type": "turn", "user": f"vivo en {old}"}]
    turns += _filler(rng, rng.randint(3, 8))
    turns += [{"type": "turn", "user": f"me he mudado, ahora vivo en {new}"}]
    turns += _filler(rng, rng.randint(3, 8))
    probe = {"type": "probe", "question": "¿dónde vivo ahora?",
             "expect_context": [new], "avoid_context": [old], "expect_answer": new}
    return [{"type": "meta", "name": "actualizacion", "goal": "dato actualizado: el vigente, no el viejo"}, *turns, probe]


def distractores(rng):
    people = rng.sample(NAMES, 3)
    jobs = rng.sample(JOBS, 3)
    target = rng.randint(0, 2)
    turns = [{"type": "turn", "user": f"{p} trabaja de {j}"} for p, j in zip(people, jobs)]
    rng.shuffle(turns)
    turns += _filler(rng, rng.randint(4, 10))
    probe = {"type": "probe", "question": f"¿en qué trabaja {people[target]}?",
             "expect_context": [jobs[target]],
             "avoid_context": [jobs[i] for i in range(3) if i != target], "expect_answer": jobs[target]}
    return [{"type": "meta", "name": "distractores", "goal": "varios referentes, mismo predicado"}, *turns, probe]


def multisesion(rng):
    city = rng.choice(CITIES)
    turns = [{"type": "turn", "user": f"trabajo en {city}"}]
    turns += _filler(rng, rng.randint(2, 5)) + [{"type": "session_break"}]
    turns += _filler(rng, rng.randint(3, 7)) + [{"type": "session_break"}]
    turns += _filler(rng, rng.randint(2, 5))
    probe = {"type": "probe", "question": "¿en qué ciudad trabajo?",
             "expect_context": [city], "expect_answer": city}
    return [{"type": "meta", "name": "multisesion", "goal": "persistencia tras dormir y cambiar de sesión"}, *turns, probe]


def multihop(rng):
    """Respuesta a DOS saltos, con el puente separado y ausente de la pregunta: persona→empresa→ciudad.
    La similitud (RAG) no salta —"¿en qué ciudad trabaja X?" no se parece a "Eroski está en Bilbao"—;
    la red, evocando el camino, sí. Es el terreno de la ventaja estructural."""
    person = rng.choice(NAMES)
    company = rng.choice(COMPANIES)
    city = rng.choice(CITIES)
    hop1 = {"type": "turn", "user": f"{person} trabaja en {company}"}
    hop2 = {"type": "turn", "user": f"{company} tiene su sede en {city}"}
    turns = [hop1] + _filler(rng, rng.randint(3, 8)) + [hop2] + _filler(rng, rng.randint(3, 8))
    probe = {"type": "probe", "question": f"¿en qué ciudad trabaja {person}?",
             "expect_context": [city], "expect_answer": city}
    return [{"type": "meta", "name": "multihop", "goal": "respuesta a dos saltos; el puente no está en la pregunta"},
            *turns, probe]


def multihop3(rng):
    """Respuesta a TRES saltos: hermano→empresa→ciudad. La similitud se pierde aún más lejos."""
    a, b = rng.sample(NAMES, 2)
    company, city = rng.choice(COMPANIES), rng.choice(CITIES)
    rel_a = "hermana" if a in FEMALE else "hermano"      # a respecto de b
    rel_b = "la hermana" if b in FEMALE else "el hermano"  # b (quien trabaja) respecto de a
    turns = [{"type": "turn", "user": f"{a} es {rel_a} de {b}"}]
    turns += _filler(rng, rng.randint(2, 5)) + [{"type": "turn", "user": f"{b} trabaja en {company}"}]
    turns += _filler(rng, rng.randint(2, 5)) + [{"type": "turn", "user": f"{company} tiene su sede en {city}"}]
    turns += _filler(rng, rng.randint(2, 5))
    probe = {"type": "probe", "question": f"¿en qué ciudad trabaja {rel_b} de {a}?",
             "expect_context": [city], "expect_answer": city}
    return [{"type": "meta", "name": "multihop3", "goal": "respuesta a tres saltos; el puente está lejos de la pregunta"},
            *turns, probe]


def convergencia(rng):
    """Intersección de dos pistas: la respuesta es quien cumple AMBAS; los distractores, solo una.
    RAG trae a todos (médicos + gente de la ciudad) → cuela distractores; la convergencia interseca."""
    target, dp, dc = rng.sample(NAMES, 3)
    job, city = rng.choice(JOBS), rng.choice(CITIES)
    turns = [{"type": "turn", "user": f"{target} es {job}"},
             {"type": "turn", "user": f"{target} vive en {city}"},
             {"type": "turn", "user": f"{dp} es {job}"},        # mismo oficio, otra ciudad
             {"type": "turn", "user": f"{dc} vive en {city}"}]  # misma ciudad, otro oficio
    rng.shuffle(turns)
    turns += _filler(rng, rng.randint(2, 5))
    probe = {"type": "probe", "question": f"¿quién es {job} y vive en {city}?",
             "expect_context": [target], "avoid_context": [dp, dc], "expect_answer": target}
    return [{"type": "meta", "name": "convergencia", "goal": "intersección de dos pistas; los distractores cumplen solo una"},
            *turns, probe]


GENERATORS = {"recurrencia": recurrencia, "actualizacion": actualizacion,
              "distractores": distractores, "multisesion": multisesion, "multihop": multihop,
              "multihop3": multihop3, "convergencia": convergencia}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=50, help="instancias por fenómeno")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--types", nargs="*", default=None, help="fenómenos a generar (por defecto, todos)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    GEN.mkdir(parents=True, exist_ok=True)
    for old in GEN.glob("*.jsonl"):
        old.unlink()
    gens = GENERATORS if not args.types else {t: GENERATORS[t] for t in args.types}
    total = 0
    for tipo, fn in gens.items():
        for i in range(args.per_type):
            lines = fn(rng)
            (GEN / f"{tipo}-{i:03d}.jsonl").write_text(
                "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n", encoding="utf-8")
            total += 1
    print(f"generados {total} escenarios en {GEN} ({args.per_type} por tipo × {len(GENERATORS)} tipos)")


if __name__ == "__main__":
    main()
