"""Anonimiza la conversación de la DEMO sobre evaluation/conversation_drive.py:

    uv run python -m evaluation.anonymize           # dry-run: cuenta coincidencias, NO escribe
    uv run python -m evaluation.anonymize --apply   # reescribe conversation_drive.py en sitio

- MAPPING: nombre real → nombre demo. Gako y la familia Ekimen SE QUEDAN (es el producto que se
  muestra). Los competidores públicos (Zendesk, UiPath…) se dejan salvo que se añadan aquí.
- MONEY: cifras de INGRESOS/dinero ×3. Contrataciones, %, métricas de código y recuentos NO cambian.
- Las claves se aplican de más LARGA a más corta (así «Ekimen Press» no se rompe por «Ekimen»,
  «Cámara de Comercio de Álava» antes que «Cámara de Comercio»).

El original es recuperable por git. Revisar el dry-run antes de --apply (la demo es pública)."""

import re
import sys
from pathlib import Path

TARGET = Path(__file__).parent / "conversation_drive.py"

# ── personas, clientes, instituciones, programas (Gako/Ekimen NO se tocan) ──────────────────
MAPPING = {
    # personas
    "Igor Laburu": "Igor Leroux",
    "Diego Apellániz": "Dorey Aminio",
    "David Álvarez": "David Renaud",
    # clientes
    "Zuk.eus": "Araba Post",
    "zut.eus": "Araba Post",        # typo/duplicado real del mismo cliente
    "Cocubo": "Nabar",
    "Trovo": "Velmar",
    "Iteraer SLU": "Berlan SLU",
    "Iteraer": "Berlan",            # forma corta (sin SLU) que aparece suelta en notas
    "ByteLab": "NabarLab",          # el hub tecnológico del cliente (Cocubo→Nabar)
    # instituciones / organismos
    "Cámara de Comercio de Álava": "Organización de Comercio",
    "Cámara de Comercio": "Organización de Comercio",
    "BIC Araba": "Fomento de Emprendimiento",
    "ARABAT": "Public Company",
    "Diputación Foral de Álava": "administración foral",
    "Ikusi": "Teknobide",
    "ESEUNE": "una escuela de negocios",
    "Fundación Michelin": "Fundación Industrial",
    "Álava Emprende": "Araba Emprende",
    # programas / ayudas / concursos
    "Ekintzaile": "Hasi",
    "Emprender en Álava": "Emprender en la Provincia",
    "EA26": "PE26",
    "Despega": "Impulsa",
    "Tu Idea Cuenta": "Tu Proyecto Cuenta",
    "Wrappers IA SL": "Wrap IA SL",
    # competidores (anonimizados a petición; los proveedores cloud AWS/Microsoft/Google se quedan)
    "Zendesk": "Helpix", "Intercom": "Mensae", "Zapier": "Flowzy", "Make": "Linko",
    "Landbot": "Botaria", "UiPath": "Pathbot", "Indra Minsait": "Solventia",
    "Sherpa.ai": "Cumbre.ai", "Google Cloud AI": "BigCloud AI", "OpenAI": "NovaAI",
    "Hugging Face": "ModelHub",
    # universidades → generalizadas
    "EHU-UPV": "la universidad pública",
    "Georgetown": "una universidad estadounidense",
    "Stanford": "una universidad de prestigio",
    # ciudades → ficticias (se mantiene la provincia Álava/Araba como ambiente)
    "Vitoria-Gasteiz": "Aralde",
    "Plaza de Zaldiarán": "Plaza Nagusia",
    "Donostia": "Marea",
    "Durango": "Olatza",
    # identificador fiscal real → ficticio
    "B22925135": "B00000000",
}

# ── cifras de ingresos/dinero ×3 (frases completas para no pillar números sueltos) ──────────
MONEY = {
    "9.575 euros": "28.725 euros",                                    # facturación anualizada
    "2.200 euros": "6.600 euros",                                     # MRR objetivo
    "base de 36.000 euros y acelerado de 145.000": "base de 108.000 euros y acelerado de 435.000",
    "20.000 euros de ayuda": "60.000 euros de ayuda",                # Ekintzaile/Hasi
    "por unos 14.700 euros": "por unos 44.100 euros",                # EA26/PE26
    "8.000 dólares en total": "24.000 dólares en total",             # créditos cloud
    "AWS son 1.000, de Microsoft 5.000 y de Google 2.000": "AWS son 3.000, de Microsoft 15.000 y de Google 6.000",
    "capital social es de 3.000 euros": "capital social es de 15.000 euros",  # a 15.000 por decisión
}


def _replace(old: str, new: str, text: str, ci: bool) -> tuple[str, int]:
    if not ci:
        return (text.replace(old, new), text.count(old))
    def repl(m):  # case-insensitive con preservación de caso (las notas usan labels en minúscula)
        s = m.group(0)
        return new.upper() if s.isupper() else (new[:1].upper() + new[1:] if s[:1].isupper() else new.lower())
    return re.subn(re.escape(old), repl, text, flags=re.IGNORECASE)


def transform(text: str, money: bool = True, ci: bool = False) -> tuple[str, list[tuple[str, int]]]:
    table = {**MAPPING, **MONEY} if money else dict(MAPPING)
    counts = []
    for old, new in sorted(table.items(), key=lambda kv: -len(kv[0])):
        text, n = _replace(old, new, text, ci)
        counts.append((old, new, n))
    return text, counts


def _process(path: Path, money: bool, apply: bool) -> None:
    raw = path.read_text(encoding="utf-8")
    out, counts = transform(raw, money=money, ci=not money)  # ficheros extra (solo nombres) → case-insensitive
    hits = [c for c in counts if c[2]]
    print(f"\n{path}  ({'nombres+dinero' if money else 'solo nombres'}):")
    for old, new, n in hits:
        print(f"  {old:<40}→ {new:<30}{n:>3}")
    if not hits:
        print("  (sin coincidencias)")
    if apply and out != raw:
        path.write_text(out, encoding="utf-8")
        print("  ✅ aplicado")


def main() -> None:
    args = sys.argv[1:]
    apply = "--apply" in args
    extra = [Path(a) for a in args if not a.startswith("--")]
    if extra:                       # ficheros de texto extra: SOLO nombres (el ×3 es de la conversación)
        for p in extra:
            _process(p, money=False, apply=apply)
    else:                           # por defecto: la conversación, con nombres + dinero ×3
        _process(TARGET, money=True, apply=apply)
    if not apply:
        print("\n(dry-run; nada escrito. Añade --apply cuando el mapeo esté validado.)")


if __name__ == "__main__":
    main()
