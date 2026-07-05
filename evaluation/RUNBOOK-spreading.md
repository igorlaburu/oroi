# RUNBOOK — ablación de la equiparación en el spreading

Guion para ejecutar y leer la ablación de `spread_equalizes` **de principio a fin**. Está
escrito para que lo siga un agente (p. ej. una sesión de Claude Code con un modelo económico)
sin necesitar más contexto que este fichero. El coste grande es la API de Azure (extracción +
embeddings al reconstruir las redes), no el modelo que pilota.

## Qué se mide y por qué

El spreading histórico **suma a los vecinos sin restar al origen**: un subgrafo conectado se
realimenta y queda caliente para siempre (el "motor perpetuo" — demostrado en
`tests/test_activation.py::test_equalize_kills_the_perpetual_motion`). El fix propuesto es
**equiparación por gradiente** (`spread_equalizes=True`): el calor solo fluye del más caliente
al más frío, proporcional a la diferencia, el origen cede lo que fluye, y al igualarse el flujo
para. Como ya no se fabrica energía, el nivel absoluto baja → se recalibra `boost=2.0`.

La duda que decide esta ablación: **¿la equiparación mantiene (o mejora) el recall global?**
Y de propina: ¿el flujo contra la flecha (backward atenuado) aporta, o basta con ir a favor?

Variantes (definidas en `evaluation/spreading.py::VARIANTS`):

| variante | qué es |
|---|---|
| `histórico` | dinámica actual por defecto (suma sin restar) — la referencia |
| `equipara+b2.0` | gradiente + backward atenuado por `back_factor` — el diseño propuesto |
| `eq-flecha+b2.0` | gradiente solo a favor de la flecha (`spread_back_factor=0`; la evocación conserva su backward) |

## Prerrequisitos

- `.env` en la raíz con las claves de Azure (`AZURE_API_BASE`, `AZURE_API_KEY`, deployments).
  Si falta, PARAR y pedirlas al usuario.
- No tocar NADA en `src/oroi/` — esta ablación no cambia defaults del core. Lo único
  editable es el dict `VARIANTS` de `evaluation/spreading.py` (paso 4, si hace falta).

## Pasos

### 1. Sanidad (rápido, gratis)

```bash
uv run pytest -q && uv run ruff check .
```

Si algo falla, PARAR y reportarlo — no seguir con la corrida cara.

### 2. La corrida (larga y con coste Azure)

```bash
uv run python -m evaluation.spreading 2>&1 | tee evaluation/spreading-results.txt
```

- Duración orientativa: cada variante reconstruye su red percibiendo el corpus entero
  (3 variantes × todos los escenarios). Cuenta con **decenas de minutos a >1h**.
- El script ya reintenta 3 veces por escenario y omite los que el filtro de contenido de
  Azure tire; verás `⚠️ ... omitido` — es aceptable si son pocos (≤2 por variante).
  Si se omiten muchos, PARAR y reportar.
- No lanzar variantes en paralelo (Azure en lotes pequeños, por rate-limit y coste).

### 3. Lectura de la tabla

La tabla es Recall@contexto por familia de escenarios y GLOBAL, una columna por variante.
Criterios de decisión, en orden:

1. **`equipara+b2.0` GLOBAL ≥ `histórico` − 0.02** → el fix estructural es **sano**:
   corrige el motor perpetuo sin perder recall. Recomendación: encender por defecto
   (pero NO cambiarlo tú — es decisión del usuario, solo déjalo anotado).
2. **`equipara+b2.0` GLOBAL claramente < `histórico`** (más de 0.05) → falta recalibrar:
   ir al paso 4.
3. **`eq-flecha` vs `equipara`**: si la variante solo-flecha pierde — sobre todo en las
   familias multihop y de categoría/miembros (la asociación lateral "cliente → hermanos"
   necesita el backward) — el diseño con backward atenuado **se queda**. Si empatan,
   anotarlo: sería argumento para simplificar.
4. **Mirar familias, no solo el global.** `multihop3` es la ventaja estructural del
   paradigma: si una variante la hunde, descartarla aunque suba el global. Y al revés:
   una mejora que solo mueve UNA familia es sobreajuste, no señal (principio del
   proyecto: el benchmark es termómetro, no diana).

### 4. (Solo si el paso 3.2 lo pide) recalibrar boost

Editar `VARIANTS` en `evaluation/spreading.py` dejando solo las variantes nuevas a probar
(las ya medidas no hace falta repetirlas), p. ej.:

```python
VARIANTS = {
    "equipara+b1.5": DynamicsConfig(spread_equalizes=True, boost=1.5),
    "equipara+b2.5": DynamicsConfig(spread_equalizes=True, boost=2.5),
    "equipara+b3.0": DynamicsConfig(spread_equalizes=True, boost=3.0),
}
```

Relanzar el paso 2 (guardando en `spreading-results-boost.txt`) y volver al paso 3.
Máximo UNA ronda de recalibración; si sigue perdiendo, parar y reportar — tocará pensar
(p. ej. "primar solo desde lo fresco"), y eso no se decide en este runbook.

### 5. Registro

- Pegar la(s) tabla(s) final(es) con fecha en `CLAUDE.md`, sección
  "RETOMAR → SPREADING", junto al veredicto según los criterios del paso 3.
- Dejar `evaluation/spreading-results*.txt` sin borrar (no committear: son artefactos).
- Revertir cualquier edición de `VARIANTS` (que el fichero quede con las 3 variantes
  originales: `git checkout evaluation/spreading.py` si se tocó).
- NO cambiar defaults en `src/oroi/core/config.py`.
- Resumen final para el usuario: tabla global, veredicto por criterio, y la recomendación
  (encender `spread_equalizes` por defecto o no, y con qué boost).

## Cómo lanzar esto con un modelo económico

Desde la raíz del repo:

```bash
claude --model sonnet
```

y como prompt:

> Lee evaluation/RUNBOOK-spreading.md y síguelo paso a paso de principio a fin.
> Al acabar, registra los resultados como indica el paso 5.
