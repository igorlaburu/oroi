"""Banco de evaluación de oroi (Fase 5) — AISLADO de la librería.

Este paquete NO forma parte de `oroi` (la librería de memoria): vive fuera de
`src/` y la importa como cualquier consumidor. La dependencia es unidireccional —
`evaluation` usa `oroi`; `oroi` nunca conoce `evaluation`. Aquí viven los
escenarios (corpus JSONL), las tecnologías de contraste (RAG, re-ranker) y las
métricas que ponen a prueba la hipótesis del proyecto.
"""
