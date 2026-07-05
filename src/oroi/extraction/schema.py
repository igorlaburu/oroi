"""La mini-red semántica que el extractor produce por cada turno (SPEC §5, paso 1)."""

from pydantic import BaseModel, Field, model_validator


class MiniNode(BaseModel):
    label: str
    kind: str = "entity"  # entity | attribute | event | concept
    salience: float = Field(0.5, ge=0.0, le=1.0)


class MiniEdge(BaseModel):
    src: str
    dst: str
    rel: str | None = None
    symmetric: bool = False


class MiniNetwork(BaseModel):
    nodes: list[MiniNode] = []
    edges: list[MiniEdge] = []

    @model_validator(mode="after")
    def _prune_dangling_edges(self):
        """Una arista solo vive si AMBOS extremos son nodos declarados. El extractor a veces
        nombra un sujeto implícito ("usuario", "yo") que no declara como nodo: esas aristas
        colgantes —y los bucles— se descartan aquí, en la frontera, en vez de morir en silencio
        más adentro. De paso canoniza los extremos a la etiqueta exacta del nodo (corrige el case)."""
        canon = {n.label.strip().lower(): n.label for n in self.nodes}
        kept = []
        for e in self.edges:
            src, dst = canon.get(e.src.strip().lower()), canon.get(e.dst.strip().lower())
            if src is not None and dst is not None and src != dst:
                e.src, e.dst = src, dst
                kept.append(e)
        self.edges = kept
        return self
