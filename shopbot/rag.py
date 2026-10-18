"""Retrieval-augmented generation, without a framework.

Ingest (done once, by ingest.py):
    documents -> chunks -> embeddings -> saved to disk

Query (done on every search):
    question -> embedding -> nearest chunks -> handed to the LLM as context

The "vector database" here is a NumPy array and a JSON file. With a few dozen
chunks that is all you need, and you can open both to see what is inside. At
tens of thousands of chunks you would swap this class for Chroma, pgvector,
Qdrant or similar; the idea stays the same.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config

MAX_CHUNK_CHARS = 1200


@dataclass
class Chunk:
    id: str
    kind: str  # "doc" for policy/FAQ text, "product" for catalog entries
    source: str  # file name, or product id
    title: str
    text: str


# ---------------------------------------------------------------- chunking --


def chunk_markdown(path: Path) -> list[Chunk]:
    """Split a markdown file into one chunk per '## ' section.

    Splitting on headings keeps each chunk about one topic, which is what
    makes retrieval precise. The document title is repeated at the top of
    every chunk so a section like "Refunds" still carries the context that it
    belongs to "Returns and exchanges".
    """
    text = path.read_text(encoding="utf-8")
    doc_title = path.stem.replace("_", " ").capitalize()
    title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
    if title_match:
        doc_title = title_match.group(1).strip()

    chunks: list[Chunk] = []
    sections = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
    for section in sections:
        heading, _, body = section.partition("\n")
        heading, body = heading.strip(), body.strip()
        for part_number, part in enumerate(_split_long(body)):
            suffix = f"-{part_number + 1}" if part_number else ""
            chunks.append(
                Chunk(
                    id=f"{path.stem}#{_slug(heading)}{suffix}",
                    kind="doc",
                    source=path.name,
                    title=f"{doc_title} > {heading}",
                    text=f"{doc_title} > {heading}\n\n{part}",
                )
            )
    return chunks


def _split_long(body: str) -> list[str]:
    """Break an over-long section at paragraph boundaries."""
    if len(body) <= MAX_CHUNK_CHARS:
        return [body]
    parts, current = [], ""
    for paragraph in body.split("\n\n"):
        if current and len(current) + len(paragraph) > MAX_CHUNK_CHARS:
            parts.append(current.strip())
            current = ""
        current += paragraph + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return parts


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def product_chunks(catalog_path: Path) -> list[Chunk]:
    """One chunk per product, describing it in words.

    Only the descriptive, slow-changing facts go in here. Price and stock are
    deliberately left out: they change, so the agent reads them live from the
    database instead of from a snapshot frozen at ingest time.
    """
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    chunks = []
    for product in catalog["products"]:
        colors = ", ".join(product["stock"])
        text = (
            f"{product['name']} ({product['category']})\n"
            f"{product['description']}\n"
            f"Fabric: {product['fabric']}. Fit: {product['fit_note']}\n"
            f"Colours: {colors}."
        )
        chunks.append(
            Chunk(
                id=product["id"],
                kind="product",
                source=product["id"],
                title=product["name"],
                text=text,
            )
        )
    return chunks


def load_all_chunks(
    knowledge_dir: Path = config.KNOWLEDGE_DIR, catalog_path: Path = config.CATALOG_PATH
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path))
    chunks.extend(product_chunks(catalog_path))
    return chunks


# ------------------------------------------------------------------- index --


class VectorIndex:
    """Chunks plus one embedding vector per chunk."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder_name = embedder_name

    @classmethod
    def build(cls, chunks: list[Chunk], embedder) -> "VectorIndex":
        vectors = embedder.embed([chunk.text for chunk in chunks])
        return cls(chunks, vectors, embedder.name)

    def save(self, index_dir: Path = config.INDEX_DIR) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", self.vectors)
        payload = {
            "embedder": self.embedder_name,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        (index_dir / "chunks.json").write_text(
            json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path = config.INDEX_DIR) -> "VectorIndex":
        chunks_file = index_dir / "chunks.json"
        if not chunks_file.exists():
            raise FileNotFoundError(
                f"No index at {index_dir}. Run `python ingest.py` first."
            )
        payload = json.loads(chunks_file.read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in payload["chunks"]]
        vectors = np.load(index_dir / "vectors.npy")
        return cls(chunks, vectors, payload["embedder"])

    def search(
        self, query: str, embedder, kind: str | None = None, k: int = 4
    ) -> list[tuple[Chunk, float]]:
        """Return the k chunks most similar to the query, best first.

        The query must be embedded by the same model that embedded the
        chunks; vectors from different models are not comparable.
        """
        if embedder.name != self.embedder_name:
            raise ValueError(
                f"Index was built with {self.embedder_name} but the query uses "
                f"{embedder.name}. Re-run `python ingest.py`."
            )
        query_vector = embedder.embed([query])[0]
        scores = self.vectors @ query_vector  # cosine similarity (unit vectors)
        order = np.argsort(-scores)
        results = []
        for position in order:
            chunk = self.chunks[int(position)]
            if kind and chunk.kind != kind:
                continue
            results.append((chunk, float(scores[position])))
            if len(results) == k:
                break
        return results
