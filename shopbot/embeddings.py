"""Turning text into vectors.

An embedding is a list of numbers that places a piece of text at a point in
space, so that texts with similar meaning land close together. Retrieval is
then "find the stored points nearest to the question's point".

Two embedders share one interface (`name` and `embed(texts)`):

* OpenAIEmbedder   - real semantic embeddings from the OpenAI API.
* HashingEmbedder  - an offline stand-in that only measures word overlap. It
                     needs no API key, which makes it useful for tests and for
                     seeing how much worse keyword matching is than meaning.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from . import config


class OpenAIEmbedder:
    def __init__(self, client=None, model: str | None = None):
        if client is None:
            from openai import OpenAI  # imported here so offline use needs no SDK

            client = OpenAI()
        self.client = client
        self.model = model or config.embedding_model()
        self.name = f"openai:{self.model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=texts)
        # OpenAI embeddings come back already normalised to length 1, so the
        # dot product of two of them is their cosine similarity.
        return np.array([item.embedding for item in response.data], dtype=np.float32)


class HashingEmbedder:
    """Bag-of-words vectors: each word is hashed to one of `dims` slots.

    Two texts score as similar only when they share words, so "jumper" and
    "sweater" are unrelated here. That gap is exactly what real embeddings fix.
    """

    def __init__(self, dims: int = 1024):
        self.dims = dims
        self.name = f"local:hashing-{dims}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if len(word) < 3:
                    continue
                word = _stem(word)
                slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims
                vectors[row, slot] += 1.0
            np.sqrt(vectors[row], out=vectors[row])  # damp repeated words
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors


def _stem(word: str) -> str:
    """Very rough plural stripping so 'returns' matches 'return'."""
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def make_embedder(name: str | None = None):
    """Build an embedder from an index's saved name, or from the settings."""
    if name is None:
        name = "local" if config.embedding_backend() == "local" else "openai"
    if name.startswith("local"):
        return HashingEmbedder()
    if name.startswith("openai:"):
        return OpenAIEmbedder(model=name.split(":", 1)[1])
    return OpenAIEmbedder()
