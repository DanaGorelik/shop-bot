"""See what retrieval returns for a question, without any LLM involved.

    python search.py "can I return something I bought on sale?"

This is the 'R' in RAG on its own. If the right passage is not in these
results, the agent cannot give the right answer, so this is the first place
to look when the bot gets a policy question wrong.
"""

from __future__ import annotations

import sys

from shopbot import config
from shopbot.embeddings import make_embedder
from shopbot.rag import VectorIndex


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    query = " ".join(sys.argv[1:])
    sys.stdout.reconfigure(errors="replace")
    config.load_env()
    index = VectorIndex.load()
    embedder = make_embedder(index.embedder_name)
    print(f'Query: "{query}"   (embedder: {index.embedder_name})')
    for kind, label in (("doc", "Document chunks"), ("product", "Products")):
        print(f"\n{label}")
        for chunk, score in index.search(query, embedder, kind=kind, k=4):
            first_line = chunk.text.split("\n\n", 1)[-1].replace("\n", " ")[:90]
            print(f"  {score:5.3f}  {chunk.id:<45} {first_line}")


if __name__ == "__main__":
    main()
