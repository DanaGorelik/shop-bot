"""Step 1: load the shop's data.

    python ingest.py            build the database and the vector index (needs OPENAI_API_KEY)
    python ingest.py --local    same, but with the offline word-overlap embedder (no key needed)
    python ingest.py --keep-database
                                rebuild only the index; keep the database if it exists

Run it again whenever you edit anything under data/. A plain run rebuilds the
database from the data files, which also clears all holds and handoff tickets.
--keep-database is what the server uses, so customers' holds survive updates.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

from shopbot import config, db
from shopbot.embeddings import HashingEmbedder, make_embedder
from shopbot.rag import VectorIndex, load_all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--local", action="store_true", help="use the offline embedder")
    parser.add_argument("--keep-database", action="store_true",
                        help="keep an existing database (and its holds); build it only if missing")
    arguments = parser.parse_args()
    config.load_env()
    if not arguments.local and config.embedding_backend() != "local" and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, "
                 "or run `python ingest.py --local` to try it offline.")

    if arguments.keep_database and config.DB_PATH.exists():
        print(f"Database  {config.DB_PATH.relative_to(config.ROOT)} kept as it is (holds and tickets preserved)")
    else:
        counts = db.build_database()
        print(f"Database  {config.DB_PATH.relative_to(config.ROOT)}")
        print(f"          {counts['products']} products, {counts['variants']} size/colour "
              f"variants, {counts['orders']} orders")

    chunks = load_all_chunks()
    embedder = HashingEmbedder() if arguments.local else make_embedder()
    index = VectorIndex.build(chunks, embedder)
    index.save()
    kinds = Counter(chunk.kind for chunk in chunks)
    print(f"Index     {config.INDEX_DIR.relative_to(config.ROOT)}")
    print(f"          {kinds['doc']} document chunks + {kinds['product']} product chunks, "
          f"{index.vectors.shape[1]} dimensions each, embedded with {embedder.name}")
    if embedder.name.startswith("local"):
        print("\nNote: the local embedder only matches shared words. For real semantic "
              "search, set OPENAI_API_KEY and run `python ingest.py` without --local.")


if __name__ == "__main__":
    main()
