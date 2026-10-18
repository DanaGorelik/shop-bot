"""Paths and settings. Everything tunable lives here or in the .env file."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
CATALOG_PATH = DATA_DIR / "catalog.json"
ORDERS_PATH = DATA_DIR / "orders.json"

# Everything ingest.py generates goes here. Safe to delete and rebuild.
STORAGE_DIR = ROOT / "storage"
DB_PATH = STORAGE_DIR / "shop.db"
INDEX_DIR = STORAGE_DIR / "index"

SHOP_NAME = "Olive Lane"


def load_env(path: Path = ROOT / ".env") -> None:
    """Read KEY=value lines from .env into the environment (existing values win)."""
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    _drop_unusable_ssl_keylog()


def _drop_unusable_ssl_keylog() -> None:
    """Stop a broken SSLKEYLOGFILE setting from blocking every HTTPS call.

    SSLKEYLOGFILE is a debugging variable: when it is set, Python writes TLS
    session keys to that file each time it opens a secure connection. If the
    file cannot be written, the connection fails with "Permission denied"
    before it starts. In that case we ignore the variable for this program
    only, which just means no keys are logged.
    """
    path = os.environ.get("SSLKEYLOGFILE")
    if not path:
        return
    try:
        with open(path, "a"):
            pass
    except OSError:
        del os.environ["SSLKEYLOGFILE"]
        print(f"Note: ignoring SSLKEYLOGFILE ({path}) because that file is not writable.")


def chat_model() -> str:
    # gpt-6-luna is OpenAI's cheapest current model and supports tool calling.
    # Try gpt-6.1-sol if you want to see how a stronger model changes behaviour.
    return os.environ.get("OPENAI_MODEL", "gpt-6-luna")


def embedding_model() -> str:
    return os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def embedding_backend() -> str:
    """'openai' for real semantic embeddings, 'local' for the offline stand-in."""
    return os.environ.get("EMBEDDING_BACKEND", "openai").lower()
