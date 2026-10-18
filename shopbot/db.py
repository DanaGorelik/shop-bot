"""The shop's live data: products, stock and orders, in SQLite.

This is the part of the store's data that should NOT go through RAG. Stock
and order status need exact, current answers, so the agent reads them with
ordinary database queries (wrapped as tools in tools.py).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE products (
    id TEXT PRIMARY KEY, name TEXT, category TEXT, price INTEGER,
    description TEXT, fabric TEXT, fit_note TEXT
);
CREATE TABLE variants (
    product_id TEXT, color TEXT, size TEXT, size_order INTEGER, quantity INTEGER,
    PRIMARY KEY (product_id, color, size)
);
CREATE TABLE orders (
    id TEXT PRIMARY KEY, customer_phone TEXT, status TEXT, placed_on TEXT,
    delivered_on TEXT, carrier TEXT, tracking_number TEXT,
    shipping_cost INTEGER, total INTEGER
);
CREATE TABLE order_items (
    order_id TEXT, product_id TEXT, color TEXT, size TEXT,
    quantity INTEGER, unit_price INTEGER
);
CREATE TABLE handoffs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, customer_phone TEXT,
    reason TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

# Holds placed through the chat. Kept separate from `variants.quantity`: an
# item on hold is still physically in the shop, it just is not for sale, so
# "available" is always computed as quantity minus active holds. Times are
# the shop computer's local time, as text that sorts correctly.
RESERVATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, customer_phone TEXT,
    product_id TEXT, color TEXT, size TEXT,
    status TEXT,  -- 'active' or 'cancelled'
    created_at TEXT, expires_at TEXT
);
"""

# Long-term conversation memory (see memory.py): what each customer wrote and
# what the bot answered, deleted after 30 days.
MESSAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, customer_phone TEXT,
    role TEXT, content TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS messages_by_customer ON messages (customer_phone, id);
"""


@contextmanager
def connect(db_path: Path = config.DB_PATH):
    """Open the database for one operation and close it afterwards."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"No database at {db_path}. Run `python ingest.py` first.")
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def build_database(
    db_path: Path = config.DB_PATH,
    catalog_path: Path = config.CATALOG_PATH,
    orders_path: Path = config.ORDERS_PATH,
    today: date | None = None,
) -> dict:
    """Create the database from the JSON files. Replaces any existing one."""
    today = today or date.today()
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)

    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    orders = json.loads(Path(orders_path).read_text(encoding="utf-8"))["orders"]

    connection = sqlite3.connect(db_path)
    connection.executescript(SCHEMA + RESERVATIONS_SCHEMA + MESSAGES_SCHEMA)
    prices = {}
    variant_count = 0
    for product in catalog["products"]:
        prices[product["id"]] = product["price"]
        connection.execute(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                product["id"], product["name"], product["category"], product["price"],
                product["description"], product["fabric"], product["fit_note"],
            ),
        )
        for color, sizes in product["stock"].items():
            for size, quantity in sizes.items():
                connection.execute(
                    "INSERT INTO variants VALUES (?, ?, ?, ?, ?)",
                    (product["id"], color, size, product["sizes"].index(size), quantity),
                )
                variant_count += 1

    for order in orders:
        placed = today - timedelta(days=order["placed_days_ago"])
        delivered = None
        if order["delivered_days_ago"] is not None:
            delivered = (today - timedelta(days=order["delivered_days_ago"])).isoformat()
        items_total = sum(prices[item["product_id"]] * item["quantity"] for item in order["items"])
        connection.execute(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                order["id"], order["customer_phone"], order["status"], placed.isoformat(),
                delivered, order["carrier"], order["tracking_number"],
                order["shipping_cost"], items_total + order["shipping_cost"],
            ),
        )
        for item in order["items"]:
            connection.execute(
                "INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?)",
                (
                    order["id"], item["product_id"], item["color"], item["size"],
                    item["quantity"], prices[item["product_id"]],
                ),
            )
    connection.commit()
    connection.close()
    return {
        "products": len(catalog["products"]),
        "variants": variant_count,
        "orders": len(orders),
    }
