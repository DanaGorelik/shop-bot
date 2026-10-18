"""The actions the agent can take.

Each tool has two halves:

* a schema (name, description, parameters) that is sent to the LLM so it
  knows the tool exists and how to call it, and
* a Python method that actually does the work when the LLM asks for it.

The LLM never touches the database or the index itself. It can only ask for
one of these functions to be run, and our code decides what really happens.

Most tools only read. The reservation tools change something real (an item
stops being for sale for 48 hours), so they get extra safeguards that live in
code rather than in the prompt: a hold needs two calls with a customer reply
in between, availability is re-checked inside a database transaction, and
each customer has a limit. See the "reservations" section below.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

from . import config, db
from .rag import VectorIndex

CATEGORIES = ["tops", "knitwear", "bottoms", "dresses", "outerwear", "accessories", "sleepwear"]
HOLD_HOURS = 48
MAX_HOLDS_PER_CUSTOMER = 2
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _tool(name: str, description: str, properties: dict) -> dict:
    # strict mode makes the model's arguments match the schema exactly. It
    # requires every property to be listed in `required`; optional ones are
    # expressed by allowing null.
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


TOOL_SCHEMAS = [
    _tool(
        "search_knowledge",
        "Search the shop's written policies and information: returns, exchanges, refunds, "
        "shipping and delivery, size charts, garment care, opening hours, address, payment "
        "methods, gift cards, loyalty club, alterations. Use this before answering any "
        "question about how the shop works. Returns the most relevant passages.",
        {
            "query": {
                "type": "string",
                "description": "What to look for, in English, phrased as a short question "
                "or topic, e.g. 'return window for sale items'.",
            }
        },
    ),
    _tool(
        "search_products",
        "Find products in the catalog that match a description. Use this when the customer "
        "is looking for something to buy or asks whether the shop sells something. Returns "
        "up to 5 products with id, current price and the sizes in stock per colour.",
        {
            "query": {
                "type": "string",
                "description": "What the customer wants, in English, e.g. 'warm sweater for "
                "winter' or 'light dress for a beach holiday'.",
            },
            "category": {
                "type": ["string", "null"],
                "enum": CATEGORIES,
                "description": "Limit results to one category, or null for all.",
            },
            "max_price": {
                "type": ["number", "null"],
                "description": "Highest acceptable price in shekels, or null for no limit.",
            },
        },
    ),
    _tool(
        "check_stock",
        "Get exact stock quantities for one product. Use this whenever the customer asks "
        "whether a specific size or colour is available. Never guess availability.",
        {
            "product_id": {
                "type": "string",
                "description": "Product id such as 'P-1003', from search_products.",
            },
            "color": {
                "type": ["string", "null"],
                "description": "Colour to check, or null for all colours.",
            },
            "size": {
                "type": ["string", "null"],
                "description": "Size to check (e.g. 'M' or '32'), or null for all sizes.",
            },
        },
    ),
    _tool(
        "get_order_status",
        "Look up one of this customer's orders: status, items, dates and tracking. Only "
        "orders belonging to the customer you are chatting with can be seen.",
        {
            "order_id": {
                "type": "string",
                "description": "Order number as the customer gave it, e.g. 'ORD-5001'.",
            }
        },
    ),
    _tool(
        "list_my_orders",
        "List the orders of the customer you are chatting with, newest first. Use this when "
        "they ask about an order but do not give an order number.",
        {},
    ),
    _tool(
        "handoff_to_human",
        "Pass the conversation to a human team member. Use this when the customer asks for "
        "a person, is upset, wants something you cannot do (cancel or change an order, "
        "approve an exception, start a refund), or when you cannot find the answer.",
        {
            "reason": {
                "type": "string",
                "description": "One or two sentences telling the team what the customer "
                "needs, including any order number.",
            }
        },
    ),
    _tool(
        "reserve_item",
        f"Step 1 of 2 for holding an item at the shop for {HOLD_HOURS} hours. This does NOT "
        "hold anything yet: it checks that the exact colour and size is in stock and that the "
        "customer may place a hold, and returns what would be held. After calling it, tell the "
        "customer exactly what will be held and for how long and ask them to confirm. Call "
        "confirm_reservation only after their next message agrees.",
        {
            "product_id": {"type": "string", "description": "Product id such as 'P-1013'."},
            "color": {"type": "string", "description": "Exact colour, e.g. 'burgundy'."},
            "size": {"type": "string", "description": "Exact size, e.g. 'M', '32' or 'One size'."},
        },
    ),
    _tool(
        "confirm_reservation",
        "Step 2 of 2: place the hold that reserve_item prepared. Call it only when the "
        "customer's latest message clearly agrees to the hold you described. It is refused if "
        "the customer has not replied since reserve_item, or if the item sold out meanwhile.",
        {},
    ),
    _tool(
        "cancel_reservation",
        "Cancel one of the holds of the customer you are chatting with.",
        {
            "reservation_id": {
                "type": "integer",
                "description": "Hold number, from confirm_reservation or list_my_reservations.",
            }
        },
    ),
    _tool(
        "list_my_reservations",
        "List the active holds of the customer you are chatting with, with the time each "
        "one ends. Use this when they ask what they have on hold.",
        {},
    ),
]


class ShopTools:
    def __init__(self, index: VectorIndex, embedder, db_path: Path = config.DB_PATH, clock=datetime.now):
        self.index = index
        self.embedder = embedder
        self.db_path = db_path
        self.clock = clock  # replaceable in tests, to move time forward
        # Holds prepared by reserve_item and waiting for the customer's reply:
        # customer -> details, including the turn number they were prepared in.
        self._pending: dict[str, dict] = {}
        try:  # databases built before reservations existed get the table here
            with db.connect(self.db_path) as connection:
                connection.executescript(db.RESERVATIONS_SCHEMA)
        except FileNotFoundError:
            pass

    def forget_pending(self, customer_phone: str) -> None:
        self._pending.pop(customer_phone, None)

    def run(self, name: str, arguments_json: str, customer_phone: str, turn: int = 0) -> dict:
        """Run the tool the model asked for and return a JSON-friendly result.

        `customer_phone` comes from our own code (on WhatsApp, from the
        message's sender), never from the model. That is what stops one
        customer from talking the bot into showing another customer's order.

        `turn` counts the customer's messages (the Agent supplies it). The
        reservation tools use it to tell whether the customer has replied.

        Errors are returned as data instead of raised, so the model can read
        what went wrong and recover (ask the customer again, try another tool).
        """
        handlers = {
            "search_knowledge": self.search_knowledge,
            "search_products": self.search_products,
            "check_stock": self.check_stock,
            "get_order_status": lambda **a: self.get_order_status(customer_phone, **a),
            "list_my_orders": lambda **a: self.list_my_orders(customer_phone, **a),
            "handoff_to_human": lambda **a: self.handoff_to_human(customer_phone, **a),
            "reserve_item": lambda **a: self.reserve_item(customer_phone, turn, **a),
            "confirm_reservation": lambda **a: self.confirm_reservation(customer_phone, turn, **a),
            "cancel_reservation": lambda **a: self.cancel_reservation(customer_phone, **a),
            "list_my_reservations": lambda **a: self.list_my_reservations(customer_phone, **a),
        }
        if name not in handlers:
            return {"error": f"Unknown tool '{name}'."}
        try:
            arguments = json.loads(arguments_json or "{}")
            return handlers[name](**arguments)
        except (json.JSONDecodeError, TypeError) as error:
            return {"error": f"Bad arguments for {name}: {error}"}
        except Exception as error:  # a tool failing must not crash the chat
            return {"error": f"{name} failed: {error}"}

    # ------------------------------------------------------------- RAG ----

    def search_knowledge(self, query: str) -> dict:
        hits = self.index.search(query, self.embedder, kind="doc", k=4)
        return {
            "passages": [
                {"source": chunk.source, "score": round(score, 3), "text": chunk.text}
                for chunk, score in hits
            ]
        }

    # ------------------------------------------------------- live data ----

    def search_products(
        self, query: str, category: str | None = None, max_price: float | None = None
    ) -> dict:
        # Step 1: semantic search ranks every product by how well its
        # description matches the request.
        ranked = self.index.search(query, self.embedder, kind="product", k=50)
        # Step 2: the database supplies the facts that must be current.
        products = []
        with db.connect(self.db_path) as connection:
            for chunk, score in ranked:
                row = connection.execute(
                    "SELECT * FROM products WHERE id = ?", (chunk.source,)
                ).fetchone()
                if row is None:
                    continue
                if category and row["category"] != category:
                    continue
                if max_price is not None and row["price"] > max_price:
                    continue
                products.append(
                    {
                        "product_id": row["id"],
                        "name": row["name"],
                        "category": row["category"],
                        "price_ils": row["price"],
                        "description": row["description"],
                        "fit_note": row["fit_note"],
                        "sizes_in_stock_by_colour": _sizes_in_stock(connection, row["id"], self._now()),
                        "match_score": round(score, 3),
                    }
                )
                if len(products) == 5:
                    break
        return {"products": products}

    def check_stock(
        self, product_id: str, color: str | None = None, size: str | None = None
    ) -> dict:
        with db.connect(self.db_path) as connection:
            product = connection.execute(
                "SELECT id, name, price FROM products WHERE id = ?", (product_id.upper(),)
            ).fetchone()
            if product is None:
                return {"error": f"No product with id {product_id}."}
            query = f"SELECT v.color, v.size, {AVAILABLE} AS available FROM variants v WHERE v.product_id = ?"
            parameters: list = [self._now(), product["id"]]
            if color:
                query += " AND lower(v.color) = lower(?)"
                parameters.append(color)
            if size:
                query += " AND lower(v.size) = lower(?)"
                parameters.append(size)
            rows = connection.execute(query + " ORDER BY v.color, v.size_order", parameters).fetchall()
        if not rows:
            return {
                "product_id": product["id"],
                "name": product["name"],
                "error": "This product is not made in that colour/size.",
                "available_variants": self.check_stock(product["id"])["variants"],
            }
        return {
            "product_id": product["id"],
            "name": product["name"],
            "price_ils": product["price"],
            "variants": [
                {"colour": row["color"], "size": row["size"], "in_stock": row["available"]}
                for row in rows
            ],
        }

    def get_order_status(self, customer_phone: str, order_id: str) -> dict:
        order_id = order_id.strip().upper()
        with db.connect(self.db_path) as connection:
            order = connection.execute(
                "SELECT * FROM orders WHERE id = ? AND customer_phone = ?",
                (order_id, customer_phone),
            ).fetchone()
            if order is None:
                # Same answer whether the order does not exist or belongs to
                # someone else, so the bot cannot be used to probe order ids.
                return {"error": f"No order {order_id} found for this customer."}
            items = connection.execute(
                "SELECT p.name, i.product_id, i.color, i.size, i.quantity, i.unit_price "
                "FROM order_items i JOIN products p ON p.id = i.product_id "
                "WHERE i.order_id = ?",
                (order_id,),
            ).fetchall()
        result = {
            "order_id": order["id"],
            "status": order["status"],
            "placed_on": order["placed_on"],
            "delivered_on": order["delivered_on"],
            "carrier": order["carrier"],
            "tracking_number": order["tracking_number"],
            "shipping_cost_ils": order["shipping_cost"],
            "total_ils": order["total"],
            "items": [
                {
                    "product_id": item["product_id"],
                    "name": item["name"],
                    "colour": item["color"],
                    "size": item["size"],
                    "quantity": item["quantity"],
                    "unit_price_ils": item["unit_price"],
                }
                for item in items
            ],
        }
        if order["delivered_on"]:
            # Do the date arithmetic in code; LLMs are unreliable at it.
            delivered = date.fromisoformat(order["delivered_on"])
            result["days_since_delivery"] = (date.today() - delivered).days
        return result

    def list_my_orders(self, customer_phone: str) -> dict:
        with db.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT id, status, placed_on, total FROM orders "
                "WHERE customer_phone = ? ORDER BY placed_on DESC",
                (customer_phone,),
            ).fetchall()
        return {
            "orders": [
                {
                    "order_id": row["id"],
                    "status": row["status"],
                    "placed_on": row["placed_on"],
                    "total_ils": row["total"],
                }
                for row in rows
            ]
        }

    # ---------------------------------------------------------- action ----

    def handoff_to_human(self, customer_phone: str, reason: str) -> dict:
        with db.connect(self.db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO handoffs (customer_phone, reason) VALUES (?, ?)",
                (customer_phone, reason),
            )
            connection.commit()
        return {
            "ticket": cursor.lastrowid,
            "status": "A team member has been notified and will reply in this chat "
            "within one business day.",
        }

    # ---------------------------------------------------- reservations ----
    #
    # Placing a hold is the first tool that changes something real, so the
    # important rules are enforced here rather than trusted to the prompt:
    #
    # 1. Two steps. reserve_item only prepares; confirm_reservation places the
    #    hold, and it refuses unless the customer has sent a message since the
    #    hold was prepared. The model judges whether that message means yes
    #    (it understands "sure", "כן", "go ahead"), but it cannot skip asking.
    # 2. The last item. Two customers can confirm the last unit at the same
    #    moment. The availability check and the insert run inside one write
    #    transaction, so the second one waits and then sees the item is gone.
    # 3. Limits. One unit per hold, at most two holds per customer, holds end
    #    by themselves after 48 hours (expired rows simply stop counting).

    def _now(self) -> str:
        return self.clock().strftime(TIME_FORMAT)

    def reserve_item(self, customer_phone: str, turn: int, product_id: str, color: str, size: str) -> dict:
        with db.connect(self.db_path) as connection:
            variant = connection.execute(
                f"SELECT v.product_id, v.color, v.size, p.name, p.price, {AVAILABLE} AS available "
                "FROM variants v JOIN products p ON p.id = v.product_id "
                "WHERE v.product_id = ? AND lower(v.color) = lower(?) AND lower(v.size) = lower(?)",
                (self._now(), product_id.strip().upper(), color.strip(), size.strip()),
            ).fetchone()
            if variant is None:
                stock = self.check_stock(product_id)
                return {
                    "error": "No such product, colour and size combination.",
                    "variants": stock.get("variants", []),
                }
            item = {
                "product_id": variant["product_id"], "name": variant["name"],
                "colour": variant["color"], "size": variant["size"], "price_ils": variant["price"],
            }
            if variant["available"] <= 0:
                return {
                    "error": f"{item['name']} in {item['colour']}, size {item['size']}, is out of "
                    "stock, so it cannot be held.",
                    "in_stock_by_colour": _sizes_in_stock(connection, item["product_id"], self._now()),
                }
            problem = self._hold_problem(connection, customer_phone, item)
            if problem:
                return problem
        self._pending[customer_phone] = {**item, "turn": turn}
        return {
            "status": "needs_confirmation",
            "item": item,
            "hold_hours": HOLD_HOURS,
            "next_step": "Nothing is held yet. Tell the customer exactly which item, colour and "
            f"size will be held at the shop for {HOLD_HOURS} hours and ask them to confirm. "
            "Call confirm_reservation only if their reply agrees.",
        }

    def confirm_reservation(self, customer_phone: str, turn: int) -> dict:
        pending = self._pending.get(customer_phone)
        if pending is None:
            return {"error": "There is no prepared hold to confirm. Call reserve_item first."}
        if turn <= pending["turn"]:
            return {
                "error": "The customer has not replied yet. Describe the hold and ask them to "
                "confirm; call confirm_reservation only after their answer."
            }
        if turn > pending["turn"] + 1:
            self._pending.pop(customer_phone)
            return {
                "error": "That hold was prepared earlier in the conversation and has lapsed. "
                "Call reserve_item again and ask the customer to confirm."
            }
        item = {key: pending[key] for key in ("product_id", "name", "colour", "size", "price_ils")}
        now = self.clock()
        expires = now + timedelta(hours=HOLD_HOURS)
        with db.connect(self.db_path) as connection:
            connection.isolation_level = None  # we manage the transaction ourselves
            connection.execute("BEGIN IMMEDIATE")  # take the write lock before checking
            try:
                available = connection.execute(
                    f"SELECT {AVAILABLE} FROM variants v "
                    "WHERE v.product_id = ? AND v.color = ? AND v.size = ?",
                    (now.strftime(TIME_FORMAT), item["product_id"], item["colour"], item["size"]),
                ).fetchone()[0]
                problem = self._hold_problem(connection, customer_phone, item)
                if available <= 0:
                    problem = {
                        "error": f"Sorry, the last {item['name']} in {item['colour']}, size "
                        f"{item['size']}, was just taken, so it cannot be held.",
                        "in_stock_by_colour": _sizes_in_stock(
                            connection, item["product_id"], now.strftime(TIME_FORMAT)
                        ),
                    }
                if problem:
                    connection.execute("ROLLBACK")
                    self._pending.pop(customer_phone, None)
                    return problem
                cursor = connection.execute(
                    "INSERT INTO reservations (customer_phone, product_id, color, size, status, "
                    "created_at, expires_at) VALUES (?, ?, ?, ?, 'active', ?, ?)",
                    (customer_phone, item["product_id"], item["colour"], item["size"],
                     now.strftime(TIME_FORMAT), expires.strftime(TIME_FORMAT)),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        self._pending.pop(customer_phone, None)
        return {
            "status": "reserved",
            "reservation_id": cursor.lastrowid,
            "item": item,
            "held_until": _readable(expires),
            "pickup": "Collect and pay for it at the shop before the hold ends. After that it "
            "goes back on sale automatically.",
        }

    def cancel_reservation(self, customer_phone: str, reservation_id: int) -> dict:
        with db.connect(self.db_path) as connection:
            cursor = connection.execute(
                "UPDATE reservations SET status = 'cancelled' WHERE id = ? AND customer_phone = ? "
                "AND status = 'active' AND expires_at > ?",
                (int(reservation_id), customer_phone, self._now()),
            )
            connection.commit()
        if cursor.rowcount == 0:
            # Same answer for "not yours" and "does not exist", like orders.
            return {"error": f"No active hold number {reservation_id} found for this customer."}
        return {"status": "cancelled", "reservation_id": int(reservation_id)}

    def list_my_reservations(self, customer_phone: str) -> dict:
        with db.connect(self.db_path) as connection:
            rows = _active_holds(connection, customer_phone, self._now())
        return {
            "reservations": [
                {
                    "reservation_id": row["id"], "name": row["name"], "colour": row["color"],
                    "size": row["size"],
                    "held_until": _readable(datetime.strptime(row["expires_at"], TIME_FORMAT)),
                }
                for row in rows
            ]
        }

    def _hold_problem(self, connection, customer_phone: str, item: dict) -> dict | None:
        """Why this customer may not hold this item now, or None if they may."""
        holds = _active_holds(connection, customer_phone, self._now())
        for hold in holds:
            if (hold["product_id"], hold["color"], hold["size"]) == (
                item["product_id"], item["colour"], item["size"]
            ):
                return {
                    "error": "The customer already has this exact item on hold.",
                    "reservation_id": hold["id"],
                }
        if len(holds) >= MAX_HOLDS_PER_CUSTOMER:
            return {
                "error": f"The customer already has {len(holds)} items on hold, the most allowed. "
                "They can cancel one to hold something else.",
                "current_holds": [f"#{h['id']} {h['name']} {h['color']} {h['size']}" for h in holds],
            }
        return None


# Units of a variant that can still be sold: stock minus unexpired active
# holds. Used inside SQL queries on `variants v`; the first parameter is "now".
AVAILABLE = (
    "MAX(0, v.quantity - (SELECT COUNT(*) FROM reservations r WHERE r.product_id = v.product_id "
    "AND r.color = v.color AND r.size = v.size AND r.status = 'active' AND r.expires_at > ?))"
)


def _active_holds(connection, customer_phone: str, now: str) -> list:
    return connection.execute(
        "SELECT r.id, r.product_id, r.color, r.size, r.expires_at, p.name FROM reservations r "
        "JOIN products p ON p.id = r.product_id WHERE r.customer_phone = ? "
        "AND r.status = 'active' AND r.expires_at > ? ORDER BY r.id",
        (customer_phone, now),
    ).fetchall()


def _readable(moment: datetime) -> str:
    return moment.strftime("%A %d %B at %H:%M")


def _sizes_in_stock(connection, product_id: str, now: str) -> dict:
    rows = connection.execute(
        f"SELECT v.color, v.size FROM variants v WHERE v.product_id = ? AND {AVAILABLE} > 0 "
        "ORDER BY v.color, v.size_order",
        (product_id, now),
    ).fetchall()
    by_colour: dict[str, list[str]] = {}
    for row in rows:
        by_colour.setdefault(row["color"], []).append(row["size"])
    return by_colour
