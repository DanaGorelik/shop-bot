"""Long-term conversation memory, kept in the shop database.

The agent already remembers the current conversation in RAM, in full detail:
every tool call, every tool result, the model's reasoning items. That detail
is what makes follow-up questions work, but it is lost when the program stops.

This store keeps a compact copy that survives restarts: only what the
customer wrote and what the bot finally answered. When a customer writes
after a restart, the agent loads their recent exchanges from here.

Why not save the full detail? Tool results go stale (stock, order status),
and the API's reasoning items only stay valid on OpenAI's side for a limited
time, so replaying them days later can fail. The plain text of the
conversation is always safe to send back, and the agent is told to look facts
up again rather than trust what an old message said.

What is kept is personal data, so it is kept only as long as useful:
exchanges older than RETENTION_DAYS are deleted, and `/reset` deletes a
customer's history immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from . import config, db

RETENTION_DAYS = 30
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class StoredMessage:
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime


class ConversationStore:
    def __init__(self, db_path: Path = config.DB_PATH, clock=datetime.now, retention_days: int = RETENTION_DAYS):
        self.db_path = db_path
        self.clock = clock
        self.retention_days = retention_days
        with db.connect(self.db_path) as connection:  # older databases get the table here
            connection.executescript(db.MESSAGES_SCHEMA)

    def save_exchange(self, customer_phone: str, message: str, answer: str) -> None:
        """Store one customer message and the bot's answer, and forget expired ones."""
        now = self.clock()
        with db.connect(self.db_path) as connection:
            connection.executemany(
                "INSERT INTO messages (customer_phone, role, content, created_at) VALUES (?, ?, ?, ?)",
                [
                    (customer_phone, "user", message, now.strftime(TIME_FORMAT)),
                    (customer_phone, "assistant", answer, now.strftime(TIME_FORMAT)),
                ],
            )
            connection.execute("DELETE FROM messages WHERE created_at < ?", (self._cutoff(),))
            connection.commit()

    def load(self, customer_phone: str, max_messages: int) -> list[StoredMessage]:
        """The customer's most recent messages within the retention period, oldest first."""
        with db.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE customer_phone = ? AND created_at >= ? ORDER BY id DESC LIMIT ?",
                (customer_phone, self._cutoff(), max_messages),
            ).fetchall()
        messages = [
            StoredMessage(row["role"], row["content"], datetime.strptime(row["created_at"], TIME_FORMAT))
            for row in reversed(rows)
        ]
        # Never start in the middle of an exchange with a lone answer.
        while messages and messages[0].role != "user":
            messages.pop(0)
        return messages

    def clear(self, customer_phone: str) -> None:
        with db.connect(self.db_path) as connection:
            connection.execute("DELETE FROM messages WHERE customer_phone = ?", (customer_phone,))
            connection.commit()

    def _cutoff(self) -> str:
        return (self.clock() - timedelta(days=self.retention_days)).strftime(TIME_FORMAT)


def restored_items(messages: list[StoredMessage]) -> list[dict]:
    """Turn stored messages into model input, led by a note about their age."""
    last = messages[-1].created_at
    note = (
        "The messages that follow are from earlier conversations with this customer, restored "
        f"from the shop's records. The most recent one was on {last:%A %d %B %Y at %H:%M}. "
        "Stock, prices, order status and holds may have changed since, so look them up again "
        "instead of relying on what these messages say. A hold that was offered but not "
        "confirmed in them has not been placed."
    )
    return [{"role": "system", "content": note}] + [
        {"role": message.role, "content": message.content} for message in messages
    ]
