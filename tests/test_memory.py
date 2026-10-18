"""Tests for long-term conversation memory, including a simulated restart.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from shopbot import db
from shopbot.agent import Agent, Reply
from shopbot.embeddings import HashingEmbedder
from shopbot.memory import ConversationStore, StoredMessage, restored_items
from shopbot.rag import VectorIndex, load_all_chunks
from shopbot.tools import ShopTools
from shopbot.whatsapp import Bridge, Incoming

CAROL, DAN = "+12025550101", "+12025550102"
START = datetime(2026, 10, 10, 14, 3, 0)


def says(text):
    return SimpleNamespace(output=[SimpleNamespace(type="message")], output_text=text)


def calls(name, arguments, call_id="c1"):
    item = SimpleNamespace(type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id)
    return SimpleNamespace(output=[item], output_text="")


class ScriptedClient:
    def __init__(self, *script):
        self.script, self.requests = list(script), []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.requests.append(list(kwargs["input"]))
        return self.script.pop(0)


class MemoryTestCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.db_path = Path(folder.name) / "shop.db"
        db.build_database(db_path=self.db_path)
        self.now = START
        self.store = ConversationStore(self.db_path, clock=lambda: self.now)

    def rows(self):
        connection = sqlite3.connect(self.db_path)
        self.addCleanup(connection.close)
        return connection.execute("SELECT customer_phone, role, content FROM messages ORDER BY id").fetchall()


class StoreTests(MemoryTestCase):
    def test_exchanges_come_back_in_order_per_customer(self):
        self.store.save_exchange(CAROL, "Hi", "Hello! How can I help?")
        self.store.save_exchange(DAN, "Opening hours?", "We open at 09:30.")
        self.store.save_exchange(CAROL, "Do you sell scarves?", "Yes, the Wool Scarf.")
        loaded = self.store.load(CAROL, max_messages=10)
        self.assertEqual(
            [(m.role, m.content) for m in loaded],
            [("user", "Hi"), ("assistant", "Hello! How can I help?"),
             ("user", "Do you sell scarves?"), ("assistant", "Yes, the Wool Scarf.")],
        )
        self.assertEqual(loaded[0].created_at, START)

    def test_only_the_most_recent_messages_and_never_a_lone_answer_first(self):
        for number in range(5):
            self.store.save_exchange(CAROL, f"question {number}", f"answer {number}")
        self.assertEqual([m.content for m in self.store.load(CAROL, 4)],
                         ["question 3", "answer 3", "question 4", "answer 4"])
        # an odd limit would start on an answer; that answer is dropped
        self.assertEqual([m.content for m in self.store.load(CAROL, 3)], ["question 4", "answer 4"])

    def test_history_older_than_30_days_is_ignored_and_deleted(self):
        self.store.save_exchange(CAROL, "old question", "old answer")
        self.now = START + timedelta(days=30, minutes=1)
        self.assertEqual(self.store.load(CAROL, 10), [])
        self.store.save_exchange(DAN, "new question", "new answer")  # saving also cleans up
        self.assertEqual([row[0] for row in self.rows()], [DAN, DAN])

    def test_history_within_30_days_is_kept(self):
        self.store.save_exchange(CAROL, "question", "answer")
        self.now = START + timedelta(days=29)
        self.assertEqual(len(self.store.load(CAROL, 10)), 2)

    def test_clear_removes_only_that_customer(self):
        self.store.save_exchange(CAROL, "a", "b")
        self.store.save_exchange(DAN, "c", "d")
        self.store.clear(CAROL)
        self.assertEqual([row[0] for row in self.rows()], [DAN, DAN])

    def test_database_built_before_memory_existed_gets_the_table(self):
        connection = sqlite3.connect(self.db_path)
        connection.execute("DROP TABLE messages")
        connection.commit()
        connection.close()
        store = ConversationStore(self.db_path, clock=lambda: self.now)
        store.save_exchange(CAROL, "hi", "hello")
        self.assertEqual(len(store.load(CAROL, 10)), 2)

    def test_restored_items_start_with_a_dated_staleness_note(self):
        items = restored_items([StoredMessage("user", "Hold the scarf?", START),
                                StoredMessage("assistant", "Shall I hold it?", START)])
        self.assertEqual(items[0]["role"], "system")
        self.assertIn("Saturday 10 October 2026 at 14:03", items[0]["content"])
        self.assertIn("look them up again", items[0]["content"])
        self.assertIn("has not been placed", items[0]["content"])
        self.assertEqual(items[1:], [{"role": "user", "content": "Hold the scarf?"},
                                     {"role": "assistant", "content": "Shall I hold it?"}])


class AgentMemoryTests(MemoryTestCase):
    tools = SimpleNamespace(forget_pending=lambda customer: None)

    def test_conversation_survives_a_restart(self):
        first = Agent(ScriptedClient(says("We close at 14:30 on Fridays.")), self.tools, store=self.store)
        first.reply(CAROL, "When do you close on Friday?")

        # "Restart": a brand-new agent with empty RAM, same database.
        client = ScriptedClient(says("On Saturdays we're closed."), says("You're welcome!"))
        second = Agent(client, self.tools, store=self.store)
        reply = second.reply(CAROL, "And on Saturday?")
        self.assertEqual(reply.restored, 2)
        sent = client.requests[0]
        self.assertEqual(sent[0]["role"], "system")
        self.assertEqual(sent[1:], [
            {"role": "user", "content": "When do you close on Friday?"},
            {"role": "assistant", "content": "We close at 14:30 on Fridays."},
            {"role": "user", "content": "And on Saturday?"},
        ])
        # while the session is live, RAM is used and nothing is restored twice
        again = second.reply(CAROL, "Thanks")
        self.assertEqual(again.restored, 0)
        self.assertEqual(sum(1 for item in client.requests[1] if isinstance(item, dict) and item.get("role") == "system"), 1)
        self.assertEqual(len(self.store.load(CAROL, 100)), 6)

    def test_other_customers_history_is_not_restored(self):
        Agent(ScriptedClient(says("Hello Carol")), self.tools, store=self.store).reply(CAROL, "Hi")
        client = ScriptedClient(says("Hello Dan"))
        reply = Agent(client, self.tools, store=self.store).reply(DAN, "Hi")
        self.assertEqual(reply.restored, 0)
        self.assertEqual(client.requests[0], [{"role": "user", "content": "Hi"}])

    def test_failed_reply_is_not_saved(self):
        class Broken:
            responses = SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("down")))

        with self.assertRaises(RuntimeError):
            Agent(Broken(), self.tools, store=self.store).reply(CAROL, "Hi")
        self.assertEqual(self.rows(), [])

    def test_reset_erases_the_saved_history(self):
        agent = Agent(ScriptedClient(says("Hello")), self.tools, store=self.store)
        agent.reply(CAROL, "Hi")
        agent.reset(CAROL)
        self.assertEqual(self.rows(), [])
        client = ScriptedClient(says("Hello again"))
        self.assertEqual(Agent(client, self.tools, store=self.store).reply(CAROL, "Hi").restored, 0)

    def test_old_history_is_not_restored(self):
        Agent(ScriptedClient(says("Hello")), self.tools, store=self.store).reply(CAROL, "Hi")
        self.now = START + timedelta(days=31)
        client = ScriptedClient(says("Hello"))
        self.assertEqual(Agent(client, self.tools, store=self.store).reply(CAROL, "Hi").restored, 0)

    def test_restore_is_limited_to_max_turns(self):
        for number in range(5):
            self.store.save_exchange(CAROL, f"q{number}", f"a{number}")
        client = ScriptedClient(says("ok"))
        reply = Agent(client, self.tools, max_turns=2, store=self.store).reply(CAROL, "next")
        self.assertEqual(reply.restored, 4)
        self.assertEqual([i["content"] for i in client.requests[0][1:-1]], ["q3", "a3", "q4", "a4"])

    def test_without_a_store_nothing_is_written(self):
        Agent(ScriptedClient(says("Hello")), self.tools).reply(CAROL, "Hi")
        self.assertEqual(self.rows(), [])


class HoldAcrossRestartTests(MemoryTestCase):
    def test_yes_after_a_restart_cannot_confirm_an_old_offer(self):
        embedder = HashingEmbedder()
        index = VectorIndex.build(load_all_chunks(), embedder)
        scarf = {"product_id": "P-1013", "color": "burgundy", "size": "One size"}

        before = Agent(ScriptedClient(calls("reserve_item", scarf), says("Shall I hold the burgundy scarf?")),
                       ShopTools(index, embedder, db_path=self.db_path), store=self.store)
        before.reply(CAROL, "Hold the burgundy scarf please")

        # Restart: new tools (no prepared hold in memory) and a new agent.
        client = ScriptedClient(calls("confirm_reservation", {}), says("Let me set that up again first."))
        after = Agent(client, ShopTools(index, embedder, db_path=self.db_path), store=self.store)
        reply = after.reply(CAROL, "yes")
        self.assertEqual(reply.restored, 2)
        self.assertIn("no prepared hold", reply.steps[0].result["error"])
        connection = sqlite3.connect(self.db_path)
        self.addCleanup(connection.close)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM reservations").fetchone()[0], 0)


class WhatsAppTraceTests(unittest.TestCase):
    def test_server_log_mentions_restored_history(self):
        class RememberingAgent:
            def reply(self, customer, message):
                return Reply(text="Welcome back!", restored=4)

        sent, log = [], []
        client = SimpleNamespace(send_text=lambda to, body: sent.append(body), mark_read_and_typing=lambda mid: None)
        bridge = Bridge(RememberingAgent(), client, log=log.append, clock=lambda: 1_000)
        bridge.process(Incoming("12025550101", "wamid.1", "text", "hi again", 1_000))
        self.assertIn("  . remembered 4 earlier messages from a previous session", log)
        self.assertEqual(sent, ["Welcome back!"])


if __name__ == "__main__":
    unittest.main()
