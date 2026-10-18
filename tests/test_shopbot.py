"""Tests that run offline: no API key, no network.

    python -m unittest discover -s tests -v

The OpenAI client is replaced by a scripted stand-in that returns objects
shaped like the Responses API's (`output` items with `type`, `name`,
`arguments`, `call_id`, plus `output_text`). That checks our loop, tools and
retrieval; it cannot check how a real model behaves.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from shopbot import config, db
from shopbot.agent import Agent, system_prompt
from shopbot.embeddings import HashingEmbedder
from shopbot.rag import VectorIndex, chunk_markdown, load_all_chunks
from shopbot.tools import TOOL_SCHEMAS, ShopTools

ALICE = "+12025550143"
BOB = "+12025550178"


class FakeResponses:
    """Plays back a list of prepared responses and records each request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def create(self, **kwargs):
        kwargs["input"] = list(kwargs["input"])  # snapshot; the agent keeps appending
        self.requests.append(kwargs)
        return self.script.pop(0)


def fake_client(*script):
    return SimpleNamespace(responses=FakeResponses(script))


def says(text):
    return SimpleNamespace(output=[SimpleNamespace(type="message")], output_text=text)


def calls(name, arguments, call_id="call_1"):
    return SimpleNamespace(
        output=[
            SimpleNamespace(type="reasoning", id="rs_" + call_id),
            SimpleNamespace(
                type="function_call",
                name=name,
                arguments=json.dumps(arguments),
                call_id=call_id,
            ),
        ],
        output_text="",
    )


class ShopBotTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls._tmp.name) / "shop.db"
        db.build_database(db_path=cls.db_path)
        cls.embedder = HashingEmbedder()
        cls.index = VectorIndex.build(load_all_chunks(), cls.embedder)
        cls.tools = ShopTools(cls.index, cls.embedder, db_path=cls.db_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class RetrievalTests(ShopBotTestCase):
    def test_markdown_is_split_by_section_and_keeps_document_title(self):
        chunks = chunk_markdown(config.KNOWLEDGE_DIR / "returns_and_exchanges.md")
        titles = [chunk.title for chunk in chunks]
        self.assertIn("Returns and exchanges > Refunds", titles)
        self.assertEqual(len(chunks), 8)
        refunds = next(c for c in chunks if c.title.endswith("Refunds"))
        self.assertTrue(refunds.text.startswith("Returns and exchanges > Refunds"))
        self.assertIn("7 business days", refunds.text)

    def test_policy_question_finds_the_right_section(self):
        hits = self.index.search("refund for sale items", self.embedder, kind="doc", k=3)
        self.assertEqual(hits[0][0].id, "returns_and_exchanges#sale-items")

    def test_kind_filter_separates_documents_from_products(self):
        docs = self.index.search("linen", self.embedder, kind="doc", k=5)
        products = self.index.search("linen", self.embedder, kind="product", k=5)
        self.assertTrue(all(chunk.kind == "doc" for chunk, _ in docs))
        self.assertTrue(all(chunk.kind == "product" for chunk, _ in products))

    def test_index_survives_save_and_load(self):
        with tempfile.TemporaryDirectory() as folder:
            self.index.save(Path(folder))
            loaded = VectorIndex.load(Path(folder))
        self.assertEqual(loaded.embedder_name, self.index.embedder_name)
        self.assertEqual(len(loaded.chunks), len(self.index.chunks))
        before = self.index.search("opening hours", self.embedder, k=1)[0][0].id
        after = loaded.search("opening hours", self.embedder, k=1)[0][0].id
        self.assertEqual(before, after)

    def test_query_with_a_different_embedder_is_rejected(self):
        other = HashingEmbedder(dims=64)
        with self.assertRaises(ValueError):
            self.index.search("returns", other)

    def test_prices_and_stock_are_not_frozen_into_the_index(self):
        for chunk in self.index.chunks:
            if chunk.kind == "product":
                self.assertNotIn("₪", chunk.text)
                self.assertNotIn("in stock", chunk.text.lower())


class ToolTests(ShopBotTestCase):
    def test_schemas_meet_strict_mode_rules(self):
        names = set()
        for schema in TOOL_SCHEMAS:
            names.add(schema["name"])
            parameters = schema["parameters"]
            self.assertTrue(schema["strict"])
            self.assertIs(parameters["additionalProperties"], False)
            self.assertEqual(set(parameters["required"]), set(parameters["properties"]))
        self.assertEqual(len(names), len(TOOL_SCHEMAS))

    def test_every_schema_has_a_handler(self):
        for schema in TOOL_SCHEMAS:
            result = self.tools.run(schema["name"], "not json", ALICE)
            self.assertIn("Bad arguments", result["error"])

    def test_check_stock_reports_exact_quantities(self):
        result = self.tools.check_stock("P-1003", color="charcoal", size="M")
        self.assertEqual(result["variants"], [{"colour": "charcoal", "size": "M", "in_stock": 0}])
        result = self.tools.check_stock("p-1001", color="White", size="m")
        self.assertEqual(result["variants"][0]["in_stock"], 14)

    def test_check_stock_lists_sizes_in_shop_order(self):
        result = self.tools.check_stock("P-1003", color="cream")
        self.assertEqual([v["size"] for v in result["variants"]], ["XS", "S", "M", "L", "XL"])

    def test_check_stock_handles_unknown_product_and_variant(self):
        self.assertIn("error", self.tools.check_stock("P-9999"))
        result = self.tools.check_stock("P-1013", size="XL")  # scarf is one size
        self.assertIn("error", result)
        self.assertTrue(result["available_variants"])

    def test_search_products_joins_live_price_and_stock(self):
        result = self.tools.search_products("linen shirt")
        top = result["products"][0]
        self.assertEqual(top["product_id"], "P-1002")
        self.assertEqual(top["price_ils"], 249)
        self.assertNotIn("S", top["sizes_in_stock_by_colour"]["olive"])  # olive S is sold out

    def test_search_products_applies_filters(self):
        result = self.tools.search_products("linen", category="dresses")
        # a filter narrows the candidates; ranking inside it is still by relevance
        self.assertEqual({p["category"] for p in result["products"]}, {"dresses"})
        self.assertEqual(result["products"][0]["product_id"], "P-1009")
        result = self.tools.search_products("warm winter", max_price=200)
        self.assertTrue(result["products"])
        self.assertTrue(all(p["price_ils"] <= 200 for p in result["products"]))
        self.assertEqual(self.tools.search_products("jacket", max_price=10), {"products": []})

    def test_customer_sees_own_order_with_days_since_delivery(self):
        result = self.tools.get_order_status(ALICE, "ord-5003")
        self.assertEqual(result["status"], "delivered")
        self.assertEqual(result["days_since_delivery"], 44)
        self.assertEqual(result["total_ils"], 599)
        shipped = self.tools.get_order_status(ALICE, "ORD-5001")
        self.assertEqual(shipped["total_ils"], 389 + 149)
        self.assertNotIn("days_since_delivery", shipped)

    def test_customer_cannot_see_someone_elses_order(self):
        other = self.tools.get_order_status(BOB, "ORD-5001")
        missing = self.tools.get_order_status(BOB, "ORD-0000")
        self.assertIn("error", other)
        self.assertNotIn("status", other)
        # identical wording, so the reply does not reveal that ORD-5001 exists
        self.assertEqual(
            other["error"].replace("ORD-5001", "X"), missing["error"].replace("ORD-0000", "X")
        )

    def test_list_my_orders_is_scoped_and_newest_first(self):
        mine = self.tools.list_my_orders(ALICE)["orders"]
        self.assertEqual([o["order_id"] for o in mine], ["ORD-5001", "ORD-5002", "ORD-5003"])
        self.assertEqual(len(self.tools.list_my_orders(BOB)["orders"]), 2)
        self.assertEqual(self.tools.list_my_orders("+10000000000"), {"orders": []})

    def test_handoff_is_recorded(self):
        result = self.tools.run(
            "handoff_to_human", json.dumps({"reason": "Wants to cancel ORD-5004"}), BOB
        )
        connection = sqlite3.connect(self.db_path)
        row = connection.execute(
            "SELECT customer_phone, reason FROM handoffs WHERE id = ?", (result["ticket"],)
        ).fetchone()
        connection.close()
        self.assertEqual(row, (BOB, "Wants to cancel ORD-5004"))

    def test_model_cannot_choose_whose_orders_it_reads(self):
        # Even if the model invents a customer_phone argument, it is refused.
        result = self.tools.run(
            "get_order_status",
            json.dumps({"order_id": "ORD-5001", "customer_phone": ALICE}),
            BOB,
        )
        self.assertIn("error", result)
        self.assertNotIn("status", result)


class AgentTests(ShopBotTestCase):
    def test_tool_call_is_executed_and_result_sent_back(self):
        client = fake_client(
            calls("check_stock", {"product_id": "P-1003", "color": "charcoal", "size": "M"}),
            says("Sorry, charcoal M is sold out."),
        )
        reply = Agent(client, self.tools).reply(ALICE, "Do you have the merino in charcoal M?")

        self.assertEqual(reply.text, "Sorry, charcoal M is sold out.")
        self.assertEqual([step.tool for step in reply.steps], ["check_stock"])
        self.assertEqual(reply.steps[0].result["variants"][0]["in_stock"], 0)

        first, second = client.responses.requests
        self.assertEqual(first["input"], [{"role": "user", "content": "Do you have the merino in charcoal M?"}])
        self.assertIs(first["tools"], TOOL_SCHEMAS)
        self.assertIn("Olive Lane", first["instructions"])
        # second request = user message + reasoning + function_call + our output
        kinds = [getattr(item, "type", None) or item.get("type", "user") for item in second["input"]]
        self.assertEqual(kinds, ["user", "reasoning", "function_call", "function_call_output"])
        output = second["input"][-1]
        self.assertEqual(output["call_id"], "call_1")
        self.assertEqual(json.loads(output["output"])["variants"][0]["in_stock"], 0)

    def test_several_tool_calls_in_one_response_all_get_outputs(self):
        both = SimpleNamespace(
            output=calls("list_my_orders", {}, "call_a").output
            + calls("search_knowledge", {"query": "return window"}, "call_b").output,
            output_text="",
        )
        client = fake_client(both, says("done"))
        reply = Agent(client, self.tools).reply(ALICE, "Can I return my last order?")
        self.assertEqual([s.tool for s in reply.steps], ["list_my_orders", "search_knowledge"])
        outputs = [i for i in client.responses.requests[1]["input"] if isinstance(i, dict) and i.get("type") == "function_call_output"]
        self.assertEqual([o["call_id"] for o in outputs], ["call_a", "call_b"])

    def test_order_tools_use_the_senders_identity(self):
        client = fake_client(calls("get_order_status", {"order_id": "ORD-5001"}), says("ok"))
        reply = Agent(client, self.tools).reply(BOB, "Where is ORD-5001?")
        self.assertIn("error", reply.steps[0].result)

    def test_conversation_is_remembered_per_customer(self):
        client = fake_client(says("Hi!"), says("Still here."), says("Hello Bob."))
        agent = Agent(client, self.tools)
        agent.reply(ALICE, "first")
        agent.reply(ALICE, "second")
        agent.reply(BOB, "hello")
        second, bob = client.responses.requests[1], client.responses.requests[2]
        self.assertEqual(second["input"][0], {"role": "user", "content": "first"})
        self.assertEqual(second["input"][-1], {"role": "user", "content": "second"})
        self.assertEqual(len(second["input"]), 3)
        self.assertEqual(bob["input"], [{"role": "user", "content": "hello"}])

        agent.reset(ALICE)
        client.responses.script.append(says("Fresh start."))
        agent.reply(ALICE, "third")
        self.assertEqual(client.responses.requests[3]["input"], [{"role": "user", "content": "third"}])

    def test_old_turns_are_dropped_whole(self):
        client = fake_client(
            calls("list_my_orders", {}), says("one"), says("two"), says("three"), says("four")
        )
        agent = Agent(client, self.tools, max_turns=2)
        for message in ("m1", "m2", "m3", "m4"):
            agent.reply(ALICE, message)
        last = client.responses.requests[-1]["input"]
        users = [i["content"] for i in last if isinstance(i, dict) and i.get("role") == "user"]
        self.assertEqual(users, ["m2", "m3", "m4"])
        # m1's tool call and its output left together; nothing is orphaned
        self.assertFalse(any(isinstance(i, dict) and i.get("type") == "function_call_output" for i in last))
        self.assertFalse(any(getattr(i, "type", "") == "function_call" for i in last))

    def test_loop_stops_after_max_steps_and_forces_an_answer(self):
        client = fake_client(
            calls("list_my_orders", {}, "c1"),
            calls("list_my_orders", {}, "c2"),
            calls("list_my_orders", {}, "c3"),
            says("Here is what I found."),
        )
        reply = Agent(client, self.tools, max_steps=3).reply(ALICE, "loop")
        self.assertEqual(reply.text, "Here is what I found.")
        self.assertEqual(len(reply.steps), 3)
        requests = client.responses.requests
        self.assertEqual(len(requests), 4)
        self.assertEqual(requests[-1]["tool_choice"], "none")
        self.assertTrue(all("tool_choice" not in r for r in requests[:-1]))

    def test_failed_request_does_not_corrupt_memory(self):
        class Boom:
            def create(self, **kwargs):
                raise RuntimeError("network down")

        agent = Agent(SimpleNamespace(responses=Boom()), self.tools)
        with self.assertRaises(RuntimeError):
            agent.reply(ALICE, "hello")
        agent.client = fake_client(says("Back."))
        agent.reply(ALICE, "hello again")
        self.assertEqual(
            agent.client.responses.requests[0]["input"], [{"role": "user", "content": "hello again"}]
        )

    def test_system_prompt_carries_todays_date(self):
        from datetime import date

        self.assertIn("Sunday, 11 October 2026", system_prompt(date(2026, 10, 11)))


if __name__ == "__main__":
    unittest.main()
