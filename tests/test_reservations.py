"""Tests for holding items at the shop: the first tool that changes something real.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from shopbot import db
from shopbot.agent import Agent
from shopbot.embeddings import HashingEmbedder
from shopbot.rag import VectorIndex, load_all_chunks
from shopbot.tools import ShopTools

CAROL, DAN, ERIN = "+12025550101", "+12025550102", "+12025550103"
START = datetime(2026, 10, 12, 18, 0, 0)
SCARF = {"product_id": "P-1013", "color": "burgundy", "size": "One size"}  # 13 in stock
LAST_COAT = {"product_id": "P-1012", "color": "yellow", "size": "M"}  # exactly 1 in stock


class ReservationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = HashingEmbedder()
        cls.index = VectorIndex.build(load_all_chunks(), cls.embedder)

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.db_path = Path(folder.name) / "shop.db"
        db.build_database(db_path=self.db_path)
        self.now = START
        self.tools = self.make_tools()

    def make_tools(self):
        return ShopTools(self.index, self.embedder, db_path=self.db_path, clock=lambda: self.now)

    def hold(self, customer, item=SCARF, tools=None):
        """Prepare in turn 1 and confirm in turn 2, as a well-behaved agent would."""
        tools = tools or self.tools
        prepared = tools.reserve_item(customer, 1, **item)
        self.assertEqual(prepared.get("status"), "needs_confirmation", prepared)
        return tools.confirm_reservation(customer, 2)

    def in_stock(self, item):
        return self.tools.check_stock(item["product_id"], item["color"], item["size"])["variants"][0]["in_stock"]

    def rows(self):
        connection = sqlite3.connect(self.db_path)
        self.addCleanup(connection.close)
        return connection.execute("SELECT customer_phone, product_id, color, size, status FROM reservations").fetchall()


class TwoStepTests(ReservationTestCase):
    def test_reserve_only_prepares(self):
        result = self.tools.reserve_item(CAROL, 1, **SCARF)
        self.assertEqual(result["status"], "needs_confirmation")
        self.assertEqual(result["item"]["name"], "Wool Scarf")
        self.assertEqual(result["hold_hours"], 48)
        self.assertIn("Nothing is held yet", result["next_step"])
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.in_stock(SCARF), 13)

    def test_confirming_in_the_same_turn_is_refused(self):
        self.tools.reserve_item(CAROL, 1, **SCARF)
        refused = self.tools.confirm_reservation(CAROL, 1)
        self.assertIn("has not replied yet", refused["error"])
        self.assertEqual(self.rows(), [])
        # the prepared hold survives, so the customer's "yes" can still confirm it
        confirmed = self.tools.confirm_reservation(CAROL, 2)
        self.assertEqual(confirmed["status"], "reserved")

    def test_confirmation_places_the_hold_and_reduces_availability(self):
        result = self.hold(CAROL)
        self.assertEqual(result["status"], "reserved")
        self.assertEqual(result["held_until"], "Wednesday 14 October at 18:00")
        self.assertEqual(self.rows(), [(CAROL, "P-1013", "burgundy", "One size", "active")])
        self.assertEqual(self.in_stock(SCARF), 12)

    def test_confirmation_must_come_in_the_very_next_message(self):
        self.tools.reserve_item(CAROL, 1, **SCARF)
        lapsed = self.tools.confirm_reservation(CAROL, 3)
        self.assertIn("lapsed", lapsed["error"])
        self.assertIn("no prepared hold", self.tools.confirm_reservation(CAROL, 4)["error"])
        self.assertEqual(self.rows(), [])

    def test_nothing_to_confirm(self):
        self.assertIn("Call reserve_item first", self.tools.confirm_reservation(CAROL, 5)["error"])

    def test_one_customers_preparation_cannot_be_confirmed_by_another(self):
        self.tools.reserve_item(CAROL, 1, **SCARF)
        self.assertIn("no prepared hold", self.tools.confirm_reservation(DAN, 2)["error"])

    def test_input_is_matched_loosely_but_stored_exactly(self):
        prepared = self.tools.reserve_item(CAROL, 1, product_id=" p-1013 ", color="Burgundy", size="one size")
        self.assertEqual((prepared["item"]["colour"], prepared["item"]["size"]), ("burgundy", "One size"))
        self.tools.confirm_reservation(CAROL, 2)
        self.assertEqual(self.rows()[0][2:4], ("burgundy", "One size"))


class RuleTests(ReservationTestCase):
    def test_sold_out_variant_cannot_be_held(self):
        result = self.tools.reserve_item(CAROL, 1, product_id="P-1013", color="grey", size="One size")
        self.assertIn("out of stock", result["error"])
        self.assertEqual(result["in_stock_by_colour"], {"burgundy": ["One size"]})

    def test_unknown_combination_lists_what_exists(self):
        result = self.tools.reserve_item(CAROL, 1, product_id="P-1013", color="pink", size="One size")
        self.assertIn("No such product", result["error"])
        self.assertEqual({v["colour"] for v in result["variants"]}, {"grey", "burgundy"})
        self.assertIn("No such product", self.tools.reserve_item(CAROL, 1, "P-9999", "red", "M")["error"])

    def test_at_most_two_holds_per_customer(self):
        self.hold(CAROL, SCARF)
        self.hold(CAROL, {"product_id": "P-1001", "color": "white", "size": "M"})
        third = self.tools.reserve_item(CAROL, 1, product_id="P-1009", color="white", size="S")
        self.assertIn("the most allowed", third["error"])
        self.assertEqual(len(third["current_holds"]), 2)
        self.assertEqual(self.hold(DAN, SCARF)["status"], "reserved")  # limits are per customer

    def test_same_item_cannot_be_held_twice_by_one_customer(self):
        first = self.hold(CAROL)
        again = self.tools.reserve_item(CAROL, 3, **SCARF)
        self.assertIn("already has this exact item", again["error"])
        self.assertEqual(again["reservation_id"], first["reservation_id"])

    def test_holding_the_last_unit_removes_it_from_product_search(self):
        self.hold(CAROL, LAST_COAT)
        self.assertEqual(self.in_stock(LAST_COAT), 0)
        coat = next(p for p in self.tools.search_products("waterproof rain coat")["products"] if p["product_id"] == "P-1012")
        self.assertNotIn("M", coat["sizes_in_stock_by_colour"]["yellow"])
        self.assertIn("out of stock", self.tools.reserve_item(DAN, 1, **LAST_COAT)["error"])


class CancelListExpiryTests(ReservationTestCase):
    def test_cancel_own_hold_frees_the_item(self):
        number = self.hold(CAROL, LAST_COAT)["reservation_id"]
        self.assertEqual(self.tools.cancel_reservation(CAROL, number), {"status": "cancelled", "reservation_id": number})
        self.assertEqual(self.in_stock(LAST_COAT), 1)
        self.assertIn("No active hold", self.tools.cancel_reservation(CAROL, number)["error"])

    def test_cannot_cancel_or_see_someone_elses_hold(self):
        number = self.hold(CAROL)["reservation_id"]
        other = self.tools.cancel_reservation(DAN, number)
        missing = self.tools.cancel_reservation(DAN, 9999)
        self.assertEqual(other["error"].replace(str(number), "N"), missing["error"].replace("9999", "N"))
        self.assertEqual(self.tools.list_my_reservations(DAN), {"reservations": []})
        self.assertEqual(self.rows()[0][4], "active")

    def test_list_shows_active_holds_with_end_time(self):
        number = self.hold(CAROL)["reservation_id"]
        self.assertEqual(
            self.tools.list_my_reservations(CAROL),
            {"reservations": [{"reservation_id": number, "name": "Wool Scarf", "colour": "burgundy",
                               "size": "One size", "held_until": "Wednesday 14 October at 18:00"}]},
        )

    def test_holds_end_by_themselves_after_48_hours(self):
        number = self.hold(CAROL, LAST_COAT)["reservation_id"]
        self.now = START + timedelta(hours=47, minutes=59)
        self.assertEqual(self.in_stock(LAST_COAT), 0)
        self.now = START + timedelta(hours=48, minutes=1)
        self.assertEqual(self.in_stock(LAST_COAT), 1)
        self.assertEqual(self.tools.list_my_reservations(CAROL), {"reservations": []})
        self.assertIn("No active hold", self.tools.cancel_reservation(CAROL, number)["error"])
        self.assertEqual(self.hold(DAN, LAST_COAT)["status"], "reserved")  # back on sale


class LastItemTests(ReservationTestCase):
    def test_item_sold_out_between_prepare_and_confirm(self):
        self.tools.reserve_item(CAROL, 1, **LAST_COAT)
        self.assertEqual(self.hold(DAN, LAST_COAT)["status"], "reserved")
        late = self.tools.confirm_reservation(CAROL, 2)
        self.assertIn("was just taken", late["error"])
        self.assertIn("no prepared hold", self.tools.confirm_reservation(CAROL, 2)["error"])

    def test_two_customers_confirming_the_last_unit_at_once(self):
        # Each customer talks to their own server thread, as on WhatsApp with
        # several workers. Both saw "1 in stock" and both say yes together.
        customers = [CAROL, DAN, ERIN]
        tools = {customer: self.make_tools() for customer in customers}
        for customer in customers:
            self.assertEqual(tools[customer].reserve_item(customer, 1, **LAST_COAT)["status"], "needs_confirmation")
        start = threading.Barrier(len(customers))
        results = {}

        def confirm(customer):
            start.wait()
            results[customer] = tools[customer].confirm_reservation(customer, 2)

        threads = [threading.Thread(target=confirm, args=(c,)) for c in customers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        winners = [c for c, r in results.items() if r.get("status") == "reserved"]
        losers = [c for c, r in results.items() if "was just taken" in r.get("error", "")]
        self.assertEqual(len(winners), 1, results)
        self.assertEqual(len(losers), 2, results)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.in_stock(LAST_COAT), 0)


class AgentFlowTests(ReservationTestCase):
    """The agent passes turn numbers, so a model that skips asking is stopped."""

    def script(self, *responses):
        responses = list(responses)
        return SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: responses.pop(0)))

    @staticmethod
    def calls(name, arguments, call_id):
        item = SimpleNamespace(type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id)
        return SimpleNamespace(output=[item], output_text="")

    @staticmethod
    def says(text):
        return SimpleNamespace(output=[SimpleNamespace(type="message")], output_text=text)

    def test_model_that_confirms_without_asking_is_refused(self):
        client = self.script(
            self.calls("reserve_item", SCARF, "c1"),
            self.calls("confirm_reservation", {}, "c2"),  # tries to skip the question
            self.says("Shall I hold the burgundy Wool Scarf for 48 hours?"),
            self.calls("confirm_reservation", {}, "c3"),
            self.says("Done! It's held until Wednesday 14 October at 18:00."),
        )
        agent = Agent(client, self.tools)
        first = agent.reply(CAROL, "Hold the burgundy scarf please")
        self.assertEqual([s.tool for s in first.steps], ["reserve_item", "confirm_reservation"])
        self.assertIn("has not replied yet", first.steps[1].result["error"])
        self.assertEqual(self.rows(), [])

        second = agent.reply(CAROL, "yes")
        self.assertEqual(second.steps[0].result["status"], "reserved")
        self.assertEqual(len(self.rows()), 1)

    def test_reset_forgets_a_prepared_hold(self):
        client = self.script(
            self.calls("reserve_item", SCARF, "c1"), self.says("Confirm?"),
            self.calls("confirm_reservation", {}, "c2"), self.says("Sorry, there is nothing to confirm."),
        )
        agent = Agent(client, self.tools)
        agent.reply(CAROL, "Hold the burgundy scarf")
        agent.reset(CAROL)
        reply = agent.reply(CAROL, "yes")
        self.assertIn("no prepared hold", reply.steps[0].result["error"])
        self.assertEqual(self.rows(), [])


class OldDatabaseTests(ReservationTestCase):
    def test_database_built_before_reservations_gets_the_table(self):
        connection = sqlite3.connect(self.db_path)
        connection.execute("DROP TABLE reservations")
        connection.commit()
        connection.close()
        tools = self.make_tools()
        self.assertEqual(tools.check_stock("P-1013", "burgundy")["variants"][0]["in_stock"], 13)
        self.assertEqual(self.hold(CAROL, tools=tools)["status"], "reserved")


if __name__ == "__main__":
    unittest.main()
