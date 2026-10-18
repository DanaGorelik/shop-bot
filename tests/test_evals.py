"""Tests for the evaluation runner. Offline: the agent's model and the judge are scripted.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from shopbot import config, db, evals
from shopbot.agent import Agent
from shopbot.embeddings import HashingEmbedder
from shopbot.rag import VectorIndex, load_all_chunks
from shopbot.tools import ShopTools

ALICE, BOB = "+12025550143", "+12025550178"


def says(text):
    return SimpleNamespace(output=[SimpleNamespace(type="message")], output_text=text)


def calls(name, arguments, call_id="c1"):
    item = SimpleNamespace(type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id)
    return SimpleNamespace(output=[item], output_text="")


def verdict(passed, reasoning="Because."):
    return SimpleNamespace(output_text=json.dumps({"reasoning": reasoning, "pass": passed}))


class ScriptedClient:
    """Agent calls get the next scripted response; judge calls (they pass `text=`) get the verdict."""

    def __init__(self, agent_script, judge_response=None):
        self.agent_script, self.judge_response = list(agent_script), judge_response
        self.judge_requests = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        if "text" in kwargs:
            self.judge_requests.append(kwargs)
            if isinstance(self.judge_response, Exception):
                raise self.judge_response
            return self.judge_response
        return self.agent_script.pop(0)


class CasesFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.customers, cls.cases = evals.load_cases()

    def test_file_loads_and_covers_every_category(self):
        self.assertGreaterEqual(len(self.cases), 30)
        categories = {case["category"] for case in self.cases}
        self.assertEqual(
            categories, {"policy", "products", "orders", "privacy", "handoff", "grounding", "conversation", "actions"}
        )
        self.assertTrue(all(case.get("criteria") for case in self.cases))

    def test_customers_are_the_sample_customers(self):
        self.assertEqual((self.customers["alice"], self.customers["bob"]), (ALICE, BOB))

    def test_expected_facts_exist_in_the_shop_data(self):
        # A case that expects text the data does not contain could never pass.
        data = "".join(p.read_text(encoding="utf-8") for p in config.KNOWLEDGE_DIR.glob("*.md"))
        data += config.CATALOG_PATH.read_text(encoding="utf-8") + config.ORDERS_PATH.read_text(encoding="utf-8")
        data = data.casefold()
        for case in self.cases:
            for wanted in case.get("must_contain", []):
                options = wanted if isinstance(wanted, list) else [wanted]
                self.assertTrue(any(o.casefold() in data for o in options), (case["id"], wanted))

    def test_every_tool_is_exercised_by_some_case(self):
        used = set()
        for case in self.cases:
            used |= set(case.get("must_call", [])) | set(case.get("must_call_any", []))
        self.assertEqual(used, evals.TOOL_NAMES)

    def test_select_by_category_or_id_fragment(self):
        self.assertEqual({c["category"] for c in evals.select(self.cases, "privacy")}, {"privacy"})
        self.assertEqual([c["id"] for c in evals.select(self.cases, "friday")], ["policy-friday-hours"])
        self.assertEqual(evals.select(self.cases, None), self.cases)
        self.assertEqual(evals.select(self.cases, "no-such-thing"), [])


class ValidationTests(unittest.TestCase):
    def load(self, *cases):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cases.json"
            path.write_text(json.dumps({"customers": {"new": "+1", "alice": ALICE}, "cases": list(cases)}))
            return evals.load_cases(path)

    def good(self, **changes):
        return {"id": "a", "category": "policy", "messages": ["hi"], "criteria": "Is polite.", **changes}

    def test_a_valid_case_loads(self):
        self.assertEqual(len(self.load(self.good())[1]), 1)

    def test_mistakes_are_reported_with_the_case_id(self):
        broken = [
            (self.good(must_call=["search_knowlege"]), "unknown tool"),
            (self.good(customer="carol"), "unknown customer"),
            (self.good(messages=[]), "messages"),
            (self.good(messages=["  "]), "messages"),
            (self.good(must_contian=["x"]), "unknown field"),
            ({"id": "a", "category": "policy", "messages": ["hi"]}, "no expectations"),
            ({"id": "a", "messages": ["hi"], "criteria": "x"}, "missing category"),
        ]
        for case, expected in broken:
            with self.assertRaises(ValueError) as caught:
                self.load(case)
            self.assertIn(expected, str(caught.exception))
            self.assertIn("a:", str(caught.exception))

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self.load(self.good(), self.good())
        self.assertIn("Duplicate", str(caught.exception))


class RuleTests(unittest.TestCase):
    def test_all_rules_pass(self):
        case = {
            "must_call": ["get_order_status"], "must_not_call": ["handoff_to_human"],
            "must_call_any": ["search_products", "search_knowledge"], "max_tool_calls": 2,
            "must_contain": ["SP48213907", ["14:30", "2:30"]], "must_not_contain": ["cancelled"],
        }
        answer = "It shipped, tracking sp48213907. We close at 2:30 PM."
        self.assertEqual(evals.check_rules(case, answer, ["get_order_status", "search_knowledge"]), [])

    def test_each_broken_rule_gets_its_own_line(self):
        case = {
            "must_call": ["get_order_status"], "must_not_call": ["handoff_to_human"],
            "must_call_any": ["search_products", "search_knowledge"], "max_tool_calls": 0,
            "must_contain": ["SP48213907", ["14:30", "2:30"]], "must_not_contain": ["Cancelled"],
        }
        failures = evals.check_rules(case, "Your order was cancelled.", ["handoff_to_human"])
        self.assertEqual(
            failures,
            [
                "did not call get_order_status",
                "called handoff_to_human, which it should not",
                "called none of search_products, search_knowledge",
                "made 1 tool call(s), at most 0 expected",
                "reply does not contain 'SP48213907'",
                "reply does not contain '14:30' or '2:30'",
                "reply contains 'Cancelled'",
            ],
        )

    def test_matching_ignores_case_and_works_in_hebrew(self):
        self.assertEqual(evals.check_rules({"must_contain": ["YELLOW"]}, "It comes in Yellow.", []), [])
        self.assertEqual(evals.check_rules({"must_contain": ["14:30"]}, "אנחנו סוגרים ב-14:30 ביום שישי", []), [])


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls._tmp.name) / "shop.db"
        db.build_database(db_path=cls.db_path)
        embedder = HashingEmbedder()
        cls.tools = ShopTools(VectorIndex.build(load_all_chunks(), embedder), embedder, db_path=cls.db_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def run_case(self, case, agent_script, judge_response=verdict(True), customer=ALICE, use_judge=True):
        client = ScriptedClient(agent_script, judge_response)
        judge = (lambda c, t: evals.ask_judge(client, "judge-model", c, t)) if use_judge else None
        case = {"id": "x", "category": "orders", **case}
        result = evals.run_case(case, customer, Agent(client, self.tools), judge)
        return result, client

    def test_passing_case_runs_tools_and_asks_the_judge(self):
        case = {
            "messages": ["Where is my order ORD-5001?"], "must_call": ["get_order_status"],
            "must_contain": ["SP48213907"], "criteria": "Says the order has shipped.",
        }
        script = [calls("get_order_status", {"order_id": "ORD-5001"}), says("It has shipped. Tracking: SP48213907")]
        result, client = self.run_case(case, script, verdict(True, "It says shipped."))
        self.assertTrue(result.passed)
        self.assertEqual(result.failures, [])
        self.assertEqual(result.tools, ["get_order_status"])
        self.assertEqual(result.judge_reason, "It says shipped.")
        (request,) = client.judge_requests
        self.assertEqual(request["model"], "judge-model")
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertIn("Customer: Where is my order ORD-5001?", request["input"])
        self.assertIn("It has shipped. Tracking: SP48213907", request["input"])
        self.assertIn("Says the order has shipped.", request["input"])

    def test_rule_failure_and_judge_failure_are_both_reported(self):
        case = {"messages": ["Cancel ORD-5004"], "must_call": ["handoff_to_human"], "criteria": "Must not claim it is cancelled."}
        result, _ = self.run_case(case, [says("Done, your order is cancelled.")], verdict(False, "It claims the order was cancelled."))
        self.assertFalse(result.passed)
        self.assertEqual(
            result.failures,
            ["did not call handoff_to_human", "judge: It claims the order was cancelled."],
        )
        self.assertEqual(result.answer, "Done, your order is cancelled.")

    def test_case_customer_identity_reaches_the_tools(self):
        case = {"messages": ["Where is ORD-5001?"], "must_not_contain": ["SP48213907"]}
        client = ScriptedClient([calls("get_order_status", {"order_id": "ORD-5001"}), says("I can't find that order.")])
        agent = Agent(client, self.tools)
        result = evals.run_case({"id": "x", "category": "privacy", **case}, BOB, agent)
        self.assertTrue(result.passed)
        # Bob's lookup of Alice's order came back as an error, so the model never saw the details
        tool_output = [i for i in agent._history[BOB][0] if isinstance(i, dict) and i.get("type") == "function_call_output"]
        self.assertIn("No order ORD-5001 found", tool_output[0]["output"])

    def test_multi_turn_case_grades_the_last_reply_with_full_transcript(self):
        case = {"messages": ["What time do you close on Friday?", "And on Saturday?"], "criteria": "Says closed on Saturday."}
        script = [
            calls("search_knowledge", {"query": "friday hours"}), says("We close at 14:30 on Fridays."),
            says("We're closed on Saturdays."),
        ]
        result, client = self.run_case(case, script)
        self.assertTrue(result.passed)
        self.assertEqual(result.answer, "We're closed on Saturdays.")
        self.assertEqual(result.tools, ["search_knowledge"])
        prompt = client.judge_requests[0]["input"]
        self.assertIn("Assistant: We close at 14:30 on Fridays.", prompt)
        self.assertIn("Customer: And on Saturday?", prompt)
        self.assertEqual(prompt.count("We're closed on Saturdays."), 1)  # only as the reply to grade

    def test_without_judge_only_rules_apply(self):
        case = {"messages": ["hi"], "max_tool_calls": 0, "criteria": "Anything."}
        result, client = self.run_case(case, [says("Hello!")], use_judge=False)
        self.assertTrue(result.passed)
        self.assertEqual(client.judge_requests, [])

    def test_agent_error_fails_the_case_without_stopping_the_run(self):
        class Broken:
            def reply(self, customer, message):
                raise RuntimeError("rate limited")

        result = evals.run_case({"id": "x", "category": "policy", "messages": ["hi"], "criteria": "c"}, ALICE, Broken())
        self.assertFalse(result.passed)
        self.assertIn("agent error", result.failures[0])
        self.assertIn("rate limited", result.failures[0])

    def test_judge_error_or_bad_json_fails_the_case(self):
        case = {"messages": ["hi"], "criteria": "Is polite."}
        result, _ = self.run_case(case, [says("Hello!")], RuntimeError("model not found"))
        self.assertFalse(result.passed)
        self.assertIn("judge error", result.failures[0])
        result, _ = self.run_case(case, [says("Hello!")], SimpleNamespace(output_text="not json"))
        self.assertFalse(result.passed)
        self.assertIn("judge error", result.failures[0])

    def test_run_all_gives_each_case_a_fresh_agent_and_keeps_order(self):
        customers, cases = evals.load_cases()
        made = []

        class Echo:
            def __init__(self):
                self.seen = []
                made.append(self)

            def reply(self, customer, message):
                self.seen.append((customer, message))
                return SimpleNamespace(text="ok", steps=[])

        printed = []
        results = evals.run_all(cases, customers, Echo, judge=None, workers=4, on_result=printed.append)
        self.assertEqual([r.id for r in results], [c["id"] for c in cases])
        self.assertEqual(printed, results)
        self.assertEqual(len(made), len(cases))
        # every agent saw exactly the messages of one case, and nothing else
        self.assertEqual(sorted([m for _, m in a.seen] for a in made), sorted(c["messages"] for c in cases))
        by_id = {c["id"]: c for c in cases}
        bob_agent = next(a for a in made if a.seen[0][1] == by_id["handoff-cancel-order"]["messages"][0])
        self.assertEqual(bob_agent.seen[0][0], BOB)


class ReportTests(unittest.TestCase):
    def results(self, **passed):
        return [evals.CaseResult(id=name, category=name.split("-")[0], passed=ok, failures=[] if ok else ["judge: nope"],
                                 answer="A reply\non two lines", tools=["search_knowledge"], seconds=1.5)
                for name, ok in passed.items()]

    def test_summary_counts_by_category(self):
        text = evals.summary(self.results(**{"policy-a": True, "policy-b": False, "orders-a": True}))
        self.assertIn("policy  1/2", text)
        self.assertIn("orders  1/1", text)
        self.assertIn("Total: 2/3 passed (67%)", text)

    def test_failed_case_shows_reason_tools_and_reply(self):
        (passed, failed) = self.results(**{"policy-a": True, "policy-b": False})
        self.assertEqual(evals.describe_result(passed), "PASS  policy-a  (1.5s)")
        text = evals.describe_result(failed)
        self.assertIn("FAIL  policy-b", text)
        self.assertIn("- judge: nope", text)
        self.assertIn("tools:  search_knowledge", text)
        self.assertIn("reply:  A reply", text)
        self.assertIn("reply:", evals.describe_result(passed, show_answer=True))

    def test_colour_is_opt_in_green_for_pass_red_for_fail(self):
        (passed, failed) = self.results(**{"policy-a": True, "policy-b": False})
        self.assertTrue(evals.describe_result(passed, color=True).startswith("\033[32mPASS\033[0m  policy-a"))
        self.assertTrue(evals.describe_result(failed, color=True).startswith("\033[31mFAIL\033[0m  policy-b"))
        self.assertNotIn("\033", evals.describe_result(passed))
        self.assertNotIn("\033", evals.describe_result(failed))

    def test_results_are_saved_and_compared_with_the_previous_run(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            first = self.results(**{"policy-a": True, "policy-b": False, "orders-a": True})
            first_path = evals.save_results(first, "gpt-6-luna", "gpt-6.1-sol", folder, datetime(2026, 10, 11, 23, 0, 0))
            self.assertIsNone(evals.previous_run(first_path))
            saved = json.loads(first_path.read_text(encoding="utf-8"))
            self.assertEqual((saved["passed"], saved["total"], saved["agent_model"]), (2, 3, "gpt-6-luna"))
            self.assertEqual(saved["cases"][1]["failures"], ["judge: nope"])

            second = self.results(**{"policy-a": False, "policy-b": True, "orders-a": True, "orders-new": False})
            second_path = evals.save_results(second, "gpt-6.1-sol", "gpt-6.1-sol", folder, datetime(2026, 10, 12, 9, 0, 0))
            previous = evals.previous_run(second_path)
            text = evals.compare(previous, second)
            self.assertIn("2026-10-11_230000.json, agent gpt-6-luna, 2/3", text)
            self.assertIn("newly failing: policy-a", text)
            self.assertIn("newly passing: policy-b", text)
            self.assertNotIn("orders-new", text)  # new case: nothing to compare with
            self.assertIn("no case changed", evals.compare(previous, first))


if __name__ == "__main__":
    unittest.main()
