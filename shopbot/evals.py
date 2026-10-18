"""Evaluation: replay customer conversations and grade what the agent does.

Chatting with the bot tells you how it behaved once. An evaluation set tells
you how it behaves across everything you care about, every time you change
something. Each case in evals/cases.json is a conversation plus expectations,
and each reply is graded in two ways:

* Rule checks, done in code. Did it call the right tools? Does the reply
  contain the tracking number? Does it avoid leaking another customer's
  order? These are exact and free, but they can only look for literal text.

* A judge, which is a second LLM call. It reads the reply next to a criterion
  written in plain language ("says the item cannot be returned") and decides
  pass or fail. This handles meaning, wording and other languages, at the
  price of being a model's opinion: read its reasons, especially on failures.

A case passes only if every rule check passes and the judge agrees.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from . import config
from .tools import TOOL_SCHEMAS

CASES_PATH = config.ROOT / "evals" / "cases.json"
RESULTS_DIR = config.ROOT / "evals" / "results"

TOOL_NAMES = {schema["name"] for schema in TOOL_SCHEMAS}
EXPECTATION_FIELDS = (
    "must_call", "must_not_call", "must_call_any", "max_tool_calls",
    "must_contain", "must_not_contain", "criteria",
)
KNOWN_FIELDS = {"id", "category", "customer", "messages", *EXPECTATION_FIELDS}

JUDGE_INSTRUCTIONS = """\
You grade replies written by a customer-service assistant for a clothing shop. You are given \
the conversation, the assistant's final reply, and a criterion written by the shop owner. \
Decide whether the final reply satisfies the criterion.

Grade only against the criterion. The criterion is the source of truth about the shop, so do \
not use your own ideas of what shops usually do. Extra detail, a different tone or a different \
length is not a reason to fail unless the criterion says so. When the criterion says the reply \
must not do something and the reply does it, that is a fail. The reply may be in any language; \
grade its meaning.

Write one or two sentences of reasoning first, then the verdict."""

JUDGE_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "pass": {"type": "boolean"},
            },
            "required": ["reasoning", "pass"],
            "additionalProperties": False,
        },
    }
}


def judge_model() -> str:
    # The judge should be at least as capable as the model it grades.
    return os.environ.get("OPENAI_JUDGE_MODEL", "gpt-6.1-sol")


@dataclass
class CaseResult:
    id: str
    category: str
    passed: bool
    failures: list[str] = field(default_factory=list)  # one line per failed check
    answer: str = ""
    tools: list[str] = field(default_factory=list)  # tool names, in call order
    judge_reason: str = ""
    seconds: float = 0.0


# ---------------------------------------------------------------- cases ----


def load_cases(path: Path = CASES_PATH) -> tuple[dict[str, str], list[dict]]:
    """Read and validate the cases file. Returns (customers, cases)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    customers, cases = data["customers"], data["cases"]
    seen = set()
    for case in cases:
        name = case.get("id", "(case without id)")
        if name in seen:
            raise ValueError(f"Duplicate case id: {name}")
        seen.add(name)
        unknown = set(case) - KNOWN_FIELDS
        if unknown:
            raise ValueError(f"{name}: unknown field(s) {sorted(unknown)}")
        if not case.get("category"):
            raise ValueError(f"{name}: missing category")
        messages = case.get("messages")
        if not messages or not all(isinstance(m, str) and m.strip() for m in messages):
            raise ValueError(f"{name}: 'messages' must be a non-empty list of texts")
        if case.get("customer", "new") not in customers:
            raise ValueError(f"{name}: unknown customer '{case['customer']}'")
        for key in ("must_call", "must_not_call", "must_call_any"):
            wrong = set(case.get(key, [])) - TOOL_NAMES
            if wrong:
                raise ValueError(f"{name}: unknown tool(s) in {key}: {sorted(wrong)}")
        if not any(key in case for key in EXPECTATION_FIELDS):
            raise ValueError(f"{name}: has no expectations, so it could never fail")
    return customers, cases


def select(cases: list[dict], only: str | None) -> list[dict]:
    """Keep cases whose id contains `only` or whose category equals it."""
    if not only:
        return cases
    return [case for case in cases if only in case["id"] or only == case["category"]]


# --------------------------------------------------------------- checks ----


def check_rules(case: dict, answer: str, tools: list[str]) -> list[str]:
    """Return a line for every rule the reply breaks; empty means all passed."""
    failures = []
    text = answer.casefold()
    for name in case.get("must_call", []):
        if name not in tools:
            failures.append(f"did not call {name}")
    for name in case.get("must_not_call", []):
        if name in tools:
            failures.append(f"called {name}, which it should not")
    any_of = case.get("must_call_any")
    if any_of and not set(any_of) & set(tools):
        failures.append(f"called none of {', '.join(any_of)}")
    limit = case.get("max_tool_calls")
    if limit is not None and len(tools) > limit:
        failures.append(f"made {len(tools)} tool call(s), at most {limit} expected")
    for wanted in case.get("must_contain", []):
        options = wanted if isinstance(wanted, list) else [wanted]
        if not any(option.casefold() in text for option in options):
            failures.append(f"reply does not contain {' or '.join(repr(o) for o in options)}")
    for banned in case.get("must_not_contain", []):
        if banned.casefold() in text:
            failures.append(f"reply contains {banned!r}")
    return failures


def ask_judge(client, model: str, case: dict, transcript: list[tuple[str, str]]) -> tuple[bool, str]:
    """Have a second model grade the final reply against the case's criterion."""
    lines = ["Conversation:"]
    for number, (message, reply) in enumerate(transcript):
        lines.append(f"Customer: {message}")
        if number < len(transcript) - 1:
            lines.append(f"Assistant: {reply}")
    lines += ["", "Final reply to grade:", transcript[-1][1], "", "Criterion:", case["criteria"]]
    response = client.responses.create(
        model=model,
        instructions=JUDGE_INSTRUCTIONS,
        input="\n".join(lines),
        text=JUDGE_FORMAT,
    )
    verdict = json.loads(response.output_text)
    return bool(verdict["pass"]), str(verdict["reasoning"])


# ---------------------------------------------------------------- running ----


def run_case(case: dict, customer: str, agent, judge=None) -> CaseResult:
    """Play one conversation through a fresh agent and grade the last reply.

    `judge` is a function (case, transcript) -> (passed, reason), or None to
    skip the judged criterion and apply the rule checks only.
    """
    started = time.monotonic()
    result = CaseResult(id=case["id"], category=case["category"], passed=False)
    transcript: list[tuple[str, str]] = []
    try:
        for message in case["messages"]:
            reply = agent.reply(customer, message)
            transcript.append((message, reply.text))
            result.tools += [step.tool for step in reply.steps]
        result.answer = transcript[-1][1]
    except Exception as error:
        result.failures.append(f"agent error: {error!r}")
        result.seconds = round(time.monotonic() - started, 1)
        return result

    result.failures += check_rules(case, result.answer, result.tools)
    if judge and case.get("criteria"):
        try:
            agreed, result.judge_reason = judge(case, transcript)
            if not agreed:
                result.failures.append(f"judge: {result.judge_reason}")
        except Exception as error:
            result.failures.append(f"judge error: {error!r}")
    result.passed = not result.failures
    result.seconds = round(time.monotonic() - started, 1)
    return result


def run_all(cases, customers, make_agent, judge=None, workers: int = 4, on_result=None):
    """Run the cases, a few at a time, and return results in case order.

    `make_agent()` must return a new Agent each time, so that no case can see
    another case's conversation memory.
    """

    def one(case):
        customer = customers[case.get("customer", "new")]
        return run_case(case, customer, make_agent(), judge)

    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for result in pool.map(one, cases):
            results.append(result)
            if on_result:
                on_result(result)
    return results


# -------------------------------------------------------------- reporting ----


GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


def describe_result(result: CaseResult, show_answer: bool = False, color: bool = False) -> str:
    verdict = "PASS" if result.passed else "FAIL"
    if color:
        verdict = f"{GREEN if result.passed else RED}{verdict}{RESET}"
    lines = [f"{verdict}  {result.id}  ({result.seconds}s)"]
    if not result.passed:
        lines += [f"        - {failure}" for failure in result.failures]
    if not result.passed or show_answer:
        lines.append(f"        tools:  {', '.join(result.tools) or 'none'}")
        answer = result.answer.replace("\n", "\n                ") or "(no reply)"
        lines.append(f"        reply:  {answer}")
    return "\n".join(lines)


def summary(results: list[CaseResult]) -> str:
    total, passed = Counter(), Counter()
    for result in results:
        total[result.category] += 1
        passed[result.category] += result.passed
    width = max(len(category) for category in total)
    lines = ["By category"]
    for category in total:
        lines.append(f"  {category:<{width}}  {passed[category]}/{total[category]}")
    done = sum(passed.values())
    count = len(results)
    lines.append(f"\nTotal: {done}/{count} passed ({round(100 * done / count)}%)")
    return "\n".join(lines)


def save_results(results, agent_model: str, judge_name: str | None, folder: Path = RESULTS_DIR, now=None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now()
    path = folder / f"{now:%Y-%m-%d_%H%M%S}.json"
    payload = {
        "run_at": now.isoformat(timespec="seconds"),
        "agent_model": agent_model,
        "judge_model": judge_name,
        "passed": sum(result.passed for result in results),
        "total": len(results),
        "cases": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def previous_run(current: Path) -> dict | None:
    """The most recent saved run before `current`, if there is one."""
    earlier = sorted(p for p in current.parent.glob("*.json") if p.name < current.name)
    if not earlier:
        return None
    data = json.loads(earlier[-1].read_text(encoding="utf-8"))
    data["file"] = earlier[-1].name
    return data


def compare(previous: dict, results: list[CaseResult]) -> str:
    """Say which cases changed between the previous run and this one.

    This is the reason to keep results: after a change you want to know not
    only the score, but exactly which answers got better and which got worse.
    """
    before = {case["id"]: case["passed"] for case in previous["cases"]}
    worse = [r.id for r in results if before.get(r.id) is True and not r.passed]
    better = [r.id for r in results if before.get(r.id) is False and r.passed]
    lines = [
        f"Compared with the previous run ({previous['file']}, agent {previous['agent_model']}, "
        f"{previous['passed']}/{previous['total']}):"
    ]
    if not worse and not better:
        lines.append("  no case changed between pass and fail")
    if worse:
        lines.append("  newly failing: " + ", ".join(worse))
    if better:
        lines.append("  newly passing: " + ", ".join(better))
    return "\n".join(lines)
