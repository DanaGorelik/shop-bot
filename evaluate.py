"""Step 4: measure the agent.

    python evaluate.py                       run every case in evals/cases.json
    python evaluate.py --only privacy        one category, or any part of a case id
    python evaluate.py --model gpt-6.1-sol   try a different model for the agent
    python evaluate.py --no-judge            rule checks only (no second model)
    python evaluate.py --show-replies        print the reply for passing cases too

Each case is a customer conversation with expectations. The script plays it
through the real agent, checks the reply, and prints a scorecard. Results are
saved under evals/results/, and each run is compared with the one before it,
so after any change you can see which answers got better or worse.

Run it before and after changing a prompt, the shop documents, the chunking
or the model. One run makes roughly 100 model calls and takes a few minutes.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from shopbot import config, db, evals
from shopbot.agent import Agent
from shopbot.embeddings import make_embedder
from shopbot.rag import VectorIndex
from shopbot.tools import ShopTools


def color_supported() -> bool:
    """True when PASS/FAIL can be shown in colour.

    Colour is switched off when the output is not a terminal (for example
    when redirected to a file) or when the NO_COLOR variable is set. Older
    Windows consoles need to be told to interpret colour codes first.
    """
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return False
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # standard output
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
        except Exception:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="run cases whose id contains this, or with this category")
    parser.add_argument("--model", help="model for the agent (default: OPENAI_MODEL or gpt-6-luna)")
    parser.add_argument("--judge-model", help="model for the judge (default: OPENAI_JUDGE_MODEL or gpt-6.1-sol)")
    parser.add_argument("--no-judge", action="store_true", help="skip the judged criteria")
    parser.add_argument("--workers", type=int, default=4, help="cases to run at the same time")
    parser.add_argument("--show-replies", action="store_true", help="show replies for passing cases too")
    arguments = parser.parse_args()

    sys.stdout.reconfigure(errors="replace")
    color = color_supported()
    config.load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    from openai import OpenAI

    try:
        customers, cases = evals.load_cases()
        index = VectorIndex.load()
    except (FileNotFoundError, ValueError) as error:
        sys.exit(str(error))
    cases = evals.select(cases, arguments.only)
    if not cases:
        sys.exit(f"No case matches '{arguments.only}'.")

    client = OpenAI()
    embedder = make_embedder(index.embedder_name)
    agent_model = arguments.model or config.chat_model()
    judge_name = None if arguments.no_judge else (arguments.judge_model or evals.judge_model())

    judge = None
    if judge_name:
        def judge(case, transcript):
            return evals.ask_judge(client, judge_name, case, transcript)

    # Every run gets a brand-new copy of the shop database. Order dates are
    # relative to today, so "delivered 9 days ago" stays true whenever you
    # run this, and test conversations never leave handoff tickets behind.
    with tempfile.TemporaryDirectory() as folder:
        db_path = Path(folder) / "shop.db"
        db.build_database(db_path=db_path)

        def make_agent():
            return Agent(client, ShopTools(index, embedder, db_path=db_path), model=agent_model)

        print(f"Evaluating {len(cases)} cases  |  agent: {agent_model}  |  "
              f"judge: {judge_name or 'off'}\n")
        results = evals.run_all(
            cases, customers, make_agent, judge, workers=arguments.workers,
            on_result=lambda result: print(evals.describe_result(result, arguments.show_replies, color)),
        )

    print("\n" + evals.summary(results))
    if arguments.only:
        print("\n(partial run: not saved)")
    else:
        path = evals.save_results(results, agent_model, judge_name)
        print(f"Saved to {path.relative_to(config.ROOT)}")
        previous = evals.previous_run(path)
        if previous:
            print("\n" + evals.compare(previous, results))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
