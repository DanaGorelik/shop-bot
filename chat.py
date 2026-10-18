"""Step 2: talk to the agent in your terminal.

    python chat.py                         chat as the first sample customer
    python chat.py --phone +12025550178    chat as the second sample customer
    python chat.py --quiet                 hide the tool-call trace

You play the customer. The lines starting with a dot show what the agent did
behind the scenes before answering: which tools it chose and what came back.

Conversations are saved in the shop database, so if you quit and come back
as the same customer, the bot remembers what you talked about.

Commands inside the chat:  /reset  forget the conversation (also the saved copy)    /quit  exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from shopbot import config
from shopbot.agent import Agent, Step
from shopbot.embeddings import make_embedder
from shopbot.memory import ConversationStore
from shopbot.rag import VectorIndex
from shopbot.tools import ShopTools


def describe(step: Step) -> str:
    """One readable line per tool call."""
    arguments = json.dumps(step.arguments, ensure_ascii=False)
    result = step.result
    if "error" in result:
        summary = f"error: {result['error']}"
    elif "passages" in result:
        summary = ", ".join(
            f"{p['text'].splitlines()[0]} ({p['score']})" for p in result["passages"]
        )
    elif "products" in result:
        summary = ", ".join(p["name"] for p in result["products"]) or "no products"
    elif "variants" in result:
        summary = ", ".join(
            f"{v['colour']} {v['size']}: {v['in_stock']}" for v in result["variants"][:8]
        )
    elif "orders" in result:
        summary = ", ".join(f"{o['order_id']} {o['status']}" for o in result["orders"])
    elif "status" in result and "order_id" in result:
        summary = f"{result['order_id']} is {result['status']}"
    elif "item" in result:
        item = result["item"]
        number = f" #{result['reservation_id']}" if "reservation_id" in result else ""
        summary = f"{result['status']}{number}: {item['name']}, {item['colour']}, {item['size']}"
    elif "reservations" in result:
        summary = ", ".join(
            f"#{r['reservation_id']} {r['name']} {r['colour']} {r['size']}" for r in result["reservations"]
        ) or "no holds"
    else:
        summary = json.dumps(result, ensure_ascii=False)[:120]
    return f"  . {step.tool}({arguments})\n      -> {summary}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phone", default="+12025550143", help="customer to chat as")
    parser.add_argument("--quiet", action="store_true", help="hide the tool-call trace")
    arguments = parser.parse_args()

    sys.stdout.reconfigure(errors="replace")
    config.load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    from openai import OpenAI

    try:
        index = VectorIndex.load()
    except FileNotFoundError as error:
        sys.exit(str(error))
    tools = ShopTools(index, make_embedder(index.embedder_name))
    agent = Agent(OpenAI(), tools, store=ConversationStore())

    print(f"{config.SHOP_NAME} assistant  |  model: {agent.model}  |  "
          f"you are customer {arguments.phone}")
    print("Type a message, or /reset, /quit.\n")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message == "/quit":
            break
        if message == "/reset":
            agent.reset(arguments.phone)
            print("(conversation forgotten, including the saved copy)\n")
            continue
        try:
            reply = agent.reply(arguments.phone, message)
        except Exception as error:
            print(f"(request failed: {error})\n")
            continue
        if not arguments.quiet:
            if reply.restored:
                print(f"  . remembered {reply.restored} earlier messages from a previous session")
            for step in reply.steps:
                print(describe(step))
        print(f"Bot: {reply.text}\n")


if __name__ == "__main__":
    main()
