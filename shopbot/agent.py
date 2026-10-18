"""The agent loop.

A chatbot sends the conversation to an LLM and returns the text it writes.
An agent adds a loop around that:

    1. Send the conversation and the list of tools to the LLM.
    2. If the LLM answers with tool calls, run them and append the results.
    3. Go back to 1, until the LLM answers with plain text.

Everything "agentic" about this bot is in `Agent.reply` below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

from . import config
from .memory import ConversationStore, restored_items
from .tools import TOOL_SCHEMAS, ShopTools


def system_prompt(today: date | None = None) -> str:
    today = today or date.today()
    return f"""\
You are the customer assistant for {config.SHOP_NAME}, an independent clothing shop with one \
physical store and an online store. You talk with customers on WhatsApp. They write to ask \
about products, sizes and availability, their orders, and how returns, delivery and the shop \
itself work. Today is {today.strftime('%A, %d %B %Y')}.

Where your facts come from
You have no built-in knowledge of this shop. Everything specific to it (policies, prices, \
stock, opening hours, order details) has to come from a tool result in this conversation. \
Look things up before you answer, even when the question seems simple, because customers act \
on what you tell them. If the tools do not give you the answer, say that you are not sure and \
offer to pass the question to the team; a wrong policy or a wrong stock answer costs the shop \
more than "let me check with the team".

Stock and prices change during the day, so check them again if the customer returns to a \
product later in the conversation instead of repeating what an earlier result said.

Using the tools
Policy and information questions go to search_knowledge. Requests for something to buy go to \
search_products, and a question about a particular size or colour goes to check_stock. \
Questions that mix both (for example whether a particular order can still be returned) need \
both the order details and the policy text, so look up each. Write tool queries in English \
even when the customer writes in another language, because the shop's data is in English.

You can only see the orders of the customer you are chatting with. If an order number is not \
found, ask them to check the number rather than guessing.

Holding items at the shop
You can hold an in-stock item at the shop for a customer for 48 hours, free, one colour and \
size per hold. Placing a hold takes two steps on purpose. First call reserve_item, which checks \
the item but holds nothing. Then tell the customer exactly what will be held (product, colour, \
size) and for how long, and ask them to confirm. Only if their next message agrees, call \
confirm_reservation. If they hesitate, change their mind or ask something else, do not confirm. \
Never tell a customer an item is held until confirm_reservation has succeeded, and pass on the \
time the hold ends. Customers can ask what they have on hold (list_my_reservations) or cancel a \
hold (cancel_reservation).

What you cannot do
You cannot place, change or cancel orders, issue refunds, create return labels or make \
exceptions to a policy. When a customer needs one of those, or asks for a person, or is \
clearly unhappy, use handoff_to_human and tell them a team member will continue in this chat.

How to write
This is WhatsApp, so write the way a helpful shop assistant would text: a few short sentences, \
plain text, no headings, tables or markdown symbols. When listing a few products, put each on \
its own line with its price. Reply in the language the customer writes in. Prices are in \
shekels; write them like ₪249. Answer what was asked first, then add a detail only if the \
customer will need it for their next step."""


@dataclass
class Step:
    """One tool call the agent made, kept so you can see how it reached its answer."""

    tool: str
    arguments: dict
    result: dict


@dataclass
class Reply:
    text: str
    steps: list[Step] = field(default_factory=list)
    restored: int = 0  # earlier messages loaded from long-term memory for this reply


class Agent:
    def __init__(
        self,
        client,
        tools: ShopTools,
        model: str | None = None,
        max_steps: int = 6,
        max_turns: int = 12,
        store: ConversationStore | None = None,
    ):
        self.client = client
        self.tools = tools
        self.model = model or config.chat_model()
        self.max_steps = max_steps  # tool rounds allowed per customer message
        self.max_turns = max_turns  # past exchanges remembered per customer
        # Two kinds of memory:
        # - _history, in RAM: customer -> list of turns. A turn is every item
        #   produced while answering one message (the message, the model's tool
        #   calls, the tool results, the final text). Full detail, lost on exit.
        # - store, in the database (optional): just the messages and final
        #   answers, which survive a restart. See memory.py.
        self._history: dict[str, list[list]] = {}
        self.store = store
        # How many messages each customer has sent. Tools receive it, which is
        # how confirm_reservation can tell that the customer has replied.
        self._turn_numbers: dict[str, int] = {}

    def reset(self, customer_phone: str) -> None:
        self._history.pop(customer_phone, None)
        if self.store:
            self.store.clear(customer_phone)
        forget = getattr(self.tools, "forget_pending", None)
        if forget:
            forget(customer_phone)

    def reply(self, customer_phone: str, message: str) -> Reply:
        turn_number = self._turn_numbers.get(customer_phone, 0) + 1
        self._turn_numbers[customer_phone] = turn_number
        restored = self._restore(customer_phone)
        past = [item for turn in self._history.get(customer_phone, []) for item in turn]
        turn: list = [{"role": "user", "content": message}]
        steps: list[Step] = []
        text = ""

        for _ in range(self.max_steps):
            response = self._ask_model(past + turn)
            # Keep everything the model produced. Besides text and tool calls
            # this includes its reasoning items, which the API expects back
            # together with the tool results.
            turn += response.output
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                text = response.output_text
                break
            for call in calls:
                result = self.tools.run(call.name, call.arguments, customer_phone, turn_number)
                steps.append(Step(call.name, _safe_json(call.arguments), result))
                turn.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )
        else:
            # Still calling tools after max_steps rounds: stop the loop and
            # make the model answer with what it has.
            response = self._ask_model(past + turn, tool_choice="none")
            turn += response.output
            text = response.output_text

        # Forget whole turns only, so a tool call is never separated from its result.
        turns = self._history.setdefault(customer_phone, [])
        turns.append(turn)
        del turns[: -self.max_turns]
        if self.store:
            self.store.save_exchange(customer_phone, message, text)
        return Reply(text=text, steps=steps, restored=restored)

    def _restore(self, customer_phone: str) -> int:
        """After a restart, load this customer's recent exchanges from the database.

        Only happens when RAM holds nothing for them, so during a live
        conversation the detailed in-memory history is used instead.
        """
        if not self.store or customer_phone in self._history:
            return 0
        messages = self.store.load(customer_phone, max_messages=2 * self.max_turns)
        if not messages:
            return 0
        self._history[customer_phone] = [restored_items(messages)]
        return len(messages)

    def _ask_model(self, items: list, **extra):
        return self.client.responses.create(
            model=self.model,
            instructions=system_prompt(),
            tools=TOOL_SCHEMAS,
            input=items,
            **extra,
        )


def _safe_json(text: str) -> dict:
    try:
        return json.loads(text or "{}")
    except json.JSONDecodeError:
        return {"_unparsed": text}
