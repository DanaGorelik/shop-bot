"""Step 3: put the agent on WhatsApp.

    python whatsapp.py            start the webhook server on port 8000
    python whatsapp.py --port 9000

This runs a small web server on your computer. Meta's WhatsApp Cloud API
calls it whenever someone messages your test number, and the agent's answer
is sent back through the same API. Meta can only reach a public HTTPS
address, so you also run ngrok, which forwards your account's fixed address
to this server. The README walks through the setup.

The console shows every conversation with the same tool-call trace as chat.py.
"""

from __future__ import annotations

import argparse
import os
import sys

from chat import describe
from shopbot import config
from shopbot.agent import Agent
from shopbot.embeddings import make_embedder
from shopbot.memory import ConversationStore
from shopbot.rag import VectorIndex
from shopbot.tools import ShopTools
from shopbot.whatsapp import WEBHOOK_PATH, Bridge, WhatsAppClient, make_server

SAMPLE_CUSTOMER = "+12025550143"  # owns ORD-5001, ORD-5002 and ORD-5003 in the sample data
REQUIRED = {
    "OPENAI_API_KEY": "your OpenAI key",
    "WHATSAPP_ACCESS_TOKEN": "access token from the API Setup page",
    "WHATSAPP_PHONE_NUMBER_ID": "Phone number ID from the API Setup page (not the phone number)",
    "WHATSAPP_APP_SECRET": "App secret from App settings > Basic",
    "WHATSAPP_VERIFY_TOKEN": "any password-like string you choose",
}


def normalise_number(text: str) -> str:
    return "+" + "".join(character for character in text if character.isdigit())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    arguments = parser.parse_args()

    sys.stdout.reconfigure(errors="replace")
    config.load_env()
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        lines = "\n".join(f"  {name}  ({REQUIRED[name]})" for name in missing)
        sys.exit(f"These settings are missing from .env:\n{lines}\nSee the WhatsApp section of the README.")
    from openai import OpenAI

    try:
        index = VectorIndex.load()
    except FileNotFoundError as error:
        sys.exit(str(error))
    agent = Agent(OpenAI(), ShopTools(index, make_embedder(index.embedder_name)), store=ConversationStore())
    client = WhatsAppClient(
        os.environ["WHATSAPP_ACCESS_TOKEN"],
        os.environ["WHATSAPP_PHONE_NUMBER_ID"],
        api_version=os.environ.get("WHATSAPP_API_VERSION", "v25.0"),
    )

    aliases = {}
    my_number = os.environ.get("DEMO_MY_NUMBER", "").strip()
    if my_number:
        aliases[normalise_number(my_number)] = SAMPLE_CUSTOMER

    bridge = Bridge(agent, client, aliases=aliases, describe=describe)
    bridge.start()
    try:
        server = make_server(
            bridge,
            verify_token=os.environ["WHATSAPP_VERIFY_TOKEN"],
            app_secret=os.environ["WHATSAPP_APP_SECRET"],
            port=arguments.port,
        )
    except OSError as error:
        sys.exit(f"Could not start on port {arguments.port}: {error}. Try --port 9000.")

    print(f"{config.SHOP_NAME} WhatsApp server  |  model: {agent.model}")
    print(f"Listening on http://localhost:{arguments.port}{WEBHOOK_PATH}")
    if aliases:
        print(f"Demo: messages from {next(iter(aliases))} are treated as sample customer {SAMPLE_CUSTOMER}")
    print("\nNext, in a second terminal, open the tunnel:")
    print(f"    ngrok http {arguments.port}")
    print(f"Meta's Callback URL is your fixed ngrok address followed by {WEBHOOK_PATH};")
    print("it is set once in Meta's dashboard and stays the same.")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
