"""The WhatsApp channel: Meta's WhatsApp Cloud API on one side, the agent on the other.

How a message travels:

    customer's phone
        -> WhatsApp
        -> Meta sends an HTTPS POST (a "webhook") to our server      [make_server]
        -> we check the signature, answer 200 at once, queue the message
        -> a background worker asks the agent for a reply            [Bridge]
        -> we POST the reply to Meta's Graph API                     [WhatsAppClient]
        -> WhatsApp -> customer's phone

Nothing in agent.py, tools.py or rag.py knows WhatsApp exists. The channel's
whole job is to turn "a message from this phone number" into a call to
`agent.reply(customer, text)` and deliver what comes back.

Only the standard library is used, so there is nothing new to install. A
production service would normally sit behind a real web server or framework.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import queue
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

MAX_TEXT_LENGTH = 4096  # WhatsApp's limit for one text message
MAX_WEBHOOK_BYTES = 3 * 1024 * 1024  # Meta sends payloads of up to 3 MB
STALE_AFTER_SECONDS = 600  # ignore messages older than this (see Bridge.process)
WEBHOOK_PATH = "/webhook"


class WhatsAppError(Exception):
    """The Graph API refused a request or could not be reached."""


@dataclass
class Incoming:
    sender: str  # the customer's number as Meta sent it (digits); also where we reply
    message_id: str
    kind: str  # "text", "image", "audio", ...
    text: str
    timestamp: int  # seconds since 1970, when the customer sent it
    name: str = ""  # WhatsApp profile name, for the log only


# ------------------------------------------------------------- receiving ----


def verify_signature(app_secret: str, raw_body: bytes, header: str | None) -> bool:
    """Check that a webhook really comes from Meta.

    Meta signs every POST with our app secret: the X-Hub-Signature-256 header
    holds "sha256=" + HMAC-SHA256(app secret, raw request body). Anyone can
    find a public URL and POST to it, and the sender's phone number in the
    body is what decides whose orders the bot will show. Without this check a
    forged request could impersonate any customer.
    """
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header[len("sha256="):])


def extract_messages(payload: dict) -> list[Incoming]:
    """Pull customer messages out of a webhook payload.

    The same webhook also carries delivery receipts ("sent", "delivered",
    "read") under `statuses`; those are skipped. One POST may batch several
    messages. The payload is outside data, so every level is read defensively.
    """
    found: list[Incoming] = []
    if not isinstance(payload, dict) or payload.get("object") != "whatsapp_business_account":
        return found
    for entry in payload.get("entry") or []:
        for change in (entry or {}).get("changes") or []:
            if (change or {}).get("field") != "messages":
                continue
            value = change.get("value") or {}
            names = {
                contact.get("wa_id"): (contact.get("profile") or {}).get("name", "")
                for contact in value.get("contacts") or []
                if isinstance(contact, dict)
            }
            for message in value.get("messages") or []:
                if not isinstance(message, dict):
                    continue
                sender = str(message.get("from") or "")
                message_id = str(message.get("id") or "")
                if not sender or not message_id:
                    continue
                kind = str(message.get("type") or "unknown")
                text = ""
                if kind == "text":
                    text = str((message.get("text") or {}).get("body") or "")
                try:
                    timestamp = int(message.get("timestamp") or 0)
                except (TypeError, ValueError):
                    timestamp = 0
                found.append(
                    Incoming(sender, message_id, kind, text, timestamp, names.get(sender, ""))
                )
    return found


# --------------------------------------------------------------- sending ----


def split_message(text: str, limit: int = MAX_TEXT_LENGTH) -> list[str]:
    """Split a long reply into WhatsApp-sized pieces at natural breaks."""
    text = text.strip()
    parts: list[str] = []
    while len(text) > limit:
        cut = limit
        for separator in ("\n\n", "\n", " "):
            position = text.rfind(separator, 0, limit + 1)
            if position >= limit // 2:
                cut = position
                break
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


class WhatsAppClient:
    """Sends messages through the Graph API."""

    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        api_version: str = "v25.0",
        base_url: str = "https://graph.facebook.com",
    ):
        self.access_token = access_token
        self.url = f"{base_url}/{api_version}/{phone_number_id}/messages"

    def send_text(self, to: str, body: str) -> None:
        for part in split_message(body):
            self._post(
                {
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": to,
                    "type": "text",
                    "text": {"body": part},
                }
            )

    def mark_read_and_typing(self, message_id: str) -> None:
        """Show the blue ticks and a 'typing...' indicator while the agent works.

        WhatsApp clears the indicator when our reply arrives, or after 25 seconds.
        """
        self._post(
            {
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id,
                "typing_indicator": {"type": "text"},
            }
        )

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                pass
            hint = ""
            if error.code == 401:
                hint = " (the access token has probably expired; generate a new one)"
            raise WhatsAppError(f"WhatsApp API answered {error.code}: {detail}{hint}") from None
        except urllib.error.URLError as error:
            raise WhatsAppError(f"Could not reach the WhatsApp API: {error.reason}") from None


# ---------------------------------------------------------------- bridge ----


class Bridge:
    """Connects incoming WhatsApp messages to the agent.

    The web server must answer Meta within seconds, but an agent turn with a
    few tool calls can take longer. So the server only queues the message, and
    one background worker takes messages off the queue and handles them one at
    a time. One worker keeps each customer's messages in order with no locks;
    the cost is that customers wait behind each other. A real service would
    run a queue per customer or a job system.
    """

    NOT_TEXT = "Sorry, I can only read text messages for now. Could you type your question?"
    FAILED = "Sorry, something went wrong on our side. Please try again in a moment."

    def __init__(
        self,
        agent,
        client,
        aliases: dict[str, str] | None = None,
        log=print,
        describe=str,
        clock=time.time,
    ):
        self.agent = agent
        self.client = client
        self.aliases = aliases or {}
        self.log = log  # prints one line to the server console
        self.describe = describe  # turns a tool-call Step into a readable line
        self.clock = clock
        self._queue: queue.Queue[Incoming] = queue.Queue()
        self._seen: OrderedDict[str, None] = OrderedDict()

    def start(self) -> None:
        threading.Thread(target=self._work, daemon=True, name="whatsapp-worker").start()

    def submit(self, incoming: Incoming) -> None:
        self._queue.put(incoming)

    def wait_until_idle(self) -> None:
        self._queue.join()

    def _work(self) -> None:
        while True:
            incoming = self._queue.get()
            try:
                self.process(incoming)
            except Exception as error:  # the worker must survive anything
                self.log(f"  ! unexpected error: {error!r}")
            finally:
                self._queue.task_done()

    def customer_for(self, sender: str) -> str:
        """The identity the agent's tools see: the sender's number as +digits.

        This comes from Meta's signed webhook, never from the message text.
        `aliases` exists only for the demo, so your own phone can act as a
        sample customer who has orders.
        """
        number = "+" + "".join(character for character in sender if character.isdigit())
        return self.aliases.get(number, number)

    def process(self, incoming: Incoming) -> None:
        # Meta re-sends a webhook when it is unsure we received it, so the same
        # message can arrive twice. Answer each message id once.
        if incoming.message_id in self._seen:
            return
        self._seen[incoming.message_id] = None
        if len(self._seen) > 2000:
            self._seen.popitem(last=False)

        # Meta keeps retrying undelivered webhooks for days. After the server
        # has been off, do not answer yesterday's messages.
        age = self.clock() - incoming.timestamp
        if incoming.timestamp and age > STALE_AFTER_SECONDS:
            self.log(f"  (skipped a message from {incoming.sender} that is {int(age // 60)} min old)")
            return

        customer = self.customer_for(incoming.sender)
        who = f"{incoming.name} " if incoming.name else ""
        self.log(f"\n{who}{incoming.sender}: {incoming.text or '[' + incoming.kind + ']'}")

        if incoming.kind != "text" or not incoming.text.strip():
            self._send(incoming.sender, self.NOT_TEXT)
            return
        if incoming.text.strip().lower() == "/reset":
            self.agent.reset(customer)
            self._send(incoming.sender, "Conversation cleared.")
            return

        try:
            self.client.mark_read_and_typing(incoming.message_id)
        except WhatsAppError as error:
            self.log(f"  ! could not show typing indicator: {error}")

        try:
            reply = self.agent.reply(customer, incoming.text)
        except Exception as error:
            self.log(f"  ! agent failed: {error!r}")
            self._send(incoming.sender, self.FAILED)
            return
        if getattr(reply, "restored", 0):
            self.log(f"  . remembered {reply.restored} earlier messages from a previous session")
        for step in reply.steps:
            try:
                self.log(self.describe(step))
            except Exception:  # the trace is for us; never let it block the reply
                self.log(f"  . {step.tool}")
        self._send(incoming.sender, reply.text or self.FAILED)

    def _send(self, to: str, text: str) -> None:
        try:
            self.client.send_text(to, text)
            self.log(f"Bot: {text}")
        except WhatsAppError as error:
            self.log(f"  ! could not send reply: {error}")


# ---------------------------------------------------------------- server ----


def make_server(bridge: Bridge, verify_token: str, app_secret: str, port: int, host: str = "127.0.0.1"):
    """Build the HTTP server that Meta calls. Call .serve_forever() to run it."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/":
                return self._answer(200, "Shop bot webhook server is running.")
            if url.path != WEBHOOK_PATH:
                return self._answer(404, "Not found")
            # One-time handshake: when you save the callback URL in Meta's
            # dashboard, Meta calls it with the verify token you typed there
            # and expects the random "challenge" value echoed back.
            query = parse_qs(url.query)
            mode = query.get("hub.mode", [""])[0]
            token = query.get("hub.verify_token", [""])[0]
            challenge = query.get("hub.challenge", [""])[0]
            if mode == "subscribe" and hmac.compare_digest(token.encode(), verify_token.encode()):
                bridge.log("Webhook verified by Meta.")
                return self._answer(200, challenge)
            return self._answer(403, "Verify token does not match")

        def do_POST(self):
            if urlsplit(self.path).path != WEBHOOK_PATH:
                return self._answer(404, "Not found")
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._answer(400, "Bad Content-Length")
            if length <= 0 or length > MAX_WEBHOOK_BYTES:
                return self._answer(413, "Body missing or too large")
            raw_body = self.rfile.read(length)
            if not verify_signature(app_secret, raw_body, self.headers.get("X-Hub-Signature-256")):
                bridge.log("  ! rejected a webhook with a missing or wrong signature")
                return self._answer(401, "Bad signature")
            try:
                payload = json.loads(raw_body)
            except ValueError:
                return self._answer(400, "Body is not JSON")
            for incoming in extract_messages(payload):
                bridge.submit(incoming)
            # Answer immediately; the reply to the customer is sent later by
            # the worker through the Graph API, not in this HTTP response.
            self._answer(200, "ok")

        def _answer(self, status: int, text: str) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *arguments) -> None:
            pass  # the Bridge prints the interesting lines

    return ThreadingHTTPServer((host, port), Handler)
